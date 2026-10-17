import asyncio
import importlib.util
import sys
import unittest
from argparse import Namespace
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

_MODULE_PATH = Path(__file__).with_name("dual_frontend.py")
_SPEC = importlib.util.spec_from_file_location("dual_frontend", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
dual_frontend = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dual_frontend)


def module(name, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    return result


class DualFrontendTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.engine = SimpleNamespace(
            model_config=object(), get_supported_tasks=AsyncMock(return_value=("generate",))
        )
        self.app = SimpleNamespace(state=SimpleNamespace())
        self.grpc = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
        self.health = SimpleNamespace(set_not_serving=Mock())
        self.shutdown = AsyncMock()
        self.sock = Mock()
        self.args = Namespace(
            smg_grpc_host=None, smg_grpc_port=50051, host="127.0.0.1", port=8000,
            enable_ssl_refresh=True, uvicorn_log_level="debug", disable_uvicorn_access_log=True,
            ssl_keyfile="key", ssl_certfile="cert", ssl_ca_certs="ca", ssl_cert_reqs=2,
            ssl_ciphers="cipher", h11_max_incomplete_event_size=123, h11_max_header_count=45,
            tool_parser_plugin=None, reasoning_parser_plugin=None,
        )

        async def init(engine, state, args, tasks):
            self.assertIs(engine, self.engine)
            self.assertIs(state, self.app.state)
            self.assertEqual(tasks, ("generate",))
            self.events.append("app_state_initialized")

        def create(engine, host, port, app_state):
            self.assertEqual(self.events, ["app_state_initialized"])
            self.assertIs(engine, self.engine)
            self.assertIs(app_state, self.app.state)
            self.events.append("grpc_created")
            return self.grpc, self.health, f"{host}:{port}"

        self.init = AsyncMock(side_effect=init)
        self.build = Mock(return_value=self.app)
        self.http = AsyncMock(return_value=self.shutdown)
        self.create = Mock(side_effect=create)
        self.modules = {
            "vllm": module("vllm", envs=SimpleNamespace(VLLM_HTTP_TIMEOUT_KEEP_ALIVE=37)),
            "vllm.entrypoints.launchers.app": module("app", build_app=self.build),
            "vllm.entrypoints.launchers.api_server.app_state": module("state", init_app_state=self.init),
            "vllm.entrypoints.launchers.launcher": module("launcher", serve_http=self.http),
            "vllm.entrypoints.launchers.utils.server_utils": module(
                "utils", get_uvicorn_log_config=Mock(return_value={"version": 1})
            ),
        }

    def serve(self):
        return asyncio.run(dual_frontend._serve_frontends(
            self.engine, "127.0.0.1:8000", self.sock, self.args, custom_option=True
        ))

    def test_shared_app_state_after_init_and_http_settings(self):
        with patch.dict(sys.modules, self.modules), patch.object(
            dual_frontend, "_create_grpc_server", self.create
        ):
            self.assertIs(self.serve(), self.shutdown)
        self.build.assert_called_once_with(self.args, ("generate",), self.engine.model_config)
        self.init.assert_awaited_once()
        self.create.assert_called_once_with(self.engine, "127.0.0.1", 50051, self.app.state)
        self.http.assert_awaited_once_with(
            self.app, sock=self.sock, enable_ssl_refresh=True, host="127.0.0.1", port=8000,
            log_level="debug", access_log=False, timeout_keep_alive=37, ssl_keyfile="key",
            ssl_certfile="cert", ssl_ca_certs="ca", ssl_cert_reqs=2, ssl_ciphers="cipher",
            h11_max_incomplete_event_size=123, h11_max_header_count=45,
            log_config={"version": 1}, custom_option=True,
        )
        self.grpc.start.assert_awaited_once()
        self.health.set_not_serving.assert_called_once()
        self.grpc.stop.assert_awaited_once_with(grace=5.0)

    def test_app_state_init_failure_does_not_start_frontends(self):
        self.init.side_effect = ValueError("app state failed")
        with patch.dict(sys.modules, self.modules), patch.object(
            dual_frontend, "_create_grpc_server", self.create
        ), self.assertRaisesRegex(ValueError, "app state failed"):
            self.serve()
        self.create.assert_not_called()
        self.http.assert_not_awaited()

    def test_http_failure_stops_grpc(self):
        self.http.side_effect = RuntimeError("http failed")
        with patch.dict(sys.modules, self.modules), patch.object(
            dual_frontend, "_create_grpc_server", self.create
        ), self.assertRaisesRegex(RuntimeError, "http failed"):
            self.serve()
        self.grpc.stop.assert_awaited_once_with(grace=5.0)
        self.health.set_not_serving.assert_called_once()

    def test_worker_shutdown_order(self):
        @asynccontextmanager
        async def build_engine(args):
            self.events.append("engine_enter")
            yield self.engine
            self.events.append("engine_exit")

        async def shutdown():
            self.events.append("http_shutdown")

        async def frontends(*args, **kwargs):
            self.assertIs(args[0], self.engine)
            return shutdown()

        self.modules.update({
            "vllm.entrypoints.launchers.api_server.entry": module(
                "entry", build_async_engine_client=build_engine
            ),
            "vllm.reasoning": module("reasoning", ReasoningParserManager=Mock()),
            "vllm.tool_parsers": module("tools", ToolParserManager=Mock()),
        })
        with patch.dict(sys.modules, self.modules), patch.object(
            dual_frontend, "_serve_frontends", frontends
        ):
            asyncio.run(dual_frontend.run_server_worker("address", self.sock, self.args))
        self.assertEqual(self.events, ["engine_enter", "engine_exit", "http_shutdown"])
        self.sock.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
