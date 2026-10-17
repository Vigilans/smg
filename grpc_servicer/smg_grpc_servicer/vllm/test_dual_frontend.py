import asyncio
import importlib.util
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

_MODULE_PATH = Path(__file__).with_name("dual_frontend.py")
_SPEC = importlib.util.spec_from_file_location("dual_frontend", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
dual_frontend = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dual_frontend)


def test_frontends_share_one_engine_and_stop_grpc_with_http():
    async def run():
        engine = object()
        grpc_server = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
        health = SimpleNamespace(set_not_serving=Mock())
        shutdown = AsyncMock()
        args = Namespace(smg_grpc_host=None, smg_grpc_port=50051, host="127.0.0.1")

        with patch.object(
            dual_frontend,
            "_create_grpc_server",
            return_value=(grpc_server, health, "127.0.0.1:50051"),
        ) as create_grpc:
            build_http = AsyncMock(return_value=shutdown)
            result = await dual_frontend._serve_frontends(
                engine,
                "127.0.0.1:8000",
                Mock(),
                args,
                build_http=build_http,
            )

        assert result is shutdown
        create_grpc.assert_called_once_with(engine, "127.0.0.1", 50051)
        assert build_http.await_args.args[0] is engine
        grpc_server.start.assert_awaited_once()
        health.set_not_serving.assert_called_once()
        grpc_server.stop.assert_awaited_once_with(grace=5.0)

    asyncio.run(run())
