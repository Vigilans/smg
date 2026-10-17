"""Run vLLM's HTTP API and the SMG gRPC service on one AsyncLLM."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import socket
import time
from typing import Any

logger = logging.getLogger(__name__)


def _create_grpc_server(engine: Any, host: str, port: int, app_state: Any):
    import grpc
    from grpc_health.v1 import health_pb2_grpc
    from grpc_reflection.v1alpha import reflection
    from smg_grpc_proto import vllm_engine_pb2, vllm_engine_pb2_grpc

    from .health_servicer import VllmHealthServicer
    from .servicer import VllmEngineServicer

    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
            ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
            ("grpc.keepalive_permit_without_calls", True),
        ]
    )
    vllm_engine_pb2_grpc.add_VllmEngineServicer_to_server(
        VllmEngineServicer(
            engine,
            time.time(),
            getattr(app_state, "openai_serving_decisions", None),
            getattr(app_state, "serving_structured_decisions", None),
        ),
        server,
    )
    health_servicer = VllmHealthServicer(engine)
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    service_names = (
        vllm_engine_pb2.DESCRIPTOR.services_by_name["VllmEngine"].full_name,
        "grpc.health.v1.Health",
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    address = f"{host}:{port}"
    if server.add_insecure_port(address) == 0:
        raise RuntimeError(f"Failed to bind SMG gRPC server to {address}")
    return server, health_servicer, address


async def _serve_frontends(
    engine: Any,
    listen_address: str,
    sock: socket.socket,
    args: argparse.Namespace,
    **uvicorn_kwargs: Any,
):
    from vllm import envs
    from vllm.entrypoints.launchers.app import build_app
    from vllm.entrypoints.launchers.api_server.app_state import init_app_state
    from vllm.entrypoints.launchers.launcher import serve_http
    from vllm.entrypoints.launchers.utils.server_utils import get_uvicorn_log_config

    log_config = get_uvicorn_log_config(args)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config
    supported_tasks = await engine.get_supported_tasks()
    logger.info("Supported tasks: %s", supported_tasks)
    app = build_app(args, supported_tasks, engine.model_config)
    await init_app_state(engine, app.state, args, supported_tasks)

    grpc_host = args.smg_grpc_host or args.host or "0.0.0.0"
    grpc_server, health_servicer, grpc_address = _create_grpc_server(
        engine, grpc_host, args.smg_grpc_port, app.state
    )
    await grpc_server.start()
    logger.info("SMG gRPC server started on %s", grpc_address)

    try:
        logger.info("Starting vLLM server on %s", listen_address)
        return await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            ssl_ciphers=args.ssl_ciphers,
            h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
            h11_max_header_count=args.h11_max_header_count,
            **uvicorn_kwargs,
        )
    finally:
        health_servicer.set_not_serving()
        await grpc_server.stop(grace=5.0)
        logger.info("SMG gRPC server stopped")


async def run_server_worker(
    listen_address: str,
    sock: socket.socket,
    args: argparse.Namespace,
    **uvicorn_kwargs: Any,
) -> None:
    from vllm.entrypoints.launchers.api_server.entry import build_async_engine_client
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)
    if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:
        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)

    async with build_async_engine_client(args) as engine:
        shutdown_task = await _serve_frontends(
            engine, listen_address, sock, args, **uvicorn_kwargs
        )
    try:
        await shutdown_task
    finally:
        sock.close()


async def run_server(args: argparse.Namespace, **uvicorn_kwargs: Any) -> None:
    from vllm.entrypoints.launchers.api_server.entry import setup_server
    from vllm.utils.system_utils import decorate_logs

    decorate_logs("APIServer", skip_if_decorated=True)

    def _interrupt_init(*_: Any) -> None:
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, _interrupt_init)
    listen_address, sock = setup_server(args, reuse_port=False)
    await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)


def make_arg_parser():
    from vllm.entrypoints.common.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        description="vLLM HTTP API with an SMG gRPC frontend on the same engine."
    )
    parser = make_arg_parser(parser)
    parser.add_argument(
        "--smg-grpc-host",
        default=None,
        help="SMG gRPC bind host; defaults to the HTTP --host value.",
    )
    parser.add_argument(
        "--smg-grpc-port",
        type=int,
        required=True,
        help="SMG gRPC bind port.",
    )
    return parser


def main() -> None:
    import uvloop
    from vllm.entrypoints.common.cli_args import validate_parsed_serve_args
    from vllm.entrypoints.common.startup import cli_env_setup

    cli_env_setup()
    args = make_arg_parser().parse_args()
    if args.model_tag is not None:
        args.model = args.model_tag
    if args.grpc:
        raise ValueError("--grpc replaces HTTP; omit it when using the dual frontend")
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
