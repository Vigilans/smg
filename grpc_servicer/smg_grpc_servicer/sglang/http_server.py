"""Run SGLang HTTP and SMG-native gRPC against one TokenizerManager."""

import argparse
from collections.abc import AsyncIterator
from concurrent import futures
from contextlib import asynccontextmanager

import grpc
from grpc_health.v1 import health_pb2_grpc
from grpc_reflection.v1alpha import reflection
from smg_grpc_proto import sglang_scheduler_pb2, sglang_scheduler_pb2_grpc

from smg_grpc_servicer.sglang.health_servicer import SGLangHealthServicer
from smg_grpc_servicer.sglang.http_adapter import HttpTokenizerManagerAdapter
from smg_grpc_servicer.sglang.metadata import model_info
from smg_grpc_servicer.sglang.servicer import SGLangSchedulerServicer


class SharedGrpcHandle:
    def __init__(self, server, servicer):
        self.server = server
        self.servicer = servicer

    async def stop(self):
        self.servicer.health_servicer.set_not_serving()
        await self.servicer.shutdown()
        await self.server.stop(5)


async def start_shared_grpc(tokenizer_manager, server_args, scheduler_info, host, port):
    adapter = HttpTokenizerManagerAdapter(tokenizer_manager)
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
            ("grpc.http2.min_ping_interval_without_data_ms", 10000),
            ("grpc.keepalive_permit_without_calls", True),
        ],
    )
    health = SGLangHealthServicer(adapter, scheduler_info)
    health_pb2_grpc.add_HealthServicer_to_server(health, server)
    servicer = SGLangSchedulerServicer(
        request_manager=adapter,
        server_args=server_args,
        model_config=tokenizer_manager.model_config,
        model_info=model_info(tokenizer_manager, server_args),
        scheduler_info=scheduler_info,
        health_servicer=health,
        tokenizer=tokenizer_manager.tokenizer,
    )
    sglang_scheduler_pb2_grpc.add_SglangSchedulerServicer_to_server(servicer, server)
    service_names = (
        sglang_scheduler_pb2.DESCRIPTOR.services_by_name["SglangScheduler"].full_name,
        "grpc.health.v1.Health",
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)
    if server.add_insecure_port(f"{host}:{port}") == 0:
        raise RuntimeError(f"failed to bind SMG gRPC listener on {host}:{port}")
    await server.start()
    health.set_serving()
    return SharedGrpcHandle(server, servicer)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smg-grpc-host", default="127.0.0.1")
    parser.add_argument("--smg-grpc-port", type=int, required=True)
    args, sglang_argv = parser.parse_known_args(argv)
    if not 1 <= args.smg_grpc_port <= 65535:
        parser.error("--smg-grpc-port must be between 1 and 65535")

    from sglang.srt.server_args import prepare_server_args

    server_args = prepare_server_args(sglang_argv)
    from sglang.srt.entrypoints import http_server

    original_lifespan = http_server.app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app) -> AsyncIterator[None]:
        async with original_lifespan(app):
            state = http_server._global_state
            handle = await start_shared_grpc(
                state.tokenizer_manager,
                server_args,
                state.scheduler_info,
                args.smg_grpc_host,
                args.smg_grpc_port,
            )
            try:
                yield
            finally:
                await handle.stop()

    http_server.app.router.lifespan_context = lifespan
    http_server.launch_server(server_args)


if __name__ == "__main__":
    main()
