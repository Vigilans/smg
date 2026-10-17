"""Native decision RPC tests using installed vLLM and generated protobufs."""

import asyncio
import importlib.util
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

for dependency in ("grpc", "vllm", "smg_grpc_proto"):
    if importlib.util.find_spec(dependency) is None:
        raise unittest.SkipTest(f"requires installed {dependency}; run in the vLLM worker environment")

from smg_grpc_proto import vllm_engine_pb2
from vllm.entrypoints.serve.exception_handling.error_response import create_error_response

from smg_grpc_servicer.vllm.servicer import VllmEngineServicer

ENGINE = SimpleNamespace(
    vllm_config=SimpleNamespace(
        kv_transfer_config=None,
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )
)
SYSTEMONE = {"model": "m", "state": "s", "questions": {"q": {"type": "noul"}}}


class Handler:
    def __init__(self, result=None, delay=0.0):
        self.result = result
        self.delay = delay
        self.requests = []
        self.cancelled = False

    async def create_decision(self, request, raw_request):
        self.requests.append((request, raw_request))
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return self.result


def servicer(systemone=None):
    result = object.__new__(VllmEngineServicer)
    result.engine = ENGINE
    result.openai_serving_decisions = None
    result.serving_structured_decisions = systemone
    return result


def call(servicer, body):
    request = vllm_engine_pb2.NativeApiRequest(request_json=json.dumps(body))
    return asyncio.run(servicer.SystemOne(request, None))


class NativeDecisionRpcTests(unittest.TestCase):
    def test_success_returns_handler_json(self):
        from vllm.entrypoints.generate.structured_decisions.protocol import (
            DecisionUsage,
            StructuredDecisionResponse,
        )

        handler = Handler(
            StructuredDecisionResponse(
                id="decision-1",
                model="m",
                answers={"q": {"noul": 0.75}},
                usage=DecisionUsage(input_tokens=3, output_tokens=1),
                diagnostics={},
            )
        )
        response = call(servicer(handler), SYSTEMONE)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.response_json)["answers"], {"q": {"noul": 0.75}})
        request, raw_request = handler.requests[0]
        self.assertEqual(request.state, "s")
        self.assertIsNone(raw_request)

    def test_error_response_keeps_status(self):
        handler = Handler(create_error_response(ValueError("bad question")))
        response = call(servicer(handler), SYSTEMONE)
        self.assertEqual(response.status_code, 400)
        self.assertIn("bad question", json.loads(response.response_json)["error"]["message"])

    def test_invalid_body_is_bad_request(self):
        handler = Handler()
        response = call(servicer(handler), {"model": "m", "state": "s"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(json.loads(response.response_json)["error"]["code"], 400)
        self.assertEqual(handler.requests, [])

    def test_missing_handler_is_not_implemented(self):
        response = call(servicer(), SYSTEMONE)
        self.assertEqual(response.status_code, 501)
        decisions = asyncio.run(
            servicer().Decisions(vllm_engine_pb2.NativeApiRequest(request_json="{}"), None)
        )
        self.assertEqual(decisions.status_code, 501)

    def test_cancellation_reaches_handler(self):
        handler = Handler(delay=60)
        request = vllm_engine_pb2.NativeApiRequest(request_json=json.dumps(SYSTEMONE))

        async def cancel():
            task = asyncio.create_task(servicer(handler).SystemOne(request, None))
            await asyncio.sleep(0.05)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(cancel())
        self.assertTrue(handler.cancelled)

    def test_server_info_advertises_handlers(self):
        both = servicer(Handler())
        both.openai_serving_decisions = Handler()
        request = vllm_engine_pb2.GetServerInfoRequest()
        with patch("smg_grpc_servicer.mm_shm.shm_namespace_id", return_value=""):
            info = asyncio.run(both.GetServerInfo(request, None))
            self.assertEqual(info.native_decision_apis, "decisions,systemone")
            info = asyncio.run(servicer().GetServerInfo(request, None))
            self.assertEqual(info.native_decision_apis, "")


if __name__ == "__main__":
    unittest.main()
