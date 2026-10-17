import asyncio
from array import array
import time
from types import SimpleNamespace

from grpc_health.v1 import health_pb2
from sglang.srt.managers.io_struct import (
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.sampling.sampling_params import SamplingParams

from smg_grpc_servicer.sglang.http_adapter import HttpTokenizerManagerAdapter
from smg_grpc_servicer.sglang.health_servicer import SGLangHealthServicer


def _tokenized(*, stream=True, n=1, mm_inputs=None):
    return TokenizedGenerateReqInput(
        rid="rid",
        input_text="prompt",
        input_ids=array("q", [1, 2]),
        input_embeds=None,
        mm_inputs=mm_inputs,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=8, n=n),
        return_logprob=True,
        logprob_start_len=0,
        top_logprobs_num=1,
        token_ids_logprob=None,
        stream=stream,
        routed_dp_rank=2,
        priority=3,
        require_reasoning=True,
        cache_salt="tenant",
    )


class Manager:
    def __init__(self, outputs):
        self.outputs = outputs
        self.incremental_streaming_output = False
        self.rid_to_state = {}
        self.sent = []
        self.aborted = []
        self.gracefully_exit = False
        self.last_receive_tstamp = time.time()

    def _init_req_state(self, request, raw_request):
        assert raw_request is None
        self.rid_to_state[request.rid] = SimpleNamespace(time_stats=object())

    def _send_one_request(self, request):
        self.sent.append(request)

    async def _wait_one_response(self, request, raw_request):
        assert raw_request is None
        for output in self.outputs:
            yield output

    async def _collect_batch_responses(self, generators):
        tasks = [asyncio.create_task(generator.__anext__()) for generator in generators]
        try:
            return await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def abort_request(self, request_id):
        if request_id in self.rid_to_state:
            self.aborted.append(request_id)


def test_streaming_dispatches_same_tokenized_object_and_slices_cumulative_output():
    mm_inputs = object()
    outputs = [
        {
            "output_ids": [7],
            "meta_info": {
                "finish_reason": None,
                "prompt_tokens": 2,
                "input_token_logprobs": [(-0.1, 1, None), (-0.2, 2, None)],
                "input_top_logprobs": [[(-0.1, 1, None)], [(-0.2, 2, None)]],
                "output_token_logprobs": [(-0.3, 7, None)],
                "output_top_logprobs": [[(-0.3, 7, None)]],
            },
        },
        {
            "output_ids": [7, 8],
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 2,
                "input_token_logprobs": [(-0.1, 1, None), (-0.2, 2, None)],
                "input_top_logprobs": [[(-0.1, 1, None)], [(-0.2, 2, None)]],
                "output_token_logprobs": [(-0.3, 7, None), (-0.4, 8, None)],
                "output_top_logprobs": [[(-0.3, 7, None)], [(-0.4, 8, None)]],
            },
        },
    ]
    manager = Manager(outputs)
    tokenized = _tokenized(mm_inputs=mm_inputs)
    adapter = HttpTokenizerManagerAdapter(manager)

    async def collect():
        return [item async for item in adapter.generate_request(tokenized, "grpc-rid")]

    result = asyncio.run(collect())
    assert [item["token_ids"] for item in result] == [[7], [8]]
    assert result[0]["input_logprobs"]["token_logprobs_idx"] == [1, 2]
    assert "input_logprobs" not in result[1]
    assert result[0]["output_logprobs"]["token_logprobs_idx"] == [7]
    assert result[1]["output_logprobs"]["token_logprobs_idx"] == [8]
    assert result[1]["all_token_ids"] == [7, 8]
    assert result[1]["all_output_logprobs"]["token_logprobs_idx"] == [7, 8]
    assert result[1]["all_input_logprobs"]["token_logprobs_idx"] == [1, 2]
    assert manager.sent == [tokenized]
    assert manager.sent[0] is tokenized
    assert manager.sent[0].mm_inputs is mm_inputs
    assert not hasattr(tokenized, "received_time")


def test_parallel_sampling_cancel_aborts_base_and_children_before_cleanup():
    manager = Manager([{"output_ids": [7], "meta_info": {"finish_reason": None}}])
    adapter = HttpTokenizerManagerAdapter(manager)

    async def cancel_after_first():
        generator = adapter.generate_request(_tokenized(n=2), "base")
        await generator.__anext__()
        await generator.aclose()

    asyncio.run(cancel_after_first())
    assert {"base-0", "base-1"}.issubset(manager.aborted)


def test_parallel_non_streaming_assigns_distinct_indices():
    manager = Manager(
        [{"output_ids": [7], "meta_info": {"finish_reason": {"type": "stop"}}}]
    )
    adapter = HttpTokenizerManagerAdapter(manager)

    async def collect():
        return [
            item async for item in adapter.generate_request(_tokenized(stream=False, n=2), "base")
        ]

    result = asyncio.run(collect())
    assert [output["index"] for output in result[0]] == [0, 1]
    assert len(manager.sent) == 2


def test_parallel_non_streaming_dispatches_all_children_concurrently():
    both_sent = asyncio.Event()

    class ConcurrentManager(Manager):
        def _send_one_request(self, request):
            super()._send_one_request(request)
            if len(self.sent) == 2:
                both_sent.set()

        async def _wait_one_response(self, request, raw_request):
            await asyncio.wait_for(both_sent.wait(), timeout=1)
            yield {"output_ids": [7], "meta_info": {"finish_reason": {"type": "stop"}}}

    manager = ConcurrentManager([])
    adapter = HttpTokenizerManagerAdapter(manager)

    async def collect():
        return [
            item async for item in adapter.generate_request(_tokenized(stream=False, n=2), "base")
        ]

    asyncio.run(collect())
    assert len(manager.sent) == 2


def test_incremental_manager_preserves_repeated_token_ids():
    manager = Manager(
        [
            {"output_ids": [7], "meta_info": {"finish_reason": None}},
            {"output_ids": [7], "meta_info": {"finish_reason": {"type": "stop"}}},
        ]
    )
    manager.incremental_streaming_output = True
    adapter = HttpTokenizerManagerAdapter(manager)

    async def collect():
        return [item async for item in adapter.generate_request(_tokenized(), "base")]

    result = asyncio.run(collect())
    assert [item["token_ids"] for item in result] == [[7], [7]]
    assert result[-1]["all_token_ids"] == [7, 7]


def test_incremental_manager_accumulates_logprobs_through_empty_finish():
    manager = Manager(
        [
            {
                "output_ids": [7],
                "meta_info": {
                    "finish_reason": None,
                    "output_token_logprobs": [(-0.3, 7, None)],
                    "output_top_logprobs": [[(-0.3, 7, None)]],
                },
            },
            {
                "output_ids": [7],
                "meta_info": {
                    "finish_reason": None,
                    "output_token_logprobs": [(-0.4, 7, None)],
                    "output_top_logprobs": [[(-0.4, 7, None)]],
                },
            },
            {"output_ids": [], "meta_info": {"finish_reason": {"type": "stop"}}},
        ]
    )
    manager.incremental_streaming_output = True
    adapter = HttpTokenizerManagerAdapter(manager)

    async def collect():
        return [item async for item in adapter.generate_request(_tokenized(), "base")]

    result = asyncio.run(collect())
    assert [item["token_ids"] for item in result] == [[7], [7], []]
    assert result[-1]["all_token_ids"] == [7, 7]
    assert result[-1]["all_output_logprobs"]["token_logprobs_idx"] == [7, 7]
    assert result[-1]["all_output_logprobs"]["token_logprobs_val"] == [-0.3, -0.4]


def test_out_of_band_base_abort_expands_to_exact_child_rids():
    release = asyncio.Event()

    class BlockingManager(Manager):
        async def _wait_one_response(self, request, raw_request):
            await release.wait()
            yield {"output_ids": [], "meta_info": {"finish_reason": {"type": "abort"}}}

    manager = BlockingManager([])
    adapter = HttpTokenizerManagerAdapter(manager)

    async def run():
        generator = adapter.generate_request(_tokenized(n=2), "base")
        pending = asyncio.create_task(generator.__anext__())
        while not {"base-0", "base-1"}.issubset(manager.rid_to_state):
            await asyncio.sleep(0)
        assert await adapter.abort_request("base") is True
        assert set(manager.aborted) == {"base-0", "base-1"}
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await generator.aclose()

    asyncio.run(run())


def test_embedding_preserves_prompt_tokens_from_http_output():
    manager = Manager([{"embedding": [0.25, 0.75], "prompt_tokens": 17}])
    tokenized = TokenizedEmbeddingReqInput(
        rid="embed",
        input_text="prompt",
        input_ids=array("q", [1, 2]),
        mm_inputs=None,
        token_type_ids=[0, 0],
        sampling_params=SamplingParams(max_new_tokens=0),
    )
    adapter = HttpTokenizerManagerAdapter(manager)

    async def run():
        future = await adapter.embedding_request(tokenized, "grpc-embed")
        return await future

    result = asyncio.run(run())
    assert result["embedding"] == [0.25, 0.75]
    assert result["prompt_tokens"] == 17
    assert manager.sent == [tokenized]


def test_shared_adapter_does_not_shutdown_http_manager():
    manager = SimpleNamespace(shutdown=lambda: (_ for _ in ()).throw(AssertionError()))
    asyncio.run(HttpTokenizerManagerAdapter(manager).shutdown())


def test_health_service_reads_forwarded_idle_and_busy_state():
    manager = Manager([])
    manager.last_receive_tstamp = time.time() - 60
    adapter = HttpTokenizerManagerAdapter(manager)
    health = SGLangHealthServicer(adapter, {})
    health.set_serving()
    request = health_pb2.HealthCheckRequest(service=health.SGLANG_SERVICE)
    context = SimpleNamespace(set_code=lambda _code: None, set_details=lambda _details: None)

    idle = asyncio.run(health.Check(request, context))
    assert idle.status == health_pb2.HealthCheckResponse.SERVING

    manager.rid_to_state["busy"] = object()
    busy = asyncio.run(health.Check(request, context))
    assert busy.status == health_pb2.HealthCheckResponse.NOT_SERVING
