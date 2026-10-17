"""Expose an HTTP TokenizerManager to the SMG gRPC servicer."""

import asyncio
import copy
import time

from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput


def _logprobs(entries):
    entries = entries or []
    return {
        "token_logprobs_val": [entry[0] for entry in entries],
        "token_logprobs_idx": [entry[1] for entry in entries],
    }


def _top_logprobs(rows):
    rows = rows or []
    return {
        "top_logprobs_val": [[entry[0] for entry in row] if row else [] for row in rows],
        "top_logprobs_idx": [[entry[1] for entry in row] if row else [] for row in rows],
    }


class _OutputAdapter:
    def __init__(self, streaming, manager_incremental):
        self.streaming = streaming
        self.manager_incremental = manager_incremental
        self.output_offset = 0
        self.output_ids = []
        self.input_logprobs_sent = False
        self.input_logprobs = None
        self.output_logprob_entries = []
        self.output_top_logprob_rows = []

    def convert(self, output):
        converted = dict(output)
        meta = converted.get("meta_info", {})
        output_ids = list(converted.get("output_ids", []))
        previous_offset = self.output_offset
        if self.streaming and self.manager_incremental:
            token_ids = output_ids
            self.output_ids.extend(output_ids)
            self.output_offset += len(output_ids)
        elif self.streaming:
            token_ids = output_ids[self.output_offset :]
            self.output_ids = output_ids
            self.output_offset = len(output_ids)
        else:
            token_ids = output_ids
            self.output_ids = output_ids
            self.output_offset = len(output_ids)
        converted["token_ids"] = token_ids
        converted["all_token_ids"] = list(self.output_ids)
        converted["finished"] = meta.get("finish_reason") is not None

        input_entries = meta.get("input_token_logprobs")
        if input_entries and (not self.streaming or not self.input_logprobs_sent):
            input_logprobs = _logprobs(input_entries)
            input_logprobs.update(_top_logprobs(meta.get("input_top_logprobs")))
            converted["input_logprobs"] = input_logprobs
            self.input_logprobs = input_logprobs
            self.input_logprobs_sent = True
        if self.input_logprobs is not None:
            converted["all_input_logprobs"] = self.input_logprobs

        output_entries = meta.get("output_token_logprobs")
        if output_entries:
            if self.streaming and self.manager_incremental:
                self.output_logprob_entries.extend(output_entries)
                self.output_top_logprob_rows.extend(meta.get("output_top_logprobs") or [])
            else:
                self.output_logprob_entries = list(output_entries)
                self.output_top_logprob_rows = list(meta.get("output_top_logprobs") or [])
            if self.streaming and not self.manager_incremental:
                output_entries = output_entries[previous_offset:]
                top_rows = (meta.get("output_top_logprobs") or [])[previous_offset:]
            else:
                top_rows = meta.get("output_top_logprobs")
            output_logprobs = _logprobs(output_entries)
            output_logprobs.update(_top_logprobs(top_rows))
            converted["output_logprobs"] = output_logprobs
        if self.output_logprob_entries:
            all_output_logprobs = _logprobs(self.output_logprob_entries)
            all_output_logprobs.update(_top_logprobs(self.output_top_logprob_rows))
            converted["all_output_logprobs"] = all_output_logprobs
        return converted


class HttpTokenizerManagerAdapter:
    """Submit SMG-tokenized requests through an existing TokenizerManager."""

    def __init__(self, tokenizer_manager):
        self.manager = tokenizer_manager
        self._parallel_children = {}

    @property
    def gracefully_exit(self):
        return self.manager.gracefully_exit

    @property
    def last_receive_tstamp(self):
        return self.manager.last_receive_tstamp

    @property
    def rid_to_state(self):
        return self.manager.rid_to_state

    def auto_create_handle_loop(self):
        self.manager.auto_create_handle_loop()

    async def _generate_one(self, obj, request_id):
        obj.rid = request_id
        output_adapter = _OutputAdapter(
            obj.stream,
            self.manager.incremental_streaming_output,
        )
        original = GenerateReqInput(
            rid=request_id,
            input_ids=list(obj.input_ids),
            sampling_params={},
            stream=obj.stream,
            return_logprob=obj.return_logprob,
            logprob_start_len=obj.logprob_start_len,
            top_logprobs_num=obj.top_logprobs_num,
            token_ids_logprob=obj.token_ids_logprob,
            bootstrap_host=obj.bootstrap_host,
            bootstrap_port=obj.bootstrap_port,
            bootstrap_room=obj.bootstrap_room,
            lora_id=obj.lora_id,
            routed_dp_rank=obj.routed_dp_rank,
            routing_key=obj.routing_key,
            require_reasoning=obj.require_reasoning,
            priority=obj.priority,
            extra_key=obj.extra_key,
            cache_salt=getattr(obj, "cache_salt", None),
            received_time=time.time(),
        )
        original.is_single = True
        original.batch_size = 1
        self.manager._init_req_state(original, None)
        obj.time_stats = self.manager.rid_to_state[request_id].time_stats
        completed = False
        try:
            self.manager._send_one_request(obj)
            async for output in self.manager._wait_one_response(original, None):
                converted = output_adapter.convert(output)
                completed = converted["finished"]
                yield converted
        except BaseException:
            if not completed:
                self.manager.abort_request(request_id)
            self.manager.rid_to_state.pop(request_id, None)
            raise

    async def generate_request(self, obj, request_id=None, grpc_context=None):
        request_id = request_id or obj.rid
        n = getattr(obj.sampling_params, "n", 1)
        if n <= 1:
            async for output in self._generate_one(obj, request_id):
                yield output
            return

        generators = []
        child_rids = {f"{request_id}-{index}" for index in range(n)}
        self._parallel_children[request_id] = child_rids
        for index in range(n):
            item = copy.copy(obj)
            item.sampling_params = copy.copy(obj.sampling_params)
            item.sampling_params.n = 1
            generators.append(self._generate_one(item, f"{request_id}-{index}"))

        tasks = {}
        completed = False
        try:
            if not obj.stream:
                outputs = await self.manager._collect_batch_responses(generators)
                for index, output in enumerate(outputs):
                    output["index"] = index
                completed = True
                yield outputs
                return

            tasks = {
                asyncio.create_task(generator.__anext__()): (index, generator)
                for index, generator in enumerate(generators)
            }
            while tasks:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    index, generator = tasks.pop(task)
                    try:
                        output = task.result()
                    except StopAsyncIteration:
                        continue
                    output["index"] = index
                    yield output
                    tasks[asyncio.create_task(generator.__anext__())] = (index, generator)
            completed = True
        finally:
            if not completed:
                await self.abort_request(request_id)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.gather(
                *(generator.aclose() for generator in generators),
                return_exceptions=True,
            )
            self._parallel_children.pop(request_id, None)

    async def embedding_request(self, obj, request_id=None):
        request_id = request_id or obj.rid
        obj.rid = request_id
        original = EmbeddingReqInput(
            rid=request_id,
            input_ids=list(obj.input_ids),
            sampling_params={},
            lora_id=obj.lora_id,
            priority=obj.priority,
            dimensions=obj.dimensions,
            received_time=time.time(),
        )
        original.is_single = True
        original.batch_size = 1
        self.manager._init_req_state(original, None)
        obj.time_stats = self.manager.rid_to_state[request_id].time_stats
        future = asyncio.get_running_loop().create_future()
        try:
            self.manager._send_one_request(obj)
        except BaseException as exc:
            self.manager.abort_request(request_id)
            self.manager.rid_to_state.pop(request_id, None)
            future.set_exception(exc)
            return future

        async def run():
            try:
                async for output in self.manager._wait_one_response(original, None):
                    future.set_result(output)
                    return
                future.set_exception(RuntimeError("embedding request produced no response"))
            except BaseException as exc:
                self.manager.abort_request(request_id)
                self.manager.rid_to_state.pop(request_id, None)
                future.set_exception(exc)

        asyncio.create_task(run())
        return future

    async def abort_request(self, request_id):
        child_rids = self._parallel_children.get(request_id)
        if child_rids is not None:
            existed = any(rid in self.manager.rid_to_state for rid in child_rids)
            for rid in child_rids:
                self.manager.abort_request(rid)
            return existed
        existed = request_id in self.manager.rid_to_state
        if existed:
            self.manager.abort_request(request_id)
        return existed

    def _cleanup_request_state(self, request_id):
        self.manager.rid_to_state.pop(request_id, None)

    def get_server_info(self):
        return {
            "active_requests": len(self.manager.rid_to_state),
            "paused": self.manager.is_pause,
            "last_receive_time": self.manager.last_receive_tstamp,
        }

    async def get_loads(self, include, dp_rank=None):
        return await self.manager.get_loads(include=include, dp_rank=dp_rank)

    async def send_communicator_req(self, req, communicator_name, timeout=30.0):
        communicator = getattr(self.manager, communicator_name)
        return await asyncio.wait_for(communicator(req), timeout=timeout)

    async def shutdown(self):
        return None
