import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _model_info():
    path = Path(__file__).parents[1] / "smg_grpc_servicer/sglang/metadata.py"
    spec = importlib.util.spec_from_file_location("test_sglang_metadata", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.model_info


def test_model_metadata_uses_model_and_tokenizer_not_scheduler_pool_capacity():
    model_info = _model_info()
    tokenizer_manager = SimpleNamespace(
        served_model_name="qwen",
        max_req_input_len=262143,
        tokenizer=SimpleNamespace(pad_token_id=151643, bos_token_id=151643),
        model_config=SimpleNamespace(
            context_len=262144,
            vocab_size=151936,
            is_multimodal=True,
            hf_eos_token_id={151643, 151645},
            hf_config=SimpleNamespace(
                model_type="qwen3_vl",
                architectures=["Qwen3VLForConditionalGeneration"],
                id2label=None,
                num_labels=0,
            ),
        ),
    )
    server_args = SimpleNamespace(model_path="/models/qwen")
    scheduler_info = {"max_total_num_tokens": 580304, "supports_vision": False}

    info = model_info(tokenizer_manager, server_args)

    assert info["max_context_length"] == 262144
    assert info["max_context_length"] != scheduler_info["max_total_num_tokens"]
    assert info["supports_vision"] is True
    assert info["vocab_size"] == 151936
    assert info["eos_token_ids"] == [151643, 151645]
    assert info["pad_token_id"] == 151643
    assert info["bos_token_id"] == 151643
