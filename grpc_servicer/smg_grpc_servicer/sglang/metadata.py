"""Model metadata for an SMG gRPC listener sharing SGLang HTTP state."""

import json


def model_info(tokenizer_manager, server_args):
    model_config = tokenizer_manager.model_config
    hf_config = model_config.hf_config
    tokenizer = tokenizer_manager.tokenizer
    id2label = getattr(hf_config, "id2label", None)
    num_labels = getattr(hf_config, "num_labels", 0) or 0
    if not id2label and num_labels:
        id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
    return {
        "model_name": tokenizer_manager.served_model_name or server_args.model_path,
        "max_context_length": model_config.context_len,
        "vocab_size": model_config.vocab_size,
        "supports_vision": model_config.is_multimodal,
        "model_type": getattr(hf_config, "model_type", None),
        "architectures": getattr(hf_config, "architectures", None),
        "max_req_input_len": tokenizer_manager.max_req_input_len,
        "eos_token_ids": sorted(model_config.hf_eos_token_id),
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "id2label_json": json.dumps(id2label) if id2label else "",
        "num_labels": num_labels or (len(id2label) if id2label else 0),
    }
