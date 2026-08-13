def kv_cache_capacity_planner(
    model_parameters,
    bytes_per_parameter,
    num_layers,
    sequence_length,
    num_attention_heads,
    num_kv_heads,
    head_dim,
    bytes_per_element,
    memory_per_gpu,
    reserved_memory,
    memory_bandwidth,
):
    model_bytes = model_parameters * bytes_per_parameter
    kv_bytes_per_request = (
        2 * num_layers * sequence_length * num_kv_heads * head_dim * bytes_per_element
    )

    available = memory_per_gpu - reserved_memory - model_bytes
    max_batch_size = max(0, available // kv_bytes_per_request)

    total_bytes = model_bytes + max_batch_size * kv_bytes_per_request
    seconds_per_token = total_bytes / memory_bandwidth

    if max_batch_size == 0:
        tokens_per_second = 0.0
    else:
        tokens_per_second = max_batch_size / seconds_per_token

    return {
        "model_bytes": int(model_bytes),
        "kv_bytes_per_request": int(kv_bytes_per_request),
        "max_batch_size": int(max_batch_size),
        "seconds_per_token": float(seconds_per_token),
        "tokens_per_second": float(tokens_per_second),
    }
