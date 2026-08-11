import numpy as np

def coalescing_analysis(byte_addresses, access_width_bytes, cache_line_bytes):
    addr = np.asarray(byte_addresses).astype(np.int64)

    all_lines = []
    unique_bytes = set()

    for a in addr.tolist():
        start_line = a // cache_line_bytes
        end_line = (a + access_width_bytes - 1) // cache_line_bytes
        all_lines.append(np.arange(start_line, end_line + 1, dtype=np.int64))
        for offset in range(access_width_bytes):
            unique_bytes.add(a + offset)

    line_ids = np.unique(np.concatenate(all_lines)).astype(np.int64)
    transaction_count = int(line_ids.size)

    U = len(unique_bytes)
    C = transaction_count
    L = cache_line_bytes
    useful_byte_fraction = float(U) / float(C * L)

    return {
        "line_ids": line_ids,
        "transaction_count": transaction_count,
        "useful_byte_fraction": useful_byte_fraction
    }