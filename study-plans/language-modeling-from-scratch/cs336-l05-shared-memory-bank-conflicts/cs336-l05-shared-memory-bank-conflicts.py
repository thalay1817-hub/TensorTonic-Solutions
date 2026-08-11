import numpy as np

def bank_conflict_analysis(byte_addresses, num_banks, bank_width_bytes):
    addr = np.asarray(byte_addresses).astype(np.int64, copy=True)

    bank_ids = (addr // bank_width_bytes) % num_banks
    bank_ids = bank_ids.astype(np.int64)

    conflict_degree = np.empty_like(bank_ids)

    for b in np.unique(bank_ids):
        mask = bank_ids == b
        distinct_count = np.unique(addr[mask]).size
        conflict_degree[mask] = distinct_count

    return {
        "bank_ids": bank_ids,
        "conflict_degree": conflict_degree.astype(np.int64)
    }