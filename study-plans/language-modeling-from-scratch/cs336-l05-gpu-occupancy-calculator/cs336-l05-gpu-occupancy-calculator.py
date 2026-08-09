import math


def gpu_occupancy(threads_per_block, registers_per_thread,
                               shared_mem_per_block, max_threads_per_sm,
                               max_warps_per_sm, max_blocks_per_sm,
                               max_registers_per_sm, max_shared_mem_per_sm,
                               warp_size):
    # Warps allocated per block (ceiling division)
    w = (threads_per_block + warp_size - 1) // warp_size

    # Effective threads the hardware actually allocates per block
    t_eff = w * warp_size

    # Limit from max threads per SM
    limit_threads = max_threads_per_sm // t_eff

    # Limit from max warps per SM
    limit_warps = max_warps_per_sm // w

    # Limit from max blocks per SM (direct hardware cap)
    limit_blocks = max_blocks_per_sm

    # Limit from registers; zero registers per thread is nonbinding
    if registers_per_thread == 0:
        limit_registers = math.inf
    else:
        registers_per_block = registers_per_thread * t_eff
        limit_registers = max_registers_per_sm // registers_per_block

    # Limit from shared memory; zero shared memory per block is nonbinding
    if shared_mem_per_block == 0:
        limit_shared_mem = math.inf
    else:
        limit_shared_mem = max_shared_mem_per_sm // shared_mem_per_block

    b = min(limit_threads, limit_warps, limit_blocks, limit_registers, limit_shared_mem)
    b = int(b)  # b is guaranteed finite since limit_blocks is always a finite int

    resident_warps = b * w
    occupancy = resident_warps / max_warps_per_sm

    return {
        "blocks_per_sm": b,
        "resident_warps": resident_warps,
        "occupancy": occupancy,
    }