#pragma once

#include <algorithm>
#include <cstdint>

inline std::uint64_t lower_pair_index(std::uint64_t pair, std::uint64_t j) {
    return ((pair & ~(j - 1)) << 1) | (pair & (j - 1));
}

inline void order_pair(std::uint32_t* data, std::uint64_t lower, std::uint64_t upper, bool ascending) {
    const std::uint32_t a = data[lower];
    const std::uint32_t b = data[upper];
    const std::uint32_t low = std::min(a, b);
    const std::uint32_t high = std::max(a, b);
    data[lower] = ascending ? low : high;
    data[upper] = ascending ? high : low;
}

inline void compare_exchange_stage(std::uint32_t* data, std::uint64_t count, std::uint64_t global_offset,
                                   std::uint64_t j, std::uint64_t k) {
    const std::uint64_t pairs = count >> 1;
    for (std::uint64_t pair = 0; pair < pairs; ++pair) {
        const std::uint64_t i = lower_pair_index(pair, j);
        order_pair(data, i, i + j, ((global_offset + i) & k) == 0);
    }
}

inline void bitonic_merge_steps(std::uint32_t* data, std::uint64_t count, std::uint64_t global_offset,
                                std::uint64_t k, std::uint64_t j_start) {
    for (std::uint64_t j = j_start; j > 0; j >>= 1) {
        compare_exchange_stage(data, count, global_offset, j, k);
    }
}

inline void bitonic_sort_serial(std::uint32_t* data, std::uint64_t count) {
    for (std::uint64_t k = 2; k <= count; k <<= 1) {
        bitonic_merge_steps(data, count, 0, k, k >> 1);
    }
}

inline void omp_compare_exchange_stage(std::uint32_t* data, std::uint64_t n, std::uint64_t global_offset,
                                       std::uint64_t j, std::uint64_t k) {
    const std::int64_t pairs = static_cast<std::int64_t>(n >> 1);
#pragma omp for schedule(static)
    for (std::int64_t pair = 0; pair < pairs; ++pair) {
        const std::uint64_t i = lower_pair_index(static_cast<std::uint64_t>(pair), j);
        order_pair(data, i, i + j, ((global_offset + i) & k) == 0);
    }
}

inline void omp_keep_min_or_max(std::uint32_t* mine, const std::uint32_t* theirs, std::uint64_t n, bool keep_min) {
    const std::int64_t count = static_cast<std::int64_t>(n);
#pragma omp for schedule(static)
    for (std::int64_t i = 0; i < count; ++i) {
        mine[i] = keep_min ? std::min(mine[i], theirs[i]) : std::max(mine[i], theirs[i]);
    }
}

inline std::uint64_t effective_tile(std::uint64_t tile, std::uint64_t n) { return tile == 0 ? 0 : std::min(tile, n); }

inline void omp_local_merge_steps(std::uint32_t* data, std::uint64_t n, std::uint64_t global_offset, std::uint64_t k,
                                  std::uint64_t j_start, std::uint64_t tile) {
    const std::uint64_t full_pass_limit = tile == 0 ? 1 : tile;
    std::uint64_t j = j_start;
    for (; j >= full_pass_limit; j >>= 1) {
        omp_compare_exchange_stage(data, n, global_offset, j, k);
    }
    if (j == 0) {
        return;
    }
    const std::int64_t tiles = static_cast<std::int64_t>(n / tile);
#pragma omp for schedule(static)
    for (std::int64_t t = 0; t < tiles; ++t) {
        const std::uint64_t start = static_cast<std::uint64_t>(t) * tile;
        bitonic_merge_steps(data + start, tile, global_offset + start, k, j);
    }
}

inline void omp_local_full_sort_phase(std::uint32_t* data, std::uint64_t n, std::uint64_t global_offset,
                                      std::uint64_t tile) {
    const std::int64_t tiles = static_cast<std::int64_t>(n / tile);
#pragma omp for schedule(static)
    for (std::int64_t t = 0; t < tiles; ++t) {
        const std::uint64_t start = static_cast<std::uint64_t>(t) * tile;
        for (std::uint64_t k = 2; k <= tile; k <<= 1) {
            bitonic_merge_steps(data + start, tile, global_offset + start, k, k >> 1);
        }
    }
}

inline void omp_local_bitonic_phase(std::uint32_t* data, std::uint64_t n, std::uint64_t global_offset,
                                    std::uint64_t tile) {
    std::uint64_t k = 2;
    if (tile >= 2) {
        omp_local_full_sort_phase(data, n, global_offset, tile);
        k = tile << 1;
    }
    for (; k <= n; k <<= 1) {
        omp_local_merge_steps(data, n, global_offset, k, k >> 1, tile);
    }
}
