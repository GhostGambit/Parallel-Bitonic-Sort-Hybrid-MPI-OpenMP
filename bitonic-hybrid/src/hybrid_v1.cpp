#include "bitonic_kernels.hpp"
#include "common.hpp"

namespace {

void hybrid_v1_sort(Buffers& buffers, const RunContext& ctx, Times& times) {
    std::uint32_t* chunk = buffers.chunk.data();
    std::uint32_t* received = buffers.received.data();
    const std::uint64_t n = ctx.n;
    const std::uint64_t offset = ctx.offset;
    const std::uint64_t tile = effective_tile(ctx.config.tile, n);
    double comm = 0.0;
    const Timer total;
#pragma omp parallel
    {
        omp_local_bitonic_phase(chunk, n, offset, tile);
        for (std::uint64_t k = n << 1; k <= ctx.total; k <<= 1) {
            std::uint64_t j = k >> 1;
            for (; j >= n; j >>= 1) {
                const int partner = ctx.rank ^ static_cast<int>(j / n);
                const bool ascending = (offset & k) == 0;
                const bool keep_min = (ctx.rank < partner) == ascending;
#pragma omp master
                comm += exchange_chunks(chunk, received, n, partner);
#pragma omp barrier
                omp_keep_min_or_max(chunk, received, n, keep_min);
            }
            omp_local_merge_steps(chunk, n, offset, k, j, tile);
        }
    }
    times.comm = comm;
    times.comp = total.elapsed() - comm;
}

}

int main(int argc, char** argv) { return run_hybrid(argc, argv, HybridVersion{"v1", false, false, hybrid_v1_sort}); }
