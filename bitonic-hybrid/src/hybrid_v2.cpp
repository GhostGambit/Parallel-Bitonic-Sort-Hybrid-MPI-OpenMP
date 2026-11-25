#include "bitonic_kernels.hpp"
#include "common.hpp"

namespace {

void parallel_std_sort(std::vector<std::uint32_t>& data, std::vector<std::uint32_t>& scratch) {
    const std::uint64_t n = data.size();
    const int parts = omp_get_max_threads();
    const auto bound = [n, parts](int part) {
        return n * static_cast<std::uint64_t>(part) / static_cast<std::uint64_t>(parts);
    };
    std::uint32_t* source = data.data();
    std::uint32_t* target = scratch.data();
#pragma omp parallel for schedule(static, 1)
    for (int part = 0; part < parts; ++part) {
        std::sort(source + bound(part), source + bound(part + 1));
    }
    for (int width = 1; width < parts; width *= 2) {
#pragma omp parallel for schedule(dynamic, 1)
        for (int left = 0; left < parts; left += 2 * width) {
            const int middle = std::min(left + width, parts);
            const int right = std::min(left + 2 * width, parts);
            std::merge(source + bound(left), source + bound(middle), source + bound(middle), source + bound(right),
                       target + bound(left));
        }
        std::swap(source, target);
    }
    if (source != data.data()) {
        data.swap(scratch);
    }
}

void local_sort(Buffers& buffers, const RunContext& ctx) {
    if (ctx.config.local_sort == LocalSort::std_sort) {
        parallel_std_sort(buffers.chunk, buffers.scratch);
        return;
    }
    std::uint32_t* chunk = buffers.chunk.data();
    const std::uint64_t tile = effective_tile(ctx.config.tile, ctx.n);
#pragma omp parallel
    omp_local_bitonic_phase(chunk, ctx.n, 0, tile);
}

std::uint64_t merge_path_split(const std::uint32_t* a, const std::uint32_t* b, std::uint64_t n,
                               std::uint64_t diagonal) {
    std::uint64_t low = diagonal > n ? diagonal - n : 0;
    std::uint64_t high = std::min(diagonal, n);
    while (low < high) {
        const std::uint64_t middle = low + (high - low) / 2;
        if (a[middle] <= b[diagonal - middle - 1]) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low;
}

void merge_segment(const std::uint32_t* a, const std::uint32_t* b, std::uint64_t n, std::uint64_t diagonal,
                   std::uint64_t count, std::uint32_t* out) {
    std::uint64_t from_a = merge_path_split(a, b, n, diagonal);
    std::uint64_t from_b = diagonal - from_a;
    for (std::uint64_t written = 0; written < count; ++written) {
        const bool take_a = from_b >= n || (from_a < n && a[from_a] <= b[from_b]);
        out[written] = take_a ? a[from_a++] : b[from_b++];
    }
}

void parallel_merge_split(const std::uint32_t* mine, const std::uint32_t* theirs, std::uint32_t* out,
                          std::uint64_t n, bool keep_low) {
    const std::uint64_t first_output = keep_low ? 0 : n;
#pragma omp parallel
    {
        const std::uint64_t threads = static_cast<std::uint64_t>(omp_get_num_threads());
        const std::uint64_t thread = static_cast<std::uint64_t>(omp_get_thread_num());
        const std::uint64_t begin = n * thread / threads;
        const std::uint64_t end = n * (thread + 1) / threads;
        merge_segment(mine, theirs, n, first_output + begin, end - begin, out + begin);
    }
}

class BlockStream {
public:
    BlockStream(const std::uint32_t* mine, std::uint32_t* theirs, std::uint64_t n, std::uint64_t block, int partner,
                bool keep_low)
        : block_(block), receives_(n / block), sends_(n / block), arrived_(n / block, 0) {
        const Timer timer;
        const std::uint64_t blocks = n / block;
        for (std::uint64_t q = 0; q < blocks; ++q) {
            MPI_Irecv(theirs + q * block, static_cast<int>(block), MPI_UINT32_T, partner, static_cast<int>(q),
                      MPI_COMM_WORLD, &receives_[q]);
        }
        for (std::uint64_t s = 0; s < blocks; ++s) {
            const std::uint64_t q = keep_low ? blocks - 1 - s : s;
            MPI_Isend(mine + q * block, static_cast<int>(block), MPI_UINT32_T, partner, static_cast<int>(q),
                      MPI_COMM_WORLD, &sends_[s]);
        }
        wait_seconds_ += timer.elapsed();
    }

    void require(std::uint64_t index) {
        const std::uint64_t q = index / block_;
        if (!arrived_[q]) {
            const Timer timer;
            MPI_Wait(&receives_[q], MPI_STATUS_IGNORE);
            arrived_[q] = 1;
            wait_seconds_ += timer.elapsed();
        }
    }

    double finish() {
        const Timer timer;
        MPI_Waitall(static_cast<int>(receives_.size()), receives_.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall(static_cast<int>(sends_.size()), sends_.data(), MPI_STATUSES_IGNORE);
        return wait_seconds_ + timer.elapsed();
    }

private:
    std::uint64_t block_;
    std::vector<MPI_Request> receives_;
    std::vector<MPI_Request> sends_;
    std::vector<char> arrived_;
    double wait_seconds_ = 0.0;
};

void stream_merge_low(const std::uint32_t* mine, const std::uint32_t* theirs, std::uint32_t* out, std::uint64_t n,
                      BlockStream& stream) {
    std::uint64_t from_a = 0;
    std::uint64_t from_b = 0;
    for (std::uint64_t written = 0; written < n; ++written) {
        stream.require(from_b);
        out[written] = mine[from_a] <= theirs[from_b] ? mine[from_a++] : theirs[from_b++];
    }
}

void stream_merge_high(const std::uint32_t* mine, const std::uint32_t* theirs, std::uint32_t* out, std::uint64_t n,
                       BlockStream& stream) {
    std::uint64_t from_a = n;
    std::uint64_t from_b = n;
    for (std::uint64_t written = n; written > 0; --written) {
        stream.require(from_b - 1);
        out[written - 1] = mine[from_a - 1] >= theirs[from_b - 1] ? mine[--from_a] : theirs[--from_b];
    }
}

double overlapped_merge_split(const std::uint32_t* mine, std::uint32_t* theirs, std::uint32_t* out, std::uint64_t n,
                              std::uint64_t block, int partner, bool keep_low) {
    BlockStream stream(mine, theirs, n, block, partner, keep_low);
    if (keep_low) {
        stream_merge_low(mine, theirs, out, n, stream);
    } else {
        stream_merge_high(mine, theirs, out, n, stream);
    }
    return stream.finish();
}

void hybrid_v2_sort(Buffers& buffers, const RunContext& ctx, Times& times) {
    const Timer total;
    double comm = 0.0;
    const std::uint64_t n = ctx.n;
    const std::uint64_t block = std::min(ctx.config.block, n);
    local_sort(buffers, ctx);
    for (int k = 2; k <= ctx.procs; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int partner = ctx.rank ^ j;
            const bool ascending = (ctx.rank & k) == 0;
            const bool keep_low = (ctx.rank < partner) == ascending;
            const std::uint32_t* mine = buffers.chunk.data();
            std::uint32_t* theirs = buffers.received.data();
            std::uint32_t* out = buffers.scratch.data();
            if (ctx.config.overlap) {
                comm += overlapped_merge_split(mine, theirs, out, n, block, partner, keep_low);
            } else {
                comm += exchange_chunks(mine, theirs, n, partner);
                parallel_merge_split(mine, theirs, out, n, keep_low);
            }
            buffers.chunk.swap(buffers.scratch);
        }
    }
    times.comm = comm;
    times.comp = total.elapsed() - comm;
}

}

int main(int argc, char** argv) { return run_hybrid(argc, argv, HybridVersion{"v2", true, true, hybrid_v2_sort}); }
