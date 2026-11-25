#pragma once

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

enum class VerifyMode { full, fast, none };
enum class LocalSort { bitonic, std_sort };

inline constexpr int max_full_verify_exp = 22;

inline bool is_power_of_two(std::uint64_t value) { return value != 0 && (value & (value - 1)) == 0; }

inline int log2_exact(std::uint64_t value) {
    if (!is_power_of_two(value)) {
        throw std::invalid_argument("log2_exact expects a power of two");
    }
    int result = 0;
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    return result;
}

struct Config {
    int exp = 20;
    std::uint64_t seed = 12345;
    int threads = omp_get_max_threads();
    std::uint64_t tile = 4096;
    VerifyMode verify = VerifyMode::fast;
    int runs = 1;
    std::string csv;
    LocalSort local_sort = LocalSort::bitonic;
    bool overlap = false;
    std::uint64_t block = 65536;

    std::uint64_t total_elements() const { return std::uint64_t{1} << exp; }
};

inline const char* local_sort_name(LocalSort mode) { return mode == LocalSort::bitonic ? "bitonic" : "std"; }

inline std::uint64_t parse_unsigned(const std::string& flag, const std::string& text) {
    std::size_t used = 0;
    std::uint64_t value = 0;
    try {
        value = std::stoull(text, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (text.empty() || text[0] == '-' || used != text.size()) {
        throw std::invalid_argument(flag + " expects a non-negative integer, got '" + text + "'");
    }
    return value;
}

inline std::uint64_t parse_in_range(const std::string& flag, const std::string& text, std::uint64_t low,
                                    std::uint64_t high) {
    const std::uint64_t value = parse_unsigned(flag, text);
    if (value < low || value > high) {
        throw std::invalid_argument(flag + " must be in [" + std::to_string(low) + ", " + std::to_string(high) + "]");
    }
    return value;
}

inline VerifyMode parse_verify(const std::string& text) {
    if (text == "full") return VerifyMode::full;
    if (text == "fast") return VerifyMode::fast;
    if (text == "none") return VerifyMode::none;
    throw std::invalid_argument("--verify must be full, fast or none, got '" + text + "'");
}

inline LocalSort parse_local_sort(const std::string& text) {
    if (text == "bitonic") return LocalSort::bitonic;
    if (text == "std") return LocalSort::std_sort;
    throw std::invalid_argument("--local-sort must be bitonic or std, got '" + text + "'");
}

inline Config parse_config(int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--overlap") {
            config.overlap = true;
            continue;
        }
        const std::vector<std::string> valued = {"--exp",  "--seed", "--threads",    "--tile", "--verify",
                                                 "--runs", "--csv",  "--local-sort", "--block"};
        if (std::find(valued.begin(), valued.end(), flag) == valued.end()) {
            throw std::invalid_argument("unknown flag '" + flag + "'");
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for " + flag);
        }
        const std::string value = argv[++i];
        if (flag == "--exp") {
            config.exp = static_cast<int>(parse_in_range(flag, value, 1, 34));
        } else if (flag == "--seed") {
            config.seed = parse_unsigned(flag, value);
        } else if (flag == "--threads") {
            config.threads = static_cast<int>(parse_in_range(flag, value, 1, 1024));
        } else if (flag == "--tile") {
            const std::uint64_t tile = parse_unsigned(flag, value);
            if (tile != 0 && (tile < 2 || !is_power_of_two(tile))) {
                throw std::invalid_argument("--tile must be 0 or a power of two >= 2, got " + value);
            }
            config.tile = tile;
        } else if (flag == "--verify") {
            config.verify = parse_verify(value);
        } else if (flag == "--runs") {
            config.runs = static_cast<int>(parse_in_range(flag, value, 1, 1000));
        } else if (flag == "--csv") {
            config.csv = value;
        } else if (flag == "--local-sort") {
            config.local_sort = parse_local_sort(value);
        } else {
            const std::uint64_t block = parse_unsigned(flag, value);
            if (!is_power_of_two(block)) {
                throw std::invalid_argument("--block must be a power of two, got " + value);
            }
            config.block = block;
        }
    }
    if (config.verify == VerifyMode::full && config.exp > max_full_verify_exp) {
        throw std::invalid_argument("--verify full is only allowed for --exp <= " + std::to_string(max_full_verify_exp));
    }
    return config;
}

inline std::uint64_t splitmix64(std::uint64_t state) {
    state += 0x9E3779B97F4A7C15ULL;
    state = (state ^ (state >> 30)) * 0xBF58476D1CE4E5B9ULL;
    state = (state ^ (state >> 27)) * 0x94D049BB133111EBULL;
    return state ^ (state >> 31);
}

inline std::uint32_t value_at(std::uint64_t seed, std::uint64_t global_index) {
    return static_cast<std::uint32_t>(splitmix64(seed + global_index * 0x9E3779B97F4A7C15ULL) >> 32);
}

inline void fill_chunk(std::uint32_t* data, std::uint64_t n, std::uint64_t offset, std::uint64_t seed) {
    const std::int64_t count = static_cast<std::int64_t>(n);
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < count; ++i) {
        data[i] = value_at(seed, offset + static_cast<std::uint64_t>(i));
    }
}

struct Checksum {
    std::uint64_t sum = 0;
    std::uint64_t sumsq = 0;

    bool operator==(const Checksum& other) const { return sum == other.sum && sumsq == other.sumsq; }
};

inline Checksum checksum_chunk(const std::uint32_t* data, std::uint64_t n) {
    std::uint64_t sum = 0;
    std::uint64_t sumsq = 0;
    const std::int64_t count = static_cast<std::int64_t>(n);
#pragma omp parallel for schedule(static) reduction(+ : sum, sumsq)
    for (std::int64_t i = 0; i < count; ++i) {
        const std::uint64_t value = data[i];
        sum += value;
        sumsq += value * value;
    }
    return Checksum{sum, sumsq};
}

inline bool is_sorted_chunk(const std::uint32_t* data, std::uint64_t n) {
    std::uint64_t violations = 0;
    const std::int64_t count = static_cast<std::int64_t>(n);
#pragma omp parallel for schedule(static) reduction(+ : violations)
    for (std::int64_t i = 1; i < count; ++i) {
        violations += data[i - 1] > data[i] ? 1 : 0;
    }
    return violations == 0;
}

class Timer {
public:
    Timer() : start_(std::chrono::steady_clock::now()) {}
    double elapsed() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    }

private:
    std::chrono::steady_clock::time_point start_;
};

struct Times {
    double init = 0.0;
    double comp = 0.0;
    double comm = 0.0;

    double total_sort() const { return comp + comm; }
};

inline void append_csv_row(const std::string& path, const std::vector<std::string>& fields) {
    const bool is_new = !std::ifstream(path).good();
    std::ofstream out(path, std::ios::app);
    if (!out) {
        throw std::runtime_error("cannot open CSV file " + path);
    }
    if (is_new) {
        out << "version,exp,N,procs,threads,tile,block,local_sort,overlap,run,t_init,t_comp,t_comm,t_sort,verified\n";
    }
    for (std::size_t i = 0; i < fields.size(); ++i) {
        out << (i ? "," : "") << fields[i];
    }
    out << "\n";
}

inline std::string format_seconds(double seconds) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(6) << seconds;
    return text.str();
}

struct RunReport {
    std::string version;
    int exp;
    int procs;
    int threads;
    std::uint64_t tile;
    std::uint64_t block;
    std::string local_sort;
    bool overlap;
    int run;
    Times times;
    std::string verified;
};

inline void publish_report(const RunReport& report, const std::string& csv_path) {
    const std::uint64_t n = std::uint64_t{1} << report.exp;
    std::cout << report.version << " exp=" << report.exp << " N=" << n << " P=" << report.procs
              << " T=" << report.threads << " tile=" << report.tile << " block=" << report.block
              << " local_sort=" << report.local_sort << " overlap=" << (report.overlap ? 1 : 0)
              << " run=" << report.run << " init=" << format_seconds(report.times.init)
              << " comp=" << format_seconds(report.times.comp) << " comm=" << format_seconds(report.times.comm)
              << " sort=" << format_seconds(report.times.total_sort()) << " verify=" << report.verified << std::endl;
    if (!csv_path.empty()) {
        append_csv_row(csv_path, {report.version, std::to_string(report.exp), std::to_string(n),
                                  std::to_string(report.procs), std::to_string(report.threads),
                                  std::to_string(report.tile), std::to_string(report.block), report.local_sort,
                                  report.overlap ? "1" : "0", std::to_string(report.run),
                                  format_seconds(report.times.init), format_seconds(report.times.comp),
                                  format_seconds(report.times.comm), format_seconds(report.times.total_sort()),
                                  report.verified});
    }
}

struct RunContext {
    Config config;
    int rank;
    int procs;
    std::uint64_t total;
    std::uint64_t n;
    std::uint64_t offset;
};

struct Buffers {
    std::vector<std::uint32_t> chunk;
    std::vector<std::uint32_t> received;
    std::vector<std::uint32_t> scratch;
};

struct HybridVersion {
    std::string name;
    bool needs_scratch;
    bool uses_merge_options;
    std::function<void(Buffers&, const RunContext&, Times&)> sort;
};

inline double exchange_chunks(const std::uint32_t* mine, std::uint32_t* theirs, std::uint64_t n, int partner) {
    const Timer timer;
    MPI_Sendrecv(mine, static_cast<int>(n), MPI_UINT32_T, partner, 0, theirs, static_cast<int>(n), MPI_UINT32_T,
                 partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return timer.elapsed();
}

inline Times reduce_times_max(const Times& local) {
    const double values[3] = {local.init, local.comp, local.comm};
    double worst[3] = {0.0, 0.0, 0.0};
    MPI_Reduce(values, worst, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return Times{worst[0], worst[1], worst[2]};
}

inline bool verify_full_distributed(const std::vector<std::uint32_t>& chunk, const RunContext& ctx) {
    std::vector<std::uint32_t> gathered(ctx.rank == 0 ? ctx.total : 0);
    MPI_Gather(chunk.data(), static_cast<int>(ctx.n), MPI_UINT32_T, gathered.data(), static_cast<int>(ctx.n),
               MPI_UINT32_T, 0, MPI_COMM_WORLD);
    if (ctx.rank != 0) {
        return false;
    }
    std::vector<std::uint32_t> expected(ctx.total);
    fill_chunk(expected.data(), ctx.total, 0, ctx.config.seed);
    std::sort(expected.begin(), expected.end());
    return gathered == expected;
}

inline bool verify_fast_distributed(const std::vector<std::uint32_t>& chunk, const RunContext& ctx,
                                    const Checksum& before) {
    int locally_ok = is_sorted_chunk(chunk.data(), ctx.n) ? 1 : 0;
    const std::uint32_t my_last = chunk[ctx.n - 1];
    std::uint32_t previous_last = 0;
    const int next = ctx.rank + 1 < ctx.procs ? ctx.rank + 1 : MPI_PROC_NULL;
    const int previous = ctx.rank > 0 ? ctx.rank - 1 : MPI_PROC_NULL;
    MPI_Sendrecv(&my_last, 1, MPI_UINT32_T, next, 1, &previous_last, 1, MPI_UINT32_T, previous, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    if (ctx.rank > 0 && previous_last > chunk[0]) {
        locally_ok = 0;
    }
    const Checksum after = checksum_chunk(chunk.data(), ctx.n);
    const std::uint64_t local_sums[4] = {before.sum, before.sumsq, after.sum, after.sumsq};
    std::uint64_t global_sums[4] = {0, 0, 0, 0};
    MPI_Reduce(local_sums, global_sums, 4, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    int all_ok = 0;
    MPI_Reduce(&locally_ok, &all_ok, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
    return all_ok != 0 && global_sums[0] == global_sums[2] && global_sums[1] == global_sums[3];
}

inline std::string verify_distributed(const std::vector<std::uint32_t>& chunk, const RunContext& ctx,
                                      const Checksum& before) {
    if (ctx.config.verify == VerifyMode::none) {
        return "SKIP";
    }
    const bool ok = ctx.config.verify == VerifyMode::full ? verify_full_distributed(chunk, ctx)
                                                          : verify_fast_distributed(chunk, ctx, before);
    return ok ? "PASS" : "FAIL";
}

inline std::string layout_error(const Config& config, int procs) {
    const std::uint64_t total = config.total_elements();
    const std::uint64_t procs_count = static_cast<std::uint64_t>(procs);
    if (!is_power_of_two(procs_count)) {
        return "the number of MPI processes must be a power of two, got " + std::to_string(procs);
    }
    if (total / procs_count < 2) {
        return "N/P must be at least 2";
    }
    if (total / procs_count > static_cast<std::uint64_t>(INT_MAX)) {
        return "N/P exceeds the MPI message count limit";
    }
    return "";
}

inline int run_hybrid(int argc, char** argv, const HybridVersion& version) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int procs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::cerr << version.name << ": MPI does not provide MPI_THREAD_FUNNELED" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    Config config;
    std::string error;
    try {
        config = parse_config(argc, argv);
        error = layout_error(config, procs);
    } catch (const std::exception& problem) {
        error = problem.what();
    }
    if (!error.empty()) {
        if (rank == 0) std::cerr << version.name << ": " << error << std::endl;
        MPI_Finalize();
        return 1;
    }
    omp_set_num_threads(config.threads);
    const std::uint64_t total = config.total_elements();
    const std::uint64_t n = total / static_cast<std::uint64_t>(procs);
    const RunContext ctx{config, rank, procs, total, n, static_cast<std::uint64_t>(rank) * n};
    Buffers buffers;
    buffers.chunk.resize(n);
    buffers.received.resize(n);
    buffers.scratch.resize(version.needs_scratch ? n : 0);
    bool all_ok = true;
    for (int run = 0; run < config.runs; ++run) {
        Times times;
        const Timer init_timer;
        fill_chunk(buffers.chunk.data(), n, ctx.offset, config.seed);
        times.init = init_timer.elapsed();
        const Checksum before = config.verify == VerifyMode::fast ? checksum_chunk(buffers.chunk.data(), n) : Checksum{};
        MPI_Barrier(MPI_COMM_WORLD);
        version.sort(buffers, ctx, times);
        MPI_Barrier(MPI_COMM_WORLD);
        const Times worst = reduce_times_max(times);
        const std::string verified = verify_distributed(buffers.chunk, ctx, before);
        if (rank == 0) {
            const std::uint64_t block = version.uses_merge_options && config.overlap ? std::min(config.block, n) : 0;
            const std::string local_sort = version.uses_merge_options ? local_sort_name(config.local_sort) : "na";
            publish_report(RunReport{version.name, config.exp, procs, config.threads, config.tile, block, local_sort,
                                     version.uses_merge_options && config.overlap, run, worst, verified},
                           config.csv);
            all_ok = all_ok && verified != "FAIL";
        }
    }
    MPI_Finalize();
    return all_ok ? 0 : 1;
}
