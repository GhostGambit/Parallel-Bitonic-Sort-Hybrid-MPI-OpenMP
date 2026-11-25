#include "bitonic_kernels.hpp"
#include "common.hpp"

namespace {

struct SequentialResult {
    double seconds;
    bool ok;
};

RunReport sequential_report(const std::string& version, const Config& config, int run, double init,
                            const SequentialResult& result) {
    const std::string verified = config.verify == VerifyMode::none ? "SKIP" : (result.ok ? "PASS" : "FAIL");
    return RunReport{version, config.exp, 1, 1, 0, 0, "na", false, run, Times{init, result.seconds, 0.0}, verified};
}

}

int main(int argc, char** argv) {
    Config config;
    try {
        config = parse_config(argc, argv);
    } catch (const std::exception& problem) {
        std::cerr << "seq_main: " << problem.what() << std::endl;
        return 1;
    }
    const std::uint64_t total = config.total_elements();
    std::vector<std::uint32_t> data(total);
    std::vector<std::uint32_t> reference(total);
    bool all_ok = true;
    for (int run = 0; run < config.runs; ++run) {
        const Timer init_timer;
        fill_chunk(data.data(), total, 0, config.seed);
        const double init = init_timer.elapsed();
        reference = data;
        const Checksum before = checksum_chunk(data.data(), total);

        const Timer bitonic_timer;
        bitonic_sort_serial(data.data(), total);
        SequentialResult bitonic{bitonic_timer.elapsed(), true};

        const Timer std_timer;
        std::sort(reference.begin(), reference.end());
        SequentialResult standard{std_timer.elapsed(), true};

        if (config.verify == VerifyMode::full) {
            bitonic.ok = data == reference;
            standard.ok = is_sorted_chunk(reference.data(), total);
        } else if (config.verify == VerifyMode::fast) {
            bitonic.ok = is_sorted_chunk(data.data(), total) && checksum_chunk(data.data(), total) == before;
            standard.ok = is_sorted_chunk(reference.data(), total) && checksum_chunk(reference.data(), total) == before;
        }
        publish_report(sequential_report("seq_bitonic", config, run, init, bitonic), config.csv);
        publish_report(sequential_report("seq_stdsort", config, run, init, standard), config.csv);
        all_ok = all_ok && bitonic.ok && standard.ok;
    }
    return all_ok ? 0 : 1;
}
