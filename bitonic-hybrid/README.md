# Parallel Bitonic Sort (Hybrid MPI + OpenMP)

Bitonic sort of 32-bit unsigned integers on a hybrid distributed and shared-memory model. MPI splits the array into P chunks, and OpenMP threads share the work inside each rank. N = 2^exp and P must be powers of two.

## Requirements

- A C++17 compiler with OpenMP (GCC 9 or newer)
- An MPI implementation with an `mpicxx` wrapper and `mpirun` or `mpiexec`: Open MPI, MPICH or MS-MPI
- GNU make and bash
- Python 3 with pandas and matplotlib, for the plots

On Ubuntu or Debian:

```bash
sudo apt install build-essential openmpi-bin libopenmpi-dev python3-pandas python3-matplotlib
```

## Build

```bash
make all
make clean
```

`make all` builds `bin/seq_main`, `bin/hybrid_v1` and `bin/hybrid_v2` with `-std=c++17 -O3 -march=native -fopenmp -Wall -Wextra`. You can also build each target on its own: `make seq_main`, `make hybrid_v1` or `make hybrid_v2`.

## Binaries

| Binary | What it does |
|---|---|
| `bin/seq_main` | Sequential bitonic sort on one core, plus `std::sort` as a reference |
| `bin/hybrid_v1` | The global bitonic network. Stages with j < N/P run locally with OpenMP (optionally cache-tiled); stages with j ≥ N/P exchange whole chunks with `MPI_Sendrecv` |
| `bin/hybrid_v2` | Each rank sorts its chunk locally first, then a bitonic network runs over the ranks with merge-split. An optional overlapped mode exchanges blocks with `MPI_Isend`/`MPI_Irecv` |

Examples:

```bash
./bin/seq_main --exp 24 --verify fast
mpirun -np 4 ./bin/hybrid_v1 --exp 24 --threads 2 --tile 4096 --verify fast
mpirun -np 4 ./bin/hybrid_v2 --exp 24 --threads 2 --local-sort std --verify fast
mpirun -np 4 ./bin/hybrid_v2 --exp 24 --threads 2 --local-sort std --overlap --block 65536 --verify fast
```

### Flags

All three binaries accept the same flags. Flags that a binary doesn't use are accepted and ignored.

| Flag | Default | Meaning |
|---|---|---|
| `--exp E` | 20 | Array size N = 2^E, with E between 1 and 34 |
| `--seed S` | 12345 | Generator seed. Element g is the upper 32 bits of splitmix64(S + g·0x9E3779B97F4A7C15), so the input is the same for any P |
| `--threads T` | `omp_get_max_threads()` | OpenMP threads per rank |
| `--tile B` | 4096 | Cache tile in elements: 0 turns tiling off, otherwise a power of two ≥ 2. Used by V1 and by the bitonic local sort of V2 |
| `--verify M` | fast | `full` gathers the result and compares it with `std::sort` (only allowed when E ≤ 22). `fast` checks that every chunk is sorted, checks the boundaries between ranks, and compares sum and sum-of-squares checksums. `none` skips verification |
| `--runs R` | 1 | Number of repetitions. The data is regenerated before each run, and each run is reported and logged |
| `--csv PATH` | none | Appends one row per run to PATH, writing the header if the file is new |
| `--local-sort X` | bitonic | V2 local phase: `bitonic` (tiled bitonic kernels) or `std` (parallel `std::sort` with pairwise merges) |
| `--overlap` | off | V2 only: block-pipelined non-blocking exchange with a streaming merge on the master thread |
| `--block B` | 65536 | Overlap block size in elements, a power of two. It is clamped to N/P |

Each run prints one line on rank 0:

```
v1 exp=24 N=16777216 P=4 T=2 tile=4096 block=0 local_sort=na overlap=0 run=0 init=... comp=... comm=... sort=... verify=PASS
```

Every time is the maximum over ranks. `sort` is `comp + comm` and does not include data generation or verification.

CSV columns: `version,exp,N,procs,threads,tile,block,local_sort,overlap,run,t_init,t_comp,t_comm,t_sort,verified`.

## Tests

```bash
./scripts/test_correctness.sh
```

The script builds everything, then runs the full-verification matrix: exp ∈ {16, 18, 20}, `seq_main`, V1 over P ∈ {1,2,4,8} × T ∈ {1,2,4} × tile ∈ {0, 1024, 4096}, and V2 over P × T × local sort, each with overlap off and with overlap on at block 1024. It ends with fast-verified runs at exp 24. It prints PASS, FAIL or SKIP for each case and a total, and exits non-zero if anything fails. With Open MPI it adds `--oversubscribe` only when P × T is larger than `nproc`.

## Benchmarks

```bash
./scripts/run_benchmarks.sh --max-cores 16 --runs 5 --sizes "24 26 27 28"
./scripts/run_benchmarks.sh --quick
```

| Option | Default | Meaning |
|---|---|---|
| `--max-cores C` | `nproc` | Largest total core count tested, from {1, 2, 4, 8, 16} |
| `--runs R` | 5 | Repetitions inside each binary |
| `--sizes "..."` | `"24 26 27"` | Exponents to benchmark. 28 runs only when at least 4 GB of memory is free |
| `--out PATH` | `results/bench.csv` | Main CSV. `tile_sweep.csv` and `block_sweep.csv` are written next to it |
| `--quick` | off | Only exp 24 with 2 runs, as a smoke test |

For every size and every split P × T = c, the script runs `seq_main`, V1 (tile 4096), V2 bitonic, V2 std, and V2 std with overlap at block 65536. It then runs a V1 tile sweep at P = 1, T = max cores, and a V2 overlap block sweep. Pinning depends on the MPI flavour:

- Open MPI: `--map-by slot:PE=T --bind-to core -x OMP_PROC_BIND=close -x OMP_PLACES=cores`
- MPICH: `-bind-to core:T`
- MS-MPI: no pinning, only `OMP_NUM_THREADS`

The script stops if any run fails verification. Set `--max-cores` to the number of physical cores so the runs don't land on hyper-threads.

## Plots

```bash
python3 scripts/plot_results.py
```

The script reads `results/bench.csv`, `results/tile_sweep.csv` and `results/block_sweep.csv`. For each size it writes these files to `plots/` at 200 dpi:

- `time_vs_cores_<exp>.png`
- `speedup_vs_cores_<exp>.png`
- `efficiency_vs_cores_<exp>.png`
- `split_comparison_<exp>.png`
- `breakdown_<exp>.png`

It also writes `tile_size_vs_time.png`, `block_size_vs_time.png` and `results/summary.csv`, which holds the best configuration per size and version with its speedup and efficiency.

## Project structure

```
bitonic-hybrid/
  Makefile
  include/common.hpp            config parsing, generator, checksums, timing, CSV, shared MPI driver and verification
  include/bitonic_kernels.hpp   serial and OpenMP compare-exchange stages, tiled local phases
  src/seq_main.cpp              sequential bitonic and std::sort baselines
  src/hybrid_v1.cpp             global network, OpenMP local stages, MPI_Sendrecv remote stages
  src/hybrid_v2.cpp             local sort, merge-split network, merge-path, overlapped exchange
  scripts/test_correctness.sh   correctness matrix
  scripts/run_benchmarks.sh     benchmark runner with pinning
  scripts/plot_results.py       plots and summary table
  results/                      CSV output
  plots/                        PNG output
  report/report.md              project report
```
