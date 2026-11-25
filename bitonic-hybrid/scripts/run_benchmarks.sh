#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

max_cores=$(nproc)
runs=5
sizes="24 26 27"
out=results/bench.csv

while (($# > 0)); do
    case $1 in
        --max-cores) max_cores=$2; shift 2 ;;
        --runs) runs=$2; shift 2 ;;
        --sizes) sizes=$2; shift 2 ;;
        --out) out=$2; shift 2 ;;
        --quick) sizes="24"; runs=2; shift ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

results_dir=$(dirname "$out")
tile_csv=$results_dir/tile_sweep.csv
block_csv=$results_dir/block_sweep.csv
mkdir -p "$results_dir"
rm -f "$out" "$tile_csv" "$block_csv"

make all

if command -v mpirun >/dev/null 2>&1; then
    launcher=mpirun
else
    launcher=mpiexec
fi
version_text=$("$launcher" --version 2>&1 || true)
if grep -qi "open mpi" <<<"$version_text"; then
    flavour=openmpi
elif grep -qiE "hydra|mpich" <<<"$version_text"; then
    flavour=mpich
else
    flavour=msmpi
fi
physical_cores=$max_cores
if command -v lscpu >/dev/null 2>&1; then
    physical_cores=$(lscpu -p=core,socket | grep -v '^#' | sort -u | wc -l)
fi
echo "launcher=$launcher flavour=$flavour max_cores=$max_cores physical_cores=$physical_cores runs=$runs sizes=$sizes"

launch_command() {
    local procs=$1 threads=$2
    case $flavour in
        openmpi)
            local hwthreads=()
            if ((procs * threads > physical_cores)); then
                hwthreads=(--use-hwthread-cpus)
            fi
            echo "$launcher -np $procs ${hwthreads[*]} --map-by slot:PE=$threads --bind-to core -x OMP_NUM_THREADS=$threads -x OMP_PROC_BIND=close -x OMP_PLACES=cores"
            ;;
        mpich)
            echo "$launcher -np $procs -bind-to core:$threads -env OMP_NUM_THREADS $threads -env OMP_PROC_BIND close -env OMP_PLACES cores"
            ;;
        *)
            echo "$launcher -n $procs -env OMP_NUM_THREADS $threads"
            ;;
    esac
}

memory_available_kb() {
    local available
    available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    if [[ -z $available ]]; then
        available=$(awk '/^MemFree:/ {print $2}' /proc/meminfo)
    fi
    echo "${available:-0}"
}

core_counts=()
for c in 1 2 4 8 16; do
    if ((c <= max_cores)); then
        core_counts+=("$c")
    fi
done
largest=${core_counts[-1]}

jobs=()
add_job() {
    jobs+=("$1|$2|$3|$4")
}

for exp in $sizes; do
    if ((exp >= 28)); then
        available_kb=$(memory_available_kb)
        if ((available_kb < 4 * 1024 * 1024)); then
            echo "skipping exp=$exp: only $((available_kb / 1024)) MB available, 4096 MB required"
            continue
        fi
    fi
    common="--exp $exp --runs $runs --verify fast"
    add_job "$out" 1 1 "./bin/seq_main $common"
    for c in "${core_counts[@]}"; do
        for ((procs = 1; procs <= c; procs *= 2)); do
            threads=$((c / procs))
            add_job "$out" "$procs" "$threads" "./bin/hybrid_v1 $common --threads $threads --tile 4096"
            add_job "$out" "$procs" "$threads" "./bin/hybrid_v2 $common --threads $threads --local-sort bitonic"
            add_job "$out" "$procs" "$threads" "./bin/hybrid_v2 $common --threads $threads --local-sort std"
            add_job "$out" "$procs" "$threads" "./bin/hybrid_v2 $common --threads $threads --local-sort std --overlap --block 65536"
        done
    done
    for tile in 0 1024 4096 16384 65536 262144; do
        add_job "$tile_csv" 1 "$largest" "./bin/hybrid_v1 $common --threads $largest --tile $tile"
    done
    if ((largest >= 4)); then
        block_procs=$((largest / 2))
    else
        block_procs=2
    fi
    block_threads=$((largest / block_procs > 0 ? largest / block_procs : 1))
    for block in 4096 65536 1048576; do
        add_job "$block_csv" "$block_procs" "$block_threads" "./bin/hybrid_v2 $common --threads $block_threads --local-sort std --overlap --block $block"
    done
done

total=${#jobs[@]}
index=0
for job in "${jobs[@]}"; do
    index=$((index + 1))
    IFS='|' read -r csv procs threads command <<<"$job"
    if [[ $command == ./bin/seq_main* ]]; then
        full_command="$command --csv $csv"
    else
        full_command="$(launch_command "$procs" "$threads") $command --csv $csv"
    fi
    echo "[$index/$total] P=$procs T=$threads $command"
    output=$(OMP_NUM_THREADS=$threads $full_command 2>&1) || {
        printf '%s\n' "$output"
        echo "run failed, aborting" >&2
        exit 1
    }
    printf '%s\n' "$output" | sed 's/^/    /'
    if grep -q 'verify=FAIL' <<<"$output"; then
        echo "verification failed, aborting" >&2
        exit 1
    fi
done

echo "done: $out $tile_csv $block_csv"
