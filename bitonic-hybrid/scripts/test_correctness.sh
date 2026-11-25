#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
make all

cores=$(nproc)
if command -v mpirun >/dev/null 2>&1; then
    launcher=mpirun
else
    launcher=mpiexec
fi
flavour=other
if "$launcher" --version 2>&1 | grep -qi "open mpi"; then
    flavour=openmpi
fi

pass=0
fail=0
skip=0

launch() {
    local procs=$1 threads=$2
    shift 2
    local extra=()
    if [[ $flavour == openmpi && $((procs * threads)) -gt $cores ]]; then
        extra=(--oversubscribe)
    fi
    "$launcher" -n "$procs" "${extra[@]}" "$@"
}

record() {
    printf '%-4s %s\n' "$1" "$2"
    case $1 in
        PASS) pass=$((pass + 1)) ;;
        FAIL) fail=$((fail + 1)) ;;
        SKIP) skip=$((skip + 1)) ;;
    esac
}

check() {
    local label=$1
    shift
    local output
    if output=$("$@" 2>&1) && grep -q 'verify=PASS' <<<"$output" && ! grep -q 'verify=FAIL' <<<"$output"; then
        record PASS "$label"
    else
        record FAIL "$label"
        printf '%s\n' "$output" | sed 's/^/      /'
    fi
}

check_mpi() {
    local label=$1 exp=$2 procs=$3 threads=$4
    shift 4
    if (((1 << exp) / procs < 2)); then
        record SKIP "$label"
        return
    fi
    check "$label" launch "$procs" "$threads" "$@" --exp "$exp" --threads "$threads"
}

for exp in 16 18 20; do
    check "seq_main exp=$exp" ./bin/seq_main --exp "$exp" --verify full
    for procs in 1 2 4 8; do
        for threads in 1 2 4; do
            for tile in 0 1024 4096; do
                check_mpi "v1 exp=$exp P=$procs T=$threads tile=$tile" "$exp" "$procs" "$threads" \
                    ./bin/hybrid_v1 --tile "$tile" --verify full
            done
            for local_sort in bitonic std; do
                check_mpi "v2 exp=$exp P=$procs T=$threads local=$local_sort" "$exp" "$procs" "$threads" \
                    ./bin/hybrid_v2 --local-sort "$local_sort" --verify full
                check_mpi "v2 exp=$exp P=$procs T=$threads local=$local_sort overlap block=1024" "$exp" "$procs" \
                    "$threads" ./bin/hybrid_v2 --local-sort "$local_sort" --overlap --block 1024 --verify full
            done
        done
    done
done

check_mpi "v1 exp=24 P=4 T=2 fast" 24 4 2 ./bin/hybrid_v1 --verify fast
check_mpi "v2 exp=24 P=4 T=2 fast" 24 4 2 ./bin/hybrid_v2 --verify fast
check_mpi "v2 exp=24 P=4 T=2 fast overlap" 24 4 2 ./bin/hybrid_v2 --local-sort std --overlap --verify fast

total=$((pass + fail + skip))
echo "TOTAL $total  PASS $pass  FAIL $fail  SKIP $skip"
if ((fail > 0)); then
    exit 1
fi
