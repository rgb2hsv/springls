#!/usr/bin/env bash
# Guards the existing 2D methods against unintended result changes: runs short headless Enright
# tests and compares their CSVs, minus the wall-clock step_seconds column, byte for byte.
#
#   tests/baseline/enright2d-baseline.sh capture <dir>   # record the reference
#   tests/baseline/enright2d-baseline.sh compare <dir>   # re-run and diff against it (exit 1 on change)
#   tests/baseline/enright2d-baseline.sh threads <dir>   # multi-object methods: 1 thread vs all
#
# BUILD (default: build) selects the build directory.
set -euo pipefail

mode=${1:-}
ref=${2:-}
if [[ -z "$mode" || -z "$ref" ]]; then
    echo "usage: $0 capture|compare|threads <dir>" >&2
    exit 2
fi
root=$(cd "$(dirname "$0")/../.." && pwd)
build=${BUILD:-$root/build}
# A program: build/demo/<name> (the per-configuration programs) or build/<name> (toybox), .exe on Windows.
exe() {
    local d
    for d in "$build/demo" "$build"; do
        if [[ -x "$d/$1.exe" ]]; then echo "$d/$1.exe"; return; fi
        if [[ -x "$d/$1" ]]; then echo "$d/$1"; return; fi
    done
    echo "$build/$1"
}

methods=(springls adaptive-springls dweave adaptive-dweave)
common=(--no-gui -g 32 -T 1 -d 2)

# Drops the step_seconds column (found by header name) from CSV $1 into $2.
normalize() {
    awk -F, -v OFS=, '
        NR == 1 { for (i = 1; i <= NF; i++) if ($i == "step_seconds") drop = i }
        { out = ""; for (i = 1; i <= NF; i++) if (i != drop) out = out (out == "" ? "" : OFS) $i; print out }
    ' "$1" > "$2"
}

run_all() {
    local out=$1
    mkdir -p "$out"
    for m in "${methods[@]}"; do
        "$(exe enright-2d-$m)" "${common[@]}" --csv "$out/$m.csv" > "$out/$m.log" 2>&1
        normalize "$out/$m.csv" "$out/$m.norm"
    done
    "$(exe toybox)" -r dweave --grid-type adaptive --interface-tracking eulerian,semi-lagrangian "${common[@]}" --pacing lockstep --csv "$out/sl.csv" > "$out/sl.log" 2>&1
    normalize "$out/sl.csv" "$out/sl.norm"
}

case "$mode" in
    capture)
        run_all "$ref"
        echo "captured baseline in $ref"
        ;;
    compare)
        tmp=$(mktemp -d)
        trap 'rm -rf "$tmp"' EXIT
        run_all "$tmp"
        status=0
        for f in "${methods[@]}" sl; do
            if ! diff -q "$ref/$f.norm" "$tmp/$f.norm" > /dev/null; then
                echo "CHANGED: $f"
                diff "$ref/$f.norm" "$tmp/$f.norm" | head -10
                status=1
            fi
        done
        [[ $status -eq 0 ]] && echo "baseline unchanged"
        exit $status
        ;;
    threads)
        # Multi-object methods give the same results on one thread as on all of them.
        mkdir -p "$ref"
        for n in 1 0; do
            "$(exe toybox)" "${common[@]}" -m multi -j "$n" --csv "$ref/multi-j$n.csv" > "$ref/multi-j$n.log" 2>&1
            normalize "$ref/multi-j$n.csv" "$ref/multi-j$n.norm"
        done
        if diff -q "$ref/multi-j1.norm" "$ref/multi-j0.norm" > /dev/null && diff -q "$ref/multi-j1-labels.csv" "$ref/multi-j0-labels.csv" > /dev/null; then
            echo "thread count does not change results"
        else
            echo "CHANGED with thread count"
            diff "$ref/multi-j1.norm" "$ref/multi-j0.norm" | head -10
            exit 1
        fi
        ;;
    *)
        echo "unknown mode $mode" >&2
        exit 2
        ;;
esac
