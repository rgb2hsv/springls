#!/usr/bin/env bash
# Guards the existing 3D methods against unintended result changes: runs short headless 3D Enright
# tests (enright-3d-<method> for four methods, the toybox semi-Lagrangian pair) and compares their CSVs, minus the wall-clock
# step_seconds column, byte for byte. Capture also writes an enright-3d-dweave checkpoint (dweave, t = 0.1)
# into <dir>/checkpoint for testing --resume later; compare does not check it.
#
#   tests/baseline/enright3d-baseline.sh capture <dir>   # record the reference
#   tests/baseline/enright3d-baseline.sh compare <dir>   # re-run and diff against it (exit 1 on change)
#
# BUILD (default: build) selects the build directory.
set -euo pipefail

mode=${1:-}
ref=${2:-}
if [[ -z "$mode" || -z "$ref" ]]; then
    echo "usage: $0 capture|compare <dir>" >&2
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
common=(--no-gui -g 16 -T 0.5 -d 2 --steps 20)

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
        "$(exe enright-3d-$m)" "${common[@]}" --csv "$out/$m.csv" > "$out/$m.log" 2>&1
        normalize "$out/$m.csv" "$out/$m.norm"
    done
    "$(exe toybox)" --domain 3d -r dweave --grid-type adaptive --interface-tracking eulerian,semi-lagrangian "${common[@]}" --pacing lockstep --csv "$out/sl.csv" > "$out/sl.log" 2>&1
    normalize "$out/sl.csv" "$out/sl.norm"
}

case "$mode" in
    capture)
        run_all "$ref"
        # Checkpoint for resume tests: the run must pass t = 0.1, which takes more than 20 steps.
        mkdir -p "$ref/checkpoint"
        "$(exe enright-3d-dweave)" --no-gui -g 16 -T 0.5 -d 2 --checkpoint-at 0.1 \
            --checkpoint-dir "$ref/checkpoint" --stop-at 0.1 > "$ref/checkpoint.log" 2>&1
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
    *)
        echo "unknown mode $mode" >&2
        exit 2
        ;;
esac
