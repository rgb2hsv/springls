#!/usr/bin/env bash
# Guards the mesh scenario (mesh-3d-dweave) against unintended result changes: runs it headless on the metasphere mesh
# for 2 steps and compares its statistics, minus the wall-clock times, byte for byte.
#
#   tests/baseline/viewer-baseline.sh capture <dir>   # record the reference
#   tests/baseline/viewer-baseline.sh compare <dir>   # re-run and diff against it (exit 1 on change)
#
# BUILD (default: build) selects the build directory. Only the run-to-run varying parts are dropped:
# ", init <t> s" on the first line and the trailing ", <t> s" of each step line.
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

# Runs it headless; writes its raw output to $1.log and the normalized statistics to $1.norm.
run_one() {
    local out=$1
    mkdir -p "$(dirname "$out")"
    "$(exe mesh-3d-dweave)" "$root/assets/model/metasphere.ply" --no-gui --no-cache --steps 2 > "$out.log" 2>&1
    sed -E 's/, init [0-9.]+ s//; s/, [0-9.]+ s$//' "$out.log" > "$out.norm"
}

case "$mode" in
    capture)
        run_one "$ref/viewer"
        echo "captured baseline in $ref"
        ;;
    compare)
        tmp=$(mktemp -d)
        trap 'rm -rf "$tmp"' EXIT
        run_one "$tmp/viewer"
        if diff -q "$ref/viewer.norm" "$tmp/viewer.norm" > /dev/null; then
            echo "baseline unchanged"
        else
            echo "CHANGED: viewer"
            diff "$ref/viewer.norm" "$tmp/viewer.norm" | head -10
            exit 1
        fi
        ;;
    *)
        echo "unknown mode $mode" >&2
        exit 2
        ;;
esac
