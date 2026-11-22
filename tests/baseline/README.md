# Enright 2D baseline

`enright2d-baseline.sh` runs short headless Enright tests (grid 32, T = 1, depth 2) with `enright-2d-springls`, `enright-2d-adaptive-springls`, `enright-2d-dweave`, `enright-2d-adaptive-dweave` and the semi-Lagrangian pair (`toybox -r dweave --grid-type adaptive --interface-tracking eulerian,semi-lagrangian`), and drops the `step_seconds` timing column from each CSV.

- `tests/baseline/enright2d-baseline.sh capture build/baseline` records the reference before a change.
- `tests/baseline/enright2d-baseline.sh compare build/baseline` re-runs and fails if any result changed.
- `tests/baseline/enright2d-baseline.sh threads build/threads` runs the multi-object methods (`toybox -m multi`) on one thread and on all threads and fails if results differ.

# Enright 3D baseline

`enright3d-baseline.sh` runs short headless 3D Enright tests (grid 16, T = 0.5, depth 2, `--steps 20`) with `enright-3d-springls|adaptive-springls|dweave|adaptive-dweave` and the 3D semi-Lagrangian pair (`toybox --domain 3d -r dweave --grid-type adaptive --interface-tracking eulerian,semi-lagrangian --pacing lockstep`), and drops the `step_seconds` column (by header name) from each CSV.

- `tests/baseline/enright3d-baseline.sh capture build/baseline3d` records the reference.
- `tests/baseline/enright3d-baseline.sh compare build/baseline3d` re-runs and fails if any result changed.

`capture` also writes the checkpoint used to test `--resume` after refactors, into `build/baseline3d/checkpoint`. It is not part of `compare`. It was produced (before the toybox refactor, by `enright3d -m dweave`) as:

```
enright-3d-dweave --no-gui -g 16 -T 0.5 -d 2 --checkpoint-at 0.1 --checkpoint-dir build/baseline3d/checkpoint --stop-at 0.1
```

giving `dweave_quad_g16_d2_t0.1000.state` and `.vdb` (state saved at t = 0.10125, step 54; the checkpoint run needs more than 20 steps, so it is separate from the `--steps 20` runs).

# Mesh scenario baseline

`viewer-baseline.sh` runs `mesh-3d-dweave assets/model/metasphere.ply --no-gui --no-cache --steps 2` and compares the printed statistics (depth, cell and face counts, per-step primal/dual/iso-face/leaf counts). Only the wall-clock times, which vary run to run, are removed: `, init <t> s` on the first line and the trailing `, <t> s` of each step line.

- `tests/baseline/viewer-baseline.sh capture build/baseline-viewer` records the reference.
- `tests/baseline/viewer-baseline.sh compare build/baseline-viewer` re-runs and fails if the statistics changed.

Both scripts take `BUILD` (default `build`) to pick the build directory. The viewer run takes about 20 s.
