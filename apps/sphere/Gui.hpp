#pragma once
#include <core/Cli.hpp>

namespace sphere_gui
{
// The sphere scenario's window: a panel on the left (play / step, time, metrics, layer toggles, iso spacing and
// arrow sliders, follow object, reset view) and the surface view to its right. The simulation initializes and
// steps on a worker thread, which also takes the snapshots.
// --screenshot path: a hidden window runs --steps steps (or to the end, whichever first), renders one
// 1600 x 900 frame of the whole window offscreen and writes it as PNG.
// --frames dir (with or without --no-gui): offscreen frame_NNNNNN.png images of the view (--frame-size
// square) every run.sphere.every steps (or, with --frame-time, each time the simulation time passes a multiple
// of run.sphere.frameTime) and at the last; without --steps it runs to the end.
// The camera follows the object (unless --no-follow), starting framed on it (about 3 object radii either side).
// Returns 0, or 1 when the setup fails (the error is printed, and in the interactive window also shown in the
// panel until the window is closed).
int RunSphereGui(const toybox::Invocation &inv);
}  // namespace sphere_gui
