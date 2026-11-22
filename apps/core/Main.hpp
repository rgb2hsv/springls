#pragma once
// The single entry point: every executable is `return toybox::Main(argc, argv, demo)` (see
// apps/common/Toybox.hpp, which passes the window runner of GUI builds).
#include "Cli.hpp"
#include "Demos.hpp"
#include "Experiment.hpp"

namespace toybox
{
// Opens the selection's window (GUI, --frames, --screenshot). Enright: the experiment is initialized
// (and its CSV open); mesh: it is not (the window loads the mesh on its worker thread).
using WindowRunner = int (*)(const Invocation &invocation, Experiment &experiment);

// Parses the command line and runs the selection headless, or through `window` (null: built without the GUI).
int Main(int argc, char *argv[], const Demo &demo, WindowRunner window);

// The headless drivers: run to completion with progress lines, then the summary (Enright), or
// initialization and step statistics (mesh).
int RunHeadless(const Invocation &invocation, Experiment &experiment);
}  // namespace toybox
