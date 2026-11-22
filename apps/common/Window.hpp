#pragma once
// The windows of the toybox: the 2D Enright window, the 3D Enright window, the mesh window and the sphere
// window, each reading the core's runs. Only in GUI builds (SPRINGLS_BUILD_GUI).
#include <core/Cli.hpp>
#include <core/Experiment.hpp>

namespace toybox
{
// Opens the window of the invocation's domain and scenario: interactive, or offscreen for --frames
// (Enright with --no-gui; sphere) and --screenshot (mesh, sphere). A toybox::WindowRunner.
int RunWindow(const Invocation &invocation, Experiment &experiment);
}  // namespace toybox
