#pragma once
#include "MeshTypes.hpp"

#include <filesystem>
#include <string>

namespace viewer
{
// --screenshot: instead of running interactively, load, run `steps` steps, render one frame of the full window
// (a fixed 1600 x 900, sidebars included) offscreen and write it to `path` as PNG.
struct ScreenshotOptions {
    std::filesystem::path path;  // empty: interactive
    int steps = 0;
};

// Interactive window: an ImGui sidebar with the controls and statistics, the matcap-shaded view in the
// middle and the grid slices on the right. The session loads and steps on a worker thread, so the
// window stays responsive while a step runs. initialMatcap: a matcap name (empty: the default).
// The worker builds every simulation (load, Reset, mesh and grid edits) through entry.make.
int RunViewerGui(const toybox::Entry &entry, const SessionOptions &options, const std::string &initialMatcap, bool startPaused, bool zUp,
                 const ScreenshotOptions &screenshot = {});
}  // namespace viewer
