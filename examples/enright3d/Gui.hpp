#pragma once
#include "Experiment.hpp"
#include "Renderer.hpp"

#include <filesystem>
#include <string>

namespace enright3d
{
struct DisplayOptions {
    std::string windowTitle = "Enright 3D: Springls and Doubleweave";
    std::string fontFile;
    // When set, every simulation step is rendered (panels only, no GUI widgets) to
    // <framesDir>/frame_NNNNNN.png.
    std::filesystem::path framesDir;
    float framePanelSize = 720.0f;  // panel size of saved frames, in pixels
    bool startPaused = false;
    bool exitWhenDone = false;
    // Initial view (the GUI can change both).
    Visibility visibility;
    OrbitCamera camera;
};

// Interactive window: an ImGui control sidebar and the OpenGL panels. The simulation steps on a
// worker thread, so the window stays responsive while a step runs.
int RunGui(Experiment &experiment, const DisplayOptions &options);
// No window: runs to completion, rendering frames through a hidden OpenGL context.
int RunOffscreenFrames(Experiment &experiment, const DisplayOptions &options);
}  // namespace enright3d
