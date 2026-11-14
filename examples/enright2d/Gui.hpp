#pragma once
#include "Experiment.hpp"

#include <filesystem>
#include <string>

namespace enright
{
struct DisplayOptions {
    std::string windowTitle = "Enright 2D: Springls, Adaptive, Doubleweave";
    std::string fontFile;
    // When set, every simulation step is rendered (panels only, no GUI widgets) to
    // <framesDir>/frame_NNNNNN.png.
    std::filesystem::path framesDir;
    float framePanelSize = 720.0f;  // panel size of saved frames, in pixels
    bool startPaused = false;
    int stepsPerFrame = 1;
    bool exitWhenDone = false;
};

// Interactive window: an ImGui control sidebar and the NanoVG panels.
int RunGui(Experiment &experiment, const DisplayOptions &options);
// No window: runs to completion, rendering frames through a hidden OpenGL context.
int RunOffscreenFrames(Experiment &experiment, const DisplayOptions &options);
}  // namespace enright
