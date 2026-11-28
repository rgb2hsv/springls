#include "Window.hpp"

#include "../enright2d/Gui.hpp"
#include "../enright3d/Gui.hpp"
#include "../mesh/Gui.hpp"
#include "../sphere/Gui.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace toybox
{
namespace
{
int Run2D(const Invocation &inv, Experiment &experiment)
{
    enright::DisplayOptions display;
    if (inv.slComparison) {
        display.windowTitle = inv.slOnly ? "Semi-Lagrangian DoubleWeave (Enright 2D)" : "Semi-Lagrangian vs CFL-limited DoubleWeave (Enright 2D)";
    }
    display.fontFile = inv.font;
    display.framesDir = inv.frames;
    display.framePanelSize = inv.frameSize;
    display.startPaused = inv.paused;
    display.stepsPerFrame = inv.stepsPerFrame;
    display.exitWhenDone = inv.exitWhenDone;
    if (!inv.noGui) return enright::RunGui(experiment, display);
    return enright::RunOffscreenFrames(experiment, display);
}

int Run3D(const Invocation &inv, Experiment &experiment)
{
    enright3d::DisplayOptions display;
    if (inv.slComparison) {
        display.windowTitle = inv.slOnly ? "Semi-Lagrangian DoubleWeave (Enright 3D)" : "Semi-Lagrangian vs CFL-limited DoubleWeave (Enright 3D)";
    }
    display.fontFile = inv.font;
    display.framesDir = inv.frames;
    display.framePanelSize = inv.frameSize;
    display.startPaused = inv.paused;
    display.exitWhenDone = inv.exitWhenDone;
    if (inv.zoom > 0.0f) display.camera.zoom = inv.zoom;
    // Capsules show how the two DoubleWeave constellations cover each other, so they are on by
    // default whenever a DoubleWeave method runs.
    bool dweave = false;
    for (const Entry *e : inv.entries) dweave |= e->config.representation == Representation::Dweave;
    display.visibility.capsules = (inv.capsules || dweave) && !inv.noCapsules;
    display.visibility.particles = inv.particles;
    std::vector<std::string> layers;
    std::string error;
    if (!ParseHiddenLayers(inv.hide, layers, error)) {
        std::cerr << error << std::endl;
        return 2;
    }
    for (const std::string &layer : layers) {
        bool *flag = layer == "iso"      ? &display.visibility.isoSurface
                     : layer == "primal" ? &display.visibility.primal
                     : layer == "dual"   ? &display.visibility.dual
                     : layer == "tracked"  ? &display.visibility.trackedSurface
                                         : &display.visibility.particles;
        *flag = false;
    }
    if (!inv.noGui) return enright3d::RunGui(experiment, display);
    return enright3d::RunOffscreenFrames(experiment, display);
}

int RunMesh(const Invocation &inv)
{
    viewer::ScreenshotOptions screenshot;
    screenshot.path = inv.screenshot;
    screenshot.steps = inv.stepsGiven ? static_cast<int>(inv.limits.maxSteps) : 0;
    return viewer::RunViewerGui(*inv.entries.front(), inv.run.mesh, inv.matcap, !inv.play, inv.zUp, screenshot);
}
}  // namespace

int RunWindow(const Invocation &invocation, Experiment &experiment)
{
    if (invocation.scenario == Scenario::Mesh) return RunMesh(invocation);
    if (Surface(invocation.scenario)) return sphere_gui::RunSphereGui(invocation);
    return invocation.domain == Domain::D2 ? Run2D(invocation, experiment) : Run3D(invocation, experiment);
}
}  // namespace toybox
