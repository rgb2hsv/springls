#pragma once
// The toybox command line: one parser holding the union of the old programs' flags. A demo's
// arguments are parsed first and the user's after them (user values win); the selection is resolved
// through the Registry and the default profile fills what neither gives.
#include "Configuration.hpp"
#include "Demos.hpp"
#include "Experiment.hpp"
#include "Registry.hpp"
#include "Simulation.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace toybox
{
struct Invocation {
    std::string program;
    std::vector<const Entry *> entries;  // the runs, in order
    Domain domain = Domain::D2;
    Scenario scenario = Scenario::Enright;
    bool semiLagrangian = false;  // some run is semi-Lagrangian (the profile in use)
    // Selected by axes with semi-Lagrangian tracking (sl-enright2d / sl-enright3d, --sl-only): their
    // banner, progress lines, comparison line and window titles.
    bool slComparison = false;
    bool slOnly = false;      // every run is semi-Lagrangian
    std::string methodLabel;  // for the banner: the -m value ("all methods" for all) or the selected keys

    RunOptions run;
    Pacing pacing = Pacing::Lockstep;
    Limits limits;

    bool list = false, listTuning = false;
    bool noGui = false;
    std::filesystem::path csv, frames;
    float frameSize = 720.0f;
    int threads = 0;
    bool paused = false, exitWhenDone = false;
    int stepsPerFrame = 1;
    std::string font;
    // 3D Enright view
    std::string hide;
    bool particles = false, capsules = false, noCapsules = false;
    float zoom = 0.0f;
    // mesh scenario
    std::string matcap;
    bool play = false, zUp = false;
    std::filesystem::path screenshot;
    bool stepsGiven = false;  // --steps given (the mesh window runs that many steps, then waits)
    bool noFollow = false;    // sphere window: start with the camera not following the object

    std::vector<std::string> notes;  // e.g. configurations of the selection that are not implemented
};

// Parses demo.defaultArgs and then argv. On failure returns false with the message (starting with
// "error: ", or argparse's message and the usage) and the exit code (2 for usage errors).
bool ParseInvocation(const Demo &demo, int argc, const char *const argv[], Invocation &out, std::string &error, int &exitCode);

// The sphere window's layers --hide accepts: regions, isolines, springls, contour, arrows, reference.
bool ParseSphereLayers(const std::string &hide, std::vector<std::string> &layers, std::string &error);

// The 3D window's layers --hide accepts.
bool ParseHiddenLayers(const std::string &hide, std::vector<std::string> &layers, std::string &error);
}  // namespace toybox
