// Semi-Lagrangian vs CFL-limited doubleweave springls on the 3D Enright test.
//
// Adaptive doubleweave evolves its level set toward the springls with an upwind PDE, so every
// advection sub-step is capped at tuning.maxStepVoxels (a CFL condition). Semi-Lagrangian
// doubleweave advects the springls freely and rebuilds the level set from them after each
// sub-step, so its sub-steps are limited only by how much the flow stretches the springls. Both
// run the same test, by default on a 32^3 base grid refined to depth 2 (128^3 equivalent), T = 3,
// with a schedule cap of T/50 per sub-step: the cap the semi-Lagrangian method can use, far above
// what the CFL limit lets the adaptive doubleweave take.
//
//   sl-enright3d                                   # GUI, side by side, equal compute
//   sl-enright3d --pacing same-time                # GUI, compare shapes at equal simulation time
//   sl-enright3d --no-gui                          # headless comparison, prints sub-steps and errors
//   sl-enright3d --sl-only                         # GUI, the semi-Lagrangian run alone
//   sl-enright3d --sl-only --no-gui --frames out/sl3d   # render it to PNGs
#include "Experiment.hpp"
#ifdef SPRINGLS_GUI
#include "Gui.hpp"
#endif

#include <argparse/argparse.hpp>

#include <cmath>
#include <iostream>
#include <limits>

using namespace springls;

namespace
{
int RunHeadless(enright3d::Experiment &experiment)
{
    // Progress about every 10% of the period, by the run furthest behind.
    constexpr int kProgressReports = 10;
    const double period = experiment.period();
    int nextReport = 1;
    while (experiment.step()) {
        double t = period;
        for (const enright3d::Run &run : experiment.runs()) {
            if (!run.done) t = std::min(t, run.current.time);
        }
        if (t >= nextReport * period / kProgressReports) {
            std::cout << "  t = " << t << " / " << period << " (step " << experiment.stepCount() << ")";
            for (const enright3d::Run &run : experiment.runs()) {
                std::cout << "  " << MethodKey(run.method) << ": t " << run.current.time << ", " << run.sim->getAdvectionSubstepCount()
                          << " sub-steps, volume error " << 100.0 * run.current.relativeVolumeError << " %";
            }
            std::cout << std::endl;
            nextReport = static_cast<int>(std::floor(t / period * kProgressReports)) + 1;
        }
    }
    experiment.printSummary(std::cout);
    if (experiment.runs().size() == 2) {
        const auto &cfl = experiment.runs()[0];
        const auto &sl = experiment.runs()[1];
        const uint64_t cflSteps = cfl.sim->getAdvectionSubstepCount(), slSteps = sl.sim->getAdvectionSubstepCount();
        if (slSteps > 0 && sl.seconds > 0.0) {
            std::cout << "Semi-Lagrangian: " << static_cast<double>(cflSteps) / static_cast<double>(slSteps) << "x fewer advection sub-steps, "
                      << cfl.seconds / sl.seconds << "x less compute than the CFL-limited run" << std::endl;
        }
    }
    return 0;
}
}  // namespace

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("sl-enright3d");
    program.add_description(
        "Semi-Lagrangian doubleweave (strain-limited, level set rebuilt from springls) vs adaptive doubleweave "
        "(CFL-limited level-set PDE) on the 3D Enright test.");
    program.add_argument("--sl-only").default_value(false).implicit_value(true).help("run only the semi-Lagrangian method");
    program.add_argument("--pacing").default_value(std::string("equal-compute")).help("which run advances each step: equal-compute|same-time|lockstep");
    program.add_argument("-e", "--elements").default_value(std::string("quad")).help("springl element type: quad|triangle");
    program.add_argument("--no-gui").default_value(false).implicit_value(true).help("run without a window");
    program.add_argument("-g", "--grid").default_value(32).scan<'i', int>().help("base level set grid size, voxels per side");
    program.add_argument("-d", "--max-depth").default_value(2).scan<'i', int>().help("adaptive refinement depth (finest = grid * 2^depth)");
    program.add_argument("-T", "--period").default_value(3.0).scan<'g', double>().help("deformation period T");
    program.add_argument("--max-time-step").default_value(1.0 / 50.0).scan<'g', double>().help("max advection sub-step, as a fraction of T");
    program.add_argument("--substeps-per-step").default_value(1).scan<'i', int>().help("advection sub-steps per step (and per GUI frame / saved frame)");
    program.add_argument("--set").default_value(std::vector<std::string>{}).append().help("tuning override key=value (repeatable)");
    program.add_argument("--csv").default_value(std::string()).help("write per-step metrics to this CSV file");
    program.add_argument("--frames")
        .default_value(std::string())
        .help("save the rendered panels of every step to this directory as frame_000000.png, ... (no GUI widgets)");
    program.add_argument("--frame-size").default_value(720.0f).scan<'g', float>().help("panel size of saved frames in pixels");
    program.add_argument("--hide").default_value(std::string()).help("comma-separated layers to hide initially: iso,primal,dual");
    program.add_argument("--particles").default_value(false).implicit_value(true).help("show springl particles (hidden by default)");
    program.add_argument("--no-capsules").default_value(false).implicit_value(true).help("draw springls, not capsules");
    program.add_argument("--zoom").default_value(0.0f).scan<'g', float>().help("initial view zoom (0: the default pose's)");
    program.add_argument("-j", "--threads").default_value(0).scan<'i', int>().help("OpenMP threads (0 = default)");
    program.add_argument("--paused").default_value(false).implicit_value(true).help("GUI: start paused");
    program.add_argument("--exit-when-done").default_value(false).implicit_value(true).help("GUI: close the window once every run finishes");
    program.add_argument("--font")
        .default_value(std::string(
#ifdef SPRINGLS_FONT_FILE
            SPRINGLS_FONT_FILE
#endif
            ))
        .help("TrueType font for panel text");
    program.add_argument("--checkpoint-at")
        .default_value(std::vector<double>{})
        .append()
        .scan<'g', double>()
        .help("save each run's state once its time reaches this (repeatable), to --checkpoint-dir");
    program.add_argument("--checkpoint-dir").default_value(std::string("checkpoints")).help("directory for --checkpoint-at state files");
    program.add_argument("--resume").default_value(std::string()).help("start the (single) run from this state file instead of t = 0");
    program.add_argument("--steps").default_value(0).scan<'i', int>().help("stop after this many steps from the start or resume (0: no limit)");
    program.add_argument("--max-springl-growth")
        .default_value(0.0)
        .scan<'g', double>()
        .help("abort a run once its springl count exceeds this many times its count at the start or resume (0: no limit)");
    program.add_argument("--stop-at").default_value(std::numeric_limits<double>::infinity()).scan<'g', double>().help("stop runs once they reach this time");
    program.add_argument("--verbose").default_value(false).implicit_value(true).help("log simulation progress");
    try {
        program.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << err.what() << "\n\n" << program;
        return 2;
    }

    enright3d::ExperimentOptions options;
    const bool slOnly = program.get<bool>("--sl-only");
    if (slOnly) {
        options.methods = {Method3D::SemiLagrangianDoubleweave};
    } else {
        options.methods = {Method3D::AdaptiveDoubleweave, Method3D::SemiLagrangianDoubleweave};
    }
    if (!enright3d::ParsePacing(program.get<std::string>("--pacing"), options.pacing)) {
        std::cerr << "error: unknown pacing '" << program.get<std::string>("--pacing") << "'" << std::endl;
        return 2;
    }
    if (!ParseElementType(program.get<std::string>("--elements"), options.elementType)) {
        std::cerr << "error: --elements expects quad or triangle" << std::endl;
        return 2;
    }
    options.config.gridSize = program.get<int>("--grid");
    options.maxDepth = program.get<int>("--max-depth");
    options.config.period = program.get<double>("--period");
    options.config.maxTimeStepFraction = program.get<double>("--max-time-step");
    options.verbose = program.get<bool>("--verbose");
    const int substeps = program.get<int>("--substeps-per-step");
    if (options.config.gridSize < 8 || options.config.period <= 0.0 || options.maxDepth < 0 || substeps < 1) {
        std::cerr << "error: need --grid >= 8, --period > 0, --max-depth >= 0 and --substeps-per-step >= 1" << std::endl;
        return 2;
    }
    options.tuning.emplace_back("substepsPerStep", substeps);
    for (const std::string &entry : program.get<std::vector<std::string>>("--set")) {
        const size_t eq = entry.find('=');
        try {
            if (eq == std::string::npos) throw std::invalid_argument(entry);
            options.tuning.emplace_back(entry.substr(0, eq), std::stod(entry.substr(eq + 1)));
        } catch (const std::exception &) {
            std::cerr << "error: --set expects key=value, got '" << entry << "'" << std::endl;
            return 2;
        }
    }
    options.checkpointTimes = program.get<std::vector<double>>("--checkpoint-at");
    options.checkpointDir = program.get<std::string>("--checkpoint-dir");
    options.resumeFile = program.get<std::string>("--resume");
    options.stopAt = program.get<double>("--stop-at");
    if (program.get<int>("--steps") > 0) options.maxSteps = static_cast<uint64_t>(program.get<int>("--steps"));
    if (program.get<double>("--max-springl-growth") > 0.0) options.maxSpringlGrowth = program.get<double>("--max-springl-growth");
    SetThreads(program.get<int>("--threads"));

    enright3d::Experiment experiment;
    std::string error;
    if (!experiment.init(options, error)) {
        std::cerr << "error: " << error << std::endl;
        return 1;
    }
    const std::string csv = program.get<std::string>("--csv");
    if (!csv.empty() && !experiment.openCsv(csv)) {
        std::cerr << "error: could not write " << csv << std::endl;
        return 1;
    }
    std::cout << (slOnly ? "Semi-Lagrangian doubleweave 3D" : "Semi-Lagrangian vs CFL-limited doubleweave 3D") << ": Enright " << options.config.gridSize
              << "^3 base, depth " << options.maxDepth << ", T = " << options.config.period << ", sub-step cap T * " << options.config.maxTimeStepFraction;
    if (!slOnly) std::cout << ", pacing " << enright3d::PacingKey(options.pacing);
    std::cout << ", " << MaxThreads() << " thread(s)" << std::endl;

    const bool noGui = program.get<bool>("--no-gui");
    const std::string frames = program.get<std::string>("--frames");
#ifdef SPRINGLS_GUI
    enright3d::DisplayOptions display;
    display.windowTitle = slOnly ? "Semi-Lagrangian Doubleweave (Enright 3D)" : "Semi-Lagrangian vs CFL-limited Doubleweave (Enright 3D)";
    display.fontFile = program.get<std::string>("--font");
    display.framesDir = frames;
    display.framePanelSize = program.get<float>("--frame-size");
    display.startPaused = program.get<bool>("--paused");
    display.exitWhenDone = program.get<bool>("--exit-when-done");
    if (program.get<float>("--zoom") > 0.0f) display.camera.zoom = program.get<float>("--zoom");
    // Capsules show how the two constellations cover each other (as in enright3d for doubleweave).
    display.visibility.capsules = !program.get<bool>("--no-capsules");
    display.visibility.particles = program.get<bool>("--particles");
    std::string hide = program.get<std::string>("--hide");
    while (!hide.empty()) {
        const size_t comma = hide.find(',');
        const std::string layer = hide.substr(0, comma);
        hide = (comma == std::string::npos) ? std::string() : hide.substr(comma + 1);
        bool *flag = layer == "iso"         ? &display.visibility.isoSurface
                     : layer == "primal"    ? &display.visibility.primal
                     : layer == "dual"      ? &display.visibility.dual
                     : layer == "particles" ? &display.visibility.particles
                                            : nullptr;
        if (!flag) {
            std::cerr << "error: --hide: unknown layer '" << layer << "'" << std::endl;
            return 2;
        }
        *flag = false;
    }
    if (!noGui) return enright3d::RunGui(experiment, display);
    if (!frames.empty()) return enright3d::RunOffscreenFrames(experiment, display);
#else
    if (!noGui) std::cout << "(built without the GUI; running headless)" << std::endl;
    if (!frames.empty()) {
        std::cerr << "error: --frames needs the GUI build (SPRINGLS_BUILD_GUI=ON)" << std::endl;
        return 1;
    }
#endif
    return RunHeadless(experiment);
}
