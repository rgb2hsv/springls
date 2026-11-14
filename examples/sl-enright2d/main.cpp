// Semi-Lagrangian vs CFL-limited doubleweave springls on the 2D Enright test.
//
// Adaptive doubleweave evolves its level set toward the springls with an upwind PDE, so every
// advection sub-step is capped at a fraction of a cell (tuning.maxStepPixels, a CFL condition).
// Semi-Lagrangian doubleweave advects the springls freely and rebuilds the level set from them
// after each sub-step, so its sub-steps are limited only by how much the flow stretches the
// springls. Both run the same test, by default at 128^2, T = 8, depth 2, with a schedule cap of
// T/50 per sub-step: the cap the semi-Lagrangian method can use, far above what the CFL limit lets
// the adaptive doubleweave take.
//
//   sl-enright2d                              # GUI, side by side, equal compute
//   sl-enright2d --pacing same-time           # GUI, compare shapes at equal simulation time
//   sl-enright2d --no-gui                     # headless comparison, prints sub-steps and errors
//   sl-enright2d --sl-only                    # GUI, the semi-Lagrangian run alone
//   sl-enright2d --sl-only --no-gui --frames out/sl   # render the semi-Lagrangian run to PNGs
#include "Experiment.hpp"
#ifdef SPRINGLS_GUI
#include "Gui.hpp"
#endif

#include <argparse/argparse.hpp>

#include <cmath>
#include <iostream>

using namespace springls;

namespace
{
int RunHeadless(enright::Experiment &experiment)
{
    // Progress about every 10% of the period, by the run furthest behind.
    constexpr int kProgressReports = 10;
    const double period = experiment.config().period;
    int nextReport = 1;
    while (experiment.step()) {
        double t = period;
        for (const enright::Run &run : experiment.runs()) {
            if (!run.done) t = std::min(t, run.current.time);
        }
        if (t >= nextReport * period / kProgressReports) {
            std::cout << "  t = " << t << " / " << period << " (step " << experiment.stepCount() << ")" << std::endl;
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
    argparse::ArgumentParser program("sl-enright2d");
    program.add_description(
        "Semi-Lagrangian doubleweave (strain-limited, level set rebuilt from springls) vs adaptive doubleweave "
        "(CFL-limited level-set PDE) on the 2D Enright test.");
    program.add_argument("--sl-only").default_value(false).implicit_value(true).help("run only the semi-Lagrangian method");
    program.add_argument("--pacing").default_value(std::string("equal-compute")).help("which run advances each step: equal-compute|same-time|lockstep");
    program.add_argument("--no-gui").default_value(false).implicit_value(true).help("run without a window");
    program.add_argument("-g", "--grid").default_value(128).scan<'i', int>().help("base level set grid size");
    program.add_argument("-T", "--period").default_value(8.0).scan<'g', double>().help("deformation period T");
    program.add_argument("-d", "--max-depth").default_value(2).scan<'i', int>().help("adaptive refinement depth (finest = grid * 2^depth)");
    program.add_argument("--max-time-step").default_value(1.0 / 50.0).scan<'g', double>().help("max advection sub-step, as a fraction of T");
    program.add_argument("--substeps-per-step").default_value(1).scan<'i', int>().help("advection sub-steps per step (and per GUI frame / saved frame)");
    program.add_argument("--set").default_value(std::vector<std::string>{}).append().help("tuning override key=value (repeatable)");
    program.add_argument("--csv").default_value(std::string()).help("write per-step metrics to this CSV file");
    program.add_argument("--frames")
        .default_value(std::string())
        .help("save the rendered panels of every step to this directory as frame_000000.png, ... (no GUI widgets)");
    program.add_argument("--frame-size").default_value(720.0f).scan<'g', float>().help("panel size of saved frames in pixels");
    program.add_argument("-j", "--threads").default_value(0).scan<'i', int>().help("OpenMP threads (0 = default)");
    program.add_argument("--paused").default_value(false).implicit_value(true).help("GUI: start paused");
    program.add_argument("--steps-per-frame").default_value(1).scan<'i', int>().help("GUI: simulation steps per displayed frame");
    program.add_argument("--exit-when-done").default_value(false).implicit_value(true).help("GUI: close the window once every run finishes");
    program.add_argument("--font")
        .default_value(std::string(
#ifdef SPRINGLS_FONT_FILE
            SPRINGLS_FONT_FILE
#endif
            ))
        .help("TrueType font for panel text");
    program.add_argument("--verbose").default_value(false).implicit_value(true).help("log simulation progress");
    try {
        program.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << err.what() << "\n\n" << program;
        return 2;
    }

    enright::ExperimentOptions options;
    const bool slOnly = program.get<bool>("--sl-only");
    if (slOnly) {
        options.methods = {Method::SemiLagrangianDoubleweave};
    } else {
        options.methods = {Method::AdaptiveDoubleweave, Method::SemiLagrangianDoubleweave};
    }
    if (!enright::ParsePacing(program.get<std::string>("--pacing"), options.pacing)) {
        std::cerr << "error: unknown pacing '" << program.get<std::string>("--pacing") << "'" << std::endl;
        return 2;
    }
    options.config.gridSize = program.get<int>("--grid");
    options.config.period = program.get<double>("--period");
    options.config.maxTimeStepFraction = program.get<double>("--max-time-step");
    options.maxDepth = program.get<int>("--max-depth");
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
    SetThreads(program.get<int>("--threads"));

    enright::Experiment experiment;
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
    std::cout << (slOnly ? "Semi-Lagrangian doubleweave" : "Semi-Lagrangian vs CFL-limited doubleweave") << ": Enright " << options.config.gridSize
              << "^2, T = " << options.config.period << ", depth " << options.maxDepth << ", sub-step cap T * " << options.config.maxTimeStepFraction;
    if (!slOnly) std::cout << ", pacing " << enright::PacingKey(options.pacing);
    std::cout << ", " << MaxThreads() << " thread(s)" << std::endl;

    const bool noGui = program.get<bool>("--no-gui");
    const std::string frames = program.get<std::string>("--frames");
#ifdef SPRINGLS_GUI
    enright::DisplayOptions display;
    display.windowTitle = slOnly ? "Semi-Lagrangian Doubleweave (Enright 2D)" : "Semi-Lagrangian vs CFL-limited Doubleweave (Enright 2D)";
    display.fontFile = program.get<std::string>("--font");
    display.framesDir = frames;
    display.framePanelSize = program.get<float>("--frame-size");
    display.startPaused = program.get<bool>("--paused");
    display.stepsPerFrame = program.get<int>("--steps-per-frame");
    display.exitWhenDone = program.get<bool>("--exit-when-done");
    if (!noGui) return enright::RunGui(experiment, display);
    if (!frames.empty()) return enright::RunOffscreenFrames(experiment, display);
#else
    if (!noGui) std::cout << "(built without the GUI; running headless)" << std::endl;
    if (!frames.empty()) {
        std::cerr << "error: --frames needs the GUI build (SPRINGLS_BUILD_GUI=ON)" << std::endl;
        return 1;
    }
#endif
    return RunHeadless(experiment);
}
