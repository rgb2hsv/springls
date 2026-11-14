// Enright 2D deformation test for the CFL-limited methods: spring level sets, adaptive springls,
// doubleweave and adaptive doubleweave. Runs all four side by side (default) or any one of them,
// with a GUI or headless. The semi-Lagrangian doubleweave has its own example, sl-enright2d.
//
//   enright2d                                  # GUI, all four methods
//   enright2d --method doubleweave --no-gui    # one method, headless, prints final errors
//   enright2d --no-gui --frames out/frames     # headless, saves every step as a PNG
//   enright2d --set contractDistance=1.0       # override a tuning parameter
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
std::string MethodChoices()
{
    std::string choices = "all";
    for (Method m : kAllMethods) choices += std::string("|") + MethodKey(m);
    return choices;
}

int RunHeadless(enright::Experiment &experiment)
{
    // Progress about every 10% of the period.
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
    return 0;
}
}  // namespace

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("enright2d");
    program.add_description(
        "2D Enright deformation test for spring level sets, adaptive springls, doubleweave and adaptive doubleweave "
        "(the semi-Lagrangian doubleweave is in sl-enright2d).");
    program.add_argument("-m", "--method").default_value(std::string("all")).help("method to run: " + MethodChoices());
    program.add_argument("--no-gui").default_value(false).implicit_value(true).help("run without a window");
    program.add_argument("-g", "--grid").default_value(64).scan<'i', int>().help("level set grid size (base grid for adaptive methods)");
    program.add_argument("-T", "--period").default_value(8.0).scan<'g', double>().help("deformation period T");
    program.add_argument("-d", "--max-depth").default_value(4).scan<'i', int>().help("adaptive refinement depth (finest = grid * 2^depth)");
    program.add_argument("--pacing").default_value(std::string("lockstep")).help("which runs advance each step: lockstep|equal-compute|same-time");
    program.add_argument("--max-time-step").default_value(1.0 / 800.0).scan<'g', double>().help("max advection sub-step, as a fraction of T");
    program.add_argument("--set").default_value(std::vector<std::string>{}).append().help("tuning override key=value (repeatable; see --list-tuning)");
    program.add_argument("--list-tuning").default_value(false).implicit_value(true).help("list tuning parameter names and exit");
    program.add_argument("--csv").default_value(std::string()).help("write per-step metrics to this CSV file");
    program.add_argument("--frames")
        .default_value(std::string())
        .help("save the rendered panels of every time step to this directory as frame_000000.png, ... (no GUI widgets)");
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
    if (program.get<bool>("--list-tuning")) {
        for (const std::string &key : TuningKeys()) std::cout << key << '\n';
        return 0;
    }

    enright::ExperimentOptions options;
    const std::string method = program.get<std::string>("--method");
    if (method == "all") {
        options.methods.assign(kAllMethods.begin(), kAllMethods.end());
    } else {
        Method m;
        if (!ParseMethod(method, m)) {
            std::cerr << "error: unknown method '" << method << "' (expected " << MethodChoices() << ")" << std::endl;
            return 2;
        }
        options.methods = {m};
    }
    options.config.gridSize = program.get<int>("--grid");
    options.config.period = program.get<double>("--period");
    options.config.maxTimeStepFraction = program.get<double>("--max-time-step");
    options.maxDepth = program.get<int>("--max-depth");
    options.verbose = program.get<bool>("--verbose");
    if (!enright::ParsePacing(program.get<std::string>("--pacing"), options.pacing)) {
        std::cerr << "error: unknown pacing '" << program.get<std::string>("--pacing") << "'" << std::endl;
        return 2;
    }
    if (options.config.gridSize < 8 || options.config.period <= 0.0 || options.maxDepth < 0) {
        std::cerr << "error: need --grid >= 8, --period > 0 and --max-depth >= 0" << std::endl;
        return 2;
    }
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
    std::cout << "Enright 2D: " << (method == "all" ? "all methods" : method) << ", grid " << options.config.gridSize << ", T = " << options.config.period
              << ", max depth " << options.maxDepth << ", " << MaxThreads() << " thread(s)" << std::endl;

    const bool noGui = program.get<bool>("--no-gui");
    const std::string frames = program.get<std::string>("--frames");
#ifdef SPRINGLS_GUI
    enright::DisplayOptions display;
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
