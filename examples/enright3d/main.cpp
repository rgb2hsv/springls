// Enright 3D deformation test for spring level sets and doubleweave springls, with quad (default)
// or triangle springls. Runs doubleweave (default), single springls, or both side by side, with a
// GUI or headless.
//
//   enright3d                                       # GUI, doubleweave with quads
//   enright3d -m all -e triangle                    # GUI, both methods with triangles
//   enright3d -m springls --no-gui --csv out.csv    # headless, prints final errors
//   enright3d --no-gui --frames out/frames          # headless, saves every step as a PNG
//   enright3d --set fillDistance=0.4                # override a tuning parameter
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
std::string MethodChoices()
{
    std::string choices = "all";
    for (Method3D m : kAllMethods3D) choices += std::string("|") + MethodKey(m);
    return choices;
}

int RunHeadless(enright3d::Experiment &experiment)
{
    // Progress about every 10% of the period.
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
                std::cout << "  " << MethodKey(run.method) << ": " << run.current.springlCount << " springls, volume error "
                          << 100.0 * run.current.relativeVolumeError << " %";
            }
            std::cout << std::endl;
            nextReport = static_cast<int>(std::floor(t / period * kProgressReports)) + 1;
        }
    }
    experiment.printSummary(std::cout);
    return 0;
}
}  // namespace

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("enright3d");
    program.add_description("3D Enright deformation test for spring level sets and doubleweave springls.");
    program.add_argument("-m", "--method").default_value(std::string("doubleweave")).help("method to run: " + MethodChoices());
    program.add_argument("-e", "--elements").default_value(std::string("quad")).help("springl element type: quad|triangle");
    program.add_argument("--no-gui").default_value(false).implicit_value(true).help("run without a window");
    program.add_argument("-g", "--grid")
        .default_value(0)
        .scan<'i', int>()
        .help("level set grid size, voxels per side (0: 128 for uniform methods, 32 base for adaptive)");
    program.add_argument("-d", "--max-depth")
        .default_value(3)
        .scan<'i', int>()
        .help("adaptive refinement depth (finest = grid * 2^depth; default 3: 256^3 equivalent)");
    program.add_argument("-T", "--period").default_value(3.0).scan<'g', double>().help("deformation period T");
    program.add_argument("--max-time-step").default_value(1.0 / 800.0).scan<'g', double>().help("max advection sub-step, as a fraction of T");
    program.add_argument("--set").default_value(std::vector<std::string>{}).append().help("tuning override key=value (repeatable; see --list-tuning)");
    program.add_argument("--list-tuning").default_value(false).implicit_value(true).help("list tuning parameter names and exit");
    program.add_argument("--csv").default_value(std::string()).help("write per-step metrics to this CSV file");
    program.add_argument("--frames")
        .default_value(std::string())
        .help("save the rendered panels of every time step to this directory as frame_000000.png, ... (no GUI widgets)");
    program.add_argument("--frame-size").default_value(720.0f).scan<'g', float>().help("panel size of saved frames in pixels");
    program.add_argument("--hide").default_value(std::string()).help("comma-separated layers to hide initially: iso,primal,dual");
    program.add_argument("--particles").default_value(false).implicit_value(true).help("show springl particles (hidden by default)");
    program.add_argument("--capsules")
        .default_value(false)
        .implicit_value(true)
        .help("draw springl capsules instead of springls (the default when a doubleweave method runs)");
    program.add_argument("--no-capsules").default_value(false).implicit_value(true).help("draw springls, not capsules, for doubleweave too");
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
    if (program.get<bool>("--list-tuning")) {
        for (const std::string &key : TuningKeys3D()) std::cout << key << '\n';
        return 0;
    }

    enright3d::ExperimentOptions options;
    const std::string method = program.get<std::string>("--method");
    if (method == "all") {
        options.methods.assign(kAllMethods3D.begin(), kAllMethods3D.end());
    } else {
        Method3D m;
        if (!ParseMethod(method, m)) {
            std::cerr << "error: unknown method '" << method << "' (expected " << MethodChoices() << ")" << std::endl;
            return 2;
        }
        options.methods = {m};
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
    if ((options.config.gridSize != 0 && options.config.gridSize < 8) || options.config.period <= 0.0 || options.maxDepth < 0) {
        std::cerr << "error: need --grid >= 8 (or 0), --period > 0 and --max-depth >= 0" << std::endl;
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
    std::cout << "Enright 3D: " << (method == "all" ? "all methods" : method) << ", " << ElementTypeKey(options.elementType)
              << " springls, T = " << options.config.period << ", max depth " << options.maxDepth << ", " << MaxThreads() << " thread(s)" << std::endl;

    const bool noGui = program.get<bool>("--no-gui");
    const std::string frames = program.get<std::string>("--frames");
#ifdef SPRINGLS_GUI
    enright3d::DisplayOptions display;
    display.fontFile = program.get<std::string>("--font");
    display.framesDir = frames;
    display.framePanelSize = program.get<float>("--frame-size");
    display.startPaused = program.get<bool>("--paused");
    display.exitWhenDone = program.get<bool>("--exit-when-done");
    if (program.get<float>("--zoom") > 0.0f) display.camera.zoom = program.get<float>("--zoom");
    // Capsules show how the two doubleweave constellations cover each other, so they are on by
    // default whenever a doubleweave method runs.
    bool doubleweave = false;
    for (const Method3D m : options.methods)
        doubleweave |= (m == Method3D::Doubleweave || m == Method3D::AdaptiveDoubleweave || m == Method3D::SemiLagrangianDoubleweave);
    display.visibility.capsules = (program.get<bool>("--capsules") || doubleweave) && !program.get<bool>("--no-capsules");
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
