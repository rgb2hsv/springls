#include "Cli.hpp"

#include "Profiles.hpp"

#include <argparse/argparse.hpp>

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <sstream>

namespace toybox
{
namespace
{
std::string DefaultFont()
{
#ifdef SPRINGLS_FONT_FILE
    return SPRINGLS_FONT_FILE;
#else
    return std::string();
#endif
}

// Cached initial states of the mesh scenario go in the build directory.
std::string DefaultCacheDir()
{
#ifdef SPRINGLS_CACHE_DIR
    return SPRINGLS_CACHE_DIR;
#else
    return "springls-cache";
#endif
}

// The mesh loaded when none is given: assets/model/metasphere.ply under the working directory, else under
// the source tree's assets.
std::string DefaultMesh()
{
    const std::string relative = "assets/model/metasphere.ply";
#ifdef SPRINGLS_ASSET_DIR
    if (!std::filesystem::exists(relative)) return std::string(SPRINGLS_ASSET_DIR) + "/model/metasphere.ply";
#endif
    return relative;
}

// The stone scenario's mesh: assets/model/stone.ply, found like DefaultMesh's.
std::string DefaultStoneMesh()
{
    const std::string relative = "assets/model/stone.ply";
#ifdef SPRINGLS_ASSET_DIR
    if (!std::filesystem::exists(relative)) return std::string(SPRINGLS_ASSET_DIR) + "/model/stone.ply";
#endif
    return relative;
}

template <class T, size_t N>
std::string Choices(const T (&values)[N])
{
    std::string s;
    for (T v : values) s += (s.empty() ? "" : "|") + std::string(Key(v));
    return s;
}

std::unique_ptr<argparse::ArgumentParser> MakeParser(const Demo &demo, bool withHelp)
{
    auto parser =
        std::make_unique<argparse::ArgumentParser>(demo.program, "1.0", withHelp ? argparse::default_arguments::all : argparse::default_arguments::none);
    argparse::ArgumentParser &p = *parser;
    p.add_description(demo.description);
    p.add_epilog(
        "Defaults depend on what runs (the demo's values, else the profile of the domain, scenario and interface tracking). "
        "--grid is the level-set grid size in the enright scenario and the base cells along the mesh's longest axis in the mesh scenario.");
    // Selection
    p.add_argument("mesh")
        .nargs(argparse::nargs_pattern::optional)
        .help(
            "mesh scenario: PLY mesh to load (default: assets/model/metasphere.ply); sphere / stone scenario: closed triangle mesh (default: an icosphere / "
            "assets/model/stone.ply)");
    p.add_argument("-m", "--method").help("method key(s) within the domain and scenario, comma separated; all; 2D: multi (see --list)");
    p.add_argument("-r", "--representation").help("representation(s): " + Choices(kRepresentations));
    p.add_argument("--domain").help("domain: " + Choices(kDomains) + " (default 2d; 3d for the mesh scenario, surf for sphere and stone)");
    p.add_argument("--grid-type").help("grid type(s): " + Choices(kGridTypes));
    p.add_argument("--objects").help("objects: " + Choices(kObjects) + " (default single)");
    p.add_argument("--scenario").help("scenario: " + Choices(kScenarios) + " (default enright)");
    p.add_argument("--interface-tracking").help("interface tracking(s): " + Choices(kInterfaceTrackings) + " (default eulerian)");
    p.add_argument("--sl-only").default_value(false).implicit_value(true).help("only the semi-Lagrangian run (--interface-tracking semi-lagrangian)");
    p.add_argument("--list").default_value(false).implicit_value(true).help("print the implemented configurations (markdown) and exit");
    p.add_argument("--list-tuning").default_value(false).implicit_value(true).help("list tuning parameter names and exit");
    // Shared
    p.add_argument("--no-gui").default_value(false).implicit_value(true).help("run without a window");
    p.add_argument("-g", "--grid")
        .scan<'i', int>()
        .help("enright: level set grid size (0: per method); mesh: base cells along the longest axis (0: from the mesh)");
    p.add_argument("-T", "--period").scan<'g', double>().help("deformation period T");
    p.add_argument("-d", "--max-depth").help("adaptive refinement depth (finest = grid * 2^depth); mesh: an integer or auto");
    p.add_argument("--max-time-step").scan<'g', double>().help("max advection sub-step, as a fraction of T (0: none; 2D Eulerian default none, 3D 1/800)");
    p.add_argument("--substeps-per-step").scan<'i', int>().help("advection sub-steps per step (and per GUI frame / saved frame); 0: the method's default");
    p.add_argument("--pacing").help("which runs advance each step: lockstep|equal-compute|same-time");
    p.add_argument("--set").default_value(std::vector<std::string>{}).append().help("tuning override key=value (repeatable; see --list-tuning)");
    p.add_argument("--csv").help("write per-step metrics to this CSV file");
    p.add_argument("--frames")
        .help("save the rendered panels of every step (sphere: every --every steps and the last) to this directory as frame_000000.png, ... (no GUI widgets)");
    p.add_argument("--frame-size").scan<'g', float>().help("panel size of saved frames in pixels (default 720)");
    p.add_argument("-j", "--threads").scan<'i', int>().help("OpenMP threads (0 = default)");
    p.add_argument("--paused").default_value(false).implicit_value(true).help("GUI: start paused");
    p.add_argument("--steps-per-frame").scan<'i', int>().help("2D GUI: simulation steps per displayed frame (default 1)");
    p.add_argument("--exit-when-done").default_value(false).implicit_value(true).help("GUI: close the window once every run finishes");
    p.add_argument("--font").help("TrueType font for panel text");
    p.add_argument("--verbose").default_value(false).implicit_value(true).help("log simulation progress");
    p.add_argument("--steps").scan<'i', int>().help(
        "enright: stop after this many steps from the start or resume (0: no limit); mesh: steps to run after initialization (default 0; GUI: then "
        "wait); sphere: steps to run (default: the whole loop; 0: none; GUI: then wait)");
    // 3D Enright and mesh
    p.add_argument("-e", "--elements", "--element").help("springl element type: quad|triangle");
    // 3D Enright
    p.add_argument("--hide").help(
        "comma-separated layers to hide initially: 3D Enright: iso,primal,dual,particles; sphere: "
        "regions,isolines,springls,particles,vertices,normals,contour,arrows,reference,wireframe");
    p.add_argument("--particles").default_value(false).implicit_value(true).help("show springl particles (hidden by default)");
    p.add_argument("--capsules")
        .default_value(false)
        .implicit_value(true)
        .help("draw springl capsules instead of springls (the default when a DoubleWeave method runs)");
    p.add_argument("--no-capsules").default_value(false).implicit_value(true).help("draw springls, not capsules, for DoubleWeave too");
    p.add_argument("--zoom").scan<'g', float>().help("initial view zoom (0: the default pose's; sphere: a factor on the first framing)");
    p.add_argument("--checkpoint-at")
        .default_value(std::vector<double>{})
        .append()
        .scan<'g', double>()
        .help("save each run's state (2D: a .state file; 3D: .state + .vdb level sets) once its time reaches this (repeatable), to --checkpoint-dir");
    p.add_argument("--checkpoint-dir").help("directory for --checkpoint-at state files (default checkpoints)");
    p.add_argument("--resume").help("start the (single) run from this .state file (3D: and the .vdb beside it) instead of t = 0");
    p.add_argument("--max-springl-growth")
        .scan<'g', double>()
        .help("abort a run once its springl count exceeds this many times its count at the start or resume (0: no limit)");
    p.add_argument("--stop-at").scan<'g', double>().help("stop runs once they reach this time");
    // Mesh
    p.add_argument("--cell-scale").scan<'g', float>().help("finest cell size as a multiple of the mean edge length (default 1)");
    p.add_argument("--matcap").help("GUI: matcap shading name");
    p.add_argument("--play").default_value(false).implicit_value(true).help("GUI: start stepping after initialization (default: stop there)");
    p.add_argument("--settle").scan<'i', int>().help("initialization: at most this many relax / contract / fill iterations (default 20)");
    p.add_argument("--z-up").default_value(false).implicit_value(true).help("GUI: the mesh's up axis is +z");
    p.add_argument("--no-cache").default_value(false).implicit_value(true).help("initialize from the mesh, without reading or writing the cache");
    p.add_argument("--cache-dir").help("directory of cached initial states (default " + DefaultCacheDir() + ")");
    p.add_argument("--screenshot").help("GUI: load, run --steps steps, write the window to this PNG and exit");
    // Sphere
    p.add_argument("--object").help("sphere: the object that travels the loop: puzzle|star (default puzzle)");
    p.add_argument("--flow").help(
        "sphere / stone: the flow that carries the object: rotation (the rigid rotation projected onto the surface) or hodge (its divergence-free "
        "part by the Helmholtz-Hodge decomposition) (default rotation; hodge for stone)");
    p.add_argument("--speed").scan<'g', double>().help("sphere: speed of the object's center under the rotation, mesh units per unit time (default 1)");
    p.add_argument("--face-velocity")
        .default_value(false)
        .implicit_value(true)
        .help(
            "stone / hodge flow: the exactly divergence-free piecewise-constant velocity per input face, instead of the default smoothed one "
            "(interpolated from vertex averages)");
    p.add_argument("--loop-distance").scan<'g', double>().help("sphere: geodesic distance of the object's center from the loop center (0: automatic)");
    p.add_argument("--radius").scan<'g', double>().help("sphere: object size r (0: loop distance / 6)");
    p.add_argument("--max-level").scan<'i', int>().help("sphere: adaptive refinement levels near the springls (default 3)");
    p.add_argument("--every").scan<'i', int>().help("sphere: frames every this many steps, and at the last (default 1)");
    p.add_argument("--frame-time")
        .scan<'g', double>()
        .help(
            "sphere: with --frames, a frame each time the simulation time passes a multiple of this (evenly spaced in time, so the motion's "
            "speed shows), and at the first and last steps; replaces --every");
    p.add_argument("--no-follow").default_value(false).implicit_value(true).help("sphere GUI: start with the camera not following the object");
    return parser;
}

// Values from the user's arguments, else the demo's.
struct Args {
    const argparse::ArgumentParser &user;
    const argparse::ArgumentParser &demo;
    template <class T>
    std::optional<T> get(const std::string &name) const
    {
        if (auto v = user.present<T>(name)) return v;
        return demo.present<T>(name);
    }
    bool flag(const std::string &name) const { return user.get<bool>(name) || demo.get<bool>(name); }
    template <class T>
    std::vector<T> list(const std::string &name) const
    {
        std::vector<T> v = demo.get<std::vector<T>>(name);
        const std::vector<T> u = user.get<std::vector<T>>(name);
        v.insert(v.end(), u.begin(), u.end());
        return v;
    }
};

std::vector<std::string> SplitList(const std::string &s)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

template <class T, size_t N>
bool ParseAxis(const std::string &axis, const std::string &value, const T (&all)[N], std::vector<T> &out, std::string &error)
{
    out.clear();
    for (const std::string &key : SplitList(value)) {
        T v;
        if (!Parse(key, v)) {
            error = "error: unknown " + axis + " '" + key + "' (expected " + Choices(all) + ")";
            return false;
        }
        if (std::find(out.begin(), out.end(), v) == out.end()) out.push_back(v);
    }
    if (out.empty()) {
        error = "error: --" + axis + " needs a value";
        return false;
    }
    return true;
}

template <class T, size_t N>
std::vector<T> AllOf(const T (&all)[N])
{
    return std::vector<T>(std::begin(all), std::end(all));
}

template <class T>
bool Contains(const std::vector<T> &values, T v)
{
    return std::find(values.begin(), values.end(), v) != values.end();
}

std::string MethodChoices(Domain domain, Scenario scenario)
{
    std::string s = "all";
    for (const Entry &e : Registry::Instance().entries()) {
        if (e.config.domain == domain && e.config.scenario == scenario) s += "|" + e.key;
    }
    if (domain == Domain::D2 && scenario == Scenario::Enright) s += "|multi";
    return s;
}

bool ParseInt(const std::string &s, int &value)
{
    try {
        size_t used = 0;
        value = std::stoi(s, &used);
        return used == s.size();
    } catch (const std::exception &) {
        return false;
    }
}

// Which flags apply where.
struct FlagRule {
    const char *name;
    bool enright2d, enright3d, mesh, sphere;
};
constexpr FlagRule kFlagRules[] = {
    {"--period", true, true, false, false},
    {"--max-time-step", true, true, false, false},
    {"--substeps-per-step", true, true, false, false},
    {"--pacing", true, true, false, false},
    {"--csv", true, true, false, true},
    {"--frames", true, true, false, true},
    {"--frame-size", true, true, false, true},
    {"--paused", true, true, false, false},
    {"--exit-when-done", true, true, false, false},
    {"--font", true, true, false, false},
    {"--verbose", true, true, false, true},
    {"--sl-only", true, true, false, false},
    {"--steps-per-frame", true, false, false, false},
    {"--hide", false, true, false, true},
    {"--particles", false, true, false, false},
    {"--capsules", false, true, false, false},
    {"--no-capsules", false, true, false, false},
    {"--zoom", false, true, false, true},
    {"--checkpoint-at", true, true, false, false},
    {"--checkpoint-dir", true, true, false, false},
    {"--resume", true, true, false, false},
    {"--max-springl-growth", true, true, false, false},
    {"--stop-at", true, true, false, false},
    {"--elements", false, true, true, false},
    {"mesh", false, false, true, true},
    {"--cell-scale", false, false, true, false},
    {"--matcap", false, false, true, false},
    {"--play", false, false, true, true},
    {"--settle", false, false, true, false},
    {"--z-up", false, false, true, false},
    {"--no-cache", false, false, true, false},
    {"--cache-dir", false, false, true, false},
    {"--screenshot", false, false, true, true},
    {"--grid", true, true, true, false},
    {"--max-depth", true, true, true, false},
    {"--object", false, false, false, true},
    {"--flow", false, false, false, true},
    {"--speed", false, false, false, true},
    {"--loop-distance", false, false, false, true},
    {"--face-velocity", false, false, false, true},
    {"--radius", false, false, false, true},
    {"--max-level", false, false, false, true},
    {"--every", false, false, false, true},
    {"--frame-time", false, false, false, true},
    {"--no-follow", false, false, false, true},
};

// The selection: runs of the axis product (or of the -m keys) in the one domain and scenario.
bool Select(const Args &args, Invocation &out, std::string &error)
{
    const argparse::ArgumentParser &user = args.user, &demo = args.demo;
    // Selection axes other than domain / scenario: a user -m replaces the demo's whole selection, a user
    // axis flag replaces the demo's -m and that axis.
    const char *axes[] = {"--representation", "--grid-type", "--objects", "--interface-tracking"};
    bool userAxes = user.get<bool>("--sl-only");
    for (const char *a : axes) userAxes |= user.is_used(a);
    if (user.is_used("--method") && userAxes) {
        error = "error: -m/--method can only be combined with --domain and --scenario";
        return false;
    }
    const argparse::ArgumentParser *selection = user.is_used("--method") ? &user : nullptr;
    std::optional<std::string> method = selection ? user.present<std::string>("--method") : std::nullopt;
    if (!selection && !userAxes) method = demo.present<std::string>("--method");
    auto axis = [&](const char *name) -> std::optional<std::string> {
        if (method) return std::nullopt;
        if (auto v = user.present<std::string>(name)) return v;
        return demo.present<std::string>(name);
    };
    std::optional<std::string> representation = axis("--representation"), gridType = axis("--grid-type"), objects = axis("--objects"),
                               tracking = axis("--interface-tracking");
    if (user.get<bool>("--sl-only")) tracking = "semi-lagrangian";

    // One domain and one scenario.
    std::vector<Scenario> scenarios = {Scenario::Enright};
    if (auto s = args.get<std::string>("--scenario")) {
        if (!ParseAxis("scenario", *s, kScenarios, scenarios, error)) return false;
    }
    Domain defaultDomain = Domain::D2;
    if (scenarios.size() == 1 && scenarios[0] == Scenario::Mesh) defaultDomain = Domain::D3;
    if (scenarios.size() == 1 && Surface(scenarios[0])) defaultDomain = Domain::Surface;
    std::vector<Domain> domains = {defaultDomain};
    if (auto d = args.get<std::string>("--domain")) {
        if (!ParseAxis("domain", *d, kDomains, domains, error)) return false;
    }
    if (domains.size() != 1 || scenarios.size() != 1) {
        error = "error: one domain and scenario per invocation (each has its own window)";
        return false;
    }
    out.domain = domains[0];
    out.scenario = scenarios[0];
    const Registry &registry = Registry::Instance();

    if (method) {
        for (const std::string &key : SplitList(*method)) {
            std::vector<const Entry *> add;
            if (key == "all" || key == "multi") {
                for (const Entry &e : registry.entries()) {
                    if (e.config.domain != out.domain || e.config.scenario != out.scenario) continue;
                    // all = the springl methods (not the basic level set ones); 2D: all = the four CFL-limited
                    // single-object springl methods, multi = the six multi-object methods.
                    if (key == "all" && e.config.representation == Representation::LevelSet && e.config.objects == Objects::Single) continue;
                    if (out.domain == Domain::D2 && out.scenario == Scenario::Enright) {
                        const bool multi = e.config.objects == Objects::Multiple;
                        if (key == "all" && (multi || e.config.tracking != InterfaceTracking::Eulerian)) continue;
                        if (key == "multi" && !multi) continue;
                    } else if (key == "multi") {
                        continue;
                    }
                    add.push_back(&e);
                }
            } else if (const Entry *e = registry.findKey(out.domain, out.scenario, key)) {
                add.push_back(e);
            }
            if (add.empty()) {
                error = "error: unknown method '" + key + "' for " + Key(out.domain) + " " + Key(out.scenario) + " (expected " +
                        MethodChoices(out.domain, out.scenario) + ")";
                return false;
            }
            out.entries.insert(out.entries.end(), add.begin(), add.end());
        }
        out.methodLabel = *method == "all" ? "all methods" : *method;
    } else {
        std::vector<Representation> reps;
        std::vector<GridType> grids;
        std::vector<Objects> objs = {Objects::Single};
        std::vector<InterfaceTracking> tracks = {InterfaceTracking::Eulerian};
        if (representation && !ParseAxis("representation", *representation, kRepresentations, reps, error)) return false;
        if (gridType && !ParseAxis("grid-type", *gridType, kGridTypes, grids, error)) return false;
        if (objects && !ParseAxis("objects", *objects, kObjects, objs, error)) return false;
        if (tracking && !ParseAxis("interface-tracking", *tracking, kInterfaceTrackings, tracks, error)) return false;
        // Axes not given take the values implemented for the given ones (all values when there are none).
        auto matchesGiven = [&](const Configuration &c) {
            return c.domain == out.domain && c.scenario == out.scenario && (!representation || Contains(reps, c.representation)) &&
                   (!gridType || Contains(grids, c.grid)) && Contains(objs, c.objects) && Contains(tracks, c.tracking);
        };
        if (!representation) {
            for (Representation r : kRepresentations) {
                for (const Entry &e : registry.entries()) {
                    if (matchesGiven(e.config) && e.config.representation == r && !Contains(reps, r)) reps.push_back(r);
                }
            }
            if (reps.empty()) reps = AllOf(kRepresentations);
        }
        if (!gridType) {
            for (GridType g : kGridTypes) {
                for (const Entry &e : registry.entries()) {
                    if (matchesGiven(e.config) && e.config.grid == g && !Contains(grids, g)) grids.push_back(g);
                }
            }
            if (grids.empty()) grids = AllOf(kGridTypes);
        }
        std::vector<Configuration> skipped;
        std::vector<Configuration> product;
        for (Representation r : reps) {
            for (GridType g : grids) {
                for (Objects o : objs) {
                    for (InterfaceTracking t : tracks) product.push_back({r, out.domain, g, o, out.scenario, t});
                }
            }
        }
        for (const Configuration &c : product) {
            if (!registry.find(c)) skipped.push_back(c);
        }
        for (const Entry &e : registry.entries()) {
            if (std::find(product.begin(), product.end(), e.config) != product.end()) out.entries.push_back(&e);
        }
        if (out.entries.empty()) {
            std::ostringstream s;
            for (size_t i = 0; i < skipped.size(); i++) s << (i ? "\n" : "") << "error: not implemented: " << Describe(skipped[i]);
            s << "\nnearest implemented configurations:";
            for (const Entry *e : registry.nearest(skipped.front()))
                s << "\n  " << Describe(e->config) << "  (" << Key(e->config.domain) << " -m " << e->key << ")";
            error = s.str();
            return false;
        }
        // A note for each skipped configuration whose combination of explicitly given axis values has no
        // run (axes left to their defaults stay quiet).
        for (const Configuration &c : skipped) {
            bool covered = false;
            for (const Entry *e : out.entries) {
                const Configuration &r = e->config;
                covered |= (!representation || r.representation == c.representation) && (!gridType || r.grid == c.grid) &&
                           (!objects || r.objects == c.objects) && (!tracking || r.tracking == c.tracking);
            }
            const bool anyGiven = representation || gridType || objects || tracking;
            if (anyGiven && !covered) out.notes.push_back("not implemented, skipped: " + Describe(c));
        }
        for (const Entry *e : out.entries) out.methodLabel += (out.methodLabel.empty() ? "" : ",") + e->key;
        out.slComparison = Contains(tracks, InterfaceTracking::SemiLagrangian);
    }
    out.slOnly = true;
    for (const Entry *e : out.entries) {
        const bool sl = e->config.tracking == InterfaceTracking::SemiLagrangian;
        out.semiLagrangian |= sl;
        out.slOnly &= sl;
    }
    // The semi-Lagrangian comparison output needs a semi-Lagrangian run.
    out.slComparison = out.slComparison && out.semiLagrangian;
    return true;
}
}  // namespace

bool ParseSphereLayers(const std::string &hide, std::vector<std::string> &layers, std::string &error)
{
    layers.clear();
    for (const std::string &layer : SplitList(hide)) {
        if (layer != "regions" && layer != "isolines" && layer != "springls" && layer != "particles" && layer != "vertices" && layer != "normals" &&
            layer != "contour" && layer != "arrows" && layer != "reference" && layer != "wireframe") {
            error = "error: --hide: unknown layer '" + layer +
                    "' (expected regions, isolines, springls, particles, vertices, normals, contour, arrows, reference or wireframe)";
            return false;
        }
        layers.push_back(layer);
    }
    return true;
}

bool ParseHiddenLayers(const std::string &hide, std::vector<std::string> &layers, std::string &error)
{
    layers.clear();
    for (const std::string &layer : SplitList(hide)) {
        if (layer != "iso" && layer != "primal" && layer != "dual" && layer != "particles") {
            error = "error: --hide: unknown layer '" + layer + "'";
            return false;
        }
        layers.push_back(layer);
    }
    return true;
}

bool ParseInvocation(const Demo &demo, int argc, const char *const argv[], Invocation &out, std::string &error, int &exitCode)
{
    out = Invocation();
    out.program = demo.program;
    exitCode = 2;
    auto demoParser = MakeParser(demo, false);
    auto userParser = MakeParser(demo, true);
    try {
        std::vector<std::string> demoArgs = {demo.program};
        demoArgs.insert(demoArgs.end(), demo.defaultArgs.begin(), demo.defaultArgs.end());
        demoParser->parse_args(demoArgs);
    } catch (const std::exception &err) {
        error = std::string("error: demo ") + demo.program + ": " + err.what();
        return false;
    }
    try {
        std::vector<std::string> userArgs(argv, argv + argc);
        userParser->parse_args(userArgs);
    } catch (const std::exception &err) {
        std::ostringstream s;
        s << err.what() << "\n\n" << *userParser;
        error = s.str();
        return false;
    }
    const Args args{*userParser, *demoParser};
    if (args.flag("--list")) {
        out.list = true;
        return true;
    }
    if (!Select(args, out, error)) return false;
    if (args.flag("--list-tuning")) {
        out.listTuning = true;
        return true;
    }
    const bool mesh = out.scenario == Scenario::Mesh;
    const bool sphere = Surface(out.scenario);
    const bool d2 = out.domain == Domain::D2;
    for (const FlagRule &rule : kFlagRules) {
        if (!userParser->is_used(rule.name)) continue;
        const bool ok = sphere ? rule.sphere : (mesh ? rule.mesh : (d2 ? rule.enright2d : rule.enright3d));
        if (!ok) {
            error = std::string("error: ") + rule.name + " is not supported for the " +
                    (sphere ? std::string(Key(out.scenario)) + " scenario"
                     : mesh ? std::string("mesh scenario")
                            : std::string(Key(out.domain)) + " enright scenario");
            return false;
        }
    }

    const Profile profile = DefaultProfile(out.domain, out.scenario, out.semiLagrangian);
    RunOptions &run = out.run;
    for (const std::string &entry : args.list<std::string>("--set")) {
        const size_t eq = entry.find('=');
        try {
            if (eq == std::string::npos) throw std::invalid_argument(entry);
            run.tuning.emplace_back(entry.substr(0, eq), std::stod(entry.substr(eq + 1)));
        } catch (const std::exception &) {
            error = "error: --set expects key=value, got '" + entry + "'";
            return false;
        }
    }
    run.verbose = args.flag("--verbose");
    out.noGui = args.flag("--no-gui");
    out.threads = args.get<int>("--threads").value_or(0);
    if (auto e = args.get<std::string>("--elements")) {
        if (!springls::ParseElementType(*e, run.elementType)) {
            error = mesh ? "error: --element expects quad or triangle" : "error: --elements expects quad or triangle";
            return false;
        }
    }
    const std::optional<int> steps = args.get<int>("--steps");
    out.stepsGiven = steps.has_value();
    run.gridSize = args.get<int>("--grid").value_or(profile.gridSize);
    const std::string depth = args.get<std::string>("--max-depth").value_or(profile.maxDepth < 0 ? "auto" : std::to_string(profile.maxDepth));

    if (sphere) {
        using springls::LoopObject;
        SphereOptions &s = run.sphere;
        if (auto mesh = userParser->present<std::string>("mesh")) {
            s.session.mesh = *mesh;
        } else if (out.scenario == Scenario::Stone) {
            s.session.mesh = DefaultStoneMesh();
        }
        const std::string object = args.get<std::string>("--object").value_or("puzzle");
        if (object == "puzzle") {
            s.session.config.object = LoopObject::PuzzlePiece;
        } else if (object == "star") {
            s.session.config.object = LoopObject::Star;
        } else {
            error = "error: --object expects puzzle or star, got '" + object + "'";
            return false;
        }
        const std::string flow = args.get<std::string>("--flow").value_or(out.scenario == Scenario::Stone ? "hodge" : "rotation");
        if (flow == "rotation") {
            s.session.config.flow = springls::LoopFlow::Rotation;
        } else if (flow == "hodge") {
            s.session.config.flow = springls::LoopFlow::Hodge;
        } else {
            error = "error: --flow expects rotation or hodge, got '" + flow + "'";
            return false;
        }
        s.session.config.speed = args.get<double>("--speed").value_or(1.0);
        s.session.config.loopDistance = args.get<double>("--loop-distance").value_or(0.0);
        s.session.config.smoothVelocity = !args.flag("--face-velocity");
        s.session.config.radius = args.get<double>("--radius").value_or(0.0);
        s.session.maxLevel = args.get<int>("--max-level").value_or(springls::IntrinsicSpringls::kDefaultMaxLevel);
        if (s.session.maxLevel < 0) {
            error = "error: --max-level expects a level count >= 0";
            return false;
        }
        s.every = args.get<int>("--every").value_or(1);
        if (s.every < 1) {
            error = "error: --every expects >= 1";
            return false;
        }
        s.frameTime = args.get<double>("--frame-time").value_or(0.0);
        if (s.frameTime < 0.0) {
            error = "error: --frame-time expects a time >= 0";
            return false;
        }
        const std::vector<std::string> keys = springls::TuningKeys();
        for (const auto &entry : run.tuning) {
            if (std::find(keys.begin(), keys.end(), entry.first) == keys.end()) {
                error = "error: --set: unknown tuning key '" + entry.first + "' (see --list-tuning)";
                return false;
            }
        }
        s.session.tuning = run.tuning;
        s.session.verbose = run.verbose;
        s.threads = out.threads;
        out.hide = args.get<std::string>("--hide").value_or("");
        if (!ParseSphereLayers(out.hide, s.hidden, error)) return false;
        out.csv = args.get<std::string>("--csv").value_or("");
        out.frames = args.get<std::string>("--frames").value_or("");
        out.frameSize = args.get<float>("--frame-size").value_or(720.0f);
        out.font = DefaultFont();
        out.play = args.flag("--play");
        out.screenshot = args.get<std::string>("--screenshot").value_or("");
        out.noFollow = args.flag("--no-follow");
        out.zoom = args.get<float>("--zoom").value_or(0.0f);
        // --steps 0 takes no step (headless and in the window); without --steps the whole loop runs.
        if (steps) out.limits.maxSteps = static_cast<uint64_t>(std::max(0, *steps));
        return true;
    }

    if (mesh) {
        MeshOptions &m = run.mesh;
        m.mesh = userParser->present<std::string>("mesh").value_or(DefaultMesh());
        m.grid.cellScale = args.get<float>("--cell-scale").value_or(1.0f);
        m.grid.gridOverride = run.gridSize;
        if (depth == "auto") {
            m.grid.maxDepth = -1;
        } else if (!ParseInt(depth, m.grid.maxDepth) || m.grid.maxDepth < 0) {
            error = "error: --max-depth expects a non-negative integer or auto, got '" + depth + "'";
            return false;
        }
        if (!(m.grid.cellScale > 0.0f) || m.grid.gridOverride < 0) {
            error = "error: need --cell-scale > 0 and --grid >= 0";
            return false;
        }
        m.elementType = run.elementType;
        m.tuning = run.tuning;
        m.threads = out.threads;
        m.settleIterations = std::max(0, args.get<int>("--settle").value_or(20));
        if (!args.flag("--no-cache")) m.cacheDir = args.get<std::string>("--cache-dir").value_or(DefaultCacheDir());
        out.limits.maxSteps = static_cast<uint64_t>(std::max(0, steps.value_or(0)));
        out.matcap = args.get<std::string>("--matcap").value_or("");
        out.play = args.flag("--play");
        out.zUp = args.flag("--z-up");
        out.screenshot = args.get<std::string>("--screenshot").value_or("");
        return true;
    }

    run.period = args.get<double>("--period").value_or(profile.period);
    run.maxTimeStepFraction = args.get<double>("--max-time-step").value_or(profile.maxTimeStepFraction);
    run.substepsPerStep = args.get<int>("--substeps-per-step").value_or(profile.substepsPerStep);
    if (!ParseInt(depth, run.maxDepth)) {
        error = "error: --max-depth expects a non-negative integer, got '" + depth + "'";
        return false;
    }
    if ((run.gridSize != 0 && run.gridSize < 8) || run.period <= 0.0 || run.maxDepth < 0 || run.substepsPerStep < 0) {
        if (out.slComparison || run.substepsPerStep < 0) {
            error = "error: need --grid >= 8, --period > 0, --max-depth >= 0 and --substeps-per-step >= 0";
        } else {
            error = d2 ? "error: need --grid >= 8, --period > 0 and --max-depth >= 0" : "error: need --grid >= 8 (or 0), --period > 0 and --max-depth >= 0";
        }
        return false;
    }
    const std::string pacing = args.get<std::string>("--pacing").value_or(PacingKey(profile.pacing));
    if (!ParsePacing(pacing, out.pacing)) {
        error = "error: unknown pacing '" + pacing + "'";
        return false;
    }
    out.csv = args.get<std::string>("--csv").value_or("");
    out.frames = args.get<std::string>("--frames").value_or("");
    out.frameSize = args.get<float>("--frame-size").value_or(720.0f);
    out.paused = args.flag("--paused");
    out.exitWhenDone = args.flag("--exit-when-done");
    out.stepsPerFrame = args.get<int>("--steps-per-frame").value_or(1);
    out.font = args.get<std::string>("--font").value_or(DefaultFont());
    if (steps && *steps > 0) out.limits.maxSteps = static_cast<uint64_t>(*steps);
    // Checkpoints and run limits (Enright, 2D and 3D).
    out.limits.checkpointTimes = args.list<double>("--checkpoint-at");
    out.limits.checkpointDir = args.get<std::string>("--checkpoint-dir").value_or("checkpoints");
    out.limits.resumeFile = args.get<std::string>("--resume").value_or("");
    if (auto s = args.get<double>("--stop-at")) out.limits.stopAt = *s;
    if (auto g = args.get<double>("--max-springl-growth"); g && *g > 0.0) out.limits.maxSpringlGrowth = *g;

    if (!d2) {
        out.hide = args.get<std::string>("--hide").value_or("");
        std::vector<std::string> layers;
        if (!ParseHiddenLayers(out.hide, layers, error)) return false;
        out.particles = args.flag("--particles");
        out.capsules = args.flag("--capsules");
        out.noCapsules = args.flag("--no-capsules");
        out.zoom = args.get<float>("--zoom").value_or(0.0f);
    }
    return true;
}
}  // namespace toybox
