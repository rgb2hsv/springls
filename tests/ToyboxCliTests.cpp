#include "Check.hpp"

#include <core/Cli.hpp>
#include <core/Demos.hpp>
#include <core/Main.hpp>
#include <core/Registry.hpp>

#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace toybox;

namespace
{
struct Parsed {
    bool ok = false;
    Invocation inv;
    std::string error;
    int exitCode = 0;
};

Parsed Parse(const Demo &demo, const std::vector<std::string> &args)
{
    std::vector<const char *> argv = {demo.program.c_str()};
    for (const std::string &a : args) argv.push_back(a.c_str());
    Parsed p;
    p.ok = ParseInvocation(demo, static_cast<int>(argv.size()), argv.data(), p.inv, p.error, p.exitCode);
    return p;
}

std::string Keys(const Invocation &inv)
{
    std::string s;
    for (const Entry *e : inv.entries) s += (s.empty() ? "" : ",") + e->key;
    return s;
}

std::string Split(const std::vector<std::string> &args)
{
    std::string s;
    for (const auto &a : args) s += a + " ";
    return s;
}

// One documented command line and what it must resolve to (today's mains' values).
struct Row {
    Demo demo;
    std::vector<std::string> args;
    Domain domain;
    std::string keys;
    int grid;
    double period;
    int depth;
    double maxTimeStep;
    Pacing pacing;
    int substeps;
};

const std::string kAll2D = "springls,adaptive-springls,dweave,adaptive-dweave";
const std::string kMulti2D = "multi-levelset,adaptive-multi-levelset,multi-springls,adaptive-multi-springls,multi-dweave,adaptive-multi-dweave";
const std::string kAll3D = "springls,adaptive-springls,dweave,adaptive-dweave,semi-lagrangian-dweave";
const std::string kSl = "adaptive-dweave,semi-lagrangian-dweave";
constexpr double k800 = 1.0 / 800.0, k50 = 1.0 / 50.0, kNoCap = 0.0;  // 2D Eulerian: no time cap (CFL only)
}  // namespace

// The documented commands (README.md, docs/index.html, the baseline scripts), through the per-entry
// programs and toybox: the values the old programs used, or the profile where the spec gives one.
TEST(cli_documented_commands_resolve)
{
    using demos::Toybox;
    const auto P = [](const char *program) { return *demos::Find(program); };
    const Pacing L = Pacing::Lockstep, E = Pacing::EqualCompute, S = Pacing::SameTime;
    const std::vector<std::string> slPair = {"-r", "dweave", "--grid-type", "adaptive", "--interface-tracking", "eulerian,semi-lagrangian"};
    const auto with = [](std::vector<std::string> a, std::initializer_list<const char *> more) {
        a.insert(a.end(), more.begin(), more.end());
        return a;
    };
    const Row rows[] = {
        {Toybox(), {}, Domain::D2, kAll2D, 0, 8, 5, kNoCap, L, 0},
        {Toybox(), {"-m", "multi"}, Domain::D2, kMulti2D, 0, 8, 5, kNoCap, L, 0},
        {P("enright-2d-adaptive-dweave"), {}, Domain::D2, "adaptive-dweave", 0, 8, 5, kNoCap, L, 0},
        {Toybox(), {"--no-gui", "-g", "96", "-T", "4"}, Domain::D2, kAll2D, 96, 4, 5, kNoCap, L, 0},
        {P("enright-2d-springls"), {"--no-gui", "--csv", "out/metrics.csv"}, Domain::D2, "springls", 0, 8, 5, kNoCap, L, 0},
        {Toybox(), {"--no-gui", "--frames", "out/frames"}, Domain::D2, kAll2D, 0, 8, 5, kNoCap, L, 0},
        {Toybox(), {"--set", "contractDistance=1.0", "--set", "fillDistance=0.4"}, Domain::D2, kAll2D, 0, 8, 5, kNoCap, L, 0},
        {P("enright-2d-dweave"), {"--no-gui", "--frames", "dir"}, Domain::D2, "dweave", 0, 8, 5, kNoCap, L, 0},
        {P("enright-2d-springls"), {"--no-gui", "-g", "32", "-T", "1", "-d", "2", "--csv", "x.csv"}, Domain::D2, "springls", 32, 1, 2, kNoCap, L, 0},
        {Toybox(), {"--no-gui", "-g", "32", "-T", "1", "-d", "2", "-m", "multi", "-j", "1", "--csv", "x.csv"}, Domain::D2, kMulti2D, 32, 1, 2, kNoCap, L, 0},
        {P("enright-2d-levelset"), {}, Domain::D2, "levelset", 0, 8, 5, kNoCap, L, 0},
        {P("enright-2d-adaptive-levelset"), {}, Domain::D2, "adaptive-levelset", 0, 8, 5, kNoCap, L, 0},
        {P("enright-2d-multi-levelset"), {}, Domain::D2, "multi-levelset", 0, 8, 5, kNoCap, L, 0},
        {P("enright-2d-adaptive-multi-levelset"), {}, Domain::D2, "adaptive-multi-levelset", 0, 8, 5, kNoCap, L, 0},
        {P("enright-2d-multi-springls"), {}, Domain::D2, "multi-springls", 0, 8, 5, kNoCap, L, 0},
        {P("enright-2d-adaptive-multi-springls"), {}, Domain::D2, "adaptive-multi-springls", 0, 8, 5, kNoCap, L, 0},
        {P("enright-2d-multi-dweave"), {}, Domain::D2, "multi-dweave", 0, 8, 5, kNoCap, L, 0},
        {P("enright-2d-adaptive-multi-dweave"), {}, Domain::D2, "adaptive-multi-dweave", 0, 8, 5, kNoCap, L, 0},
        {Toybox(), slPair, Domain::D2, kSl, 128, 8, 2, k50, E, 1},
        {Toybox(), with(slPair, {"--pacing", "same-time"}), Domain::D2, kSl, 128, 8, 2, k50, S, 1},
        {Toybox(), with(slPair, {"--no-gui", "-g", "32", "-T", "1", "-d", "2", "--pacing", "lockstep", "--csv", "sl.csv"}), Domain::D2, kSl, 32, 1, 2, k50, L,
         1},
        {P("enright-2d-semi-lagrangian-dweave"), {}, Domain::D2, "semi-lagrangian-dweave", 128, 8, 2, k50, E, 1},
        {P("enright-2d-semi-lagrangian-dweave"), {"--no-gui", "--frames", "out/sl"}, Domain::D2, "semi-lagrangian-dweave", 128, 8, 2, k50, E, 1},
        {P("enright-2d-semi-lagrangian-dweave"),
         {"--no-gui", "-g", "64", "-T", "8", "-d", "4", "--max-time-step", "0.00125", "--substeps-per-step", "3", "--frames", "dir"},
         Domain::D2,
         "semi-lagrangian-dweave",
         64,
         8,
         4,
         k800,
         E,
         3},
        {P("enright-3d-dweave"), {}, Domain::D3, "dweave", 0, 3, 3, k800, L, 0},
        {P("enright-3d-adaptive-springls"), {}, Domain::D3, "adaptive-springls", 0, 3, 3, k800, L, 0},
        {Toybox(), {"--domain", "3d", "-m", "all", "-e", "triangle"}, Domain::D3, kAll3D, 32, 3, 2, k50, E, 1},  // includes SL: the SL profile
        {P("enright-3d-semi-lagrangian-dweave"), {}, Domain::D3, "semi-lagrangian-dweave", 32, 3, 2, k50, E, 1},
        {P("enright-3d-dweave"), {"--no-gui", "--frames", "out/frames", "--zoom", "2", "--hide", "iso"}, Domain::D3, "dweave", 0, 3, 3, k800, L, 0},
        {P("enright-3d-springls"), {"--no-gui", "--csv", "out/metrics.csv"}, Domain::D3, "springls", 0, 3, 3, k800, L, 0},
        {P("enright-3d-dweave"), {"--set", "fillDistance=0.4"}, Domain::D3, "dweave", 0, 3, 3, k800, L, 0},
        {P("enright-3d-adaptive-dweave"),
         {"--no-gui", "-g", "16", "-T", "0.5", "-d", "2", "--steps", "20", "--csv", "x.csv"},
         Domain::D3,
         "adaptive-dweave",
         16,
         0.5,
         2,
         k800,
         L,
         0},
        {Toybox(), with(with(slPair, {"--domain", "3d"}), {}), Domain::D3, kSl, 32, 3, 2, k50, E, 1},
        {P("enright-3d-levelset"), {}, Domain::D3, "levelset", 0, 3, 3, k800, L, 0},
        {P("enright-3d-adaptive-levelset"), {}, Domain::D3, "adaptive-levelset", 0, 3, 3, k800, L, 0},
        {Toybox(), with(slPair, {"--domain", "3d", "--sl-only", "--no-gui", "--frames", "out/sl3d"}), Domain::D3, "semi-lagrangian-dweave", 32, 3, 2, k50, E,
         1},
        {Toybox(), with(slPair, {"--domain", "3d", "--no-gui", "--pacing", "lockstep", "-g", "16", "-T", "0.5", "-d", "2", "--steps", "20", "--csv", "x.csv"}),
         Domain::D3, kSl, 16, 0.5, 2, k50, L, 1},
    };
    for (const Row &row : rows) {
        const Parsed p = Parse(row.demo, row.args);
        const bool ok = p.ok && p.inv.domain == row.domain && p.inv.scenario == Scenario::Enright && Keys(p.inv) == row.keys &&
                        p.inv.run.gridSize == row.grid && p.inv.run.period == row.period && p.inv.run.maxDepth == row.depth &&
                        p.inv.run.maxTimeStepFraction == row.maxTimeStep && p.inv.pacing == row.pacing && p.inv.run.substepsPerStep == row.substeps;
        if (!ok) {
            std::printf("  %s %s -> ok %d keys %s grid %d T %g d %d dt %g pacing %s substeps %d %s\n", row.demo.program.c_str(), Split(row.args).c_str(), p.ok,
                        Keys(p.inv).c_str(), p.inv.run.gridSize, p.inv.run.period, p.inv.run.maxDepth, p.inv.run.maxTimeStepFraction, PacingKey(p.inv.pacing),
                        p.inv.run.substepsPerStep, p.error.c_str());
        }
        CHECK(ok);
    }
}

TEST(cli_documented_flags_reach_the_invocation)
{
    using demos::Toybox;
    const auto P = [](const char *program) { return *demos::Find(program); };
    Parsed p = Parse(P("enright-2d-springls"), {"--no-gui", "--csv", "out/metrics.csv"});
    CHECK(p.ok && p.inv.noGui && p.inv.csv == "out/metrics.csv");
    p = Parse(Toybox(), {"--set", "contractDistance=1.0", "--set", "fillDistance=0.4"});
    CHECK(p.ok && p.inv.run.tuning.size() == 2 && p.inv.run.tuning[0].first == "contractDistance" && p.inv.run.tuning[1].second == 0.4);
    p = Parse(P("enright-3d-dweave"), {"-e", "triangle"});
    CHECK(p.ok && p.inv.run.elementType == springls::ElementType::Triangle);
    p = Parse(P("enright-3d-dweave"), {"--no-gui", "--frames", "out/frames", "--zoom", "2", "--hide", "iso"});
    CHECK(p.ok && p.inv.frames == "out/frames" && p.inv.zoom == 2.0f && p.inv.hide == "iso");
    p = Parse(P("enright-3d-dweave"), {"--checkpoint-at", "0.1", "--checkpoint-dir", "cp", "--stop-at", "0.1", "--steps", "5", "--max-springl-growth", "3"});
    CHECK(p.ok && p.inv.limits.checkpointTimes.size() == 1 && p.inv.limits.checkpointDir == "cp" && p.inv.limits.stopAt == 0.1 && p.inv.limits.maxSteps == 5 &&
          p.inv.limits.maxSpringlGrowth == 3.0);
    p = Parse(P("enright-3d-dweave"), {"--resume", "a.state"});
    CHECK(p.ok && p.inv.limits.resumeFile == "a.state");
    // 2D runs checkpoint and resume too.
    p = Parse(P("enright-2d-adaptive-multi-springls"), {"--checkpoint-at", "4", "--checkpoint-dir", "cp", "--stop-at", "4", "--max-springl-growth", "3"});
    CHECK(p.ok && p.inv.limits.checkpointTimes.size() == 1 && p.inv.limits.checkpointDir == "cp" && p.inv.limits.stopAt == 4.0);
    p = Parse(P("enright-2d-adaptive-multi-springls"), {"--resume", "b.state"});
    CHECK(p.ok && p.inv.limits.resumeFile == "b.state");
    p = Parse(P("enright-3d-dweave"), {"--steps", "0"});  // 0: no limit
    CHECK(p.ok && p.inv.limits.maxSteps == std::numeric_limits<uint64_t>::max());
    p = Parse(Toybox(), {"-r", "dweave", "--grid-type", "adaptive", "--interface-tracking", "eulerian,semi-lagrangian", "--paused", "--exit-when-done",
                         "--steps-per-frame", "3", "--frame-size", "512", "-j", "4"});
    CHECK(p.ok && p.inv.paused && p.inv.exitWhenDone && p.inv.stepsPerFrame == 3 && p.inv.frameSize == 512.0f && p.inv.threads == 4);
    CHECK(p.inv.slComparison);
    CHECK(!Parse(P("enright-3d-semi-lagrangian-dweave"), {}).inv.slComparison);
}

TEST(cli_viewer_commands_resolve_as_today)
{
    using demos::Toybox;
    const auto SpringlsViewer = [] { return *demos::Find("mesh-3d-dweave"); };
    Parsed p = Parse(SpringlsViewer(), {"mesh.ply", "--no-gui"});
    CHECK(p.ok && p.inv.scenario == Scenario::Mesh && p.inv.domain == Domain::D3 && Keys(p.inv) == "dweave" && p.inv.noGui);
    CHECK(p.inv.run.mesh.mesh == "mesh.ply" && p.inv.run.mesh.grid.maxDepth == -1 && p.inv.run.mesh.grid.cellScale == 1.0f &&
          p.inv.run.mesh.grid.gridOverride == 0 && p.inv.run.mesh.settleIterations == 20 && !p.inv.run.mesh.cacheDir.empty());
    CHECK(p.inv.limits.maxSteps == 0);  // --steps defaults to 0 steps
    p = Parse(SpringlsViewer(), {});
    CHECK(p.ok && p.inv.run.mesh.mesh.filename() == "metasphere.ply" && !p.inv.noGui && !p.inv.stepsGiven);
    p = Parse(SpringlsViewer(), {"assets/model/cosplay.ply"});
    CHECK(p.ok && p.inv.run.mesh.mesh == "assets/model/cosplay.ply");
    p = Parse(SpringlsViewer(), {"m.ply", "--no-gui", "--no-cache", "--steps", "2", "--max-depth", "4", "--cell-scale", "1.5", "--grid", "40", "--element",
                                 "triangle", "--settle", "3", "--set", "fillDistance=0.4"});
    CHECK(p.ok && p.inv.run.mesh.cacheDir.empty() && p.inv.limits.maxSteps == 2 && p.inv.stepsGiven && p.inv.run.mesh.grid.maxDepth == 4 &&
          p.inv.run.mesh.grid.cellScale == 1.5f && p.inv.run.mesh.grid.gridOverride == 40 && p.inv.run.elementType == springls::ElementType::Triangle &&
          p.inv.run.mesh.settleIterations == 3 && p.inv.run.tuning.size() == 1);
    p = Parse(SpringlsViewer(), {"m.ply", "--screenshot", "view.png", "--steps", "2", "--matcap", "jade", "--play", "--z-up"});
    CHECK(p.ok && p.inv.screenshot == "view.png" && p.inv.matcap == "jade" && p.inv.play && p.inv.zUp && p.inv.limits.maxSteps == 2);
    p = Parse(SpringlsViewer(), {"--max-depth", "deep"});
    CHECK(!p.ok && p.exitCode == 2 && p.error.find("--max-depth expects a non-negative integer or auto") != std::string::npos);
    // toybox selects the mesh scenario by axis.
    p = Parse(Toybox(), {"--scenario", "mesh", "--domain", "3d", "m.ply"});
    CHECK(p.ok && Keys(p.inv) == "dweave" && p.inv.run.mesh.mesh == "m.ply" && p.inv.notes.empty());
}

TEST(cli_unimplemented_selection_exit_2)
{
    const Parsed p = Parse(demos::Toybox(), {"-r", "levelset", "--domain", "3d", "--objects", "multiple"});
    CHECK(!p.ok && p.exitCode == 2);
    CHECK(p.error.find("not implemented: levelset 3d uniform multiple enright eulerian") != std::string::npos);
    CHECK(p.error.find("nearest") != std::string::npos && p.error.find("levelset") != std::string::npos);
    const Parsed surface = Parse(demos::Toybox(), {"--domain", "surf", "-r", "springls"});
    CHECK(!surface.ok && surface.exitCode == 2 && surface.error.find("not implemented: springls surf") != std::string::npos);
}

TEST(cli_mixed_domains_rejected)
{
    Parsed p = Parse(demos::Toybox(), {"--domain", "2d,3d"});
    CHECK(!p.ok && p.exitCode == 2 && p.error.find("one domain and scenario per invocation") != std::string::npos);
    p = Parse(demos::Toybox(), {"--scenario", "enright,mesh", "--domain", "3d"});
    CHECK(!p.ok && p.exitCode == 2 && p.error.find("one domain and scenario per invocation") != std::string::npos);
}

TEST(cli_flag_invalid_for_domain)
{
    Parsed p = Parse(*demos::Find("enright-2d-springls"), {"--elements", "triangle"});
    CHECK(!p.ok && p.exitCode == 2 && p.error.find("--elements") != std::string::npos);
    p = Parse(*demos::Find("mesh-3d-dweave"), {"--csv", "x.csv"});
    CHECK(!p.ok && p.exitCode == 2 && p.error.find("--csv is not supported for the mesh scenario") != std::string::npos);
    p = Parse(*demos::Find("enright-2d-springls"), {"--hide", "iso"});
    CHECK(!p.ok && p.exitCode == 2 && p.error.find("--hide") != std::string::npos);
    p = Parse(*demos::Find("enright-3d-dweave"), {"--cell-scale", "2"});
    CHECK(!p.ok && p.exitCode == 2 && p.error.find("--cell-scale") != std::string::npos);
    p = Parse(*demos::Find("enright-2d-springls"), {"extra.ply"});
    CHECK(!p.ok && p.exitCode == 2);
    p = Parse(*demos::Find("enright-3d-dweave"), {"--hide", "fog"});
    CHECK(!p.ok && p.exitCode == 2 && p.error.find("unknown layer 'fog'") != std::string::npos);
    p = Parse(*demos::Find("enright-2d-springls"), {"--pacing", "sideways"});
    CHECK(!p.ok && p.exitCode == 2 && p.error.find("unknown pacing 'sideways'") != std::string::npos);
    p = Parse(*demos::Find("enright-2d-springls"), {"-g", "4"});
    CHECK(!p.ok && p.exitCode == 2 && p.error.find("need --grid >= 8") != std::string::npos);
    p = Parse(*demos::Find("enright-2d-springls"), {"--set", "fill"});
    CHECK(!p.ok && p.exitCode == 2 && p.error.find("--set expects key=value, got 'fill'") != std::string::npos);
    p = Parse(*demos::Find("enright-2d-springls"), {"-m", "nope"});
    CHECK(!p.ok && p.exitCode == 2 && p.error.find("unknown method 'nope'") != std::string::npos);
    p = Parse(*demos::Find("enright-2d-springls"), {"--no-such-flag"});
    CHECK(!p.ok && p.exitCode == 2);
}

TEST(cli_m_with_domain)
{
    Parsed p = Parse(demos::Toybox(), {"--domain", "3d", "-m", "dweave"});
    CHECK(p.ok && p.inv.domain == Domain::D3 && p.inv.entries.size() == 1 && p.inv.entries[0]->config.domain == Domain::D3 &&
          p.inv.entries[0]->config.grid == GridType::Uniform);
    CHECK(p.ok && p.inv.run.period == 3.0 && p.inv.run.maxDepth == 3);  // the 3D Eulerian profile
    p = Parse(demos::Toybox(), {"--domain", "3d", "-m", "all"});
    CHECK(p.ok && Keys(p.inv) == kAll3D);
    p = Parse(demos::Toybox(), {"-m", "dweave,springls"});
    CHECK(p.ok && Keys(p.inv) == "dweave,springls");
}

TEST(cli_m_with_axis_rejected)
{
    const Parsed p = Parse(demos::Toybox(), {"-m", "dweave", "-r", "springls"});
    CHECK(!p.ok && p.exitCode == 2 && p.error.find("--domain and --scenario") != std::string::npos);
}

TEST(cli_partial_product_runs_implemented_and_notes_rest)
{
    const Parsed p = Parse(demos::Toybox(), {"-r", "springls,levelset", "--domain", "3d", "--objects", "single,multiple"});
    CHECK(p.ok && Keys(p.inv) == "springls,adaptive-springls,levelset,adaptive-levelset");
    std::string notes;
    for (const std::string &n : p.inv.notes) notes += n + "\n";
    CHECK(notes.find("levelset 3d uniform multiple enright eulerian") != std::string::npos);
    CHECK(notes.find("springls 3d adaptive multiple enright eulerian") != std::string::npos);
    // Axes left out only take implemented values: no note.
    const Parsed quiet = Parse(demos::Toybox(), {"--grid-type", "adaptive"});
    CHECK(quiet.ok && Keys(quiet.inv) == "adaptive-springls,adaptive-dweave,adaptive-levelset" && quiet.inv.notes.empty());
    // The basic level set methods, by axis.
    const Parsed levelSets = Parse(demos::Toybox(), {"-r", "levelset", "--objects", "single"});
    CHECK(levelSets.ok && Keys(levelSets.inv) == "levelset,adaptive-levelset" && levelSets.inv.notes.empty());
    const Parsed multi = Parse(demos::Toybox(), {"--objects", "multiple"});
    CHECK(multi.ok && multi.inv.entries.size() == 6);
}

TEST(cli_sl_profile_applies_to_mixed_selection)
{
    const Parsed p = Parse(demos::Toybox(), {"-r", "dweave", "--grid-type", "adaptive", "--interface-tracking", "eulerian,semi-lagrangian"});
    CHECK(p.ok && Keys(p.inv) == kSl);
    CHECK(p.inv.run.gridSize == 128 && p.inv.run.maxDepth == 2 && p.inv.run.maxTimeStepFraction == k50 && p.inv.pacing == Pacing::EqualCompute &&
          p.inv.run.substepsPerStep == 1 && p.inv.run.period == 8.0);
    const Parsed sl3 = Parse(demos::Toybox(), {"--domain", "3d", "--interface-tracking", "semi-lagrangian"});
    CHECK(sl3.ok && Keys(sl3.inv) == "semi-lagrangian-dweave" && sl3.inv.run.gridSize == 32 && sl3.inv.run.maxDepth == 2 && sl3.inv.run.period == 3.0);
}

TEST(cli_demo_user_args_override)
{
    Parsed p = Parse(*demos::Find("enright-3d-dweave"), {"-m", "springls", "-T", "1"});
    CHECK(p.ok && Keys(p.inv) == "springls" && p.inv.run.period == 1.0 && p.inv.domain == Domain::D3);
    p = Parse(*demos::Find("enright-2d-semi-lagrangian-dweave"), {"-g", "64"});
    CHECK(p.ok && p.inv.run.gridSize == 64 && Keys(p.inv) == "semi-lagrangian-dweave");
    p = Parse(*demos::Find("enright-2d-multi-levelset"), {"-m", "springls"});
    CHECK(p.ok && Keys(p.inv) == "springls");
    // An axis flag replaces the demo's method list.
    p = Parse(*demos::Find("enright-2d-springls"), {"-r", "dweave"});
    CHECK(p.ok && Keys(p.inv) == "dweave,adaptive-dweave");
    p = Parse(*demos::Find("enright-2d-adaptive-dweave"), {"--interface-tracking", "semi-lagrangian"});
    CHECK(p.ok && Keys(p.inv) == "semi-lagrangian-dweave" && p.inv.run.gridSize == 128);
}

TEST(cli_list_prints_matrix)
{
    Parsed p = Parse(demos::Toybox(), {"--list"});
    CHECK(p.ok && p.inv.list);
    std::ostringstream captured;
    std::streambuf *old = std::cout.rdbuf(captured.rdbuf());
    const char *argv[] = {"toybox", "--list"};
    const int code = Main(2, const_cast<char **>(argv), demos::Toybox(), nullptr);
    std::cout.rdbuf(old);
    CHECK(code == 0 && captured.str() == Registry::Instance().markdownMatrix());
}

TEST(main_headless_2d_runs_and_summarizes)
{
    std::ostringstream captured;
    std::streambuf *old = std::cout.rdbuf(captured.rdbuf());
    const char *argv[] = {"enright-2d-springls", "--no-gui", "-g", "32", "-T", "0.25"};
    const int code = Main(6, const_cast<char **>(argv), *demos::Find("enright-2d-springls"), nullptr);
    std::cout.rdbuf(old);
    const std::string out = captured.str();
    CHECK(code == 0);
    CHECK(out.rfind("Enright 2D: springls, grid 32, T = 0.25, max depth 5, ", 0) == 0);
    CHECK(out.find("Enright 2D, T = 0.25, errors at t = 0.250 (domain units):\n  Spring Level Set") != std::string::npos);
}

// The semi-Lagrangian comparison output (banner, ratio line, titles) needs a semi-Lagrangian run.
TEST(cli_sl_comparison_needs_a_semi_lagrangian_run)
{
    Parsed p = Parse(demos::Toybox(), {"-r", "springls", "--interface-tracking", "eulerian,semi-lagrangian"});
    CHECK(p.ok && Keys(p.inv) == "springls,adaptive-springls" && !p.inv.slComparison && !p.inv.semiLagrangian);
    p = Parse(demos::Toybox(), {"-r", "dweave", "--grid-type", "adaptive", "--interface-tracking", "eulerian,semi-lagrangian"});
    CHECK(p.ok && p.inv.slComparison && !p.inv.slOnly);
    p = Parse(demos::Toybox(), {"--interface-tracking", "semi-lagrangian"});
    CHECK(p.ok && p.inv.slComparison && p.inv.slOnly);
}

// Mesh tracking is semi-Lagrangian only, so its representation alone selects it (not as the SL comparison), and the
// other representations keep their Eulerian default.
TEST(cli_tracked_mesh_representation_selects_mesh_tracking)
{
    Parsed p = Parse(demos::Toybox(), {"--domain", "3d", "-r", "tracked-mesh"});
    CHECK(p.ok && Keys(p.inv) == "mesh-tracking" && !p.inv.slComparison && p.inv.notes.empty());
    p = Parse(demos::Toybox(), {"--domain", "3d", "-r", "dweave"});
    CHECK(p.ok && Keys(p.inv) == "dweave,adaptive-dweave");
}

// A skipped combination of explicitly given axis values is noted even when each value alone has runs.
TEST(cli_notes_skipped_combination_of_given_axes)
{
    const Parsed p = Parse(demos::Toybox(), {"-r", "dweave", "--grid-type", "uniform,adaptive", "--interface-tracking", "eulerian,semi-lagrangian"});
    CHECK(p.ok && Keys(p.inv) == "dweave,adaptive-dweave,semi-lagrangian-dweave");
    std::string notes;
    for (const std::string &n : p.inv.notes) notes += n + "\n";
    CHECK(notes.find("dweave 2d uniform single enright semi-lagrangian") != std::string::npos);
    // Defaulted axes stay quiet.
    const Parsed quiet = Parse(demos::Toybox(), {"--interface-tracking", "eulerian,semi-lagrangian"});
    CHECK(quiet.ok && quiet.inv.notes.empty());
}

TEST(cli_sphere_defaults)
{
    const Parsed p = Parse(*demos::Find("sphere-surf-multi-springls"), {});
    CHECK(p.ok);
    CHECK(p.inv.scenario == Scenario::Sphere && p.inv.domain == Domain::Surface && p.inv.entries.size() == 1);
    const SphereOptions &s = p.inv.run.sphere;
    CHECK(s.session.mesh.empty() && s.every == 1 && s.hidden.empty() && s.threads == 0);
    CHECK(s.session.config.object == springls::LoopObject::PuzzlePiece && s.session.maxLevel == 3);
    CHECK(!p.inv.stepsGiven && !p.inv.play && !p.inv.noFollow);
    CHECK(p.inv.limits.maxSteps == std::numeric_limits<uint64_t>::max());
}

TEST(cli_stone_defaults)
{
    // The sphere scenario's options on assets/model/stone.ply; a mesh argument still replaces it.
    const Parsed p = Parse(*demos::Find("stone-surf-multi-springls"), {});
    CHECK(p.ok && p.inv.scenario == Scenario::Stone && p.inv.domain == Domain::Surface && p.inv.entries.size() == 1);
    const SphereOptions &s = p.inv.run.sphere;
    CHECK(s.session.mesh.filename() == "stone.ply" && std::filesystem::exists(s.session.mesh));
    CHECK(s.session.config.object == springls::LoopObject::PuzzlePiece && s.session.maxLevel == 3);
    const Parsed star = Parse(demos::Toybox(), {"--scenario", "stone", "--object", "star", "--speed", "2"});
    CHECK(star.ok && star.inv.scenario == Scenario::Stone && star.inv.domain == Domain::Surface);
    CHECK(star.inv.run.sphere.session.config.object == springls::LoopObject::Star && star.inv.run.sphere.session.config.speed == 2.0);
    const Parsed other = Parse(*demos::Find("stone-surf-multi-springls"), {"other.ply"});
    CHECK(other.ok && other.inv.run.sphere.session.mesh == "other.ply");
    // The stone runs the Hodge flow by default, the sphere the rotation; --flow picks either.
    CHECK(s.session.config.flow == springls::LoopFlow::Hodge);
    const Parsed sphere = Parse(*demos::Find("sphere-surf-multi-springls"), {});
    CHECK(sphere.ok && sphere.inv.run.sphere.session.config.flow == springls::LoopFlow::Rotation);
    const Parsed sphereHodge = Parse(*demos::Find("sphere-surf-multi-springls"), {"--flow", "hodge"});
    CHECK(sphereHodge.ok && sphereHodge.inv.run.sphere.session.config.flow == springls::LoopFlow::Hodge);
    const Parsed stoneRotation = Parse(*demos::Find("stone-surf-multi-springls"), {"--flow", "rotation"});
    CHECK(stoneRotation.ok && stoneRotation.inv.run.sphere.session.config.flow == springls::LoopFlow::Rotation);
    const Parsed badFlow = Parse(*demos::Find("stone-surf-multi-springls"), {"--flow", "swirl"});
    CHECK(!badFlow.ok && badFlow.error.find("--flow") != std::string::npos);
    CHECK(!Parse(*demos::Find("enright-2d-springls"), {"--flow", "hodge"}).ok);
    // Enright-only flags are rejected for the stone as for the sphere.
    const Parsed bad = Parse(*demos::Find("stone-surf-multi-springls"), {"--period", "3"});
    CHECK(!bad.ok && bad.error.find("stone scenario") != std::string::npos);
}

TEST(cli_sphere_steps_zero_takes_no_step)
{
    const Parsed p = Parse(*demos::Find("sphere-surf-multi-springls"), {"--steps", "0"});
    CHECK(p.ok && p.inv.stepsGiven && p.inv.limits.maxSteps == 0);
    // The Enright scenarios keep 0 as "no limit".
    const Parsed e = Parse(*demos::Find("enright-2d-springls"), {"--steps", "0"});
    CHECK(e.ok && e.inv.limits.maxSteps == std::numeric_limits<uint64_t>::max());
}

TEST(cli_sphere_no_follow)
{
    const Parsed p = Parse(*demos::Find("sphere-surf-multi-springls"), {"--no-follow"});
    CHECK(p.ok && p.inv.noFollow);
    const Parsed e = Parse(*demos::Find("enright-2d-springls"), {"--no-follow"});
    CHECK(!e.ok && e.error.find("is not supported") != std::string::npos);
}

TEST(cli_sphere_hides_isolines)
{
    const Parsed p = Parse(*demos::Find("sphere-surf-multi-springls"), {"--hide", "isolines"});
    CHECK(p.ok && p.inv.run.sphere.hidden == std::vector<std::string>({"isolines"}));
}

TEST(cli_sphere_options)
{
    const Parsed p = Parse(*demos::Find("sphere-surf-multi-springls"),
                           {"m.ply",       "--object", "star",    "--speed", "2",      "--loop-distance", "0.8",   "--radius",         "0.2",
                            "--max-level", "1",        "--every", "5",       "--hide", "arrows,contour",  "--set", "fillDistance=0.4", "--csv",
                            "m.csv",       "-j",       "3",       "--steps", "7"});
    CHECK(p.ok);
    if (!p.ok) return;
    const SphereOptions &s = p.inv.run.sphere;
    CHECK(s.session.mesh == "m.ply" && s.session.config.object == springls::LoopObject::Star && s.session.config.speed == 2.0 &&
          s.session.config.loopDistance == 0.8 && s.session.config.radius == 0.2 && s.session.maxLevel == 1);
    CHECK(s.every == 5 && s.hidden == std::vector<std::string>({"arrows", "contour"}) && s.threads == 3);
    CHECK(s.session.tuning.size() == 1 && s.session.tuning[0].first == "fillDistance" && s.session.tuning[0].second == 0.4);
    CHECK(p.inv.csv == "m.csv" && p.inv.limits.maxSteps == 7 && p.inv.stepsGiven);
}

TEST(cli_sphere_rejects_other_flags)
{
    const Demo demo = *demos::Find("sphere-surf-multi-springls");
    for (const std::vector<std::string> &args : {std::vector<std::string>{"--period", "2"}, std::vector<std::string>{"--grid", "64"}}) {
        const Parsed p = Parse(demo, args);
        CHECK(!p.ok && p.error.find("is not supported for the sphere scenario") != std::string::npos);
    }
}

TEST(cli_enright_rejects_sphere_flags)
{
    const Parsed p = Parse(*demos::Find("enright-2d-springls"), {"--object", "star"});
    CHECK(!p.ok && p.error.find("is not supported") != std::string::npos);
}

TEST(cli_sphere_max_level)
{
    const Demo demo = *demos::Find("sphere-surf-multi-springls");
    CHECK(Parse(demo, {}).inv.run.sphere.session.maxLevel == 3);
    CHECK(Parse(demo, {"--max-level", "1"}).inv.run.sphere.session.maxLevel == 1);
    for (const char *gone : {"--subdivide", "--subdivide-margin"}) {
        const Parsed p = Parse(demo, {gone, "1"});
        CHECK(!p.ok);
    }
}

TEST(cli_sphere_hides_reference)
{
    const Parsed p = Parse(*demos::Find("sphere-surf-multi-springls"), {"--hide", "reference,arrows"});
    CHECK(p.ok && p.inv.run.sphere.hidden == std::vector<std::string>({"reference", "arrows"}));
}

TEST(cli_sphere_bad_hide)
{
    const Parsed p = Parse(*demos::Find("sphere-surf-multi-springls"), {"--hide", "teapot"});
    CHECK(!p.ok && p.error.find("teapot") != std::string::npos);
    CHECK(!Parse(*demos::Find("sphere-surf-multi-springls"), {"--object", "cube"}).ok);
    CHECK(!Parse(*demos::Find("sphere-surf-multi-springls"), {"--set", "nonsense=1"}).ok);
}
