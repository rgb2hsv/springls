#include "Check.hpp"

#include <core/Enright3DSimulation.hpp>
#include <core/Experiment.hpp>
#include <core/Registry.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace toybox;

namespace
{
std::vector<const Entry *> Entries(Domain domain, std::initializer_list<const char *> keys)
{
    std::vector<const Entry *> out;
    for (const char *key : keys) out.push_back(Registry::Instance().findKey(domain, Scenario::Enright, key));
    return out;
}

RunOptions Small2D()
{
    RunOptions options;
    options.gridSize = 32;
    options.period = 0.25;
    options.maxDepth = 2;
    return options;
}

RunOptions Small3D()
{
    RunOptions options;
    options.gridSize = 16;
    options.period = 0.5;
    options.maxDepth = 2;
    options.maxTimeStepFraction = 1.0 / 800.0;
    return options;
}

std::string ReadFile(const std::filesystem::path &path)
{
    std::ifstream in(path, std::ios::binary);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

// The CSV without its step_seconds column (found by header name), as the baseline scripts store it.
std::string Normalize(const std::string &csv)
{
    std::istringstream in(csv);
    std::string line, out;
    int drop = -1;
    bool header = true;
    while (std::getline(in, line)) {
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string field;
        while (std::getline(ss, field, ',')) fields.push_back(field);
        if (header) {
            for (size_t i = 0; i < fields.size(); i++) {
                if (fields[i] == "step_seconds") drop = static_cast<int>(i);
            }
            header = false;
        }
        std::string joined;
        for (size_t i = 0; i < fields.size(); i++) {
            if (static_cast<int>(i) == drop) continue;
            joined += (joined.empty() ? "" : ",") + fields[i];
        }
        out += joined + "\n";
    }
    return out;
}

const std::filesystem::path kSource = SPRINGLS_SOURCE_DIR;
}  // namespace

TEST(experiment_lockstep_steps_every_run)
{
    Experiment experiment;
    std::string error;
    CHECK(experiment.init(Entries(Domain::D2, {"springls", "dweave"}), Small2D(), Pacing::Lockstep, Limits(), error));
    CHECK(experiment.runs().size() == 2 && !experiment.done() && experiment.stepCount() == 0);
    CHECK(experiment.step());
    for (const Run &run : experiment.runs()) CHECK(run.stepped && run.stepsTaken == 1 && run.sim->iteration() == 1);
    CHECK(experiment.stepCount() == 1);
    while (experiment.step()) {
    }
    CHECK(experiment.done());
    for (const Run &run : experiment.runs()) CHECK(run.done && run.sim->complete());
}

TEST(experiment_equal_compute_picks_least_time)
{
    Experiment experiment;
    std::string error;
    CHECK(experiment.init(Entries(Domain::D2, {"springls", "dweave"}), Small2D(), Pacing::EqualCompute, Limits(), error));
    experiment.runs()[0].seconds = 1e9;  // run 0 has used far more compute
    CHECK(experiment.step());
    CHECK(!experiment.runs()[0].stepped && experiment.runs()[1].stepped);
    CHECK(experiment.runs()[0].sim->iteration() == 0 && experiment.runs()[1].sim->iteration() == 1);
    // Same time: the run furthest behind in simulation time steps.
    experiment.setPacing(Pacing::SameTime);
    CHECK(experiment.step());
    CHECK(experiment.runs()[0].stepped && !experiment.runs()[1].stepped);
}

TEST(experiment_csv_2d_matches_old_format)
{
    const auto reference = kSource / "build/baseline/springls.norm";
    if (!std::filesystem::exists(reference)) {
        std::printf("  (skipped: no 2D baseline at %s)\n", reference.string().c_str());
        return;
    }
    // The baseline's run: enright2d --no-gui -g 32 -T 1 -d 2 -m springls --csv
    RunOptions options;
    options.gridSize = 32;
    options.period = 1.0;
    options.maxDepth = 2;
    const auto csv = std::filesystem::temp_directory_path() / "toybox_experiment_2d.csv";
    Experiment experiment;
    std::string error;
    CHECK(experiment.init(Entries(Domain::D2, {"springls"}), options, Pacing::Lockstep, Limits(), error));
    CHECK(experiment.openCsv(csv));
    while (experiment.step()) {
    }
    experiment.closeCsv();
    CHECK(Normalize(ReadFile(csv)) == ReadFile(reference));
}

TEST(experiment_csv_3d_matches_old_format)
{
    const auto reference = kSource / "build/baseline3d/dweave.norm";
    if (!std::filesystem::exists(reference)) {
        std::printf("  (skipped: no 3D baseline at %s)\n", reference.string().c_str());
        return;
    }
    // The baseline's run: enright3d --no-gui -m dweave -g 16 -T 0.5 -d 2 --steps 20 --csv
    RunOptions options = Small3D();
    Limits limits;
    limits.maxSteps = 20;
    const auto csv = std::filesystem::temp_directory_path() / "toybox_experiment_3d.csv";
    Experiment experiment;
    std::string error;
    CHECK(experiment.init(Entries(Domain::D3, {"dweave"}), options, Pacing::Lockstep, limits, error));
    CHECK(experiment.openCsv(csv));
    while (experiment.step()) {
    }
    experiment.closeCsv();
    CHECK(Normalize(ReadFile(csv)) == ReadFile(reference));
}

TEST(experiment_labels_csv_for_multi)
{
    const auto csv = std::filesystem::temp_directory_path() / "toybox_experiment_multi.csv";
    const auto labels = std::filesystem::temp_directory_path() / "toybox_experiment_multi-labels.csv";
    std::filesystem::remove(labels);
    Experiment experiment;
    std::string error;
    CHECK(experiment.init(Entries(Domain::D2, {"multi-levelset", "springls"}), Small2D(), Pacing::Lockstep, Limits(), error));
    CHECK(experiment.openCsv(csv));
    experiment.step();
    experiment.closeCsv();
    const std::string text = ReadFile(labels);
    CHECK(text.rfind("method,step,time,label,area,relative_area_error\nmulti-levelset,0,0,1,", 0) == 0);
    CHECK(text.find("springls,") == std::string::npos);
    CHECK(text.find("multi-levelset,1,") != std::string::npos);
    // Single-object runs write no labels file.
    const auto plainLabels = std::filesystem::temp_directory_path() / "toybox_experiment_plain-labels.csv";
    std::filesystem::remove(plainLabels);
    Experiment plain;
    CHECK(plain.init(Entries(Domain::D2, {"springls"}), Small2D(), Pacing::Lockstep, Limits(), error));
    CHECK(plain.openCsv(std::filesystem::temp_directory_path() / "toybox_experiment_plain.csv"));
    CHECK(!std::filesystem::exists(plainLabels));
}

TEST(experiment_3d_checkpoint_and_resume)
{
    const auto dir = std::filesystem::temp_directory_path() / "toybox_experiment_checkpoints";
    std::filesystem::remove_all(dir);
    Limits limits;
    limits.checkpointTimes = {0.005};
    limits.checkpointDir = dir;
    limits.stopAt = 0.006;
    Experiment experiment;
    std::string error;
    CHECK(experiment.init(Entries(Domain::D3, {"dweave"}), Small3D(), Pacing::Lockstep, limits, error));
    while (experiment.step()) {
    }
    const Run &done = experiment.runs()[0];
    CHECK(done.done && done.sim->time() >= 0.006);
    const auto state = dir / "dweave_quad_g16_d2_t0.0050.state";
    CHECK(std::filesystem::exists(state));

    Limits resume;
    resume.resumeFile = state;
    resume.checkpointTimes = {0.005};  // already passed: not written again
    resume.checkpointDir = dir / "again";
    resume.maxSteps = 2;
    Experiment resumed;
    CHECK(resumed.init(Entries(Domain::D3, {"dweave"}), Small3D(), Pacing::Lockstep, resume, error));
    const Run &run = resumed.runs()[0];
    CHECK(run.sim->time() >= 0.005 && run.sim->iteration() > 0 && run.startSpringls == run.sim->springlCount());
    const uint64_t start = run.sim->iteration();
    while (resumed.step()) {
    }
    CHECK(run.sim->iteration() == start + 2 && run.stepsTaken == 2);
    CHECK(!std::filesystem::exists(dir / "again"));
    // Resuming needs exactly one run.
    Experiment two;
    CHECK(!two.init(Entries(Domain::D3, {"dweave", "springls"}), Small3D(), Pacing::Lockstep, resume, error));
}

TEST(experiment3d_resumes_checkpoint_written_before_refactor)
{
    const auto state = kSource / "build/baseline3d/checkpoint/dweave_quad_g16_d2_t0.1000.state";
    if (!std::filesystem::exists(state)) {
        std::printf("  (skipped: no checkpoint at %s)\n", state.string().c_str());
        return;
    }
    Limits limits;
    limits.resumeFile = state;
    limits.maxSteps = 1;
    Experiment experiment;
    std::string error;
    CHECK(experiment.init(Entries(Domain::D3, {"dweave"}), Small3D(), Pacing::Lockstep, limits, error));
    if (experiment.runs().empty()) return;
    const Run &run = experiment.runs()[0];
    // Saved by the enright3d of the baseline at t = 0.10125, step 54.
    CHECK(run.sim->iteration() == 54);
    CHECK_NEAR(run.sim->time(), 0.10125, 1e-6);  // stored in single precision
    CHECK(experiment.step() && run.sim->iteration() == 55);
}

TEST(experiment_max_steps_and_stop_at)
{
    Limits limits;
    limits.maxSteps = 3;
    Experiment experiment;
    std::string error;
    CHECK(experiment.init(Entries(Domain::D2, {"springls"}), Small2D(), Pacing::Lockstep, limits, error));
    int steps = 0;
    while (experiment.step()) steps++;
    CHECK(steps == 3 && experiment.runs()[0].sim->iteration() == 3 && experiment.done());

    Limits stop;
    stop.stopAt = 0.05;
    Experiment stopped;
    CHECK(stopped.init(Entries(Domain::D2, {"springls"}), Small2D(), Pacing::Lockstep, stop, error));
    while (stopped.step()) {
    }
    const double t = stopped.runs()[0].sim->time();
    CHECK(t >= 0.05 && t < 0.25);

    Limits none;
    none.maxSteps = 0;
    Experiment zero;
    CHECK(zero.init(Entries(Domain::D2, {"springls"}), Small2D(), Pacing::Lockstep, none, error));
    CHECK(zero.done() && !zero.step());
}

TEST(experiment_springl_growth_aborts)
{
    Limits limits;
    limits.maxSpringlGrowth = 0.5;  // any count above half the start aborts
    Experiment experiment;
    std::string error;
    CHECK(experiment.init(Entries(Domain::D3, {"dweave"}), Small3D(), Pacing::Lockstep, limits, error));
    CHECK(experiment.step());
    CHECK(experiment.runs()[0].aborted && experiment.runs()[0].done && experiment.done());
}

TEST(experiment_restart_and_summary)
{
    Experiment experiment;
    std::string error;
    CHECK(experiment.init(Entries(Domain::D2, {"springls"}), Small2D(), Pacing::Lockstep, Limits(), error));
    const uint64_t generation = experiment.generation();
    experiment.step();
    CHECK(experiment.restart(error));
    CHECK(experiment.generation() == generation + 1 && experiment.runs()[0].sim->iteration() == 0 && experiment.stepCount() == 0);
    std::ostringstream out;
    experiment.printSummary(out);
    CHECK(out.str().rfind("Enright 2D, T = 0.25, errors at t = 0.000 (domain units):\n  Spring Level Set", 0) == 0);

    Experiment threeD;
    CHECK(threeD.init(Entries(Domain::D3, {"dweave"}), Small3D(), Pacing::Lockstep, Limits(), error));
    std::ostringstream out3;
    threeD.printSummary(out3);
    CHECK(out3.str().rfind("Enright 3D, quad springls, T = 0.5, errors at t = 0.000 (domain units):\n  DoubleWeave 3D", 0) == 0);
    const ExperimentSnapshot snap = Snapshot3D(threeD);
    CHECK(snap.runs.size() == 1 && snap.period == 0.5 && snap.generation == threeD.generation() && !snap.done());
}
