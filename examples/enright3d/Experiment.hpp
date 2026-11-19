#pragma once
#include <springls/AdaptiveGrid3D.hpp>
#include <springls/EnrightTest3D.hpp>

#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace enright3d
{
// Which runs each Experiment::step() advances.
enum class Pacing {
    Lockstep,      // every unfinished run takes one step
    EqualCompute,  // the run that has used the least wall-clock time takes one step
    SameTime,      // the run furthest behind in simulation time takes one step
};
const char *PacingKey(Pacing pacing);  // "lockstep", "equal-compute", "same-time"
bool ParsePacing(const std::string &key, Pacing &pacing);

struct ExperimentOptions {
    // config.gridSize 0 picks DefaultGridSize() per method (128^3 uniform, 32^3 adaptive base).
    springls::EnrightConfig3D config;
    int maxDepth = 3;  // adaptive methods: finest cells are gridSize * 2^maxDepth equivalent
    std::vector<springls::Method3D> methods;
    springls::ElementType elementType = springls::ElementType::Quad;
    std::vector<std::pair<std::string, double>> tuning;
    bool verbose = false;
    Pacing pacing = Pacing::Lockstep;
    // Checkpoints: each run saves its state (SpringLevelSet3D::saveState) to checkpointDir once its
    // simulation time reaches each of checkpointTimes, as <name>.state (springls and scalars, cereal) and
    // <name>.vdb (every level set, OpenVDB). resumeFile (a .state, one method only) starts the run from a
    // saved state, with the .vdb beside it, instead of t = 0. Runs stop once they reach stopAt. Each checkpoint
    // also writes the iso-surface and springls as binary PLY meshes (<name>.iso.ply, <name>.springls.ply).
    std::vector<double> checkpointTimes;
    std::filesystem::path checkpointDir = "checkpoints";
    std::filesystem::path resumeFile;
    double stopAt = std::numeric_limits<double>::infinity();
    // Quick tests: runs stop after maxSteps steps from where they started (or resumed), and abort
    // once their springl count exceeds maxSpringlGrowth times its count there.
    uint64_t maxSteps = std::numeric_limits<uint64_t>::max();
    double maxSpringlGrowth = std::numeric_limits<double>::infinity();
};

// One method running the Enright test.
struct Run {
    springls::Method3D method;
    springls::EnrightConfig3D config;  // with this run's grid size
    std::unique_ptr<springls::SpringLevelSet3D> sim;
    springls::EnrightMetrics3D initial;
    springls::EnrightMetrics3D current;
    bool done = false;
    double seconds = 0.0;       // wall-clock time spent stepping (and initializing)
    double stepSeconds = 0.0;   // wall-clock time of the latest step (0 for the initial state)
    bool stepped = true;        // advanced in the latest step() (or is the initial state)
    size_t nextCheckpoint = 0;  // index into the sorted checkpoint times
    uint64_t stepsTaken = 0;    // steps since the start (or resume)
    size_t startSpringls = 0;   // springl count at the start (or resume)
    bool aborted = false;       // stopped by maxSpringlGrowth
};

// Copy of what the renderer shows of one run, so drawing never reads a simulation mid-step.
struct RunSnapshot {
    springls::Method3D method;
    springls::EnrightConfig3D config;
    // Adaptive runs: finest depth in use, maximum depth and leaf count (0 for uniform runs).
    int finestDepth = 0, maxDepth = 0;
    size_t leafCount = 0;
    // Level set for slice views: the octree of adaptive runs, the uniform grid otherwise.
    std::shared_ptr<const springls::AdaptiveGrid3D> grid;
    std::shared_ptr<const springls::VolumeF> levelSet;
    springls::Manifold3D manifold;
    springls::EnrightMetrics3D metrics;
    uint64_t iteration = 0;
    uint64_t substeps = 0;  // advection sub-steps so far
    bool done = false;
    double seconds = 0.0;
};
struct ExperimentSnapshot {
    double period = 0.0;
    uint64_t generation = 0;
    std::vector<RunSnapshot> runs;
    bool done() const
    {
        for (const RunSnapshot &run : runs) {
            if (!run.done) return false;
        }
        return true;
    }
};

// Runs the selected methods side by side, one step each per call to step().
class Experiment
{
public:
    // Creates and initializes every run; returns false (with a message) on bad options.
    bool init(const ExperimentOptions &options, std::string &error);
    // Advances the unfinished runs the pacing selects by one step; returns false when nothing
    // advanced.
    bool step();
    bool done() const;
    uint64_t stepCount() const { return mStepCount; }
    // Incremented by every init(), so views can tell a restarted run from the old one.
    uint64_t generation() const { return mGeneration; }
    // Applies `apply` to every run's simulation (e.g. toggling resampling from the GUI).
    template <class Fn>
    void forEachSimulation(Fn &&apply)
    {
        for (Run &run : mRuns) apply(*run.sim);
    }

    const std::vector<Run> &runs() const { return mRuns; }
    ExperimentSnapshot snapshot() const;
    Pacing pacing() const { return mOptions.pacing; }
    void setPacing(Pacing pacing) { mOptions.pacing = pacing; }
    const ExperimentOptions &options() const { return mOptions; }
    ExperimentOptions &options() { return mOptions; }
    double period() const { return mOptions.config.period; }

    // Metrics CSV: one row per run for the initial state and for every step it takes.
    bool openCsv(const std::filesystem::path &file);
    void printSummary(std::ostream &out) const;

private:
    ExperimentOptions mOptions;
    std::vector<Run> mRuns;
    std::ofstream mCsv;
    uint64_t mStepCount = 0;
    uint64_t mGeneration = 0;
    void writeCsvRows();
    // Saves run's state for every checkpoint time it has reached; false (with error) on failure.
    bool saveCheckpoints(Run &run, std::string &error);
};
}  // namespace enright3d
