#pragma once
// Runs the selected configurations side by side (one Simulation per run): pacing, metrics CSV,
// checkpoints, step limits and the summary. Replaces enright::Experiment and enright3d::Experiment.
#include "Simulation.hpp"

#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace toybox
{
// Which runs each Experiment::step() advances.
enum class Pacing {
    Lockstep,      // every unfinished run takes one step
    EqualCompute,  // the run that has used the least wall-clock time takes one step
    SameTime,      // the run furthest behind in simulation time takes one step
};
const char *PacingKey(Pacing pacing);  // "lockstep", "equal-compute", "same-time"
const char *PacingTitle(Pacing pacing);
bool ParsePacing(const std::string &key, Pacing &pacing);

struct Limits {
    // Checkpoints: each run saves its state to checkpointDir once its simulation time reaches each of
    // checkpointTimes (runs that support checkpoints). resumeFile (a .state, one run only) starts the run
    // from a saved state instead of t = 0. Runs stop once they reach stopAt.
    std::vector<double> checkpointTimes;
    std::filesystem::path checkpointDir = "checkpoints";
    std::filesystem::path resumeFile;
    double stopAt = std::numeric_limits<double>::infinity();
    // Runs stop after maxSteps steps from where they started (or resumed), and abort once their springl
    // count exceeds maxSpringlGrowth times its count there.
    uint64_t maxSteps = std::numeric_limits<uint64_t>::max();
    double maxSpringlGrowth = std::numeric_limits<double>::infinity();
};

struct Run {
    std::unique_ptr<Simulation> sim;
    double seconds = 0.0;      // wall-clock time spent stepping (and initializing)
    double stepSeconds = 0.0;  // wall-clock time of the latest step (0 for the initial state)
    bool done = false;
    bool stepped = true;        // advanced in the latest step() (or is the initial state)
    bool aborted = false;       // stopped by maxSpringlGrowth
    uint64_t stepsTaken = 0;    // steps since the start (or resume)
    size_t startSpringls = 0;   // springl count at the start (or resume)
    size_t nextCheckpoint = 0;  // index into the sorted checkpoint times
    template <class T>
    T *as()
    {
        return dynamic_cast<T *>(sim.get());
    }
    template <class T>
    const T *as() const
    {
        return dynamic_cast<const T *>(sim.get());
    }
};

class Experiment
{
public:
    // Creates and initializes one run per entry; returns false (with a message) on bad options.
    bool init(std::vector<const Entry *> entries, const RunOptions &options, Pacing pacing, const Limits &limits, std::string &error);
    // init() again with the current entries, options (options() may have changed them), pacing and limits.
    bool restart(std::string &error);
    // Advances the unfinished runs the pacing selects by one step; returns false when nothing advanced.
    bool step();
    bool done() const;
    // Number of step() calls that advanced (frame counter).
    uint64_t stepCount() const { return mStepCount; }
    // Incremented by every init(), so views can tell a restarted run from the old one.
    uint64_t generation() const { return mGeneration; }

    const std::vector<Run> &runs() const { return mRuns; }
    std::vector<Run> &runs() { return mRuns; }
    const std::vector<const Entry *> &entries() const { return mEntries; }
    const RunOptions &options() const { return mOptions; }
    RunOptions &options() { return mOptions; }
    const Limits &limits() const { return mLimits; }
    Pacing pacing() const { return mPacing; }
    void setPacing(Pacing pacing) { mPacing = pacing; }
    double period() const { return mOptions.period; }
    // Applies `apply` to every run's simulation (e.g. toggling resampling from a window).
    template <class Fn>
    void forEachSimulation(Fn &&apply)
    {
        for (Run &run : mRuns) apply(*run.sim);
    }

    // Metrics CSV: one row per run for the initial state and for every step it takes, plus the per-region
    // rows of multi-object runs in <stem>-labels<ext>. False when a file cannot be written or no run supports CSV.
    bool openCsv(const std::filesystem::path &file);
    void closeCsv();
    // The domain's header line and one line (or more) per run.
    void printSummary(std::ostream &out) const;

private:
    std::vector<const Entry *> mEntries;
    RunOptions mOptions;
    Pacing mPacing = Pacing::Lockstep;
    Limits mLimits;
    std::vector<Run> mRuns;
    std::ofstream mCsv;
    std::ofstream mLabelsCsv;
    uint64_t mStepCount = 0;
    uint64_t mGeneration = 0;
    void writeCsvRows();
    // Saves run's state for every checkpoint time it has reached; false (with error) on failure.
    bool saveCheckpoints(Run &run, std::string &error);
};
}  // namespace toybox
