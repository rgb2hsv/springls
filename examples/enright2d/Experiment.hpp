#pragma once
#include <springls/EnrightTest2D.hpp>

#include <filesystem>
#include <fstream>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace enright
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

struct ExperimentOptions {
    springls::EnrightConfig2D config;
    std::vector<springls::Method> methods;
    int maxDepth = 4;  // adaptive methods: finest cells are gridSize * 2^maxDepth equivalent
    std::vector<std::pair<std::string, double>> tuning;
    bool verbose = false;
    Pacing pacing = Pacing::Lockstep;
};

// One method running the Enright test.
struct Run {
    springls::Method method;
    std::unique_ptr<springls::SpringLevelSet2D> sim;
    springls::EnrightMetrics2D initial;
    springls::EnrightMetrics2D current;
    bool done = false;
    double seconds = 0.0;      // wall-clock time spent stepping (and initializing)
    double stepSeconds = 0.0;  // wall-clock time of the latest step (0 for the initial state)
    bool stepped = true;       // advanced in the latest step() (or is the initial state)
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
    Pacing pacing() const { return mOptions.pacing; }
    void setPacing(Pacing pacing) { mOptions.pacing = pacing; }
    bool done() const;
    // Number of step() calls that advanced (frame counter).
    uint64_t stepCount() const { return mStepCount; }

    const std::vector<Run> &runs() const { return mRuns; }
    const ExperimentOptions &options() const { return mOptions; }
    const springls::EnrightConfig2D &config() const { return mOptions.config; }

    // Metrics CSV: one row per run for the initial state and for every step it takes, with the
    // step's wall-clock time.
    bool openCsv(const std::filesystem::path &file);
    void printSummary(std::ostream &out) const;

private:
    ExperimentOptions mOptions;
    std::vector<Run> mRuns;
    std::ofstream mCsv;
    uint64_t mStepCount = 0;
    void writeCsvRows();
};
}  // namespace enright
