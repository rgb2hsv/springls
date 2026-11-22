#pragma once
// A 2D method running the Enright test (today's enright::Run).
#include "Simulation.hpp"

#include <scenarios/EnrightTest2D.hpp>

#include <memory>

namespace toybox
{
class Enright2DSimulation : public Simulation
{
public:
    Enright2DSimulation(const Entry &entry, const RunOptions &options);

    springls::Method method() const { return mEntry.method2D; }
    // The test as this run sees it (its grid size may differ from the other runs').
    const springls::EnrightConfig2D &config() const { return mConfig; }
    const springls::SpringLevelSet2D &sim() const { return *mSim; }
    springls::SpringLevelSet2D &sim() { return *mSim; }
    const springls::EnrightMetrics2D &initial() const { return mInitial; }
    const springls::EnrightMetrics2D &current() const { return mCurrent; }

    const Entry &entry() const override { return mEntry; }
    bool init(std::string &error) override;
    bool step() override;
    bool complete() const override { return mComplete; }
    double time() const override { return mCurrent.time; }
    uint64_t iteration() const override { return mSim ? mSim->getIteration() : 0; }
    uint64_t substeps() const override { return mSim ? mSim->getAdvectionSubstepCount() : 0; }
    size_t springlCount() const override { return mCurrent.springlCount; }
    bool supportsCsv() const override { return true; }
    void csvHeader(std::ostream &out) const override;
    void csvRow(std::ostream &out, double stepSeconds) const override;
    bool hasLabelsCsv() const override { return mSim && mSim->isLabeled(); }
    void labelsCsvHeader(std::ostream &out) const override;
    void labelsCsvRows(std::ostream &out) const override;
    void summaryHeader(std::ostream &out, double period) const override;
    void summary(std::ostream &out, double seconds) const override;
    // Checkpoints: one <key>_g<grid>_d<depth>_t<t>.state file (the 2D state holds its level sets).
    bool supportsCheckpoints() const override { return true; }
    bool saveCheckpoint(const std::filesystem::path &dir, double t, std::string &error) const override;
    bool resume(const std::filesystem::path &stateFile, std::string &error) override;

private:
    const Entry &mEntry;
    RunOptions mOptions;
    springls::EnrightConfig2D mConfig;
    std::unique_ptr<springls::SpringLevelSet2D> mSim;
    springls::EnrightMetrics2D mInitial, mCurrent;
    bool mComplete = false;
};
}  // namespace toybox
