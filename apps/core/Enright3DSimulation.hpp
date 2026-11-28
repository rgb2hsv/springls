#pragma once
// A 3D method running the Enright test (today's enright3d::Run), with checkpoints.
#include "Simulation.hpp"

#include <lod/MeshCheck3D.hpp>
#include <lod/TriangleSurface3D.hpp>
#include <scenarios/EnrightTest3D.hpp>
#include <spatial/AdaptiveGrid3D.hpp>

#include <memory>
#include <vector>

namespace toybox
{
class Experiment;

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
    // Mesh tracking: the tracked mesh (base voxels; null for the other methods).
    std::shared_ptr<const springls::TriangleSurface> trackedMesh;
    // ... and its fold-over / overlap check, per triangle of trackedMesh (null unless the check is on).
    std::shared_ptr<const springls::MeshCheck> trackedCheck;
    springls::EnrightMetrics3D metrics;
    uint64_t iteration = 0;
    uint64_t substeps = 0;  // advection sub-steps so far
    bool done = false;      // set by the experiment
    double seconds = 0.0;   // set by the experiment
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
// Snapshot of every run of a 3D Enright experiment (for the window's worker thread).
ExperimentSnapshot Snapshot3D(const Experiment &experiment);

class Enright3DSimulation : public Simulation
{
public:
    Enright3DSimulation(const Entry &entry, const RunOptions &options);

    springls::Method3D method() const { return mEntry.method3D; }
    springls::ElementType elementType() const { return mOptions.elementType; }
    const springls::EnrightConfig3D &config() const { return mConfig; }
    const springls::SpringLevelSet3D &sim() const { return *mSim; }
    springls::SpringLevelSet3D &sim() { return *mSim; }
    const springls::EnrightMetrics3D &initial() const { return mInitial; }
    const springls::EnrightMetrics3D &current() const { return mCurrent; }
    RunSnapshot snapshot() const;

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
    void summaryHeader(std::ostream &out, double period) const override;
    void summary(std::ostream &out, double seconds) const override;
    void progress(std::ostream &out, bool withTime) const override;
    // Not for mesh tracking: its mesh is not part of the checkpoint state.
    bool supportsCheckpoints() const override { return method() != springls::Method3D::MeshTracking; }
    bool saveCheckpoint(const std::filesystem::path &dir, double t, std::string &error) const override;
    bool resume(const std::filesystem::path &stateFile, std::string &error) override;

private:
    const Entry &mEntry;
    RunOptions mOptions;
    springls::EnrightConfig3D mConfig;
    std::unique_ptr<springls::SpringLevelSet3D> mSim;
    springls::EnrightMetrics3D mInitial, mCurrent;
    bool mComplete = false;
};
}  // namespace toybox
