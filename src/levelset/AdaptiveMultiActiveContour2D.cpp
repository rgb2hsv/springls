#include <levelset/AdaptiveMultiActiveContour2D.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

namespace springls
{
namespace
{
// AdaptiveGrid2D::evolveLabeled moves the zero layer by kEvolveTimeStep * clamp(v . grad, -1, 1)
// cells per iteration: velocities are scaled by 1 / kEvolveTimeStep to move exactly v dt.
constexpr float kEvolveTimeStep = 0.5f;
using Block = AdaptiveGrid2D::Block;
}  // namespace

AdaptiveMultiActiveContour2D::AdaptiveMultiActiveContour2D() { mName = "Adaptive MOGAC"; }

bool AdaptiveMultiActiveContour2D::init()
{
    if (mInitialLabels.empty()) return false;
    mSimulationTime = 0.0;
    mIteration = 0;
    mAdvectionSubstepCount = 0;
    clearSpringls();
    initLevelSet();
    return true;
}

float AdaptiveMultiActiveContour2D::maxBandSpeed(double t)
{
    mGrid.updateActive();
    float maxSpeed = 0.0f;
    for (int index : mGrid.activeLeaves()) {
        const Block &block = mGrid.blocks()[index];
        for (int b = 0; b < AdaptiveGrid2D::kBlockNodes; b++) {
            for (int a = 0; a < AdaptiveGrid2D::kBlockNodes; a++) maxSpeed = std::max(maxSpeed, particleVelocity(block.nodePosition(a, b), t).squaredNorm());
        }
    }
    return std::sqrt(maxSpeed);
}

void AdaptiveMultiActiveContour2D::evolve(double t, float dt)
{
    mGrid.updateActive();
    const auto &blocks = mGrid.blocks();
    const auto &active = mGrid.activeLeaves();
    std::vector<AdaptiveGrid2D::NodeVelocities> velocities(active.size());
    const int A = static_cast<int>(active.size());
    SPRINGLS_OMP(omp parallel for if (A >= kParallelMinBlocks))
    for (int n = 0; n < A; n++) {
        const Block &block = blocks[active[n]];
        // Cells this corner's boundary moves in the sub-step, per pseudo-time step of evolveLabeled.
        const float scale = dt / (block.h * kEvolveTimeStep);
        for (int b = 0; b < AdaptiveGrid2D::kBlockNodes; b++) {
            for (int a = 0; a < AdaptiveGrid2D::kBlockNodes; a++) {
                velocities[n][Block::index(a, b)] = scale * particleVelocity(block.nodePosition(a, b), t);
            }
        }
    }
    mGrid.evolveLabeled(1, mCurvatureWeight, backgroundCurvatureWeight, velocities, true);
}

bool AdaptiveMultiActiveContour2D::step()
{
    const float finestCell = std::ldexp(1.0f, -adaptiveTuning.maxDepth);
    for (int iter = 0; iter < tuning.advectionSubstepsPerStep; iter++) {
        double remaining = std::numeric_limits<double>::infinity();
        if (mSchedule.weight) {
            remaining = mSchedule.endTime - mSimulationTime;
            if (remaining <= 0.0) break;
            mAdvectionWeight = mSchedule.weight(mSimulationTime);
        }
        // The band moves with the boundaries before they move.
        adaptToPoints(mManifold.vertexLocations);
        // The fastest boundary corner moves at most maxStepPixels finest cells: within the half
        // cell evolveLabeled allows per iteration, and within the band margin.
        const float maxSpeed = maxBandSpeed(mSimulationTime);
        const float maxStep = tuning.maxStepPixels * finestCell;
        double dt = (maxSpeed > 1E-6f) ? maxStep / maxSpeed : 1.0;
        if (mSchedule.weight) {
            dt = std::min({dt, mSchedule.maxTimeStep, remaining});
            dt = limitStepOverInterval(dt, maxStep, maxSpeed, mSimulationTime, [this](double t) { return maxBandSpeed(t); });
        }
        evolve(mSimulationTime, static_cast<float>(dt));
        mAdvectionSubstepCount++;
        mSimulationTime = (mSchedule.weight && dt >= remaining) ? mSchedule.endTime : mSimulationTime + dt;
        extractGridContour();
    }
    mIteration++;
    if (verbose && (mIteration % kLogInterval) == 0) {
        std::cerr << mName << ": step " << mIteration << " t = " << mSimulationTime << ", " << mGrid.leaves().size() << " leaves" << std::endl;
    }
    return mIteration < mMaxIterations && !isScheduleComplete();
}
}  // namespace springls
