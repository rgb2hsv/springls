#include <levelset/LevelSetMethod2D.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

namespace springls
{
namespace
{
// AdaptiveGrid2D::evolve moves the zero layer by kEvolveTimeStep * clamp(v . grad, -1, 1) cells per
// iteration: velocities are scaled by 1 / kEvolveTimeStep to move exactly v dt.
constexpr float kEvolveTimeStep = 0.5f;
using Block = AdaptiveGrid2D::Block;
}  // namespace

// ---------------------------------------------------------------------------------------------
// Uniform grid
// ---------------------------------------------------------------------------------------------

LevelSetMethod2D::LevelSetMethod2D() { mName = "Level Set"; }

bool LevelSetMethod2D::init()
{
    if (mInitialLevelSet.empty()) return false;
    mSimulationTime = 0.0;
    mIteration = 0;
    mAdvectionSubstepCount = 0;
    clearSpringls();
    mBand.initialize(mInitialLevelSet);
    solveUniformContour();
    return true;
}

bool LevelSetMethod2D::step()
{
    // The fastest contour vertex moves at most maxStepPixels per sub-step (the zero layer moves at
    // most a pixel per update).
    advanceWithFlow(
        tuning.maxStepPixels, [this](double t) { return maxContourSpeed(mManifold.vertexLocations, t); },
        [this](double t, float dt) {
            mBand.evolve([&](int i, int j) { return dt * particleVelocity(Vec2(static_cast<float>(i), static_cast<float>(j)), t); }, dt * mCurvatureWeight,
                         1.0f);
        });
    solveUniformContour();
    mIteration++;
    if (verbose && (mIteration % kLogInterval) == 0) {
        std::cerr << mName << ": step " << mIteration << " t = " << mSimulationTime << ", band " << mBand.active().size() << std::endl;
    }
    return mIteration < mMaxIterations && !isScheduleComplete();
}

// ---------------------------------------------------------------------------------------------
// Adaptive grid
// ---------------------------------------------------------------------------------------------

AdaptiveLevelSetMethod2D::AdaptiveLevelSetMethod2D()
{
    mName = "Adaptive Level Set";
    mSplitOnRefine = false;  // no springls to split
}

bool AdaptiveLevelSetMethod2D::init()
{
    if (mInitialLevelSet.empty()) return false;
    mSimulationTime = 0.0;
    mIteration = 0;
    mAdvectionSubstepCount = 0;
    clearSpringls();
    initLevelSet();
    return true;
}

void AdaptiveLevelSetMethod2D::evolve(double t, float dt)
{
    mGrid.updateActive();
    const auto &blocks = mGrid.blocks();
    const auto &active = mGrid.activeLeaves();
    std::vector<AdaptiveGrid2D::NodeVelocities> velocities(active.size());
    const int A = static_cast<int>(active.size());
    SPRINGLS_OMP(omp parallel for if (A >= kParallelMinBlocks))
    for (int n = 0; n < A; n++) {
        const Block &block = blocks[active[n]];
        // Cells the interface at this corner moves in the sub-step, per pseudo-time step of evolve.
        const float scale = dt / (block.h * kEvolveTimeStep);
        for (int b = 0; b < AdaptiveGrid2D::kBlockNodes; b++) {
            for (int a = 0; a < AdaptiveGrid2D::kBlockNodes; a++) velocities[n][Block::index(a, b)] = scale * particleVelocity(block.nodePosition(a, b), t);
        }
    }
    mGrid.evolve(1, mCurvatureWeight, velocities);
}

bool AdaptiveLevelSetMethod2D::step()
{
    // The fastest contour vertex moves at most maxStepPixels finest cells per sub-step: within the
    // half cell evolve allows per iteration, and within the band margin.
    const float maxStep = tuning.maxStepPixels * std::ldexp(1.0f, -adaptiveTuning.maxDepth);
    advanceWithFlow(
        maxStep, [this](double t) { return maxContourSpeed(mManifold.vertexLocations, t); },
        [this](double t, float dt) {
            // The band moves with the interface before it moves.
            adaptToPoints(mManifold.vertexLocations);
            evolve(t, dt);
            extractGridContour();
        });
    mIteration++;
    if (verbose && (mIteration % kLogInterval) == 0) {
        std::cerr << mName << ": step " << mIteration << " t = " << mSimulationTime << ", " << mGrid.leaves().size() << " leaves" << std::endl;
    }
    return mIteration < mMaxIterations && !isScheduleComplete();
}
}  // namespace springls
