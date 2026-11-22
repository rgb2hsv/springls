#include <levelset/LevelSetMethod3D.hpp>

#include <levelset/IsoSurface3D.hpp>

#include <cmath>
#include <iostream>

namespace springls
{
namespace
{
// AdaptiveGrid3D::evolve moves the zero layer by kEvolveTimeStep * clamp(v . grad, -1, 1) cells per
// iteration: velocities are scaled by 1 / kEvolveTimeStep to move exactly v dt.
constexpr float kEvolveTimeStep = 0.5f;
}  // namespace

// ---------------------------------------------------------------------------------------------
// Uniform grid
// ---------------------------------------------------------------------------------------------

LevelSetMethod3D::LevelSetMethod3D(ElementType elementType) : SpringLevelSet3D(elementType) { mName = "Level Set 3D"; }

bool LevelSetMethod3D::init()
{
    if (mInitialLevelSet.empty()) return false;
    mSimulationTime = 0.0;
    mIteration = 0;
    mAdvectionSubstepCount = 0;
    mUnsignedLevelSet.clear();
    clearSpringls();
    initLevelSet();
    return true;
}

void LevelSetMethod3D::extractSurface()
{
    // Not refined toward springls: there are none.
    ExtractIsoSurface(mBand.phi(), mBand.active(), elementType(), mManifold.isoSurface);
    orientVertexNormals(mManifold.isoSurface);
}

bool LevelSetMethod3D::step()
{
    // The fastest iso-surface vertex moves at most maxStepVoxels per sub-step (the zero layer moves
    // at most a voxel per update).
    advanceWithFlow(
        tuning.maxStepVoxels, [this](double t) { return maxContourSpeed(mManifold.isoSurface.vertexes, t); },
        [this](double t, float dt) {
            mBand.evolve(
                [&](int i, int j, int k) { return dt * particleVelocity(Vec3(static_cast<float>(i), static_cast<float>(j), static_cast<float>(k)), t); },
                dt * mCurvatureWeight, 1.0f);
        });
    extractSurface();
    mIteration++;
    if (verbose && (mIteration % kLogInterval) == 0) {
        std::cerr << mName << ": step " << mIteration << " t = " << mSimulationTime << ", band " << mBand.active().size() << std::endl;
    }
    return mIteration < mMaxIterations && !isScheduleComplete();
}

// ---------------------------------------------------------------------------------------------
// Adaptive grid
// ---------------------------------------------------------------------------------------------

AdaptiveLevelSetMethod3D::AdaptiveLevelSetMethod3D(ElementType elementType) : AdaptiveSpringLevelSet3D(elementType)
{
    mName = "Adaptive Level Set 3D";
    mSplitOnRefine = false;  // no springls to split
}

bool AdaptiveLevelSetMethod3D::init()
{
    if (mInitialLevelSet.empty()) return false;
    mSimulationTime = 0.0;
    mIteration = 0;
    mAdvectionSubstepCount = 0;
    mUnsignedLevelSet.clear();
    clearSpringls();
    initLevelSet();
    return true;
}

bool AdaptiveLevelSetMethod3D::step()
{
    // The fastest iso-surface vertex moves at most maxStepVoxels finest cells per sub-step: within
    // the half cell evolve allows per iteration, and within the band margin.
    const float maxStep = tuning.maxStepVoxels * std::ldexp(1.0f, -adaptiveTuning.maxDepth);
    advanceWithFlow(
        maxStep, [this](double t) { return maxContourSpeed(mManifold.isoSurface.vertexes, t); },
        [this](double t, float dt) {
            // The band moves with the interface before it moves.
            adaptToPoints(mManifold.isoSurface.vertexes);
            mGrid.updateActive();
            mGrid.evolve(1, mCurvatureWeight, [&](const Vec3 &q, float h) -> Vec3 { return dt / (h * kEvolveTimeStep) * particleVelocity(q, t); });
            extractGridSurface();
        });
    mIteration++;
    if (verbose && (mIteration % kLogInterval) == 0) {
        std::cerr << mName << ": step " << mIteration << " t = " << mSimulationTime << ", " << mGrid.leaves().size() << " leaves" << std::endl;
    }
    return mIteration < mMaxIterations && !isScheduleComplete();
}
}  // namespace springls
