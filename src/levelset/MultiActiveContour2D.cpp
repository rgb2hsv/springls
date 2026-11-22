#include <levelset/MultiActiveContour2D.hpp>

#include <levelset/Contour2D.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

namespace springls
{
MultiActiveContour2D::MultiActiveContour2D()
{
    mName = "MOGAC";
    mLabeled = true;
}

bool MultiActiveContour2D::init()
{
    if (mInitialLabels.empty()) return false;
    mSimulationTime = 0.0;
    mIteration = 0;
    mAdvectionSubstepCount = 0;
    clearSpringls();
    mMultiBand.initialize(mInitialLabels, mInitialLabelDistance);
    mObjectLabels = ObjectLabels(mInitialLabels);
    extractContour();
    return true;
}

void MultiActiveContour2D::extractContour()
{
    const MultiNarrowBand2D &band = mMultiBand;
    ExtractMultiLabelContour([&](int i, int j, int l) { return band.signedValue(i, j, l); }, band.labels().width(), band.labels().height(), mObjectLabels,
                             mManifold.vertexLocations, mManifold.vertexIndexes, mManifold.curveLabels);
}

void MultiActiveContour2D::evolve(double t, float dt)
{
    const std::vector<Vec2i> &active = mMultiBand.active();
    const auto &candidates = mMultiBand.candidates();
    auto &delta = mMultiBand.delta();
    const int N = static_cast<int>(active.size());
    SPRINGLS_OMP(omp parallel for if (N >= kParallelMinHeavyItems))
    for (int n = 0; n < N; n++) {
        const Vec2i p = active[n];
        const Vec2 v = particleVelocity(Vec2(static_cast<float>(p.x()), static_cast<float>(p.y())), t);
        for (int c = 0; c < MultiNarrowBand2D::kCandidates; c++) {
            const int label = candidates[n][c];
            if (label == MultiNarrowBand2D::kNoLabel) continue;
            const float weight = (label == kBackgroundLabel) ? backgroundCurvatureWeight : mCurvatureWeight;
            const float speed = -mMultiBand.previousUpwindAdvection(p.x(), p.y(), label, v) + mMultiBand.previousCurvatureTerm(p.x(), p.y(), label, weight);
            delta[n][c] = speed * dt;
        }
    }
    // delta already holds cells moved this sub-step; the band clamps it to one cell.
    mMultiBand.update(1.0f);
}

bool MultiActiveContour2D::step()
{
    for (int iter = 0; iter < tuning.advectionSubstepsPerStep; iter++) {
        double remaining = std::numeric_limits<double>::infinity();
        if (mSchedule.weight) {
            remaining = mSchedule.endTime - mSimulationTime;
            if (remaining <= 0.0) break;
            mAdvectionWeight = mSchedule.weight(mSimulationTime);
        }
        // Same time step rule as springl advection: the fastest band pixel moves maxStepPixels.
        const auto bandSpeed = [this](double t) {
            float speed = 0.0f;
            for (const Vec2i &p : mMultiBand.active()) {
                speed = std::max(speed, particleVelocity(Vec2(static_cast<float>(p.x()), static_cast<float>(p.y())), t).squaredNorm());
            }
            return std::max(std::sqrt(speed), 1.0f);  // speeds below a pixel per unit time count as one
        };
        const float maxSpeed = bandSpeed(mSimulationTime);
        double dt = tuning.maxStepPixels / maxSpeed;
        if (mSchedule.weight) {
            dt = std::min({dt, mSchedule.maxTimeStep, remaining});
            dt = limitStepOverInterval(dt, tuning.maxStepPixels, maxSpeed, mSimulationTime, bandSpeed);
        }
        evolve(mSimulationTime, static_cast<float>(dt));
        mAdvectionSubstepCount++;
        // Land exactly on endTime instead of accumulating float rounding.
        mSimulationTime = (mSchedule.weight && dt >= remaining) ? mSchedule.endTime : mSimulationTime + dt;
    }
    extractContour();
    mIteration++;
    if (verbose && (mIteration % kLogInterval) == 0) {
        std::cerr << mName << ": step " << mIteration << " t = " << mSimulationTime << ", band " << mMultiBand.active().size() << std::endl;
    }
    return mIteration < mMaxIterations && !isScheduleComplete();
}
}  // namespace springls
