#include <springl/SpringLevelSet2D.hpp>

#include <levelset/Contour2D.hpp>
#include <spatial/PointGrid2D.hpp>
#include <spatial/SegmentTree2D.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>

namespace springls
{
namespace
{
// Normals within this angle of each other belong to the same side of the interface.
const float kPlaneThreshold = std::cos(80.0f * kPi / 180.0f);
// Same speed clamp as the original: the pull is at most 1.5 cells per pseudo-time step.
constexpr float kMaxPullCells = 1.5f;
// Velocity field weights below this are treated as zero.
constexpr float kWeightEpsilon = 1E-7f;
}  // namespace

SpringLevelSet2D::SpringLevelSet2D() = default;

// ---------------------------------------------------------------------------------------------
// Springl bookkeeping
// ---------------------------------------------------------------------------------------------

size_t SpringLevelSet2D::appendSpringl(const Vec2 &particle, const Vec2 &v0, const Vec2 &v1, float scale, int label)
{
    const size_t index = mManifold.springlParticles.size();
    if (mScaledSpringls) mSpringlScales.push_back(scale);
    if (mLabeled) mManifold.springlLabels.push_back(label);
    mManifold.springlParticles.push_back(particle);
    mManifold.springlVertexes.push_back(v0);
    mManifold.springlVertexes.push_back(v1);
    mManifold.velocities.push_back(Vec2::Zero());
    if (mTrackingEnabled) {
        mManifold.springlParticleTracking.push_back(-1);
        mManifold.correspondence.push_back(Vec2::Constant(std::numeric_limits<float>::infinity()));
    }
    return index;
}

void SpringLevelSet2D::clearSpringls()
{
    mManifold.springlVertexes.clear();
    mManifold.springlParticles.clear();
    mManifold.springlParticleNormals.clear();
    mManifold.springlParticleTracking.clear();
    mManifold.correspondence.clear();
    mManifold.velocities.clear();
    mManifold.springlLabels.clear();
    mSpringlScales.clear();
    mNearestNeighbor.clear();
    mSpringlGrid.invalidate();
}

void SpringLevelSet2D::compactSpringls(const std::vector<uint8_t> &keep)
{
    Manifold2D &m = mManifold;
    const size_t N = m.springlParticles.size();
    assert(keep.size() == N);
    const bool hasNormals = m.springlParticleNormals.size() == N;
    const bool hasTracking = mTrackingEnabled && m.correspondence.size() == N && m.springlParticleTracking.size() == N;
    const bool hasScales = mSpringlScales.size() == N;
    const bool hasLabels = m.springlLabels.size() == N;
    mSpringlGrid.invalidate();  // springl indexes are about to change
    size_t out = 0;
    for (size_t i = 0; i < N; i++) {
        if (!keep[i]) continue;
        m.springlParticles[out] = m.springlParticles[i];
        m.springlVertexes[2 * out] = m.springlVertexes[2 * i];
        m.springlVertexes[2 * out + 1] = m.springlVertexes[2 * i + 1];
        m.velocities[out] = m.velocities[i];
        if (hasNormals) m.springlParticleNormals[out] = m.springlParticleNormals[i];
        if (hasTracking) {
            m.correspondence[out] = m.correspondence[i];
            m.springlParticleTracking[out] = m.springlParticleTracking[i];
        }
        if (hasScales) mSpringlScales[out] = mSpringlScales[i];
        if (hasLabels) m.springlLabels[out] = m.springlLabels[i];
        out++;
    }
    m.springlParticles.resize(out);
    m.springlVertexes.resize(2 * out);
    m.velocities.resize(out);
    if (hasNormals) m.springlParticleNormals.resize(out);
    if (hasTracking) {
        m.correspondence.resize(out);
        m.springlParticleTracking.resize(out);
    }
    if (hasScales) mSpringlScales.resize(out);
    if (hasLabels) m.springlLabels.resize(out);
}

void SpringLevelSet2D::seedSpringls()
{
    for (size_t c = 0; c < mManifold.vertexIndexes.size(); c++) {
        const Curve &curve = mManifold.vertexIndexes[c];
        for (size_t k = 0; k + 1 < curve.size(); k++) {
            const Vec2 v0 = mManifold.vertexLocations[curve[k]];
            const Vec2 v1 = mManifold.vertexLocations[curve[k + 1]];
            appendSpringl(0.5f * (v0 + v1), v0, v1, std::min(contourScale(curve[k]), contourScale(curve[k + 1])), curveLabel(c));
        }
    }
}

void SpringLevelSet2D::resetTrackingIdentity()
{
    if (!mTrackingEnabled) return;
    mManifold.springlParticleTracking.resize(mManifold.springlParticles.size());
    for (size_t n = 0; n < mManifold.springlParticleTracking.size(); n++) {
        mManifold.springlParticleTracking[n] = static_cast<int>(n);
    }
}

// ---------------------------------------------------------------------------------------------
// Initialization and the time step
// ---------------------------------------------------------------------------------------------

void SpringLevelSet2D::initLevelSet()
{
    mBand.initialize(mInitialLevelSet);
    solveUniformContour();
    refineInitialContour();
}

bool SpringLevelSet2D::init()
{
    if (mLabeled ? mInitialLabels.empty() : mInitialLevelSet.empty()) return false;
    mSimulationTime = 0.0;
    mIteration = 0;
    mAdvectionSubstepCount = 0;
    initLevelSet();
    clearSpringls();
    seedSpringls();
    mManifold.velocities.assign(mManifold.springlParticles.size(), Vec2::Zero());
    if (mTrackingEnabled) {
        mManifold.correspondence = mManifold.springlParticles;
        resetTrackingIdentity();
    }
    mManifold.updateNormals();
    relax();
    updateNearestNeighbors();
    afterResample();
    return true;
}

bool SpringLevelSet2D::step()
{
    const bool logStep = verbose && (mIteration % kLogInterval) == 0;
    resetTrackingIdentity();
    int contractTotal = 0, fillTotal = 0;
    for (int iter = 0; iter < tuning.advectionSubstepsPerStep; iter++) {
        if (mSchedule.weight) {
            const double remaining = mSchedule.endTime - mSimulationTime;
            if (remaining <= 0.0) break;
            mAdvectionWeight = mSchedule.weight(mSimulationTime);
            const float timeStep = advect(advectionMaxStep(), static_cast<float>(std::min(remaining, mSchedule.maxTimeStep)));
            // Land exactly on endTime instead of accumulating float rounding.
            mSimulationTime = (timeStep >= remaining) ? mSchedule.endTime : mSimulationTime + timeStep;
        } else {
            mSimulationTime += advect(advectionMaxStep());
        }
        // Relaxation pulls neighboring arm vertexes together, which can open gaps elsewhere; a level
        // set rebuilt from the springls (rather than evolved toward them) is rebuilt first, from the
        // springls exactly as the flow left them.
        const bool rebuildFirst = rebuildsLevelSetBeforeRelax();
        if (mResampleEnabled && !rebuildFirst) relaxWithRememberedNeighbors();
        evolveLevelSetAndContour();
        if (mResampleEnabled && rebuildFirst) relaxWithRememberedNeighbors();
        if (mResampleEnabled) {
            snapshotForTracking();
            resetTrackingIdentity();
            contractTotal += contract();
            updateNearestNeighbors();
            // Keep filling until every gap is closed.
            constexpr int kMaxFillPasses = 16;
            int filled = 0, passes = 0;
            do {
                passes++;
                filled = fill();
                relax();
                fillTotal += filled;
            } while (filled > 0 && passes < kMaxFillPasses);
            mManifold.updateNormals();
            afterResample();
            if (mTrackingEnabled) updateTracking(3.0f * tuning.nearestNeighborDistance);
        } else {
            solveContourForFill();
            mManifold.updateNormals();
        }
    }
    mIteration++;
    if (logStep) {
        std::cerr << mName << ": step " << mIteration << " t = " << mSimulationTime << ", springls " << mManifold.springlCount() << ", contracted "
                  << contractTotal << ", filled " << fillTotal << std::endl;
    }
    return mIteration < mMaxIterations && !isScheduleComplete();
}

// ---------------------------------------------------------------------------------------------
// Advection and relaxation
// ---------------------------------------------------------------------------------------------

double SpringLevelSet2D::limitStepOverInterval(double dt, float maxStep, float startSpeed, double t0, const std::function<float(double)> &maxSpeed)
{
    // The flow changes over the step (the schedule weight), so the speed at its start does not bound
    // the displacement: near a reversal the start speed is about zero and the step would jump across
    // it. Shrink the step until the fastest speed sampled over it (start, middle, end; the extremes
    // of a monotone weight) moves at most maxStep.
    constexpr int kMaxPasses = 8;
    for (int pass = 0; pass < kMaxPasses; pass++) {
        const float speed = std::max({startSpeed, maxSpeed(t0 + 0.5 * dt), maxSpeed(t0 + dt)});
        if (speed * dt <= maxStep * (1.0 + 1E-6)) break;
        dt = maxStep / speed;
    }
    return dt;
}

void SpringLevelSet2D::advanceWithFlow(float maxStep, const std::function<float(double)> &maxSpeed, const std::function<void(double, float)> &substep)
{
    for (int iter = 0; iter < tuning.advectionSubstepsPerStep; iter++) {
        double remaining = std::numeric_limits<double>::infinity();
        if (mSchedule.weight) {
            remaining = mSchedule.endTime - mSimulationTime;
            if (remaining <= 0.0) break;
            mAdvectionWeight = mSchedule.weight(mSimulationTime);
        }
        const float speed = maxSpeed(mSimulationTime);
        double dt = (speed > 1E-6f) ? maxStep / speed : std::numeric_limits<double>::infinity();
        if (mSchedule.weight) dt = std::min({dt, mSchedule.maxTimeStep, remaining});
        if (!std::isfinite(dt)) dt = maxStep;  // no flow and no schedule: unit speed
        if (mSchedule.weight) dt = limitStepOverInterval(dt, maxStep, speed, mSimulationTime, maxSpeed);
        substep(mSimulationTime, static_cast<float>(dt));
        mAdvectionSubstepCount++;
        // Land exactly on endTime instead of accumulating float rounding.
        mSimulationTime = (mSchedule.weight && dt >= remaining) ? mSchedule.endTime : mSimulationTime + dt;
    }
}

float SpringLevelSet2D::maxContourSpeed(const Points2 &points, double t) const
{
    float speed = 0.0f;
    for (const Vec2 &p : points) speed = std::max(speed, particleVelocity(p, t).squaredNorm());
    return std::sqrt(speed);
}

Vec2 SpringLevelSet2D::particleVelocity(const Vec2 &position, double time) const
{
    // The schedule weight is evaluated at the stage time, so RK4 sees the time-varying flow.
    const float weight = mSchedule.weight ? mSchedule.weight(time) : mAdvectionWeight;
    if (std::abs(weight) < kWeightEpsilon) return Vec2::Zero();
    if (mVelocityFunction) return mVelocityFunction(position) * weight;
    if (mVelocityField.empty()) return Vec2::Zero();
    return mVelocityField.interpolate(position) * weight;
}

float SpringLevelSet2D::advectionTimeStep(const Points2 &k1, float maxStep) const
{
    float maxSpeed = 0.0f;
    for (const Vec2 &v : k1) maxSpeed = std::max(maxSpeed, v.squaredNorm());
    maxSpeed = std::sqrt(maxSpeed);
    const float dt = (maxSpeed > 1.0f) ? maxStep / maxSpeed : maxStep;
    if (!mSchedule.weight) return dt;
    // Speeds below a pixel per unit time count as one, as above.
    const Points2 &particles = mManifold.springlParticles;
    const auto speedAt = [&](double t) { return std::max(maxContourSpeed(particles, t), 1.0f); };
    return static_cast<float>(limitStepOverInterval(dt, maxStep, std::max(maxSpeed, 1.0f), mSimulationTime, speedAt));
}

float SpringLevelSet2D::advect(float maxStep, float maxTimeStep)
{
    Manifold2D &m = mManifold;
    const int N = static_cast<int>(m.springlParticles.size());
    const double t0 = mSimulationTime;
    // Stage 1 at the current particle positions; its magnitude picks the time step.
    Points2 k1(N);
    SPRINGLS_OMP(omp parallel for if (N >= kParallelMinLightItems))
    for (int i = 0; i < N; i++) k1[i] = particleVelocity(m.springlParticles[i], t0);
    mAdvectionSubstepCount++;
    const float timeStep = std::min(advectionTimeStep(k1, maxStep), maxTimeStep);
    const double tHalf = t0 + 0.5 * timeStep;
    const double tEnd = t0 + timeStep;
    // Displacement of a point over the step (classic RK4, or forward Euler).
    const auto displace = [&](const Vec2 &x, const Vec2 &v1) -> Vec2 {
        if (temporalScheme != TemporalScheme::RK4) return timeStep * v1;
        const Vec2 v2 = particleVelocity(x + (0.5f * timeStep) * v1, tHalf);
        const Vec2 v3 = particleVelocity(x + (0.5f * timeStep) * v2, tHalf);
        const Vec2 v4 = particleVelocity(x + timeStep * v3, tEnd);
        return (timeStep / 6.0f) * (v1 + 2.0f * v2 + 2.0f * v3 + v4);
    };
    // 1. Particles and vertexes are all advected by the flow.
    const Points2 oldParticles = m.springlParticles;
    const Points2 oldVertexes = m.springlVertexes;
    SPRINGLS_OMP(omp parallel for if (N >= kParallelMinHeavyItems))
    for (int i = 0; i < N; i++) {
        const Vec2 displacement = displace(oldParticles[i], k1[i]);
        m.springlParticles[i] = oldParticles[i] + displacement;
        for (int e = 0; e < 2; e++) {
            const Vec2 v = oldVertexes[2 * i + e];
            m.springlVertexes[2 * i + e] = v + displace(v, particleVelocity(v, t0));
        }
        m.velocities[i] = (timeStep > 0.0f) ? Vec2(displacement / timeStep) : k1[i];
    }
    // 2. Neighbors are found on the flow-advected vertexes and remembered for relaxation.
    updateNearestNeighbors();
    // 3. Each springl becomes the similarity transform (rotation + uniform scale about its
    //    advected particle) of its previous shape that best fits its advected vertexes, so it
    //    stays straight and its particle does not move.
    SPRINGLS_OMP(omp parallel for if (N >= kParallelMinLightItems))
    for (int i = 0; i < N; i++) {
        const Vec2 p = m.springlParticles[i];
        float re = 0.0f, im = 0.0f, norm = 0.0f;
        Vec2 arms[2];
        for (int e = 0; e < 2; e++) {
            arms[e] = oldVertexes[2 * i + e] - oldParticles[i];
            const Vec2 w = m.springlVertexes[2 * i + e] - p;
            re += arms[e].dot(w);  // conj(arm) * w, as complex numbers
            im += cross2(arms[e], w);
            norm += arms[e].squaredNorm();
        }
        constexpr float kMinFit = 1E-12f;
        if (norm < kMinFit || re * re + im * im < kMinFit * norm * norm) {
            // Degenerate springl: keep its shape and translate it with the particle.
            for (int e = 0; e < 2; e++) m.springlVertexes[2 * i + e] = p + arms[e];
            continue;
        }
        const float zr = re / norm, zi = im / norm;
        for (int e = 0; e < 2; e++) {
            const Vec2 u = arms[e];
            m.springlVertexes[2 * i + e] = p + Vec2(zr * u.x() - zi * u.y(), zr * u.y() + zi * u.x());
        }
    }
    m.updateNormals();
    return timeStep;
}

void SpringLevelSet2D::updateNearestNeighbors(float maxDistance)
{
    if (maxDistance < 0.0f) maxDistance = tuning.nearestNeighborDistance;
    refreshSpringlGrid();
    const Points2 &vertexes = mManifold.springlVertexes;
    const int N = static_cast<int>(vertexes.size());
    mNearestNeighbor.assign(N, -1);
    SPRINGLS_OMP(omp parallel for if (N >= kParallelMinHeavyItems))
    for (int i = 0; i < N; i++) {
        const size_t springl = static_cast<size_t>(i / 2);
        const Vec2 p = vertexes[i];
        // Nearest arm vertex of another springl this one may be attracted to.
        mNearestNeighbor[i] = mSpringlGrid.nearestVertex(vertexes, p, maxDistance * attractionScale(p, springl), springl,
                                                         [&](size_t other) { return canAttract(springl, other); });
    }
}

void SpringLevelSet2D::relax()
{
    updateNearestNeighbors();
    relaxWithRememberedNeighbors();
}

void SpringLevelSet2D::relaxWithRememberedNeighbors()
{
    if (mNearestNeighbor.size() != mManifold.springlVertexes.size()) updateNearestNeighbors();
    for (int i = 0; i < tuning.relaxIterations; i++) relax(tuning.relaxTimeStep);
}

void SpringLevelSet2D::relax(float timeStep)
{
    const int N = static_cast<int>(mManifold.springlParticles.size());
    Points2 updates(mManifold.springlVertexes.size());
    SPRINGLS_OMP(omp parallel for if (N >= kParallelMinLightItems))
    for (int i = 0; i < N; i++) relaxSpringl(i, timeStep, updates[2 * i], updates[2 * i + 1]);
    mManifold.springlVertexes = std::move(updates);
    mManifold.updateNormals();
}

void SpringLevelSet2D::relaxSpringl(size_t idx, float timeStep, Vec2 &f1, Vec2 &f2) const
{
    constexpr float kMaxForce = 0.999f;
    constexpr float kMinLength = 0.001f;
    const float scale = springlScale(idx);
    const float armRestLength = tuning.armRestLength * scale;
    const float maxArmLength = tuning.maxArmLength * scale;
    const Vec2 particle = mManifold.springlParticles[idx];
    Vec2 tangents[2], motion[2];
    float armLength[2], springForce[2];
    float moment = 0.0f;
    for (int i = 0; i < 2; i++) {
        const size_t eid = idx * 2 + i;
        const Vec2 vertex = mManifold.springlVertexes[eid];
        const Vec2 arm = vertex - particle;
        armLength[i] = arm.norm();
        tangents[i] = (armLength[i] > 1E-6f) ? Vec2(arm / armLength[i]) : arm;
        Vec2 velocity = Vec2::Zero();
        const int nbr = mNearestNeighbor[eid];
        if (nbr >= 0) {
            const Vec2 dir = mManifold.springlVertexes[nbr] - vertex;
            // Neighbors of a different size class (coarse/fine transitions) use the pair's mean size.
            const float pairScale = 0.5f * (scale + springlScale(static_cast<size_t>(nbr) / 2));
            const float w = std::atanh(
                kMaxForce *
                std::clamp((dir.norm() - tuning.neighborRestSeparation * pairScale) / ((kExtent + tuning.neighborRestSeparation) * pairScale), -1.0f, 1.0f));
            velocity += w * dir;
        }
        motion[i] = timeStep * tuning.sharpness * velocity;
        springForce[i] = timeStep * tuning.springConstant * (armRestLength - armLength[i]);
        moment += cross2(motion[i], tangents[i]);
    }
    const float cosa = std::cos(moment), sina = std::sin(moment);
    const auto rotate = [cosa, sina](const Vec2 &p) { return Vec2(p.x() * cosa + p.y() * sina, -p.x() * sina + p.y() * cosa); };
    const auto updateVertex = [&](int i) -> Vec2 {
        const float radial = std::clamp(armLength[i] + motion[i].dot(tangents[i]) + springForce[i], kMinLength, maxArmLength);
        return rotate(radial * tangents[i]) + particle;
    };
    f1 = updateVertex(0);
    f2 = updateVertex(1);
}

// ---------------------------------------------------------------------------------------------
// Level set
// ---------------------------------------------------------------------------------------------

void SpringLevelSet2D::updateUnsignedLevelSet(float maxDistance)
{
    if (maxDistance < 0.0f) maxDistance = tuning.unsignedMaxDistance;
    const Vec2i size = levelSetSize();
    mUnsignedLevelSet.resize(size.x(), size.y());
    SegmentTree2D tree;
    const size_t SN = mManifold.springlParticles.size();
    for (size_t n = 0; n < SN; n++) tree.append(mManifold.springlVertexes[2 * n], mManifold.springlVertexes[2 * n + 1], static_cast<int>(n));
    tree.finish();
    const int H = mUnsignedLevelSet.height(), W = mUnsignedLevelSet.width();
    SPRINGLS_OMP(omp parallel for if (W * H >= kParallelMinHeavyItems))
    for (int j = 0; j < H; j++) {
        for (int i = 0; i < W; i++) mUnsignedLevelSet(i, j) = tree.distance(Vec2(i, j), maxDistance);
    }
}

Vec2 SpringLevelSet2D::unsignedPull(int i, int j) const
{
    // -min(U, 1.5) * grad(U) / |grad(U)|: pulls the level set toward the springls.
    const ImageF &u = mUnsignedLevelSet;
    const Vec2 grad(0.5f * (u.at(i + 1, j) - u.at(i - 1, j)), 0.5f * (u.at(i, j + 1) - u.at(i, j - 1)));
    const float len = std::max(1E-6f, grad.norm());
    return -(std::min(u(i, j), kMaxPullCells) * grad / len);
}

Vec2 SpringLevelSet2D::signedGradientPull(float x, float y) const
{
    const ImageF &phi = mBand.phi();
    const auto a = [&](float px, float py) { return std::abs(phi.interpolate(px, py)); };
    const Vec2 grad(0.5f * (a(x + 1.0f, y) - a(x - 1.0f, y)), 0.5f * (a(x, y + 1.0f) - a(x, y - 1.0f)));
    const float len = std::max(1E-6f, grad.norm());
    return -(a(x, y) * grad / len);
}

void SpringLevelSet2D::updateSignedLevelSet(float maxStep)
{
    mBand.evolve([this](int i, int j) { return unsignedPull(i, j); }, mCurvatureWeight, maxStep);
}

void SpringLevelSet2D::solveUniformContour() { ExtractIsoContour(mBand.phi(), mManifold.vertexLocations, mManifold.vertexIndexes); }

int SpringLevelSet2D::correctSignsFromSpringls(const std::function<bool(size_t)> &include)
{
    const Manifold2D &m = mManifold;
    mManifold.updateNormals();
    SegmentTree2D springlTree;
    for (size_t n = 0; n < m.springlParticles.size(); n++) {
        if (!include || include(n)) springlTree.append(m.springlVertexes[2 * n], m.springlVertexes[2 * n + 1], static_cast<int>(n));
    }
    springlTree.finish();
    return mBand.correctSigns([&](int i, int j, float &d) {
        // Within a cell of a springl, and beside it rather than past one of its ends (where the
        // side is ambiguous): signed by the springl's outward normal.
        const Vec2 q(static_cast<float>(i), static_cast<float>(j));
        const auto hit = springlTree.nearest(q, 1.0f);
        if (!hit) return false;
        const size_t s = static_cast<size_t>(hit->id);
        constexpr float kEndTolerance = 1E-6f;
        if (distanceSqr(hit->point, m.springlVertexes[2 * s]) < kEndTolerance || distanceSqr(hit->point, m.springlVertexes[2 * s + 1]) < kEndTolerance) {
            return false;
        }
        const float side = m.springlParticleNormals[s].dot(q - hit->point);
        if (side == 0.0f) return false;
        d = (side < 0.0f) ? -std::sqrt(hit->distanceSqr) : std::sqrt(hit->distanceSqr);
        return true;
    });
}

void SpringLevelSet2D::evolveLevelSetAndContour()
{
    updateUnsignedLevelSet();
    for (int i = 0; i < tuning.signedEvolveIterations; i++) updateSignedLevelSet();
    if (tuning.springlCorrection) correctSignsFromSpringls();
    afterLevelSetUpdate();
    solveUniformContour();
    refineContour(tuning.refineIterations, tuning.refineProximity, tuning.refineStepSize);
    mManifold.updateNormals();
}

void SpringLevelSet2D::solveContourForFill()
{
    solveUniformContour();
    refineContour(tuning.refineIterations, tuning.refineProximity, tuning.refineStepSize);
}

// ---------------------------------------------------------------------------------------------
// Contour refinement
// ---------------------------------------------------------------------------------------------

void SpringLevelSet2D::refineInitialContour()
{
    // Moves iso-vertexes onto the zero crossing of the signed level set and evens out spacing.
    Points2 &verts = mManifold.vertexLocations;
    Points2 delta(verts.size(), Vec2::Zero());
    constexpr int kIterations = 10;
    for (int iter = 0; iter < kIterations; iter++) {
        for (const Curve &curve : mManifold.vertexIndexes) {
            for (size_t k = 1; k + 1 < curve.size(); k++) {
                const Vec2 cur = verts[curve[k]], next = verts[curve[k + 1]], prev = verts[curve[k - 1]];
                const Vec2 tan1 = next - cur, tan2 = cur - prev;
                const float d1 = tan1.norm(), d2 = tan2.norm();
                const Vec2 tan = normalizedOrZero(tan1 + tan2);
                const Vec2 norm(-tan.y(), tan.x());
                const float sum = d1 + d2;
                delta[curve[k]] =
                    0.1f * norm.dot(signedGradientPull(cur.x(), cur.y())) * norm + ((sum > 0.0f) ? Vec2(0.5f * tan * (d1 - d2) / sum) : Vec2::Zero());
            }
        }
        for (size_t v = 0; v < verts.size(); v++) verts[v] += delta[v];
    }
}

void SpringLevelSet2D::evenInitialContourSpacing()
{
    Points2 &verts = mManifold.vertexLocations;
    const Points2 original = verts;
    const auto pinned = [&](uint32_t v) { return v < mContourJunctions.size() && mContourJunctions[v] != 0; };
    for (const Curve &curve : mManifold.vertexIndexes) {
        const bool closed = curve.size() > 2 && curve.back() == curve.front();
        std::vector<uint32_t> loop(curve.begin(), closed ? curve.end() - 1 : curve.end());
        const size_t N = loop.size();
        if (N < 3) continue;
        // Span ends: junction vertexes, the ends of an open curve, and the first vertex of a closed
        // curve without junctions.
        std::vector<size_t> ends;
        for (size_t k = 0; k < N; k++) {
            if (pinned(loop[k]) || (!closed && (k == 0 || k + 1 == N))) ends.push_back(k);
        }
        if (ends.empty()) ends.push_back(0);
        const size_t spans = closed ? ends.size() : ends.size() - 1;
        for (size_t s = 0; s < spans; s++) {
            const size_t first = ends[s];
            size_t last = (s + 1 < ends.size()) ? ends[s + 1] : ends[0] + N;  // closed: wraps around
            if (last == first) last += N;                                     // a single end: the whole loop
            const size_t count = last - first;                                // segments in the span
            if (count < 2) continue;
            const auto at = [&](size_t k) { return original[loop[k % N]]; };
            std::vector<float> arc(count + 1, 0.0f);
            for (size_t k = 1; k <= count; k++) arc[k] = arc[k - 1] + distance(at(first + k - 1), at(first + k));
            const float total = arc[count];
            if (total <= 0.0f) continue;
            size_t segment = 0;
            for (size_t k = 1; k < count; k++) {
                const float target = total * static_cast<float>(k) / static_cast<float>(count);
                while (segment + 1 < count && arc[segment + 1] < target) segment++;
                const float length = arc[segment + 1] - arc[segment];
                const float t = (length > 0.0f) ? (target - arc[segment]) / length : 0.0f;
                verts[loop[(first + k) % N]] = at(first + segment) + t * (at(first + segment + 1) - at(first + segment));
            }
        }
    }
}

void SpringLevelSet2D::refineContour(int iterations, float proximity, float stepSize)
{
    Points2 &verts = mManifold.vertexLocations;
    Points2 delta(verts.size(), Vec2::Zero());
    // Particles the contour may snap to.
    PointGrid2D particleGrid;
    {
        Points2 points;
        std::vector<int> ids;
        for (size_t n = 0; n < mManifold.springlParticles.size(); n++) {
            if (refinesContourWith(n)) {
                points.push_back(mManifold.springlParticles[n]);
                ids.push_back(static_cast<int>(n));
            }
        }
        particleGrid.build(points, proximity, &ids);
    }
    std::vector<PointGrid2D::Neighbor> result;
    for (int iter = 0; iter < iterations; iter++) {
        for (size_t c = 0; c < mManifold.vertexIndexes.size(); c++) {
            Curve curve = mManifold.vertexIndexes[c];
            const int label = curveLabel(c);
            if (curve.size() < 2) continue;
            if (curve.back() == curve.front()) curve.pop_back();
            const size_t N = curve.size();
            for (size_t n = 0; n < N; n++) {
                const uint32_t cur = curve[n];
                const Vec2 curPt = verts[cur];
                const Vec2 nextPt = verts[curve[(n + 1) % N]];
                const Vec2 prevPt = verts[curve[(n + N - 1) % N]];
                const Vec2 tan = normalizedOrZero((nextPt - curPt) + (curPt - prevPt));
                const Vec2 norm(-tan.y(), tan.x());
                float w = 0.0f;
                delta[cur] = Vec2::Zero();
                const float vertexProximity = proximity * contourScale(cur);
                particleGrid.within(curPt, vertexProximity, result);
                // Closest particle facing the same way.
                int best = -1;
                for (const PointGrid2D::Neighbor &nbr : result) {
                    if (mLabeled && springlLabel(nbr.id) != label) continue;
                    if (norm.dot(mManifold.springlParticleNormals[nbr.id]) > kPlaneThreshold) {
                        best = nbr.id;
                        break;  // results are sorted by distance
                    }
                }
                if (best >= 0) {
                    const Vec2 closest = mManifold.springlParticles[best];
                    const Vec2 closestNorm = mManifold.springlParticleNormals[best];
                    w = 1.0f - std::clamp(distance(curPt, closest) / vertexProximity, 0.0f, 1.0f);
                    delta[cur] = -w * stepSize * (curPt - closest).dot(closestNorm) * closestNorm;
                }
                // Tangential smoothing near springls, normal smoothing away from them.
                const Vec2 delt = (nextPt + curPt + prevPt) / 3.0f - curPt;
                const Vec2 pr = delt.dot(norm) * norm;
                delta[cur] += w * (delt - pr) + (1.0f - w) * pr;
            }
        }
        for (size_t v = 0; v < verts.size(); v++) verts[v] += delta[v];
    }
}

// ---------------------------------------------------------------------------------------------
// Resampling and tracking
// ---------------------------------------------------------------------------------------------

int SpringLevelSet2D::markContractions(std::vector<uint8_t> &keep, const std::function<bool(size_t)> &include) const
{
    // One tree per region label (a single tree for single-object methods): a springl is measured
    // only against the contour of its own region.
    std::map<int, SegmentTree2D> contours;
    for (size_t c = 0; c < mManifold.vertexIndexes.size(); c++) {
        const Curve &curve = mManifold.vertexIndexes[c];
        SegmentTree2D &contour = contours[curveLabel(c)];
        for (size_t k = 0; k + 1 < curve.size(); k++) {
            contour.append(mManifold.vertexLocations[curve[k]], mManifold.vertexLocations[curve[k + 1]], static_cast<int>(curve[k + 1]));
        }
    }
    for (auto &entry : contours) entry.second.finish();
    SegmentTree2D noContour;
    noContour.finish();
    int contracted = 0;
    const size_t SN = mManifold.springlParticles.size();
    for (size_t i = 0; i < SN; i++) {
        if (!include(i)) continue;
        const Vec2 pt = mManifold.springlParticles[i];
        const float d1 = distance(mManifold.springlVertexes[2 * i], pt);
        const float d2 = distance(mManifold.springlVertexes[2 * i + 1], pt);
        const float scale = springlScale(i);
        const float contractDistance = tuning.contractDistance * scale;
        const float maxArmLength = tuning.maxArmLength * scale;
        const auto found = contours.find(springlLabel(i));
        const SegmentTree2D &contour = (found != contours.end()) ? found->second : noContour;
        const float d = contour.distance(pt, contractDistance + 1.0f);
        const bool ok = d <= contractDistance && d1 >= kMinEdgeLength && d2 >= kMinEdgeLength && d1 <= maxArmLength && d2 <= maxArmLength;
        if (!ok) {
            keep[i] = 0;
            contracted++;
        }
    }
    return contracted;
}

int SpringLevelSet2D::contract()
{
    std::vector<uint8_t> keep(mManifold.springlParticles.size(), 1);
    const int contracted = markContractions(keep, [](size_t) { return true; });
    if (contracted > 0) compactSpringls(keep);
    return contracted;
}

int SpringLevelSet2D::fill()
{
    solveContourForFill();
    // Gap tests see the springls that exist before this pass.
    refreshSpringlGrid();
    const Points2 springlVertexes = mManifold.springlVertexes;
    const Points2 &verts = mManifold.vertexLocations;
    const std::vector<int> springlLabels = mManifold.springlLabels;
    int filled = 0;
    for (size_t c = 0; c < mManifold.vertexIndexes.size(); c++) {
        const Curve &curve = mManifold.vertexIndexes[c];
        const int label = curveLabel(c);
        // Only springls of the curve's own region close its gaps.
        const auto sameRegion = [&](size_t s) { return !mLabeled || springlLabels[s] == label; };
        for (size_t k = 0; k + 1 < curve.size(); k++) {
            const uint32_t a = curve[k], b = curve[k + 1];
            const Vec2 pt = 0.5f * (verts[a] + verts[b]);
            const float scale = std::min(contourScale(a), contourScale(b));
            const float fillDistance = tuning.fillDistance * scale;
            const double gap = mSpringlGrid.closestSegment(springlVertexes, pt, fillDistance + 1.0f, sameRegion).value_or(fillDistance + 1.0f);
            if (gap > fillDistance) {
                appendSpringl(pt, verts[a], verts[b], scale, label);
                filled++;
            }
        }
    }
    return filled;
}

void SpringLevelSet2D::snapshotForTracking()
{
    mOldVertexes = mManifold.springlVertexes;
    mOldParticles = mManifold.springlParticles;
    mOldNormals = mManifold.springlParticleNormals;
    mOldCorrespondences = mManifold.correspondence;
    mOldVelocities = mManifold.velocities;
    mOldLabels = mManifold.springlLabels;
}

float SpringLevelSet2D::initialSignedValue(const Vec2 &p, int label) const
{
    const ImageI &labels = mInitialLabels;
    const ImageF &d = mInitialLabelDistance;
    const float x = std::clamp(p.x(), 0.0f, static_cast<float>(labels.width() - 1));
    const float y = std::clamp(p.y(), 0.0f, static_cast<float>(labels.height() - 1));
    const int i = std::min(static_cast<int>(x), labels.width() - 2), j = std::min(static_cast<int>(y), labels.height() - 2);
    const float fx = x - i, fy = y - j;
    const auto s = [&](int a, int b) { return (labels(a, b) == label) ? -d(a, b) : d(a, b); };
    return (1.0f - fy) * ((1.0f - fx) * s(i, j) + fx * s(i + 1, j)) + fy * ((1.0f - fx) * s(i, j + 1) + fx * s(i + 1, j + 1));
}

Vec2 SpringLevelSet2D::traceInitial(Vec2 pt, int label) const
{
    // Gradient descent on |phi0| back onto the initial interface (of region `label`).
    constexpr float kTimeStep = 0.5f;
    constexpr int kMaxIterations = 30;
    const ImageF &phi = mInitialLevelSet;
    const auto a = [&](float x, float y) { return mLabeled ? std::abs(initialSignedValue(Vec2(x, y), label)) : std::abs(phi.interpolate(x, y)); };
    for (int iter = 0; iter < kMaxIterations; iter++) {
        const float v11 = a(pt.x(), pt.y());
        const Vec2 grad =
            v11 * Vec2(0.5f * (a(pt.x() + 1.0f, pt.y()) - a(pt.x() - 1.0f, pt.y())), 0.5f * (a(pt.x(), pt.y() + 1.0f) - a(pt.x(), pt.y() - 1.0f)));
        pt -= kTimeStep * grad;
        if (grad.norm() <= 1E-5f) break;
    }
    return pt;
}

void SpringLevelSet2D::updateTracking(float maxDistance)
{
    // New springls (no correspondence yet) inherit correspondence and velocity from old springls
    // whose arm vertexes are near theirs and that face the same way.
    PointGrid2D oldVertexGrid;
    oldVertexGrid.build(mOldVertexes, maxDistance);
    std::vector<PointGrid2D::Neighbor> result;
    const float inf = std::numeric_limits<float>::infinity();
    Manifold2D &m = mManifold;
    for (size_t pid = 0; pid < m.springlParticles.size(); pid++) {
        if (!(std::isinf(m.correspondence[pid].x()) || m.springlParticleTracking[pid] < 0)) continue;
        const Vec2 pt = m.springlParticles[pid];
        const Vec2 norm = m.springlParticleNormals[pid];
        float dmin = 1E30f;
        int bestMatch = -1;
        Vec2 velocity = Vec2::Zero();
        Vec2 q[2] = {Vec2::Constant(inf), Vec2::Constant(inf)};
        for (int end = 0; end < 2; end++) {
            oldVertexGrid.within(m.springlVertexes[2 * pid + end], maxDistance, result);
            for (const PointGrid2D::Neighbor &nbr : result) {
                const size_t qid = static_cast<size_t>(nbr.id) / 2;
                if (!canTrackFrom(pid, qid)) continue;
                q[end] = mOldCorrespondences[qid];
                if (norm.dot(mOldNormals[qid]) > kPlaneThreshold) {
                    const float d = distance(mOldParticles[qid], pt);
                    if (d < dmin) {
                        bestMatch = static_cast<int>(qid);
                        dmin = d;
                    }
                    if (!std::isinf(q[end].x())) {
                        velocity += mOldVelocities[qid];
                        break;
                    }
                }
            }
        }
        if (m.springlParticleTracking[pid] < 0 && bestMatch >= 0) m.springlParticleTracking[pid] = bestMatch;
        const bool has0 = !std::isinf(q[0].x()), has1 = !std::isinf(q[1].x());
        if (has0 && has1) {
            m.velocities[pid] = 0.5f * velocity;
            m.correspondence[pid] = traceInitial(0.5f * (q[0] + q[1]), springlLabel(pid));
        } else if (has0 || has1) {
            m.velocities[pid] = velocity;
            m.correspondence[pid] = has0 ? q[0] : q[1];
        }
    }
}
}  // namespace springls
