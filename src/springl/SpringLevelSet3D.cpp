#include <core/Profile.hpp>
#include <springl/SpringLevelSet3D.hpp>

#include <spatial/PointGrid3D.hpp>
#include <spatial/TriangleTree3D.hpp>

#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <array>
#include <cassert>
#include <cmath>

namespace springls
{

namespace
{
// Normals within this angle of each other belong to the same side of the interface.
const float kPlaneThreshold = std::cos(80.0f * kPi / 180.0f);
// The unsigned field is only read near the zero crossing (LevelSet3D::evolve skips voxels with
// |phi| > 0.5 and the stencil reaches one voxel further), so it is computed only on the
// narrow-band voxels within this distance of the interface.
constexpr float kUnsignedBand = 2.5f;
// The pull toward the springls is at most this many voxels per pseudo-time step.
constexpr float kMaxPullVoxels = 1.5f;
// Velocity field weights below this are treated as zero.
constexpr float kWeightEpsilon = 1E-7f;

// Iso-surface triangles (quads split in two) for distance queries.
TriangleTree3D MakeSurfaceTree(const SurfaceMesh3D &mesh)
{
    TriangleTree3D tree;
    const int K = mesh.elementSize();
    const size_t F = mesh.faceCount();
    tree.reserve(F * (K - 2));
    const auto &v = mesh.vertexes;
    for (size_t f = 0; f < F; f++) {
        const uint32_t *idx = mesh.face(f);
        tree.append(v[idx[0]], v[idx[1]], v[idx[2]], static_cast<int>(f));
        if (K == 4) tree.append(v[idx[2]], v[idx[3]], v[idx[0]], static_cast<int>(f));
    }
    tree.finish();
    return tree;
}
}  // namespace

bool FitSimilarity3D(std::span<const Vec3> from, std::span<const Vec3> to, Eigen::Matrix3f &R, float &s)
{
    assert(from.size() == to.size());
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    double norm = 0.0;
    for (size_t k = 0; k < from.size(); k++) {
        const Vec3d a = from[k].cast<double>();
        H += a * to[k].cast<double>().transpose();
        norm += a.squaredNorm();
    }
    constexpr double kMinFit = 1E-12;
    if (norm < kMinFit) return false;
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix3d &U = svd.matrixU();
    const Eigen::Matrix3d &V = svd.matrixV();
    const double d = ((V * U.transpose()).determinant() < 0.0) ? -1.0 : 1.0;
    const Vec3d sigma = svd.singularValues();
    const double trace = sigma[0] + sigma[1] + d * sigma[2];
    if (!(trace > kMinFit * norm)) return false;
    R = (V * Vec3d(1.0, 1.0, d).asDiagonal() * U.transpose()).cast<float>();
    s = static_cast<float>(trace / norm);
    return true;
}

SpringLevelSet3D::SpringLevelSet3D(ElementType elementType) { mManifold.elementType = elementType; }

// ---------------------------------------------------------------------------------------------
// Springl bookkeeping
// ---------------------------------------------------------------------------------------------

size_t SpringLevelSet3D::appendSpringl(const Vec3 &particle, std::span<const Vec3> vertexes, float scale)
{
    const int K = static_cast<int>(vertexes.size());
    assert(K >= 3 && K <= kMaxSpringlSize);
    Manifold3D &m = mManifold;
    const size_t index = m.springlParticles.size();
    if (mScaledSpringls) mSpringlScales.push_back(scale);
    m.springlParticles.push_back(particle);
    // Springls face outward (along the level-set gradient), like the iso-surface vertex normals that
    // refineContour() matches them against.
    Vec3 normal = FanNormal(particle, vertexes.data(), K);
    const bool reverse = normal.dot(levelSetGradient(particle)) < 0.0f;
    if (reverse) normal = -normal;
    std::array<Vec3, kMaxSpringlSize> ordered;
    for (int k = 0; k < K; k++) ordered[k] = vertexes[reverse ? K - 1 - k : k];
    m.appendSpringlVertexes(std::span<const Vec3>(ordered.data(), K));
    m.springlParticleNormals.push_back(normalizedOrZero(normal));
    m.velocities.push_back(Vec3::Zero());
    if (mTrackingEnabled) {
        m.springlParticleTracking.push_back(-1);
        m.correspondence.push_back(Vec3::Constant(std::numeric_limits<float>::infinity()));
    }
    return index;
}

void SpringLevelSet3D::clearSpringls()
{
    Manifold3D &m = mManifold;
    m.springlParticles.clear();
    m.clearSpringlVertexes();
    m.springlParticleNormals.clear();
    m.springlParticleTracking.clear();
    m.correspondence.clear();
    m.velocities.clear();
    mSpringlScales.clear();
    mNearestNeighbors.clear();
    mSpringlGrid.invalidate();
}

void SpringLevelSet3D::compactSpringls(const std::vector<uint8_t> &keep)
{
    Manifold3D &m = mManifold;
    const size_t N = m.springlParticles.size();
    assert(keep.size() == N);
    const bool hasNormals = m.springlParticleNormals.size() == N;
    const bool hasTracking = mTrackingEnabled && m.correspondence.size() == N && m.springlParticleTracking.size() == N;
    const bool hasScales = mSpringlScales.size() == N;
    mSpringlGrid.invalidate();  // springl indexes are about to change
    size_t out = 0;
    uint32_t vertexOut = 0;
    for (size_t i = 0; i < N; i++) {
        if (!keep[i]) continue;
        m.springlParticles[out] = m.springlParticles[i];
        // Vertexes move down (never up), so copying in place in order is safe.
        const uint32_t begin = m.springlStart[i], end = m.springlStart[i + 1];
        m.springlStart[out] = vertexOut;
        for (uint32_t v = begin; v < end; v++, vertexOut++) {
            m.springlVertexes[vertexOut] = m.springlVertexes[v];
            m.vertexSpringl[vertexOut] = static_cast<uint32_t>(out);
        }
        m.velocities[out] = m.velocities[i];
        if (hasNormals) m.springlParticleNormals[out] = m.springlParticleNormals[i];
        if (hasTracking) {
            m.correspondence[out] = m.correspondence[i];
            m.springlParticleTracking[out] = m.springlParticleTracking[i];
        }
        if (hasScales) mSpringlScales[out] = mSpringlScales[i];
        out++;
    }
    m.springlParticles.resize(out);
    m.springlStart.resize(out + 1);
    m.springlStart[out] = vertexOut;
    m.springlVertexes.resize(vertexOut);
    m.vertexSpringl.resize(vertexOut);
    m.velocities.resize(out);
    if (hasNormals) m.springlParticleNormals.resize(out);
    if (hasTracking) {
        m.correspondence.resize(out);
        m.springlParticleTracking.resize(out);
    }
    if (hasScales) mSpringlScales.resize(out);
}

void SpringLevelSet3D::seedSpringls()
{
    const SurfaceMesh3D &iso = mManifold.isoSurface;
    const int K = elementSize();
    std::array<Vec3, kMaxElementSize> verts;
    for (size_t f = 0; f < iso.faceCount(); f++) {
        const uint32_t *idx = iso.face(f);
        float scale = std::numeric_limits<float>::max();
        for (int k = 0; k < K; k++) {
            verts[k] = iso.vertexes[idx[k]];
            scale = std::min(scale, contourScale(idx[k]));
        }
        appendSpringl(Centroid(verts.data(), K), std::span<const Vec3>(verts.data(), K), scale);
    }
}

void SpringLevelSet3D::resetTrackingIdentity()
{
    if (!mTrackingEnabled) return;
    mManifold.springlParticleTracking.resize(mManifold.springlParticles.size());
    for (size_t n = 0; n < mManifold.springlParticleTracking.size(); n++) mManifold.springlParticleTracking[n] = static_cast<int>(n);
}

// ---------------------------------------------------------------------------------------------
// Initialization and the time step
// ---------------------------------------------------------------------------------------------

bool SpringLevelSet3D::init()
{
    if (mInitialLevelSet.empty()) return false;
    mSimulationTime = 0.0;
    mIteration = 0;
    mAdvectionSubstepCount = 0;
    mUnsignedLevelSet.clear();
    initLevelSet();
    clearSpringls();
    seedSpringls();
    mManifold.velocities.assign(mManifold.springlParticles.size(), Vec3::Zero());
    if (mTrackingEnabled) {
        mManifold.correspondence = mManifold.springlParticles;
        resetTrackingIdentity();
    }
    mManifold.updateNormals();
    relax();
    updateNearestNeighbors();
    updateUnsignedLevelSet();
    return true;
}

void SpringLevelSet3D::initLevelSet()
{
    mBand.initialize(mInitialLevelSet);
    // The initial iso-surface is not refined: there are no springls to refine it toward yet.
    ExtractIsoSurface(mBand.phi(), mBand.active(), elementType(), mManifold.isoSurface);
    orientVertexNormals(mManifold.isoSurface);
}

int SpringLevelSet3D::settle(int maxIterations, float tolerance)
{
    int iterations = 0;
    while (iterations < maxIterations) {
        iterations++;
        relax();
        const int contracted = contract();
        const int filled = fill();
        mManifold.updateNormals();
        if (tuning.fixNormals) fixNormals();
        if (static_cast<double>(contracted) + filled <= static_cast<double>(tolerance) * static_cast<double>(mManifold.springlCount())) break;
    }
    if (iterations > 0) {
        mManifold.velocities.assign(mManifold.springlParticles.size(), Vec3::Zero());
        if (mTrackingEnabled) {
            mManifold.correspondence = mManifold.springlParticles;
            resetTrackingIdentity();
        }
        updateNearestNeighbors();
        updateUnsignedLevelSet();
    }
    return iterations;
}

bool SpringLevelSet3D::step()
{
    const bool logStep = verbose && (mIteration % kLogInterval) == 0;
    resetTrackingIdentity();
    int contractTotal = 0, fillTotal = 0;
    for (int iter = 0; iter < tuning.advectionSubstepsPerStep; iter++) {
        if (mSchedule.weight) {
            const double remaining = mSchedule.endTime - mSimulationTime;
            if (remaining <= 0.0) break;
            const float timeStep = advect(tuning.maxStepVoxels, static_cast<float>(std::min(remaining, mSchedule.maxTimeStep)));
            // Land exactly on endTime instead of accumulating float rounding.
            mSimulationTime = (timeStep >= remaining) ? mSchedule.endTime : mSimulationTime + timeStep;
        } else {
            mSimulationTime += advect(tuning.maxStepVoxels);
        }
        // Relaxation pulls neighboring arm vertexes together, which can open gaps elsewhere; a level
        // set rebuilt from the springls (rather than evolved toward them) is rebuilt first, from the
        // springls exactly as the flow left them.
        const bool rebuildFirst = rebuildsLevelSetBeforeRelax();
        if (mResampleEnabled && !rebuildFirst) relaxWithRememberedNeighbors();
        evolveLevelSetAndContour();
        if (mResampleEnabled && rebuildFirst) relaxWithRememberedNeighbors();
        if (mResampleEnabled) {
            // Resampling tracks against this sub-step's snapshot; composing with the tracking so far
            // keeps it relative to the start of the step.
            const std::vector<int> stepTracking = mManifold.springlParticleTracking;
            snapshotForTracking();
            resetTrackingIdentity();
            contractTotal += contract();
            // Fill does not use neighbors; relaxing after a pass that added springls recomputes them.
            int filled = 0, passes = 0;
            do {
                passes++;
                filled = fill();
                if (filled > 0) relax();
                fillTotal += filled;
            } while (filled > 0 && passes < tuning.maxFillPasses);
            mManifold.updateNormals();
            if (tuning.fixNormals) fixNormals();
            if (mTrackingEnabled) {
                updateTracking(3.0f * tuning.nearestNeighborDistance);
                for (int &t : mManifold.springlParticleTracking) {
                    if (t >= 0) t = stepTracking[t];
                }
            }
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

void SpringLevelSet3D::advanceWithFlow(float maxStep, const std::function<float(double)> &maxSpeed, const std::function<void(double, float)> &substep)
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
        substep(mSimulationTime, static_cast<float>(dt));
        mAdvectionSubstepCount++;
        // Land exactly on endTime instead of accumulating float rounding.
        mSimulationTime = (mSchedule.weight && dt >= remaining) ? mSchedule.endTime : mSimulationTime + dt;
    }
}

float SpringLevelSet3D::maxContourSpeed(const Points3 &points, double t) const
{
    float speed = 0.0f;
    for (const Vec3 &p : points) speed = std::max(speed, particleVelocity(p, t).squaredNorm());
    return std::sqrt(speed);
}

Vec3 SpringLevelSet3D::particleVelocity(const Vec3 &position, double time) const
{
    // The schedule weight is evaluated at the stage time, so RK4 sees the time-varying flow.
    const float weight = mSchedule.weight ? mSchedule.weight(time) : mAdvectionWeight;
    if (std::abs(weight) < kWeightEpsilon) return Vec3::Zero();
    if (mVelocityFunction) return mVelocityFunction(position) * weight;
    if (mVelocityField.empty()) return Vec3::Zero();
    return mVelocityField.interpolate(position) * weight;
}

float SpringLevelSet3D::advectionTimeStep(const Points3 &k1, float maxStep) const
{
    float maxSpeed = 0.0f;
    for (const Vec3 &v : k1) maxSpeed = std::max(maxSpeed, v.squaredNorm());
    maxSpeed = std::sqrt(maxSpeed);
    return (maxSpeed > 1.0f) ? maxStep / maxSpeed : maxStep;
}

float SpringLevelSet3D::advect(float maxStep, float maxTimeStep)
{
    ProfileScope profileScope("advect");
    Manifold3D &m = mManifold;
    const int N = static_cast<int>(m.springlParticles.size());
    const double t0 = mSimulationTime;
    // Stage 1 at the current particle positions; its magnitude picks the time step.
    Points3 k1(N);
    SPRINGLS_OMP(omp parallel for if (N >= kParallelMinLightItems))
    for (int i = 0; i < N; i++) k1[i] = particleVelocity(m.springlParticles[i], t0);
    mAdvectionSubstepCount++;
    const float timeStep = std::min(advectionTimeStep(k1, maxStep), maxTimeStep);
    const double tHalf = t0 + 0.5 * timeStep;
    const double tEnd = t0 + timeStep;
    // Displacement of a point over the step (classic RK4, or forward Euler).
    const auto displace = [&](const Vec3 &x, const Vec3 &v1) -> Vec3 {
        if (temporalScheme != TemporalScheme::RK4) return timeStep * v1;
        const Vec3 v2 = particleVelocity(x + (0.5f * timeStep) * v1, tHalf);
        const Vec3 v3 = particleVelocity(x + (0.5f * timeStep) * v2, tHalf);
        const Vec3 v4 = particleVelocity(x + timeStep * v3, tEnd);
        return (timeStep / 6.0f) * (v1 + 2.0f * v2 + 2.0f * v3 + v4);
    };
    // 1. Particles and vertexes are all advected by the flow.
    const Points3 oldParticles = m.springlParticles;
    const Points3 oldVertexes = m.springlVertexes;
    SPRINGLS_OMP(omp parallel for if (N >= kParallelMinHeavyItems))
    for (int i = 0; i < N; i++) {
        const Vec3 displacement = displace(oldParticles[i], k1[i]);
        m.springlParticles[i] = oldParticles[i] + displacement;
        for (uint32_t e = m.springlStart[i]; e < m.springlStart[i + 1]; e++) {
            const Vec3 v = oldVertexes[e];
            m.springlVertexes[e] = v + displace(v, particleVelocity(v, t0));
        }
        // Velocity is kept only for updateTracking(), which hands it to new springls.
        m.velocities[i] = (timeStep > 0.0f) ? Vec3(displacement / timeStep) : k1[i];
    }
    // 2. Neighbors are found on the flow-advected vertexes and remembered for relaxation.
    updateNearestNeighbors();
    // 3. Each springl becomes the similarity transform (rotation + uniform scale about its advected
    //    particle) of its previous shape that best fits its advected vertexes, so it stays rigid
    //    and its particle does not move.
    SPRINGLS_OMP(omp parallel for if (N >= kParallelMinLightItems))
    for (int i = 0; i < N; i++) {
        const Vec3 p = m.springlParticles[i];
        const uint32_t base = m.springlStart[i];
        const int K = m.springlSize(i);
        std::array<Vec3, kMaxSpringlSize> arms, advected;
        for (int e = 0; e < K; e++) {
            arms[e] = oldVertexes[base + e] - oldParticles[i];
            advected[e] = m.springlVertexes[base + e] - p;
        }
        Eigen::Matrix3f R;
        float s;
        if (FitSimilarity3D(std::span<const Vec3>(arms.data(), K), std::span<const Vec3>(advected.data(), K), R, s)) {
            for (int e = 0; e < K; e++) m.springlVertexes[base + e] = p + s * (R * arms[e]);
        } else {
            // Degenerate springl: keep its shape and translate it with the particle.
            for (int e = 0; e < K; e++) m.springlVertexes[base + e] = p + arms[e];
        }
    }
    m.updateNormals();
    return timeStep;
}

void SpringLevelSet3D::updateNearestNeighbors(float maxDistance)
{
    ProfileScope profileScope("updateNearestNeighbors");
    if (maxDistance < 0.0f) maxDistance = tuning.nearestNeighborDistance;
    const Points3 &vertexes = mManifold.springlVertexes;
    const int V = static_cast<int>(vertexes.size());
    mNearestNeighbors.resize(V);
    if (V == 0) return;
    refreshSpringlGrid();
    const size_t maxCount = static_cast<size_t>(std::max(0, tuning.maxNearestNeighbors));
    SPRINGLS_OMP(omp parallel for if (V >= kParallelMinHeavyItems))
    for (int i = 0; i < V; i++) {
        const size_t springl = mManifold.vertexSpringl[i];
        const float radius = maxDistance * attractionScale(vertexes[i], springl);
        mSpringlGrid.nearestEdges(
            mManifold.springlParticles, vertexes, vertexes[i], radius, springl, maxCount, [&](size_t other) { return canAttract(springl, other); },
            mNearestNeighbors[i]);
    }
}

void SpringLevelSet3D::relax()
{
    ProfileScope profileScope("relax (find neighbors)");
    updateNearestNeighbors();
    relaxWithRememberedNeighbors();
}

void SpringLevelSet3D::relaxWithRememberedNeighbors()
{
    ProfileScope profileScope("relax (remembered neighbors)");
    if (mNearestNeighbors.size() != mManifold.springlVertexes.size()) updateNearestNeighbors();
    for (int i = 0; i < tuning.relaxIterations; i++) relax(tuning.relaxTimeStep);
}

void SpringLevelSet3D::relax(float timeStep)
{
    const int N = static_cast<int>(mManifold.springlParticles.size());
    Points3 updates(mManifold.springlVertexes.size());
    SPRINGLS_OMP(omp parallel for if (N >= kParallelMinLightItems))
    for (int i = 0; i < N; i++) relaxSpringl(i, timeStep, updates);
    mManifold.springlVertexes = std::move(updates);
    mManifold.updateNormals();
}

void SpringLevelSet3D::relaxSpringl(size_t idx, float timeStep, Points3 &updates) const
{
    constexpr float kMaxForce = 0.999f;
    const Manifold3D &m = mManifold;
    const int K = m.springlSize(idx);
    const uint32_t base = m.springlStart[idx];
    const Points3 &vertexes = m.springlVertexes;
    const Vec3 particle = m.springlParticles[idx];
    std::array<Vec3, kMaxSpringlSize> motion, tangents;
    std::array<float, kMaxSpringlSize> armLength, springForce;
    Vec3 moment = Vec3::Zero();
    const float scale = springlScale(idx);
    const float armRestLength = tuning.armRestLength * scale, maxArmLength = tuning.maxArmLength * scale;
    for (int k = 0; k < K; k++) {
        const size_t eid = base + k;
        const Vec3 start = vertexes[eid];
        const Vec3 arm = start - particle;
        armLength[k] = arm.norm();
        tangents[k] = (armLength[k] > 1E-6f) ? Vec3(arm / armLength[k]) : arm;
        Vec3 velocity = Vec3::Zero();
        const std::vector<SpringlEdge> &nbrs = mNearestNeighbors[eid];
        for (const SpringlEdge &e : nbrs) {
            // The closest point on the neighbor edge is recomputed each iteration.
            Vec3 closest;
            const uint32_t nbase = m.springlStart[e.springl];
            const int nK = m.springlSize(e.springl);
            SegmentDistanceSqr(vertexes[nbase + e.edge], vertexes[nbase + (e.edge + 1) % nK], start, &closest);
            const Vec3 dir = closest - start;
            // Neighbors of a different size class (coarse/fine transitions) use the pair's mean size.
            const float pairScale = 0.5f * (scale + springlScale(e.springl));
            const float restSeparation = tuning.neighborRestSeparation * pairScale;
            const float w =
                std::atanh(kMaxForce * std::clamp((dir.norm() - restSeparation) / ((kExtent + tuning.neighborRestSeparation) * pairScale), -1.0f, 1.0f));
            velocity += w * dir;
        }
        if (!nbrs.empty()) velocity /= static_cast<float>(nbrs.size());
        motion[k] = timeStep * tuning.sharpness * velocity;
        springForce[k] = timeStep * tuning.springConstant * (armRestLength - armLength[k]);
        moment += motion[k].cross(tangents[k]);
    }
    // Rotate the springl about its particle by the net moment (toward the tangential motion).
    const float angle = moment.norm();
    const Eigen::Matrix3f rot = (angle >= 1E-6f) ? Eigen::Matrix3f(Eigen::AngleAxisf(-angle, moment / angle)) : Eigen::Matrix3f::Identity();
    for (int k = 0; k < K; k++) {
        const float radial = std::clamp(armLength[k] + motion[k].dot(tangents[k]) + springForce[k], kMinArmLength, maxArmLength);
        updates[base + k] = rot * (radial * tangents[k]) + particle;
    }
}

// ---------------------------------------------------------------------------------------------
// Level set
// ---------------------------------------------------------------------------------------------

void SpringLevelSet3D::updateUnsignedLevelSet(float maxDistance, const std::function<bool(size_t)> &include)
{
    if (maxDistance < 0.0f) maxDistance = tuning.unsignedMaxDistance;
    if (mBand.empty()) return;  // adaptive variants keep no uniform band
    const VolumeF &phi = mBand.phi();
    mUnsignedLevelSet.resize(phi.width(), phi.height(), phi.depth());
    mUnsignedLevelSet.fill(maxDistance);
    // Dense queries near the interface: a BVH over the springls' triangle fans.
    const Manifold3D &m = mManifold;
    const size_t N = m.springlParticles.size();
    TriangleTree3D tree;
    tree.reserve(m.springlVertexes.size());
    for (size_t n = 0; n < N; n++) {
        if (include && !include(n)) continue;
        const Vec3 *v = m.springl(n);
        const int K = m.springlSize(n);
        for (int k = 0; k < K; k++) tree.append(m.springlParticles[n], v[k], v[(k + 1) % K], static_cast<int>(n));
    }
    tree.finish();
    if (tree.empty()) return;
    const std::vector<Vec3i> &active = mBand.active();
    const int AS = static_cast<int>(active.size());
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 256) if (AS >= kParallelMinHeavyItems))
    for (int i = 0; i < AS; i++) {
        const Vec3i p = active[i];
        if (std::abs(phi(p)) <= kUnsignedBand) mUnsignedLevelSet(p) = tree.distance(p.cast<float>(), maxDistance);
    }
}

Vec3 SpringLevelSet3D::unsignedPull(int i, int j, int k) const
{
    // -min(U, 1.5) * grad(U) / |grad(U)|: pulls the level set toward the springls.
    const VolumeF &u = mUnsignedLevelSet;
    const Vec3 grad(0.5f * (u.at(i + 1, j, k) - u.at(i - 1, j, k)), 0.5f * (u.at(i, j + 1, k) - u.at(i, j - 1, k)),
                    0.5f * (u.at(i, j, k + 1) - u.at(i, j, k - 1)));
    const float len = std::max(1E-6f, grad.norm());
    return -(std::min(u(i, j, k), kMaxPullVoxels) * grad / len);
}

Vec3 SpringLevelSet3D::levelSetGradient(const Vec3 &p) const
{
    const VolumeF &phi = mBand.phi();
    const auto v = [&](float dx, float dy, float dz) { return phi.interpolate(p.x() + dx, p.y() + dy, p.z() + dz); };
    const Vec3 grad(0.5f * (v(1, 0, 0) - v(-1, 0, 0)), 0.5f * (v(0, 1, 0) - v(0, -1, 0)), 0.5f * (v(0, 0, 1) - v(0, 0, -1)));
    return grad / std::max(1E-6f, grad.norm());
}

void SpringLevelSet3D::updateSignedLevelSet(float maxStep)
{
    mBand.evolve([this](int i, int j, int k) { return unsignedPull(i, j, k); }, mCurvatureWeight, maxStep);
}

int SpringLevelSet3D::correctSignsFromSpringls(const std::function<bool(size_t)> &include)
{
    const Manifold3D &m = mManifold;
    TriangleTree3D springlTree;
    springlTree.reserve(m.springlVertexes.size());
    for (size_t n = 0; n < m.springlParticles.size(); n++) {
        if (include && !include(n)) continue;
        const Vec3 *v = m.springl(n);
        const int K = m.springlSize(n);
        for (int k = 0; k < K; k++) springlTree.append(m.springlParticles[n], v[k], v[(k + 1) % K], static_cast<int>(n));
    }
    springlTree.finish();
    return mBand.correctSigns([&](int i, int j, int k, float &d) {
        // Within a cell of a springl and over its element rather than past its outer edges (where
        // the side is ambiguous): signed by the springl's outward normal.
        const Vec3 q(static_cast<float>(i), static_cast<float>(j), static_cast<float>(k));
        const auto hit = springlTree.nearest(q, 1.0f);
        if (!hit) return false;
        const size_t s = static_cast<size_t>(hit->id);
        constexpr float kEdgeTolerance = 1E-10f;
        const Vec3 *v = m.springl(s);
        const int K = m.springlSize(s);
        for (int e = 0; e < K; e++) {
            if (SegmentDistanceSqr(v[e], v[(e + 1) % K], hit->point) < kEdgeTolerance) return false;
        }
        const float side = m.springlParticleNormals[s].dot(q - hit->point);
        if (side == 0.0f) return false;
        d = (side < 0.0f) ? -std::sqrt(hit->distanceSqr) : std::sqrt(hit->distanceSqr);
        return true;
    });
}

void SpringLevelSet3D::evolveLevelSetAndContour()
{
    updateUnsignedLevelSet();
    for (int i = 0; i < tuning.signedEvolveIterations; i++) updateSignedLevelSet();
    if (tuning.springlCorrection) correctSignsFromSpringls();
    afterLevelSetUpdate();
    solveContour();
}

// ---------------------------------------------------------------------------------------------
// Iso-surface
// ---------------------------------------------------------------------------------------------

void SpringLevelSet3D::orientVertexNormals(SurfaceMesh3D &mesh) const
{
    const int N = static_cast<int>(mesh.vertexes.size());
    SPRINGLS_OMP(omp parallel for if (N >= kParallelMinHeavyItems))
    for (int n = 0; n < N; n++) {
        if (mesh.normals[n].dot(levelSetGradient(mesh.vertexes[n])) < 0.0f) mesh.normals[n] = -mesh.normals[n];
    }
}

void SpringLevelSet3D::solveContour(SurfaceMesh3D &mesh) const
{
    ExtractIsoSurface(mBand.phi(), mBand.active(), elementType(), mesh);
    orientVertexNormals(mesh);
    refineContour(mesh, tuning.refineIterations, tuning.refineProximity, tuning.refineStepSize);
}

void SpringLevelSet3D::refineContour(SurfaceMesh3D &mesh, int iterations, float proximity, float stepSize) const
{
    ProfileScope profileScope("refineContour");
    const int N = static_cast<int>(mesh.vertexes.size());
    if (N == 0) return;
    Points3 &points = mesh.vertexes;
    const Points3 &normals = mesh.normals;
    const Manifold3D &m = mManifold;
    // Edge-adjacent vertexes, as a compressed (CSR) table, each vertex's list sorted and unique:
    // bucket both directions of every face edge by vertex (counting pass, then fill), then sort and
    // dedupe each short list in parallel.
    const int K = mesh.elementSize();
    const size_t F = mesh.faceCount();
    std::vector<uint32_t> nbrStart(N + 1, 0);
    for (size_t f = 0; f < F; f++) {
        const uint32_t *idx = mesh.face(f);
        for (int k = 0; k < K; k++) nbrStart[idx[k] + 1] += 2;  // its next and previous corner
    }
    for (int n = 0; n < N; n++) nbrStart[n + 1] += nbrStart[n];
    std::vector<uint32_t> raw(nbrStart[N]);
    {
        std::vector<uint32_t> fillAt(nbrStart.begin(), nbrStart.end() - 1);
        for (size_t f = 0; f < F; f++) {
            const uint32_t *idx = mesh.face(f);
            for (int k = 0; k < K; k++) {
                raw[fillAt[idx[k]]++] = idx[(k + 1) % K];
                raw[fillAt[idx[k]]++] = idx[(k + K - 1) % K];
            }
        }
    }
    std::vector<uint32_t> uniqueCount(N);
    SPRINGLS_OMP(omp parallel for if (N >= kParallelMinLightItems))
    for (int n = 0; n < N; n++) {
        const auto begin = raw.begin() + nbrStart[n], end = raw.begin() + nbrStart[n + 1];
        std::sort(begin, end);
        uniqueCount[n] = static_cast<uint32_t>(std::unique(begin, end) - begin);
    }
    std::vector<uint32_t> nbrs;
    {
        std::vector<uint32_t> start(N + 1, 0);
        for (int n = 0; n < N; n++) start[n + 1] = start[n] + uniqueCount[n];
        nbrs.resize(start[N]);
        SPRINGLS_OMP(omp parallel for if (N >= kParallelMinLightItems))
        for (int n = 0; n < N; n++) std::copy(raw.begin() + nbrStart[n], raw.begin() + nbrStart[n] + uniqueCount[n], nbrs.begin() + start[n]);
        nbrStart.swap(start);
    }
    // Particles near each vertex, gathered once: vertexes move far less than kSlack during
    // refinement, and each pass still requires a particle within `proximity`. Only springls facing
    // the same way as the vertex can ever be matched.
    constexpr float kSlack = 0.5f;
    PointGrid3D particleGrid;
    {
        Points3 pts;
        std::vector<int> ids;
        for (size_t n = 0; n < m.springlParticles.size(); n++) {
            if (refinesContourWith(n)) {
                pts.push_back(m.springlParticles[n]);
                ids.push_back(static_cast<int>(n));
            }
        }
        particleGrid.build(pts, proximity + kSlack, &ids);
    }
    std::vector<std::vector<uint32_t>> candidates(N);
    std::vector<float> vertexProximity(N);
    SPRINGLS_OMP(omp parallel for if (N >= kParallelMinHeavyItems))
    for (int n = 0; n < N; n++) {
        vertexProximity[n] = proximity * contourScale(static_cast<uint32_t>(n));
        std::vector<PointGrid3D::Neighbor> result;
        particleGrid.within(points[n], vertexProximity[n] + kSlack * contourScale(static_cast<uint32_t>(n)), result);
        for (const PointGrid3D::Neighbor &r : result) {
            if (normals[n].dot(m.springlParticleNormals[r.id]) > kPlaneThreshold) candidates[n].push_back(static_cast<uint32_t>(r.id));
        }
    }
    Points3 newPoints = points;
    for (int iter = 0; iter < iterations; iter++) {
        SPRINGLS_OMP(omp parallel for if (N >= kParallelMinLightItems))
        for (int n = 0; n < N; n++) {
            const Vec3 pt = points[n];
            const Vec3 norm = normals[n];
            float w = 0.0f;
            newPoints[n] = pt;
            const float reach = vertexProximity[n];
            float dmin = reach;
            int closest = -1;
            for (uint32_t c : candidates[n]) {
                const float d = distance(m.springlParticles[c], pt);
                if (d <= dmin) {
                    dmin = d;
                    closest = static_cast<int>(c);
                }
            }
            if (closest >= 0) {
                const Vec3 closestNorm = m.springlParticleNormals[closest];
                w = 1.0f - std::clamp(dmin / reach, 0.0f, 1.0f);
                newPoints[n] = pt - w * stepSize * (pt - m.springlParticles[closest]).dot(closestNorm) * closestNorm;
            }
            // Tangential smoothing near springls, normal smoothing away from them.
            const uint32_t count = nbrStart[n + 1] - nbrStart[n];
            if (count > 2) {
                Vec3 center = Vec3::Zero();
                for (uint32_t k = nbrStart[n]; k < nbrStart[n + 1]; k++) center += points[nbrs[k]];
                center /= static_cast<float>(count);
                const Vec3 delta = center - pt;
                const Vec3 pr = delta.dot(norm) * norm;
                newPoints[n] += w * (delta - pr) + (1.0f - w) * pr;
            }
        }
        std::swap(points, newPoints);
    }
    mesh.updateVertexNormals();
    orientVertexNormals(mesh);
}

// ---------------------------------------------------------------------------------------------
// Resampling and tracking
// ---------------------------------------------------------------------------------------------

int SpringLevelSet3D::markContractions(std::vector<uint8_t> &keep, const std::function<bool(size_t)> &include) const
{
    const TriangleTree3D tree = MakeSurfaceTree(mManifold.isoSurface);
    const Manifold3D &m = mManifold;
    const int SN = static_cast<int>(m.springlParticles.size());
    assert(keep.size() == static_cast<size_t>(SN));
    constexpr float kQueryMargin = 1.0f;
    int contracted = 0;
    SPRINGLS_OMP(omp parallel for reduction(+ : contracted) if (SN >= kParallelMinHeavyItems))
    for (int i = 0; i < SN; i++) {
        if (!include(i)) continue;
        if (isUndersized(i)) {
            keep[i] = 0;
            contracted++;
            continue;
        }
        const float scale = springlScale(i), scale2 = scale * scale;
        const float contractDistance = tuning.contractDistance * scale;
        const Vec3 pt = m.springlParticles[i];
        float minArm = std::numeric_limits<float>::max(), maxArm = 0.0f;
        float minEdge = std::numeric_limits<float>::max(), maxEdge = 0.0f;
        float area = 0.0f;
        const Vec3 *v = m.springl(i);
        const int K = m.springlSize(i);
        for (int k = 0; k < K; k++) {
            const Vec3 v1 = v[k];
            const Vec3 v2 = v[(k + 1) % K];
            const float arm = distance(v1, pt), edge = distance(v1, v2);
            minArm = std::min(minArm, arm);
            maxArm = std::max(maxArm, arm);
            minEdge = std::min(minEdge, edge);
            maxEdge = std::max(maxEdge, edge);
            area += 0.5f * (v1 - pt).cross(v2 - pt).norm();
        }
        // Dual springls (DoubleWeave) have the valence of their iso-surface vertex and are far from
        // flat at irregular vertexes, so they also contract when folded: when a triangle of their
        // fan faces against the fan normal. A folded springl is on both sides of itself, and the
        // springl correction of the adaptive grid would stamp wrong signs around it.
        bool folded = false;
        if (!m.springlConstellations.empty() && m.springlConstellations[i] == static_cast<uint8_t>(Constellation::Dual)) {
            Vec3 fan = Vec3::Zero();
            for (int k = 0; k < K; k++) fan += (v[k] - pt).cross(v[(k + 1) % K] - pt);
            for (int k = 0; k < K; k++) folded |= (v[k] - pt).cross(v[(k + 1) % K] - pt).dot(fan) <= 0.0f;
        }
        const float aspect = (maxEdge > 0.0f) ? minEdge / maxEdge : 0.0f;
        const float d = tree.empty() ? std::numeric_limits<float>::infinity() : tree.distance(pt, contractDistance + kQueryMargin * scale);
        const bool ok = d <= contractDistance && minArm >= kMinArmLength && maxArm <= tuning.maxArmLength * scale && area >= tuning.minArea * scale2 &&
                        area < tuning.maxArea * scale2 && aspect >= tuning.minAspectRatio && !folded;
        if (!ok) {
            keep[i] = 0;
            contracted++;
        }
    }
    return contracted;
}

int SpringLevelSet3D::contract()
{
    std::vector<uint8_t> keep(mManifold.springlParticles.size(), 1);
    const int contracted = markContractions(keep, [](size_t) { return true; });
    if (contracted > 0) compactSpringls(keep);
    return contracted;
}

int SpringLevelSet3D::fill()
{
    // Gap tests see the springls that exist before this pass (springls are appended afterwards),
    // so the springl arrays are read in place.
    refreshSpringlGrid();
    const SurfaceMesh3D &iso = mManifold.isoSurface;
    const int K = elementSize();
    const int F = static_cast<int>(iso.faceCount());
    const auto faceScale = [&](int f) {
        float scale = std::numeric_limits<float>::max();
        for (int k = 0; k < K; k++) scale = std::min(scale, contourScale(iso.face(f)[k]));
        return scale;
    };
    std::vector<uint8_t> isGap(F, 0);
    SPRINGLS_OMP(omp parallel for if (F >= kParallelMinHeavyItems))
    for (int f = 0; f < F; f++) {
        std::array<Vec3, kMaxElementSize> verts;
        for (int k = 0; k < K; k++) verts[k] = iso.vertexes[iso.face(f)[k]];
        isGap[f] = !mSpringlGrid.anyElementWithin(mManifold.springlParticles, mManifold.springlVertexes, Centroid(verts.data(), K),
                                                  tuning.fillDistance * faceScale(f), [](size_t) { return true; });
    }
    int filled = 0;
    for (int f = 0; f < F; f++) {
        if (!isGap[f]) continue;
        std::array<Vec3, kMaxElementSize> verts;
        for (int k = 0; k < K; k++) verts[k] = iso.vertexes[iso.face(f)[k]];
        appendSpringl(Centroid(verts.data(), K), std::span<const Vec3>(verts.data(), K), faceScale(f));
        filled++;
    }
    return filled;
}

void SpringLevelSet3D::fixNormals()
{
    ProfileScope profileScope("fixNormals");
    Manifold3D &m = mManifold;
    const int N = static_cast<int>(m.springlParticles.size());
    SPRINGLS_OMP(omp parallel for if (N >= kParallelMinHeavyItems))
    for (int n = 0; n < N; n++) {
        Vec3 &norm = m.springlParticleNormals[n];
        if (norm.dot(levelSetGradient(m.springlParticles[n])) < 0.0f) {
            norm = -norm;
            // Reverse the vertex cycle.
            std::reverse(m.springlVertexes.begin() + m.springlStart[n], m.springlVertexes.begin() + m.springlStart[n + 1]);
        }
    }
}

void SpringLevelSet3D::snapshotForTracking()
{
    mOldVertexes = mManifold.springlVertexes;
    mOldVertexSpringl = mManifold.vertexSpringl;
    mOldParticles = mManifold.springlParticles;
    mOldNormals = mManifold.springlParticleNormals;
    mOldCorrespondences = mManifold.correspondence;
    mOldVelocities = mManifold.velocities;
}

Vec3 SpringLevelSet3D::traceInitial(Vec3 pt) const
{
    // Gradient descent on |phi0| back onto the initial interface.
    constexpr float kTimeStep = 0.5f;
    constexpr int kMaxIterations = 30;
    const VolumeF &phi = mInitialLevelSet;
    const auto a = [&](const Vec3 &p) { return std::abs(phi.interpolate(p)); };
    for (int iter = 0; iter < kMaxIterations; iter++) {
        const Vec3 grad = a(pt) * Vec3(0.5f * (a(pt + Vec3::UnitX()) - a(pt - Vec3::UnitX())), 0.5f * (a(pt + Vec3::UnitY()) - a(pt - Vec3::UnitY())),
                                       0.5f * (a(pt + Vec3::UnitZ()) - a(pt - Vec3::UnitZ())));
        pt -= kTimeStep * grad;
        if (grad.norm() <= 1E-5f) break;
    }
    return pt;
}

void SpringLevelSet3D::updateTracking(float maxDistance)
{
    ProfileScope profileScope("updateTracking");
    // New springls (no correspondence yet) inherit correspondence and velocity from old springls
    // whose arm vertexes are near theirs and that face the same way.
    if (mOldVertexes.empty()) return;
    PointGrid3D oldVertexGrid;
    oldVertexGrid.build(mOldVertexes, maxDistance);
    std::vector<PointGrid3D::Neighbor> result;
    Manifold3D &m = mManifold;
    for (size_t pid = 0; pid < m.springlParticles.size(); pid++) {
        if (!(std::isinf(m.correspondence[pid].x()) || m.springlParticleTracking[pid] < 0)) continue;
        const Vec3 pt = m.springlParticles[pid];
        const Vec3 norm = m.springlParticleNormals[pid];
        Vec3 velocity = Vec3::Zero();
        Vec3 q = Vec3::Zero();
        int count = 0;
        float dmin = 1E30f;
        int bestMatch = -1;
        for (uint32_t e = m.springlStart[pid]; e < m.springlStart[pid + 1]; e++) {
            oldVertexGrid.within(m.springlVertexes[e], maxDistance, result);
            for (const PointGrid3D::Neighbor &nbr : result) {
                const size_t qid = mOldVertexSpringl[nbr.id];
                if (!canTrackFrom(pid, qid) || !(norm.dot(mOldNormals[qid]) > kPlaneThreshold)) continue;
                const float d = distance(mOldParticles[qid], pt);
                if (d < dmin) {
                    bestMatch = static_cast<int>(qid);
                    dmin = d;
                }
                const Vec3 &oldQ = mOldCorrespondences[qid];
                if (!std::isinf(oldQ.x())) {
                    q += oldQ;
                    velocity += mOldVelocities[qid];
                    count++;
                    break;
                }
            }
        }
        if (m.springlParticleTracking[pid] < 0 && bestMatch >= 0) m.springlParticleTracking[pid] = bestMatch;
        if (count > 0) {
            m.velocities[pid] = velocity / static_cast<float>(count);
            m.correspondence[pid] = traceInitial(q / static_cast<float>(count));
        }
    }
}
}  // namespace springls
