#include <core/Profile.hpp>
#include <dweave/SemiLagrangianDweaveLevelSet3D.hpp>

#include <springl/SpringlSignField3D.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>

namespace springls
{
namespace
{
// Without a schedule (or when nothing else bounds the step) a sub-step is at most this long.
constexpr float kFallbackMaxTimeStep = 1.0f;
using Block = AdaptiveGrid3D::Block;

// Whether the ray from p along `dir` crosses triangle (a, b, c) (Moller-Trumbore).
bool RayHitsTriangle(const Vec3 &p, const Vec3 &dir, const Vec3 &a, const Vec3 &b, const Vec3 &c)
{
    const Vec3 e1 = b - a, e2 = c - a;
    const Vec3 h = dir.cross(e2);
    const float det = e1.dot(h);
    if (std::abs(det) < 1E-12f) return false;
    const float inv = 1.0f / det;
    const Vec3 s = p - a;
    const float u = inv * s.dot(h);
    if (u < 0.0f || u > 1.0f) return false;
    const Vec3 q = s.cross(e1);
    const float v = inv * dir.dot(q);
    if (v < 0.0f || u + v > 1.0f) return false;
    return inv * e2.dot(q) > 0.0f;
}
}  // namespace

SemiLagrangianDweaveLevelSet3D::SemiLagrangianDweaveLevelSet3D(ElementType elementType) : AdaptiveDweaveLevelSet3D(elementType)
{
    mName = "Semi-Lagrangian DoubleWeave 3D";
}

bool SemiLagrangianDweaveLevelSet3D::step()
{
    // The strain estimate reads attraction neighbors; splitting springls clears them.
    if (mNearestNeighbors.size() != mManifold.springlVertexes.size()) updateNearestNeighbors();
    return AdaptiveDweaveLevelSet3D::step();
}

void SemiLagrangianDweaveLevelSet3D::evolveGrid()
{
    ProfileScope profileScope("rebuild (SL evolveGrid)");
    mManifold.updateNormals();  // refineContour matches iso-surface vertexes to springls by normal
    SignFieldInput3D input;
    input.grid = &mGrid;
    input.particles = &mManifold.springlParticles;
    input.vertexes = &mManifold.springlVertexes;
    input.start = &mManifold.springlStart;
    for (size_t n = 0; n < mManifold.springlParticles.size(); n++) {
        if (levelSetIncludes(n)) input.springls.push_back(static_cast<uint32_t>(n));
    }
    float finestCell = std::numeric_limits<float>::infinity();
    for (int leaf : mGrid.leaves()) finestCell = std::min(finestCell, mGrid.blocks()[leaf].h);
    SignFieldParams3D params;
    params.capsuleCells = slTuning.capsuleCells;
    params.distanceClamp = std::min(adaptiveTuning.bandWidth, kDistanceCells * (std::isfinite(finestCell) ? finestCell : 1.0f));
    SpringlSignField3D field;
    field.build(input, params);
    FillGridFromSignField(mGrid, field, adaptiveTuning.bandWidth, params.distanceClamp);
    mGrid.synchronize();
    mGrid.updateActive();
    extractGridSurface();
    if (eraseTinyComponents() > 0) {
        mGrid.synchronize();
        mGrid.updateActive();
        extractGridSurface();
    }
}

int SemiLagrangianDweaveLevelSet3D::eraseTinyComponents()
{
    ProfileScope profileScope("rebuild: eraseTinyComponents");
    const SurfaceMesh3D &mesh = mManifold.isoSurface;
    const size_t F = mesh.faceCount();
    const int K = mesh.elementSize();
    if (F == 0) return 0;
    // Components: vertexes joined through faces.
    std::vector<uint32_t> parent(mesh.vertexes.size());
    std::iota(parent.begin(), parent.end(), 0u);
    const auto find = [&](uint32_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    for (size_t f = 0; f < F; f++) {
        const uint32_t *face = mesh.face(f);
        for (int k = 1; k < K; k++) {
            const uint32_t a = find(face[0]), b = find(face[k]);
            if (a != b) parent[std::max(a, b)] = std::min(a, b);
        }
    }
    struct Component {
        std::vector<size_t> faces;
        double volume = 0.0;
        float h = std::numeric_limits<float>::infinity();
        Vec3 lo = Vec3::Constant(std::numeric_limits<float>::infinity()), hi = Vec3::Constant(-std::numeric_limits<float>::infinity());
    };
    std::unordered_map<uint32_t, Component> components;
    for (size_t f = 0; f < F; f++) {
        const uint32_t *face = mesh.face(f);
        Component &c = components[find(face[0])];
        c.faces.push_back(f);
        // Fan triangles about the first vertex: the divergence theorem gives the enclosed volume.
        for (int k = 1; k + 1 < K; k++) {
            const Vec3 &a = mesh.vertexes[face[0]], &b = mesh.vertexes[face[k]], &d = mesh.vertexes[face[k + 1]];
            c.volume += static_cast<double>(a.dot(b.cross(d))) / 6.0;
        }
        for (int k = 0; k < K; k++) {
            c.h = std::min(c.h, contourScale(face[k]));
            c.lo = c.lo.cwiseMin(mesh.vertexes[face[k]]);
            c.hi = c.hi.cwiseMax(mesh.vertexes[face[k]]);
        }
    }
    // A slightly skewed ray direction avoids grazing mesh edges and vertexes exactly.
    const Vec3 dir = Vec3(1.0f, 0.00137f, 0.00291f).normalized();
    auto &blocks = mGrid.blocks();
    int erased = 0;
    for (const auto &entry : components) {
        const auto &c = entry.second;  // not a structured binding: the lambdas below capture it (OpenMP in Clang)
        if (std::abs(c.volume) >= static_cast<double>(slTuning.minComponentCells) * c.h * c.h * c.h) continue;
        const auto enclosed = [&](const Vec3 &p) {
            bool inside = false;
            for (size_t f : c.faces) {
                const uint32_t *face = mesh.face(f);
                for (int k = 1; k + 1 < K; k++) {
                    if (RayHitsTriangle(p, dir, mesh.vertexes[face[0]], mesh.vertexes[face[k]], mesh.vertexes[face[k + 1]])) inside = !inside;
                }
            }
            return inside;
        };
        std::vector<std::pair<int, int>> nodes;  // (block, node) enclosed by the component
        for (int leaf : mGrid.leaves()) {
            const Block &block = blocks[leaf];
            const Vec3 blockHi = block.origin + Vec3::Constant(block.size());
            if ((block.origin.array() > c.hi.array()).any() || (blockHi.array() < c.lo.array()).any()) continue;
            for (int z = 0; z < AdaptiveGrid3D::kBlockNodes; z++) {
                for (int y = 0; y < AdaptiveGrid3D::kBlockNodes; y++) {
                    for (int x = 0; x < AdaptiveGrid3D::kBlockNodes; x++) {
                        const Vec3 q = block.nodePosition(x, y, z);
                        if ((q.array() < c.lo.array()).any() || (q.array() > c.hi.array()).any() || !enclosed(q)) continue;
                        nodes.emplace_back(leaf, Block::index(x, y, z));
                    }
                }
            }
        }
        // The component's interior sign is the majority sign of the nodes it encloses.
        int vote = 0;
        for (const auto &[leaf, node] : nodes) vote += blocks[leaf].value(node) < 0.0f ? -1 : 1;
        if (vote == 0) continue;
        const bool interiorInside = vote < 0;
        for (const auto &[leaf, node] : nodes) {
            Block &block = blocks[leaf];
            const float v = block.value(node);
            if ((v < 0.0f) != interiorInside) continue;
            block.materialize();
            block.phi[node] = -v;
        }
        erased++;
    }
    return erased;
}

float SemiLagrangianDweaveLevelSet3D::advectionTimeStep(const Points3 &k1, float) const
{
    ProfileScope profileScope("advectionTimeStep");
    const Manifold3D &m = mManifold;
    const size_t N = m.springlParticles.size();
    const Points3 &particles = m.springlParticles;
    const Points3 &vertexes = m.springlVertexes;
    const Points3 &normals = m.springlParticleNormals;
    const double t = mSimulationTime;
    const float maxStepCells = slTuning.maxStepCells;
    const bool haveNormals = normals.size() == N;
    // Neighbor pairs need stage-1 velocities for every springl and current neighbor links.
    const bool pairs = haveNormals && k1.size() == N && mNearestNeighbors.size() == vertexes.size();
    // Primal and dual constellations do not attract each other, so their divergence is measured
    // separately: each particle against the nearest springl of the other constellation.
    const bool crossPairs = haveNormals && k1.size() == N && m.springlConstellations.size() == N;
    SpringlHashGrid3D springlGrid;
    if (crossPairs) springlGrid.update(particles, vertexes, m.springlStart);
    const float crossReach = 2.0f * tuning.maxArmLength;
    // Only same-facing pairs measure stretching: opposite-facing pairs are fronts colliding (or a
    // sheet's two sides), whose relative speed says nothing about how far the springls spread.
    const auto sameFacing = [&](size_t a, size_t b) { return normals[a].dot(normals[b]) > 0.0f; };
    const auto pairStrain = [&](size_t i, size_t j) {
        const float gap = std::max(distance(particles[i], particles[j]), kMinPairGapScale * springlScale(i));
        return (k1[i] - k1[j]).norm() / gap;
    };
    float maxStrain = 0.0f;
    float timeStep = std::numeric_limits<float>::infinity();
    const int count = static_cast<int>(N);
    SPRINGLS_OMP(omp parallel for reduction(max : maxStrain) reduction(min : timeStep) if (count >= kParallelMinHeavyItems))
    for (int i = 0; i < count; i++) {
        const size_t s = static_cast<size_t>(i);
        const Vec3 &p = particles[s];
        const Vec3 vp = (s < k1.size()) ? k1[s] : particleVelocity(p, t);
        // Stretch within the springl (each arm)...
        const uint32_t begin = m.springlStart[s], end = m.springlStart[s + 1];
        for (uint32_t v = begin; v < end; v++) {
            const float arm = distance(vertexes[v], p);
            if (arm > 0.0f) maxStrain = std::max(maxStrain, (particleVelocity(vertexes[v], t) - vp).norm() / arm);
        }
        // ...between attraction neighbors...
        if (pairs) {
            for (uint32_t v = begin; v < end; v++) {
                for (const SpringlEdge &e : mNearestNeighbors[v]) {
                    if (e.springl != s && sameFacing(s, e.springl)) maxStrain = std::max(maxStrain, pairStrain(s, e.springl));
                }
            }
        }
        // ...and against the nearest springl of the other constellation.
        if (crossPairs) {
            const uint8_t mine = m.springlConstellations[s];
            std::vector<SpringlEdge> nearestOther;
            springlGrid.nearestEdges(
                particles, vertexes, p, crossReach * springlScale(s), s, 1, [&](size_t o) { return m.springlConstellations[o] != mine && sameFacing(s, o); },
                nearestOther);
            if (!nearestOther.empty()) maxStrain = std::max(maxStrain, pairStrain(s, nearestOther.front().springl));
        }
        const float speed = vp.norm();
        if (speed > 0.0f && std::isfinite(maxStepCells)) timeStep = std::min(timeStep, maxStepCells * springlScale(s) / speed);
    }
    if (maxStrain > 0.0f) timeStep = std::min(timeStep, slTuning.maxStrain / maxStrain);
    if (!mSchedule.weight) timeStep = std::min(timeStep, kFallbackMaxTimeStep);
    // Nothing bounded the step (no strain, no maxStepCells), e.g. a rigid translation: fall back to
    // the unscheduled cap, or advect would move by inf. advect() still applies the schedule's cap.
    if (!std::isfinite(timeStep)) timeStep = kFallbackMaxTimeStep;
    return timeStep;
}
}  // namespace springls
