#include "Check.hpp"

#include <mesh/MeshDistance3D.hpp>
#include <mesh/MeshSplat3D.hpp>
#include <springl/AdaptiveSpringLevelSet3D.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <vector>

using namespace springls;

namespace
{
constexpr float kBand = 3.0f;
constexpr int kDepth = 3;
const Vec3 kCenter(16.2f, 16.1f, 15.9f);

struct TriangleMesh {
    Points3 positions;
    std::vector<std::array<uint32_t, 3>> triangles;
};

// Icosahedron subdivided `levels` times (20 * 4^levels faces), vertexes on the sphere; outward
// wound unless `inward`.
void AddIcosphere(TriangleMesh &mesh, const Vec3 &center, float radius, int levels, bool inward)
{
    const float t = (1.0f + std::sqrt(5.0f)) / 2.0f;
    Points3 v = {{-1, t, 0},  {1, t, 0},  {-1, -t, 0}, {1, -t, 0}, {0, -1, t},  {0, 1, t},
                 {0, -1, -t}, {0, 1, -t}, {t, 0, -1},  {t, 0, 1},  {-t, 0, -1}, {-t, 0, 1}};
    std::vector<std::array<uint32_t, 3>> f = {{0, 11, 5},  {0, 5, 1},  {0, 1, 7},  {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
                                              {11, 10, 2}, {10, 7, 6}, {7, 1, 8},  {3, 9, 4},  {3, 4, 2},   {3, 2, 6}, {3, 6, 8},
                                              {3, 8, 9},   {4, 9, 5},  {2, 4, 11}, {6, 2, 10}, {8, 6, 7},   {9, 8, 1}};
    for (Vec3 &p : v) p.normalize();
    for (int level = 0; level < levels; level++) {
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> midpoints;
        const auto mid = [&](uint32_t a, uint32_t b) {
            const auto key = std::make_pair(std::min(a, b), std::max(a, b));
            const auto it = midpoints.find(key);
            if (it != midpoints.end()) return it->second;
            v.push_back((v[a] + v[b]).normalized());
            return midpoints[key] = static_cast<uint32_t>(v.size() - 1);
        };
        std::vector<std::array<uint32_t, 3>> next;
        for (const auto &tri : f) {
            const uint32_t ab = mid(tri[0], tri[1]), bc = mid(tri[1], tri[2]), ca = mid(tri[2], tri[0]);
            next.push_back({tri[0], ab, ca});
            next.push_back({tri[1], bc, ab});
            next.push_back({tri[2], ca, bc});
            next.push_back({ab, bc, ca});
        }
        f = std::move(next);
    }
    const uint32_t base = static_cast<uint32_t>(mesh.positions.size());
    for (const Vec3 &p : v) mesh.positions.push_back(center + radius * p);
    for (const auto &tri : f) {
        if (inward) {
            mesh.triangles.push_back({base + tri[0], base + tri[2], base + tri[1]});
        } else {
            mesh.triangles.push_back({base + tri[0], base + tri[1], base + tri[2]});
        }
    }
}

// Closed, outward-wound axis-aligned box.
void AddBox(TriangleMesh &mesh, const Vec3 &lo, const Vec3 &hi)
{
    const uint32_t base = static_cast<uint32_t>(mesh.positions.size());
    for (int c = 0; c < 8; c++) mesh.positions.push_back(Vec3((c & 1) ? hi.x() : lo.x(), (c & 2) ? hi.y() : lo.y(), (c & 4) ? hi.z() : lo.z()));
    // Quads counterclockwise about the outward normal: -x, +x, -y, +y, -z, +z.
    const uint32_t quads[6][4] = {{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}};
    for (const auto &q : quads) {
        mesh.triangles.push_back({base + q[0], base + q[1], base + q[2]});
        mesh.triangles.push_back({base + q[0], base + q[2], base + q[3]});
    }
}

std::shared_ptr<MeshDistance3D> Build(const TriangleMesh &mesh)
{
    auto distance = std::make_shared<MeshDistance3D>();
    CHECK(distance->build(mesh.positions, mesh.triangles));
    return distance;
}

// The ball of radius 6.3 used by several tests (1280 faces).
TriangleMesh Sphere()
{
    TriangleMesh mesh;
    AddIcosphere(mesh, kCenter, 6.3f, 3, false);
    return mesh;
}

std::unique_ptr<AdaptiveDweaveLevelSet3D> InitSim(std::shared_ptr<const MeshDistance3D> surface, const VolumeF &base)
{
    auto sim = std::make_unique<AdaptiveDweaveLevelSet3D>();
    sim->adaptiveTuning.maxDepth = kDepth;
    sim->adaptiveTuning.bandWidth = kBand;
    sim->setInitialLevelSet(base);
    sim->setInitialSurface(std::move(surface));
    sim->init();
    return sim;
}

std::unique_ptr<AdaptiveDweaveLevelSet3D> InitSim(std::shared_ptr<const MeshDistance3D> surface)
{
    return InitSim(std::move(surface), VolumeF(33, 33, 33, kBand));
}

// The r0 that initLevelSet() passes to SplatMeshIntoGrid (bandMargin() + sqrt(3) 2^-maxDepth).
float SplatRadius(const AdaptiveSpringLevelSet3D &sim)
{
    const float h = std::ldexp(1.0f, -sim.adaptiveTuning.maxDepth);
    return sim.tuning.maxStepVoxels + sim.adaptiveTuning.bandCells * h + std::sqrt(3.0f) * h;
}

bool SameSign(float a, float b) { return (a < 0.0f) == (b < 0.0f); }

// Calls visit(block, value, position) for every corner of every leaf, and visitTile(block) for tiles.
template <class Corner, class Tile>
void ForEachLeaf(const AdaptiveGrid3D &grid, Corner &&visit, Tile &&visitTile)
{
    constexpr int M = AdaptiveGrid3D::kBlockNodes;
    for (int index : grid.leaves()) {
        const AdaptiveGrid3D::Block &block = grid.blocks()[index];
        if (block.isTile()) {
            visitTile(block);
            continue;
        }
        for (int c = 0; c < M; c++) {
            for (int b = 0; b < M; b++) {
                for (int a = 0; a < M; a++) visit(block, block.at(a, b, c), block.nodePosition(a, b, c));
            }
        }
    }
}

Vec3 BlockCenter(const AdaptiveGrid3D::Block &block) { return block.origin + Vec3::Constant(0.5f * block.size()); }
}  // namespace

TEST(candidate_query_matches_bvh)
{
    const TriangleMesh mesh = Sphere();
    const auto distance = Build(mesh);
    CHECK(distance->faceCount() == mesh.triangles.size());
    std::vector<uint32_t> all(distance->faceCount());
    std::iota(all.begin(), all.end(), 0u);
    std::mt19937 random(7);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    int mismatches = 0;
    for (int n = 0; n < 2000; n++) {
        Vec3 dir(unit(random), unit(random), unit(random));
        if (dir.squaredNorm() < 1e-4f) continue;
        const Vec3 p = kCenter + dir.normalized() * (6.3f + 2.0f * unit(random));
        uint32_t closest = 0;
        const auto candidate = distance->signedDistance(p, all, std::numeric_limits<float>::infinity(), &closest);
        if (!candidate || std::abs(*candidate - distance->signedDistance(p)) > 1e-5f) mismatches++;  // the two paths round differently
        if (candidate && distance->faceDistance(closest, p) != std::abs(*candidate)) mismatches++;
    }
    CHECK(mismatches == 0);
    // Nothing within maxDistance: no answer.
    CHECK(!distance->signedDistance(kCenter, all, 1.0f));
    CHECK(!distance->signedDistance(kCenter, std::span<const uint32_t>(), 100.0f));

    // faceBounds / faceNearBox agree with the geometry.
    Vec3 lo, hi;
    distance->faceBounds(0, lo, hi);
    CHECK((lo.array() <= hi.array()).all());
    CHECK(distance->faceNearBox(0, lo, hi, 0.0f));
    const Vec3 away = hi + Vec3::Constant(5.0f);
    CHECK(!distance->faceNearBox(0, away, away + Vec3::Constant(1.0f), 1.0f));
    CHECK(distance->faceNearBox(0, away, away + Vec3::Constant(1.0f), 5.0f * std::sqrt(3.0f) + 0.01f));
}

TEST(splat_matches_bvh)
{
    const auto distance = Build(Sphere());
    const auto sim = InitSim(distance);
    const AdaptiveGrid3D &grid = sim->getGrid();
    const float r0 = SplatRadius(*sim);
    CHECK(grid.finestDepth() == kDepth);
    int checked = 0, wrongValue = 0, wrongSign = 0, wrongTile = 0;
    float worst = 0.0f;
    ForEachLeaf(
        grid,
        [&](const AdaptiveGrid3D::Block &block, float value, const Vec3 &p) {
            const float d = distance->signedDistance(p, kBand);
            if (block.key.depth == kDepth && std::abs(d) < r0) {
                checked++;
                const float error = std::abs(value - std::clamp(d, -kBand, kBand));
                worst = std::max(worst, error);
                if (error > 1e-5f) wrongValue++;
            }
            if (std::abs(d) > 1e-4f && !SameSign(value, d)) wrongSign++;
        },
        [&](const AdaptiveGrid3D::Block &block) {
            if (!SameSign(block.tileValue, distance->signedDistance(BlockCenter(block)))) wrongTile++;
        });
    CHECK(checked > 1000);
    CHECK(wrongValue == 0);
    CHECK_NEAR(worst, 0.0, 1e-5);
    CHECK(wrongSign == 0);
    CHECK(wrongTile == 0);
}

TEST(splat_keeps_thin_slab)
{
    TriangleMesh mesh;
    AddBox(mesh, Vec3(4.0f, 4.0f, 16.35f), Vec3(28.0f, 28.0f, 16.65f));
    const auto sim = InitSim(Build(mesh));
    const SurfaceMesh3D &iso = sim->getManifold().isoSurface;
    CHECK(!iso.vertexes.empty());
    int inner = 0, off = 0;
    for (const Vec3 &v : iso.vertexes) {
        if (v.x() < 6.0f || v.x() > 26.0f || v.y() < 6.0f || v.y() > 26.0f) continue;
        inner++;
        if (std::min(std::abs(v.z() - 16.35f), std::abs(v.z() - 16.65f)) > 0.05f) off++;
    }
    CHECK(inner > 0);
    CHECK(off == 0);

    // Sampled at base voxels only, the slab falls between the samples.
    VolumeF slab(33, 33, 33);
    for (int k = 0; k < 33; k++) {
        for (int j = 0; j < 33; j++) {
            for (int i = 0; i < 33; i++) slab(i, j, k) = std::abs(static_cast<float>(k) - 16.5f) - 0.15f;
        }
    }
    const auto baseOnly = InitSim(nullptr, slab);
    CHECK(baseOnly->getManifold().isoSurface.vertexes.empty());
}

TEST(splat_sphere_accuracy)
{
    const TriangleMesh mesh = Sphere();
    const float radius = 6.3f;
    // How far the faces come inside the sphere.
    float deviation = 0.0f;
    for (const auto &t : mesh.triangles) {
        const float d = std::sqrt(TriangleDistanceSqr(mesh.positions[t[0]], mesh.positions[t[1]], mesh.positions[t[2]], kCenter));
        deviation = std::max(deviation, radius - d);
    }
    const auto sim = InitSim(Build(mesh));
    const SurfaceMesh3D &iso = sim->getManifold().isoSurface;
    CHECK(!iso.vertexes.empty());
    float worst = 0.0f;
    for (const Vec3 &v : iso.vertexes) worst = std::max(worst, std::abs((v - kCenter).norm() - radius));
    CHECK(worst < 0.03f + deviation);
    CHECK(sim->countSpringls(Constellation::Primal) > 0);
    CHECK(sim->countSpringls(Constellation::Dual) > 0);
}

TEST(splat_hollow_shell_cavity)
{
    TriangleMesh mesh;
    AddIcosphere(mesh, kCenter, 8.0f, 3, false);
    AddIcosphere(mesh, kCenter, 4.0f, 3, true);
    const auto distance = Build(mesh);
    // Tiles (and corners) in the cavity and outside are positive, in the shell negative. Returns how many
    // coarse (flood-signed) leaves lie in the cavity.
    const auto check = [&](const AdaptiveGrid3D &grid) {
        int wrong = 0, outsideTiles = 0, cavityCoarse = 0;
        ForEachLeaf(
            grid,
            [&](const AdaptiveGrid3D::Block &, float value, const Vec3 &p) {
                const float r = (p - kCenter).norm();
                if ((r < 3.0f || r > 9.0f) && value <= 0.0f) wrong++;
                if (r > 5.0f && r < 7.0f && value >= 0.0f) wrong++;
            },
            [&](const AdaptiveGrid3D::Block &block) {
                const float r = (BlockCenter(block) - kCenter).norm();
                if ((r < 3.0f || r > 9.0f) && block.tileValue <= 0.0f) wrong++;
                if (r > 5.0f && r < 7.0f && block.tileValue >= 0.0f) wrong++;
                if (r > 9.0f) outsideTiles++;
            });
        for (int leaf : grid.leaves()) {
            const AdaptiveGrid3D::Block &block = grid.blocks()[leaf];
            if (block.key.depth < kDepth && (BlockCenter(block) - kCenter).norm() < 3.0f) cavityCoarse++;
        }
        CHECK(wrong == 0);
        CHECK(outsideTiles > 0);
        return cavityCoarse;
    };
    const auto sim = InitSim(distance);
    check(sim->getGrid());

    // A narrower splat leaves coarse leaves in the cavity, which only the flood signs.
    AdaptiveGrid3D grid;
    grid.initialize(VolumeF(33, 33, 33, kBand), kDepth, kBand);
    const MeshSplatStats stats = SplatMeshIntoGrid(grid, *distance, 0.3f, kBand);
    CHECK(stats.refinedLeaves > 0);
    CHECK(stats.candidateEntries > 0);
    CHECK(stats.floodComponents >= 3);  // outside, the shell's interior, the cavity
    CHECK(stats.fallbackCorners == 0);
    CHECK(check(grid) > 0);
}

// Deep grids (the viewer picks the depth from the triangle count): exact values where the evolution uses them,
// no full-mesh query per corner, and the active set of exact values, including at depths where the r0 that
// initLevelSet() passes falls below the active threshold of a finest leaf.
TEST(splat_deep_grids)
{
    const Vec3 center(4.1f, 4.05f, 3.95f);
    for (int depth : {6, 8}) {
        TriangleMesh mesh;
        AddIcosphere(mesh, center, depth == 6 ? 0.6f : 0.1f, 2, false);
        const auto distance = Build(mesh);
        AdaptiveGrid3D grid;
        grid.setActiveBand(3.0f, 0.35f);  // AdaptiveTuning's activeCells, activeMargin
        grid.initialize(VolumeF(9, 9, 9, kBand), depth, kBand);
        const float h = std::ldexp(1.0f, -depth);
        const float r0 = SpringLevelSet3D::Tuning().maxStepVoxels + 5.0f * h + std::sqrt(3.0f) * h;
        const float threshold = grid.activeThreshold(h);
        const MeshSplatStats stats = SplatMeshIntoGrid(grid, *distance, r0, kBand);
        CHECK(grid.finestDepth() == depth);
        CHECK(stats.fallbackCorners == 0);
        int checked = 0, wrongValue = 0, wrongSign = 0, inactive = 0;
        for (int leaf : grid.leaves()) {
            const AdaptiveGrid3D::Block &block = grid.blocks()[leaf];
            // Leaves with no corner within r0 or the active threshold only need their signs (block corners); at
            // depth 8 the near leaves are sampled at every other corner to keep the test quick.
            const float halfDiagonal = 0.5f * std::sqrt(3.0f) * block.size();
            const float centerDistance = std::abs(distance->signedDistance(BlockCenter(block), kBand));
            const int stride = (centerDistance - halfDiagonal > std::max(r0, threshold)) ? 8 : (depth == 8 ? 2 : 1);
            float nearest = std::numeric_limits<float>::infinity();
            for (int c = 0; c < AdaptiveGrid3D::kBlockNodes; c += stride) {
                for (int b = 0; b < AdaptiveGrid3D::kBlockNodes; b += stride) {
                    for (int a = 0; a < AdaptiveGrid3D::kBlockNodes; a += stride) {
                        const float d = distance->signedDistance(block.nodePosition(a, b, c), kBand);
                        const float value = block.at(a, b, c);
                        nearest = std::min(nearest, std::abs(d));
                        if (std::abs(d) > 1e-4f && !SameSign(value, d)) wrongSign++;
                        if (block.key.depth == depth && std::abs(d) < r0) {
                            checked++;
                            if (std::abs(value - d) > 1e-5f) wrongValue++;
                        }
                    }
                }
            }
            if (nearest < threshold - 0.01f && !block.active) inactive++;
        }
        CHECK(depth != 8 || r0 < threshold);  // the case max(r0, threshold) exists for
        CHECK(checked > 1000);
        CHECK(wrongValue == 0);
        CHECK(wrongSign == 0);
        CHECK(inactive == 0);
    }
}
