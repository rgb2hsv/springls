// BuildMeshletHierarchy: meshlet-optimizer's LOD hierarchy of a closed mesh, as springls arrays.
#include "Check.hpp"

#include <lod/MeshletHierarchy.hpp>
#include <lod/MeshletUvChecks.hpp>
#include <mesh/Icosphere.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdio>
#include <map>
#include <utility>

using namespace springls;

namespace
{
// Watertight: every directed edge has as many opposite ones (a few non-manifold edges, where four triangles
// meet, are allowed: meshlet-optimizer's levels can have them, and MeshDistance3D handles them).
// Returns the number of directed edges used more than once, or -1 when the mesh is not watertight.
int NonManifoldEdges(const MeshletLevel &level)
{
    std::map<std::pair<uint32_t, uint32_t>, int> directed;
    for (const auto &t : level.triangles) {
        for (int e = 0; e < 3; e++) directed[{t[e], t[(e + 1) % 3]}]++;
    }
    int repeated = 0;
    for (const auto &[edge, count] : directed) {
        const auto reverse = directed.find({edge.second, edge.first});
        if (reverse == directed.end() || reverse->second != count) return -1;
        repeated += count > 1;
    }
    return repeated;
}
}  // namespace

// Unit icosphere, 4 subdivisions (5120 triangles): builds into 2..6 levels.
TEST(meshlet_hierarchy_unit_sphere_levels)
{
    const LoopMesh sphere = MakeIcosphere(4);
    const auto project = [](const Vec3 &p) { return Vec3(p.normalized()); };
    const auto start = std::chrono::steady_clock::now();
    const std::vector<MeshletLevel> levels = BuildMeshletHierarchy(sphere, MeshletHierarchyParams{}, project);
    std::printf("  %zu levels in %.2f s\n", levels.size(), std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    CHECK(levels.size() >= 2 && levels.size() <= 6);
    for (size_t k = 0; k < levels.size(); k++) {
        const MeshletLevel &L = levels[k];
        std::printf("  level %zu: %zu vertexes, %zu triangles\n", k, L.vertexes.size(), L.triangles.size());
        CHECK(!L.triangles.empty() && L.faceRegion.size() == L.triangles.size() && L.vertexRegion.size() == L.vertexes.size());
        const int nonManifold = NonManifoldEdges(L);
        CHECK(nonManifold >= 0 && nonManifold <= 8);
        // Interior vertexes are projected onto the sphere; border vertexes slide along border chords
        // (meshlet-optimizer's border relaxation), so they sit slightly inside it.
        for (size_t v = 0; v < L.vertexes.size(); v++) CHECK_NEAR(L.vertexes[v].norm(), 1.0, L.vertexRegion[v] < 0 ? 0.05 : 0.02);
        CHECK(L.regionParent.empty() == (k + 1 == levels.size()));
    }
    CHECK(!levels.empty() && levels.back().triangles.size() < levels.front().triangles.size());
}

TEST(meshlet_hierarchy_respects_max_depth)
{
    MeshletHierarchyParams params;
    params.maxDepth = 1;
    const std::vector<MeshletLevel> levels = BuildMeshletHierarchy(MakeIcosphere(4), params);
    CHECK(!levels.empty() && levels.size() <= 2);
}

namespace
{
const auto kProjectToUnitSphere = [](const Vec3 &p) { return Vec3(p.normalized()); };

// The tests compare UVs before and after the springs pull the triangles together, so the repulsion and the
// fit into the atlas (a uniform scale per region) are off unless asked for.
constexpr int kRelaxSteps = MeshletHierarchyParams().uvRelaxSteps;
std::vector<MeshletLevel> BuildUvs(int relaxSteps, bool fitToAtlas = false, bool repulsion = false)
{
    MeshletHierarchyParams params;
    params.uvRelaxSteps = relaxSteps;
    params.uvFitToAtlas = fitToAtlas;
    params.uvEnforceNoOverlaps = repulsion;
    return BuildMeshletHierarchy(MakeIcosphere(4), params, kProjectToUnitSphere);
}

struct Box {
    Vec2d lo = Vec2d::Constant(std::numeric_limits<double>::infinity()), hi = -Vec2d::Constant(std::numeric_limits<double>::infinity());
    void add(const Vec2d &p)
    {
        lo = lo.cwiseMin(p);
        hi = hi.cwiseMax(p);
    }
};
std::map<int, Box> RegionBoxes(const MeshletLevel &level, const std::vector<Vec2d> &uv)
{
    std::map<int, Box> boxes;
    for (size_t f = 0; f < level.triangles.size(); f++) {
        for (int c = 0; c < 3; c++) boxes[level.faceRegion[f]].add(uv[3 * f + c]);
    }
    return boxes;
}

}  // namespace

// Double-precision UVs: every triangle has its 3D angles and, within a region, the same UV/3D scale (areas
// proportional to 3D areas); none is flipped.
TEST(meshlet_hierarchy_uvs_are_isometric)
{
    const std::vector<MeshletLevel> levels = BuildUvs(kRelaxSteps);
    CHECK(!levels.empty());
    for (size_t k = 0; k < levels.size(); k++) {
        const MeshletLevel &level = levels[k];
        CHECK(level.texcoords.size() == 3 * level.triangles.size());
        if (level.texcoords.size() != 3 * level.triangles.size()) continue;
        const MeshletUvChecks checks = CheckMeshletUvs(level);
        std::printf("  level %zu: %zu flips, %zu overlapping pairs over one texel at 4096^2 (%zu across regions, deepest %.2g texels), worst angle "
                    "error %.2g rad, area ratio %.3g..%.3g of the median\n",
                    k, checks.flips, checks.overlaps, checks.crossRegionOverlaps, checks.maxOverlapTexels, checks.maxAngleError, checks.minAreaRatio,
                    checks.maxAreaRatio);
        CHECK(checks.triangles == level.triangles.size());
        CHECK(checks.flips == 0);
        CHECK(checks.maxAngleError < 1e-6);
        CHECK(std::abs(checks.minAreaRatio - 1.0) < 1e-6 && std::abs(checks.maxAreaRatio - 1.0) < 1e-6);
    }
}

// Corner correspondences: a corner pairs with its vertex's copy across each of its triangle's two edges, only
// within the same region.
TEST(meshlet_hierarchy_corner_pairs)
{
    const std::vector<MeshletLevel> levels = BuildUvs(0);
    CHECK(!levels.empty());
    const MeshletLevel &level = levels[0];
    const std::vector<MeshletCornerPair> pairs = MeshletCornerPairs(level);
    CHECK(!pairs.empty());
    std::map<std::pair<uint32_t, int>, int> perCorner;
    for (const MeshletCornerPair &p : pairs) {
        CHECK(p.faceA != p.faceB);
        CHECK(level.faceRegion[p.faceA] == level.faceRegion[p.faceB]);
        CHECK(level.triangles[p.faceA][p.cornerA] == level.triangles[p.faceB][p.cornerB]);
        perCorner[{p.faceA, p.cornerA}]++;
        perCorner[{p.faceB, p.cornerB}]++;
    }
    for (const auto &[corner, count] : perCorner) CHECK(count <= 2);
}

// The springs pull the corners that were one atlas vertex together: the mean gap between corresponding corners
// shrinks by at least a quarter from what the placement left (rigid triangles cannot close the gaps a curved
// chart needs; on the finest level the placement is already close).
TEST(meshlet_hierarchy_uv_springs_close_gaps)
{
    const std::vector<MeshletLevel> before = BuildUvs(0), after = BuildUvs(kRelaxSteps);
    CHECK(before.size() == after.size() && !before.empty());
    for (size_t k = 0; k < before.size() && k < after.size(); k++) {
        const double gapBefore = CheckMeshletUvs(before[k]).meanCornerGap, gapAfter = CheckMeshletUvs(after[k]).meanCornerGap;
        std::printf("  level %zu: corner gap %.3g -> %.3g\n", k, gapBefore, gapAfter);
        CHECK(gapAfter < 0.75 * gapBefore);
    }
}

// After the simulation each region is scaled uniformly and moved so it fits tightly inside its original atlas
// chart: inside its bounding box, touching it on one axis, centered on the other; charts then no longer overlap.
TEST(meshlet_hierarchy_uv_charts_fit_atlas)
{
    const std::vector<MeshletLevel> levels = BuildUvs(kRelaxSteps, true);
    CHECK(!levels.empty());
    for (size_t k = 0; k < levels.size(); k++) {
        const MeshletLevel &level = levels[k];
        CHECK(level.atlasTexcoords.size() == level.texcoords.size());
        if (level.atlasTexcoords.size() != level.texcoords.size()) continue;
        const std::map<int, Box> atlas = RegionBoxes(level, level.atlasTexcoords), fitted = RegionBoxes(level, level.texcoords);
        double worstOutside = 0.0, worstSlack = 0.0, worstOffCenter = 0.0;
        for (const auto &[region, a] : atlas) {
            const Box &b = fitted.at(region);
            const Vec2d size = a.hi - a.lo, fit = b.hi - b.lo;
            const double tol = 1e-9 * size.maxCoeff();
            worstOutside = std::max({worstOutside, (a.lo - b.lo).maxCoeff() - tol, (b.hi - a.hi).maxCoeff() - tol});
            worstSlack = std::max(worstSlack, 1.0 - std::max(fit.x() / size.x(), fit.y() / size.y()));
            worstOffCenter = std::max(worstOffCenter, ((a.lo + a.hi) - (b.lo + b.hi)).cwiseAbs().maxCoeff() / size.maxCoeff());
        }
        const MeshletUvChecks checks = CheckMeshletUvs(level);
        std::printf("  level %zu: outside %.2g, slack %.2g, off center %.2g, %zu overlaps across regions\n", k, worstOutside, worstSlack, worstOffCenter,
                    checks.crossRegionOverlaps);
        CHECK(worstOutside <= 0.0);
        CHECK(worstSlack < 1e-9);
        CHECK(worstOffCenter < 1e-9);
        CHECK(checks.crossRegionOverlaps == 0);
        CHECK(checks.maxAngleError < 1e-6);
    }
}

namespace
{
// Largest uniform scale at which `points`, turned by `angle`, fit in a box of `size`.
double ScaleAt(const std::vector<Vec2d> &points, double angle, const Vec2d &size)
{
    const double c = std::cos(angle), s = std::sin(angle);
    Vec2d lo = Vec2d::Constant(std::numeric_limits<double>::infinity()), hi = -lo;
    for (const Vec2d &p : points) {
        const Vec2d q(c * p.x() - s * p.y(), s * p.x() + c * p.y());
        lo = lo.cwiseMin(q);
        hi = hi.cwiseMax(q);
    }
    const Vec2d have = hi - lo;
    return std::min(have.x() > 0.0 ? size.x() / have.x() : 1e300, have.y() > 0.0 ? size.y() / have.y() : 1e300);
}
}  // namespace

// The fit also turns each chart: its uniform scale is the largest of eight turns (k * 22.5 degrees) allows,
// never below the unrotated fit.
TEST(meshlet_hierarchy_uv_charts_fit_rotation)
{
    const std::vector<MeshletLevel> relaxed = BuildUvs(kRelaxSteps), fitted = BuildUvs(kRelaxSteps, true);
    CHECK(relaxed.size() == fitted.size() && !relaxed.empty());
    for (size_t k = 0; k < relaxed.size() && k < fitted.size(); k++) {
        const MeshletLevel &a = relaxed[k], &b = fitted[k];
        const std::map<int, Box> atlas = RegionBoxes(b, b.atlasTexcoords);
        std::map<int, std::vector<Vec2d>> points;
        std::map<int, double> scale;  // fitted / relaxed size, from one edge of the region
        for (size_t f = 0; f < a.triangles.size(); f++) {
            for (int c = 0; c < 3; c++) points[a.faceRegion[f]].push_back(a.texcoords[3 * f + c]);
            scale.try_emplace(a.faceRegion[f], (b.texcoords[3 * f + 1] - b.texcoords[3 * f]).norm() / (a.texcoords[3 * f + 1] - a.texcoords[3 * f]).norm());
        }
        double worstVsBrute = 0.0, gainOverUnrotated = 0.0;
        for (const auto &[region, pts] : points) {
            const Vec2d size = atlas.at(region).hi - atlas.at(region).lo;
            double brute = 0.0;
            for (int i = 0; i < 8; i++) brute = std::max(brute, ScaleAt(pts, std::acos(-1.0) * i / 8.0, size));
            const double unrotated = ScaleAt(pts, 0.0, size);
            CHECK(scale[region] >= unrotated * (1.0 - 1e-9));
            worstVsBrute = std::max(worstVsBrute, 1.0 - scale[region] / brute);
            gainOverUnrotated = std::max(gainOverUnrotated, scale[region] / unrotated - 1.0);
        }
        std::printf("  level %zu: fitted scale short of the best of 8 turns by %.2g, up to %.1f%% larger than unrotated\n", k, worstVsBrute,
                    100.0 * gainOverUnrotated);
        CHECK(std::abs(worstVsBrute) < 1e-9);
    }
}

// The LSCM atlas (the isometric UVs' starting point) has no flipped triangles. The sphere has no stored normals,
// and meshlet-optimizer's computed normals point inward for counter-clockwise triangles; charts must still be
// laid out unmirrored (their orientation comes from their triangles' winding).
TEST(meshlet_hierarchy_atlas_has_no_flips)
{
    const std::vector<MeshletLevel> levels = BuildUvs(0);
    CHECK(!levels.empty());
    for (size_t k = 0; k < levels.size(); k++) {
        MeshletLevel atlas = levels[k];
        atlas.texcoords = levels[k].atlasTexcoords;
        const MeshletUvChecks checks = CheckMeshletUvs(atlas);
        std::printf("  level %zu atlas: %zu of %zu triangles flipped, %zu overlapping pairs over one texel at 4096^2\n", k, checks.flips, checks.triangles,
                    checks.overlaps);
        CHECK(checks.flips == 0);
    }
}

// With the repulsion, overlapping triangles push each other apart (impulses) until none overlap: the final UVs
// have no pairs overlapping by more than a texel at 4096^2, no flips, and every triangle keeps its 3D angles.
TEST(meshlet_hierarchy_uvs_do_not_overlap)
{
    const std::vector<MeshletLevel> levels = BuildUvs(kRelaxSteps, true, true);
    CHECK(!levels.empty());
    for (size_t k = 0; k < levels.size(); k++) {
        const MeshletUvChecks checks = CheckMeshletUvs(levels[k]);
        std::printf("  level %zu: %zu overlapping pairs (deepest %.2g), %zu flips, worst angle error %.2g rad, area ratio %.3g..%.3g, corner gap mean %.2g\n",
                    k, checks.overlaps, checks.maxOverlapTexels, checks.flips, checks.maxAngleError, checks.minAreaRatio, checks.maxAreaRatio,
                    checks.meanCornerGap);
        CHECK(checks.overlaps == 0);
        CHECK(checks.flips == 0);
        CHECK(checks.maxAngleError < 1e-6);
    }
}

// The overlap test (Geometric Tools' 2D triangle-triangle queries, ported into meshlet-optimizer): separated and
// touching triangles do not overlap, a known overlap has its depth, containment and clockwise input work.
TEST(meshlet_uv_overlap_depth)
{
    using T = std::array<Vec2d, 3>;
    const T a{Vec2d(0, 0), Vec2d(1, 0), Vec2d(0, 1)};
    CHECK(UvOverlapDepth(a, T{Vec2d(2, 0), Vec2d(3, 0), Vec2d(2, 1)}) == 0.0);    // apart
    CHECK(UvOverlapDepth(a, T{Vec2d(1, 0), Vec2d(1, 1), Vec2d(0, 1)}) == 0.0);    // sharing the edge (1,0)-(0,1)
    CHECK(UvOverlapDepth(a, T{Vec2d(0, 0), Vec2d(0, -1), Vec2d(-1, 0)}) == 0.0);  // sharing a vertex only
    // Shifted by 0.25 along x: the projections overlap least along the hypotenuse's normal (1, 1) / sqrt(2), by
    // (1 - 0.25) / sqrt(2) (along x by 0.75, along y by 1).
    const double shifted = UvOverlapDepth(a, T{Vec2d(0.25, 0), Vec2d(1.25, 0), Vec2d(0.25, 1)});
    std::printf("  shifted overlap depth %.17g\n", shifted);
    CHECK(std::abs(shifted - 0.75 / std::sqrt(2.0)) < 1e-12);
    // A small triangle inside a: overlaps.
    CHECK(UvOverlapDepth(a, T{Vec2d(0.1, 0.1), Vec2d(0.2, 0.1), Vec2d(0.1, 0.2)}) > 0.0);
    // Clockwise input gives the same answer.
    CHECK(std::abs(UvOverlapDepth(T{a[0], a[2], a[1]}, T{Vec2d(0.25, 0), Vec2d(0.25, 1), Vec2d(1.25, 0)}) - shifted) < 1e-15);
}
