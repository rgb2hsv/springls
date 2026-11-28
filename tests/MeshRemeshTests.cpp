// AdaptiveRemesh3D (edge splits, collapses and flips, by independent sets and chart by chart) and CvtCharts.
#include "Check.hpp"

#include <lod/AdaptiveRemesh3D.hpp>
#include <lod/CvtCharts.hpp>
#include <lod/TriangleSurface3D.hpp>
#include <mesh/Icosphere.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>

using namespace springls;

namespace
{
// A sphere of radius r about c as a mesh (icosphere).
TriangleSurface SphereMesh(const Vec3 &c, float r, int levels)
{
    const LoopMesh m = MakeIcosphere(levels, r, c.cast<double>());
    TriangleSurface mesh;
    for (const auto &p : m.vertices) mesh.vertexes.push_back(p);
    mesh.triangles = m.triangles;
    mesh.faceRegion.assign(mesh.triangles.size(), 0);
    return mesh;
}
bool Outward(const TriangleSurface &mesh)
{
    double volume = 0.0;
    for (const auto &t : mesh.triangles) volume += mesh.vertexes[t[0]].dot((mesh.vertexes[t[1]] - mesh.vertexes[t[0]]).cross(mesh.vertexes[t[2]] - mesh.vertexes[t[0]]));
    return volume >= 0.0;
}
bool Closed(const TriangleSurface &mesh)
{
    std::map<std::pair<uint32_t, uint32_t>, int> edges;
    for (const auto &t : mesh.triangles)
        for (int e = 0; e < 3; e++) edges[std::minmax(t[e], t[(e + 1) % 3])]++;
    for (const auto &[e, n] : edges)
        if (n != 2) return false;
    return !mesh.triangles.empty();
}
}  // namespace

namespace
{
float MaxEdge(const TriangleSurface &mesh)
{
    float longest = 0.0f;
    for (const auto &t : mesh.triangles)
        for (int e = 0; e < 3; e++) longest = std::max(longest, (mesh.vertexes[t[e]] - mesh.vertexes[t[(e + 1) % 3]]).norm());
    return longest;
}
}  // namespace

// Frozen triangles are left exactly as they are (and so are their vertexes); the rest is remeshed.
TEST(remesh_keeps_frozen_triangles)
{
    TriangleSurface mesh = SphereMesh(Vec3(0, 0, 0), 4.0f, 2);
    std::vector<uint8_t> frozen(mesh.triangles.size(), 0);
    for (size_t f = 0; f < frozen.size(); f++) frozen[f] = mesh.vertexes[mesh.triangles[f][0]].z() > 0.0f;
    std::vector<std::array<Vec3, 3>> before;
    for (size_t f = 0; f < frozen.size(); f++)
        if (frozen[f]) before.push_back({mesh.vertexes[mesh.triangles[f][0]], mesh.vertexes[mesh.triangles[f][1]], mesh.vertexes[mesh.triangles[f][2]]});
    RemeshParams params;
    params.targetEdge = 0.4f * MaxEdge(mesh);
    params.frozen = &frozen;
    CHECK(RemeshSurface(mesh, params));
    CHECK(frozen.size() == mesh.triangles.size());
    std::vector<std::array<Vec3, 3>> after;
    for (size_t f = 0; f < frozen.size(); f++)
        if (frozen[f]) after.push_back({mesh.vertexes[mesh.triangles[f][0]], mesh.vertexes[mesh.triangles[f][1]], mesh.vertexes[mesh.triangles[f][2]]});
    CHECK(before == after);
    CHECK(Closed(mesh) && Outward(mesh));
}

// Chart by chart (CVT charts, a new segmentation each round) keeps the mesh closed and converges to the whole-mesh
// remesher's result: an edge on a chart border waits for a segmentation that puts it inside a chart.
TEST(remesh_by_charts_converges)
{
    const TriangleSurface start = SphereMesh(Vec3(0, 0, 0), 4.0f, 3);
    RemeshParams params;
    params.targetEdge = 0.35f * MaxEdge(start);
    TriangleSurface whole = start;
    for (int round = 0; round < 4; round++) RemeshSurface(whole, params);
    TriangleSurface charted = start;
    int previousLong = 1 << 30;
    for (uint32_t seed = 1; seed <= 4; seed++) {
        CvtChartParams chartParams;
        chartParams.maxTrianglesPerChart = 128;
        chartParams.randomSeed = seed;
        const std::vector<int> charts = SegmentCvtCharts(charted.vertexes, charted.triangles, chartParams);
        CHECK(charts.size() == charted.triangles.size());
        charted.faceRegion = charts;
        RemeshSurfaceByCharts(charted, params);
        CHECK(Closed(charted) && Outward(charted));
        int longEdges = 0;
        for (const auto &t : charted.triangles)
            for (int e = 0; e < 3; e++) longEdges += (charted.vertexes[t[e]] - charted.vertexes[t[(e + 1) % 3]]).norm() > params.splitRatio * params.targetEdge;
        std::printf("  round %u: %zu triangles, %d long half-edges\n", seed, charted.triangles.size(), longEdges);
        CHECK(longEdges < previousLong || longEdges == 0);
        previousLong = longEdges;
    }
    std::printf("  whole: %zu triangles, longest %.3f; charts: %zu triangles, longest %.3f (limit %.3f)\n", whole.triangles.size(), MaxEdge(whole),
                charted.triangles.size(), MaxEdge(charted), params.splitRatio * params.targetEdge);
    CHECK(previousLong < 0.005 * 3.0 * double(charted.triangles.size()));
    CHECK(std::abs(double(charted.triangles.size()) - double(whole.triangles.size())) < 0.2 * whole.triangles.size());
}


namespace
{
int NonManifoldEdges(const TriangleSurface &mesh)
{
    std::map<std::pair<uint32_t, uint32_t>, int> edges;
    for (const auto &t : mesh.triangles)
        for (int e = 0; e < 3; e++) edges[std::minmax(t[e], t[(e + 1) % 3])]++;
    int bad = 0;
    for (const auto &[e, n] : edges) bad += n != 2;
    return bad;
}
}  // namespace

// A sphere stretched 3x along x is refined chart by chart, round after round, with the charts made once (as mesh
// tracking does): it stays a closed manifold, every chart keeps triangles, and every long edge is split in the end,
// on the chart borders too.
TEST(remesh_by_charts_keeps_charts_when_stretched)
{
    TriangleSurface mesh = SphereMesh(Vec3(0, 0, 0), 4.0f, 4);
    const float target = 0.5f * MaxEdge(mesh);
    for (Vec3 &p : mesh.vertexes) p.x() *= 3.0f;
    mesh.faceRegion = SegmentCvtCharts(mesh.vertexes, mesh.triangles, CvtChartParams{});
    CHECK(mesh.faceRegion.size() == mesh.triangles.size());
    const int charts = *std::max_element(mesh.faceRegion.begin(), mesh.faceRegion.end()) + 1;
    RemeshParams params;
    params.targetEdge = target;
    for (uint32_t round = 1; round <= 8; round++) {
        RemeshSurfaceByCharts(mesh, params);
        const int bad = NonManifoldEdges(mesh);
        std::vector<int> count(charts, 0);
        for (int c : mesh.faceRegion) count[c]++;
        const int empty = static_cast<int>(std::count(count.begin(), count.end(), 0));
        std::printf("  round %u: %zu triangles, %d non-manifold edges, %d of %d charts empty, longest %.3f (limit %.3f)\n", round,
                    mesh.triangles.size(), bad, empty, charts, MaxEdge(mesh), params.splitRatio * target);
        CHECK(bad == 0 && empty == 0);
        CHECK(mesh.faceRegion.size() == mesh.triangles.size());
    }
    CHECK(MaxEdge(mesh) <= params.splitRatio * target * 1.001f);
}

namespace
{
// Total length of the edges between triangles of different charts.
double ChartBorderLength(const TriangleSurface &mesh, const std::vector<int> &chart)
{
    std::map<std::pair<uint32_t, uint32_t>, std::vector<int>> edges;
    for (size_t f = 0; f < mesh.triangles.size(); f++) {
        const auto &t = mesh.triangles[f];
        for (int e = 0; e < 3; e++) edges[std::minmax(t[e], t[(e + 1) % 3])].push_back(chart[f]);
    }
    double length = 0.0;
    for (const auto &[e, charts] : edges) {
        if (charts.size() == 2 && charts[0] != charts[1]) length += (mesh.vertexes[e.first] - mesh.vertexes[e.second]).norm();
    }
    return length;
}
// Whether every chart is one edge-connected piece.
bool ChartsConnected(const TriangleSurface &mesh, const std::vector<int> &chart)
{
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> edges;
    for (uint32_t f = 0; f < mesh.triangles.size(); f++) {
        const auto &t = mesh.triangles[f];
        for (int e = 0; e < 3; e++) edges[std::minmax(t[e], t[(e + 1) % 3])].push_back(f);
    }
    std::vector<std::vector<uint32_t>> next(mesh.triangles.size());
    for (const auto &[e, fs] : edges) {
        if (fs.size() == 2 && chart[fs[0]] == chart[fs[1]]) {
            next[fs[0]].push_back(fs[1]);
            next[fs[1]].push_back(fs[0]);
        }
    }
    std::vector<int> pieces(*std::max_element(chart.begin(), chart.end()) + 1, 0);
    std::vector<uint8_t> seen(mesh.triangles.size(), 0);
    for (uint32_t f = 0; f < mesh.triangles.size(); f++) {
        if (seen[f]) continue;
        pieces[chart[f]]++;
        std::vector<uint32_t> stack = {f};
        seen[f] = 1;
        while (!stack.empty()) {
            const uint32_t g = stack.back();
            stack.pop_back();
            for (uint32_t h : next[g]) {
                if (!seen[h]) {
                    seen[h] = 1;
                    stack.push_back(h);
                }
            }
        }
    }
    return std::all_of(pieces.begin(), pieces.end(), [](int n) { return n == 1; });
}
}  // namespace

// Charts finished as meshlets (L-BFGS, regularization, seam straightening): dense labels, each chart one piece,
// and shorter borders than the raw CVT labels.
TEST(cvt_charts_finished_as_meshlets)
{
    const TriangleSurface mesh = SphereMesh(Vec3(0, 0, 0), 4.0f, 4);
    CvtChartParams raw;
    raw.meshletBaseLevel = false;
    raw.chartCount = 20;
    CvtChartParams finished;  // the meshlet base level, sized for about 20 charts
    finished.maxTrianglesPerChart = 680;
    const std::vector<int> a = SegmentCvtCharts(mesh.vertexes, mesh.triangles, raw);
    const std::vector<int> b = SegmentCvtCharts(mesh.vertexes, mesh.triangles, finished);
    CHECK(a.size() == mesh.triangles.size() && b.size() == mesh.triangles.size());
    const int count = *std::max_element(b.begin(), b.end()) + 1;
    std::vector<int> used(count, 0);
    for (int l : b) used[l]++;
    CHECK(*std::min_element(b.begin(), b.end()) == 0 && std::all_of(used.begin(), used.end(), [](int n) { return n > 0; }));
    CHECK(ChartsConnected(mesh, b));
    const double la = ChartBorderLength(mesh, a), lb = ChartBorderLength(mesh, b);
    std::printf("  raw: %d charts, border %.1f; finished: %d charts, border %.1f\n", *std::max_element(a.begin(), a.end()) + 1, la, count, lb);
    CHECK(count >= 10 && count <= 40);
    CHECK(lb < la);
}

// chartFill sets the seed count: charts aiming at a larger share of the limit are fewer (and larger).
TEST(cvt_charts_fill_sets_count)
{
    const TriangleSurface mesh = SphereMesh(Vec3(0, 0, 0), 4.0f, 4);
    std::vector<int> counts;
    for (double fill : {0.25, 0.375, 0.75}) {
        CvtChartParams p;
        p.chartFill = fill;
        const std::vector<int> labels = SegmentCvtCharts(mesh.vertexes, mesh.triangles, p);
        CHECK(labels.size() == mesh.triangles.size());
        counts.push_back(*std::max_element(labels.begin(), labels.end()) + 1);
        std::printf("  chartFill %.3f: %d charts\n", fill, counts.back());
    }
    CHECK(counts[0] > counts[1] && counts[1] > counts[2]);
}
