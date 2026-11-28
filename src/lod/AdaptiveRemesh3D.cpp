#include <lod/AdaptiveRemesh3D.hpp>

#include <core/Common.hpp>

#include <algorithm>
#include <functional>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <execution>
#include <limits>
#include <memory>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace springls
{
namespace
{
constexpr uint32_t kFree = std::numeric_limits<uint32_t>::max();

float MinAngle(const Vec3 &a, const Vec3 &b, const Vec3 &c)
{
    const auto angle = [](const Vec3 &u, const Vec3 &v) { return std::atan2(u.cross(v).norm(), u.dot(v)); };
    return std::min({angle(b - a, c - a), angle(c - b, a - b), angle(a - c, b - c)});
}
Vec3 Normal(const Vec3 &a, const Vec3 &b, const Vec3 &c) { return (b - a).cross(c - a); }
uint64_t EdgeKey(uint32_t a, uint32_t b) { return (static_cast<uint64_t>(std::min(a, b)) << 32) | std::max(a, b); }
uint32_t KeyLow(uint64_t key) { return static_cast<uint32_t>(key >> 32); }
uint32_t KeyHigh(uint64_t key) { return static_cast<uint32_t>(key & 0xffffffffu); }

template <class It, class Less>
void Sort(bool parallel, It begin, It end, Less less)
{
    if (parallel)
        std::sort(std::execution::par, begin, end, less);
    else
        std::sort(begin, end, less);
}
template <class It>
void Sort(bool parallel, It begin, It end)
{
    Sort(parallel, begin, end, std::less<>());
}

// A triangle's edge: corner `corner` of `face` to the next corner.
struct HalfEdge {
    uint64_t key;
    uint32_t face;
    uint32_t corner;
};

// Edges (half-edges sorted by edge) and vertex rings (triangles around each vertex), rebuilt from the triangles.
struct Tables {
    std::vector<HalfEdge> halfEdges;
    std::vector<uint32_t> edgeStart;  // first half-edge of each edge, then the end
    std::vector<uint32_t> ringStart, ringFaces;

    void build(const TriangleSurface &mesh, bool parallel = true)
    {
        const auto &tris = mesh.triangles;
        const int F = static_cast<int>(tris.size());
        halfEdges.resize(3 * static_cast<size_t>(F));
        SPRINGLS_OMP(omp parallel for if (F >= kParallelMinLightItems))
        for (int f = 0; f < F; f++) {
            for (uint32_t c = 0; c < 3; c++) halfEdges[3 * static_cast<size_t>(f) + c] = {EdgeKey(tris[f][c], tris[f][(c + 1) % 3]), static_cast<uint32_t>(f), c};
        }
        Sort(parallel, halfEdges.begin(), halfEdges.end(), [](const HalfEdge &x, const HalfEdge &y) { return x.key < y.key || (x.key == y.key && x.face < y.face); });
        edgeStart.clear();
        for (uint32_t h = 0; h < halfEdges.size(); h++) {
            if (h == 0 || halfEdges[h].key != halfEdges[h - 1].key) edgeStart.push_back(h);
        }
        edgeStart.push_back(static_cast<uint32_t>(halfEdges.size()));
        const size_t V = mesh.vertexes.size();
        ringStart.assign(V + 1, 0);
        for (const auto &t : tris) {
            for (uint32_t v : t) ringStart[v + 1]++;
        }
        for (size_t v = 0; v < V; v++) ringStart[v + 1] += ringStart[v];
        ringFaces.resize(ringStart[V]);
        std::vector<uint32_t> fill(ringStart.begin(), ringStart.end() - 1);
        for (uint32_t f = 0; f < tris.size(); f++) {
            for (uint32_t v : tris[f]) ringFaces[fill[v]++] = f;
        }
    }
    size_t edgeCount() const { return edgeStart.empty() ? 0 : edgeStart.size() - 1; }
    std::span<const HalfEdge> edge(size_t e) const { return {halfEdges.data() + edgeStart[e], halfEdges.data() + edgeStart[e + 1]}; }
    std::span<const HalfEdge> edge(uint64_t key, bool) const
    {
        const auto lo = std::lower_bound(halfEdges.begin(), halfEdges.end(), key, [](const HalfEdge &h, uint64_t k) { return h.key < k; });
        auto hi = lo;
        while (hi != halfEdges.end() && hi->key == key) ++hi;
        return {halfEdges.data() + (lo - halfEdges.begin()), static_cast<size_t>(hi - lo)};
    }
    bool hasEdge(uint64_t key) const { return !edge(key, true).empty(); }
    std::span<const uint32_t> ring(uint32_t v) const { return {ringFaces.data() + ringStart[v], ringFaces.data() + ringStart[v + 1]}; }
};

// Claims: every item takes the smallest rank that wants it.
struct Claims {
    std::unique_ptr<std::atomic<uint32_t>[]> owner;
    size_t size = 0;
    explicit Claims(size_t n) : owner(new std::atomic<uint32_t>[n]), size(n)
    {
        for (size_t i = 0; i < n; i++) owner[i].store(kFree, std::memory_order_relaxed);
    }
    void claim(size_t item, uint32_t rank)
    {
        uint32_t current = owner[item].load(std::memory_order_relaxed);
        while (rank < current && !owner[item].compare_exchange_weak(current, rank, std::memory_order_relaxed)) {
        }
    }
    bool holds(size_t item, uint32_t rank) const { return owner[item].load(std::memory_order_relaxed) == rank; }
};

// Collects, per thread, the items `make(i, out)` produces for i in [0, n), in index order overall.
template <class T, class Make>
std::vector<T> ParallelCollect(int n, const Make &make)
{
    std::vector<std::vector<T>> parts;
    SPRINGLS_OMP(omp parallel if (n >= kParallelMinLightItems))
    {
        std::vector<T> local;
        SPRINGLS_OMP(omp for schedule(static) nowait)
        for (int i = 0; i < n; i++) make(i, local);
        SPRINGLS_OMP(omp critical(remesh_collect))
        parts.push_back(std::move(local));
    }
    std::vector<T> all;
    for (auto &p : parts) all.insert(all.end(), p.begin(), p.end());
    return all;
}

// --- Splits: edges longer than the limit, longest first; a split claims the edge's triangles.
bool SplitPass(TriangleSurface &mesh, const Tables &tables, float limit, const RemeshParams &params, const RemeshHooks &hooks)
{
    const std::vector<uint8_t> *frozen = params.frozen;
    const int E = static_cast<int>(tables.edgeCount());
    struct Candidate {
        float length;
        uint32_t edge;
    };
    std::vector<Candidate> candidates = ParallelCollect<Candidate>(E, [&](int e, std::vector<Candidate> &out) {
        const auto halves = tables.edge(static_cast<size_t>(e));
        if (frozen && std::any_of(halves.begin(), halves.end(), [&](const HalfEdge &h) { return (*frozen)[h.face] != 0; })) return;
        const uint64_t key = halves.front().key;
        const float length = (mesh.vertexes[KeyLow(key)] - mesh.vertexes[KeyHigh(key)]).norm();
        if (length > limit) out.push_back({length, static_cast<uint32_t>(e)});
    });
    if (candidates.empty()) return false;
    Sort(params.parallel, candidates.begin(), candidates.end(),
         [](const Candidate &x, const Candidate &y) { return x.length > y.length || (x.length == y.length && x.edge < y.edge); });
    const int C = static_cast<int>(candidates.size());
    Claims faces(mesh.triangles.size());
    SPRINGLS_OMP(omp parallel for if (C >= kParallelMinLightItems))
    for (int r = 0; r < C; r++) {
        for (const HalfEdge &h : tables.edge(candidates[r].edge)) faces.claim(h.face, static_cast<uint32_t>(r));
    }
    // Winners, with their new vertex and the first of their new triangles.
    struct Winner {
        uint32_t edge, vertex, firstFace;
    };
    std::vector<Winner> winners;
    uint32_t nextVertex = static_cast<uint32_t>(mesh.vertexes.size()), nextFace = static_cast<uint32_t>(mesh.triangles.size());
    for (int r = 0; r < C; r++) {
        const auto halves = tables.edge(candidates[r].edge);
        if (!std::all_of(halves.begin(), halves.end(), [&](const HalfEdge &h) { return faces.holds(h.face, static_cast<uint32_t>(r)); })) continue;
        winners.push_back({candidates[r].edge, nextVertex++, nextFace});
        nextFace += static_cast<uint32_t>(halves.size());
    }
    if (winners.empty()) return false;
    if (hooks.grow) hooks.grow(nextVertex);
    mesh.vertexes.resize(nextVertex);
    mesh.triangles.resize(nextFace);
    mesh.faceRegion.resize(nextFace);
    if (params.frozen) params.frozen->resize(nextFace, 0);
    const int W = static_cast<int>(winners.size());
    SPRINGLS_OMP(omp parallel for if (W >= kParallelMinLightItems))
    for (int w = 0; w < W; w++) {
        const Winner &win = winners[w];
        const auto halves = tables.edge(win.edge);
        const uint32_t a = KeyLow(halves.front().key), b = KeyHigh(halves.front().key), m = win.vertex;
        mesh.vertexes[m] = 0.5f * (mesh.vertexes[a] + mesh.vertexes[b]);
        uint32_t next = win.firstFace;
        for (const HalfEdge &h : halves) {
            const std::array<uint32_t, 3> t = mesh.triangles[h.face];
            const uint32_t p = t[h.corner], q = t[(h.corner + 1) % 3], r = t[(h.corner + 2) % 3];
            mesh.triangles[h.face] = {p, m, r};
            mesh.triangles[next] = {m, q, r};
            mesh.faceRegion[next] = mesh.faceRegion[h.face];
            next++;
        }
        if (hooks.split) hooks.split(m, a, b);
    }
    return true;
}

// --- Collapses: short edges and slivers' shortest edges, shortest first; a collapse claims the vertexes of the
// one-rings of both ends.
bool CollapsePass(TriangleSurface &mesh, const Tables &tables, const RemeshParams &params, const RemeshHooks &hooks)
{
    const auto &tris = mesh.triangles;
    const float shortEdge = params.collapseRatio * params.targetEdge;
    const float minAngle = params.minAngleDegrees * static_cast<float>(kPi) / 180.0f;
    const int F = static_cast<int>(tris.size());
    std::vector<std::pair<float, uint64_t>> keys = ParallelCollect<std::pair<float, uint64_t>>(F, [&](int f, auto &out) {
        const auto &t = tris[f];
        int shortest = 0;
        float lengths[3];
        for (int e = 0; e < 3; e++) {
            lengths[e] = (mesh.vertexes[t[(e + 1) % 3]] - mesh.vertexes[t[e]]).norm();
            if (lengths[e] < lengths[shortest]) shortest = e;
        }
        const bool sliver = MinAngle(mesh.vertexes[t[0]], mesh.vertexes[t[1]], mesh.vertexes[t[2]]) < minAngle;
        for (int e = 0; e < 3; e++) {
            if (lengths[e] < shortEdge || (sliver && e == shortest)) out.push_back({lengths[e], EdgeKey(t[e], t[(e + 1) % 3])});
        }
    });
    if (keys.empty()) return false;
    Sort(params.parallel, keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end(), [](const auto &x, const auto &y) { return x.second == y.second; }), keys.end());
    Sort(params.parallel, keys.begin(), keys.end());  // shortest first
    const std::vector<uint8_t> *frozen = params.frozen;

    const auto onBorder = [&](uint32_t v) {
        if (!params.keepRegionBorders) return false;
        const auto faces = tables.ring(v);
        return std::any_of(faces.begin(), faces.end(), [&](uint32_t f) { return mesh.faceRegion[f] != mesh.faceRegion[faces.front()]; });
    };
    struct Collapse {
        uint32_t a, b;  // b merges into a
        Vec3 target;
        bool midpoint;
        uint32_t e0, e1;  // the edge's triangles (removed)
        bool valid = false;
    };
    const int K = static_cast<int>(keys.size());
    std::vector<Collapse> ops(K);
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 64) if (K >= kParallelMinLightItems))
    for (int k = 0; k < K; k++) {
        Collapse &op = ops[k];
        uint32_t a = KeyLow(keys[k].second), b = KeyHigh(keys[k].second);
        const auto halves = tables.edge(keys[k].second, true);
        if (halves.size() != 2) continue;
        if (frozen) {
            bool touchesFrozen = false;
            for (uint32_t v : {a, b}) {
                for (uint32_t f : tables.ring(v)) touchesFrozen |= (*frozen)[f] != 0;
            }
            if (touchesFrozen) continue;
        }
        // Link condition: a and b share exactly the two opposite vertexes; neither has only three neighbors (a
        // tetrahedron would fold into a two-triangle pillow).
        std::vector<uint32_t> na, nb;
        for (uint32_t f : tables.ring(a)) na.insert(na.end(), tris[f].begin(), tris[f].end());
        for (uint32_t f : tables.ring(b)) nb.insert(nb.end(), tris[f].begin(), tris[f].end());
        for (auto *n : {&na, &nb}) {
            std::sort(n->begin(), n->end());
            n->erase(std::unique(n->begin(), n->end()), n->end());
            n->erase(std::remove_if(n->begin(), n->end(), [&](uint32_t v) { return v == a || v == b; }), n->end());
        }
        std::vector<uint32_t> common;
        std::set_intersection(na.begin(), na.end(), nb.begin(), nb.end(), std::back_inserter(common));
        if (common.size() != 2 || na.size() <= 3 || nb.size() <= 3) continue;
        const uint32_t e0 = halves[0].face, e1 = halves[1].face;
        // On a region border the border end stays; along a border edge, the midpoint.
        const bool borderA = onBorder(a), borderB = onBorder(b);
        const bool borderEdge = params.keepRegionBorders && mesh.faceRegion[e0] != mesh.faceRegion[e1];
        Vec3 target;
        if (borderA && borderB) {
            if (!borderEdge) continue;
            target = 0.5f * (mesh.vertexes[a] + mesh.vertexes[b]);
        } else if (borderA) {
            target = mesh.vertexes[a];
        } else if (borderB) {
            std::swap(a, b);
            target = mesh.vertexes[a];
        } else {
            target = 0.5f * (mesh.vertexes[a] + mesh.vertexes[b]);
        }
        // No remaining triangle around a or b may turn over (or degenerate).
        bool ok = true;
        for (uint32_t v : {a, b}) {
            for (uint32_t f : tables.ring(v)) {
                if (f == e0 || f == e1) continue;
                std::array<Vec3, 3> p;
                for (int c = 0; c < 3; c++) p[c] = (tris[f][c] == a || tris[f][c] == b) ? target : mesh.vertexes[tris[f][c]];
                const Vec3 before = Normal(mesh.vertexes[tris[f][0]], mesh.vertexes[tris[f][1]], mesh.vertexes[tris[f][2]]), after = Normal(p[0], p[1], p[2]);
                if (after.dot(before) <= 0.0f || after.norm() < 1e-3f * before.norm()) ok = false;
            }
        }
        if (!ok) continue;
        op = {a, b, target, target != mesh.vertexes[a], e0, e1, true};
    }
    Claims vertexes(mesh.vertexes.size());
    SPRINGLS_OMP(omp parallel for if (K >= kParallelMinLightItems))
    for (int k = 0; k < K; k++) {
        if (!ops[k].valid) continue;
        for (uint32_t v : {ops[k].a, ops[k].b}) {
            for (uint32_t f : tables.ring(v)) {
                for (uint32_t u : tris[f]) vertexes.claim(u, static_cast<uint32_t>(k));
            }
        }
    }
    std::vector<int> winners;
    for (int k = 0; k < K; k++) {
        if (!ops[k].valid) continue;
        bool all = true;
        for (uint32_t v : {ops[k].a, ops[k].b}) {
            for (uint32_t f : tables.ring(v)) {
                for (uint32_t u : tris[f]) all &= vertexes.holds(u, static_cast<uint32_t>(k));
            }
        }
        if (all) winners.push_back(k);
    }
    if (winners.empty()) return false;
    std::vector<uint8_t> removed(tris.size(), 0);
    const int W = static_cast<int>(winners.size());
    SPRINGLS_OMP(omp parallel for if (W >= kParallelMinLightItems))
    for (int w = 0; w < W; w++) {
        const Collapse &op = ops[winners[w]];
        mesh.vertexes[op.a] = op.target;
        removed[op.e0] = removed[op.e1] = 1;
        for (uint32_t f : tables.ring(op.b)) {
            if (f == op.e0 || f == op.e1) continue;
            for (uint32_t &c : mesh.triangles[f]) {
                if (c == op.b) c = op.a;
            }
        }
        if (hooks.merge) hooks.merge(op.a, op.b, op.midpoint);
    }
    size_t out = 0;
    for (size_t f = 0; f < mesh.triangles.size(); f++) {
        if (removed[f]) continue;
        mesh.triangles[out] = mesh.triangles[f];
        mesh.faceRegion[out] = mesh.faceRegion[f];
        out++;
    }
    if (frozen) {
        std::vector<uint8_t> &flags = *params.frozen;
        size_t kept = 0;
        for (size_t f = 0; f < removed.size(); f++) {
            if (!removed[f]) flags[kept++] = flags[f];
        }
        flags.resize(kept);
    }
    mesh.triangles.resize(out);
    mesh.faceRegion.resize(out);
    return true;
}

// --- Flips inside a region, largest gain of the smaller smallest angle first; a flip claims its four vertexes.
bool FlipPass(TriangleSurface &mesh, const Tables &tables, const RemeshParams &params)
{
    auto &tris = mesh.triangles;
    const std::vector<uint8_t> *frozen = params.frozen;
    struct Flip {
        float gain;
        uint32_t f1, f2, a, b, c, d;
    };
    // With frozen triangles (a chart of RemeshSurfaceByCharts), no flip may join two vertexes of frozen triangles:
    // a neighboring chart, remeshed at the same time, could make the same edge from its side.
    std::vector<uint8_t> pinned;
    if (frozen) {
        pinned.assign(mesh.vertexes.size(), 0);
        for (size_t f = 0; f < tris.size(); f++) {
            if ((*frozen)[f]) {
                for (uint32_t v : tris[f]) pinned[v] = 1;
            }
        }
    }
    const int E = static_cast<int>(tables.edgeCount());
    std::vector<Flip> flips = ParallelCollect<Flip>(E, [&](int e, std::vector<Flip> &out) {
        const auto halves = tables.edge(static_cast<size_t>(e));
        if (halves.size() != 2) return;
        const uint32_t f1 = halves[0].face, f2 = halves[1].face;
        if (params.keepRegionBorders && mesh.faceRegion[f1] != mesh.faceRegion[f2]) return;
        if (frozen && ((*frozen)[f1] || (*frozen)[f2])) return;
        const uint32_t i = halves[0].corner;
        const uint32_t a = tris[f1][i], b = tris[f1][(i + 1) % 3], c = tris[f1][(i + 2) % 3];
        int j = -1;
        for (int s = 0; s < 3; s++) {
            if (tris[f2][s] == b && tris[f2][(s + 1) % 3] == a) j = s;
        }
        if (j < 0) return;
        const uint32_t d = tris[f2][(j + 2) % 3];
        if (c == d || tables.hasEdge(EdgeKey(c, d))) return;
        if (frozen && pinned[c] && pinned[d]) return;
        const Vec3 &pa = mesh.vertexes[a], &pb = mesh.vertexes[b], &pc = mesh.vertexes[c], &pd = mesh.vertexes[d];
        const float before = std::min(MinAngle(pa, pb, pc), MinAngle(pb, pa, pd));
        const float after = std::min(MinAngle(pa, pd, pc), MinAngle(pd, pb, pc));
        const Vec3 normal = Normal(pa, pb, pc).normalized() + Normal(pb, pa, pd).normalized();
        if (!(after > before + 1e-3f) || Normal(pa, pd, pc).dot(normal) <= 0.0f || Normal(pd, pb, pc).dot(normal) <= 0.0f) return;
        out.push_back({after - before, f1, f2, a, b, c, d});
    });
    if (flips.empty()) return false;
    Sort(params.parallel, flips.begin(), flips.end(), [](const Flip &x, const Flip &y) { return x.gain > y.gain || (x.gain == y.gain && x.f1 < y.f1); });
    const int K = static_cast<int>(flips.size());
    Claims vertexes(mesh.vertexes.size());
    SPRINGLS_OMP(omp parallel for if (K >= kParallelMinLightItems))
    for (int k = 0; k < K; k++) {
        for (uint32_t v : {flips[k].a, flips[k].b, flips[k].c, flips[k].d}) vertexes.claim(v, static_cast<uint32_t>(k));
    }
    bool changed = false;
    SPRINGLS_OMP(omp parallel for reduction(|| : changed) if (K >= kParallelMinLightItems))
    for (int k = 0; k < K; k++) {
        const Flip &x = flips[k];
        const auto r = static_cast<uint32_t>(k);
        if (!(vertexes.holds(x.a, r) && vertexes.holds(x.b, r) && vertexes.holds(x.c, r) && vertexes.holds(x.d, r))) continue;
        tris[x.f1] = {x.a, x.d, x.c};
        tris[x.f2] = {x.d, x.b, x.c};
        changed = true;
    }
    return changed;
}
}  // namespace

bool RemeshSurface(TriangleSurface &mesh, const RemeshParams &params, const RemeshHooks &hooks)
{
    if (mesh.faceRegion.size() != mesh.triangles.size()) mesh.faceRegion.resize(mesh.triangles.size(), 0);
    const float limit = params.splitRatio * params.targetEdge;
    if (!(limit > 0.0f)) return false;
    bool any = false;
    Tables tables;
    for (int pass = 0; pass < params.passes; pass++) {
        bool changed = false;
        tables.build(mesh, params.parallel);
        changed |= SplitPass(mesh, tables, limit, params, hooks);
        tables.build(mesh, params.parallel);
        changed |= CollapsePass(mesh, tables, params, hooks);
        tables.build(mesh, params.parallel);
        changed |= FlipPass(mesh, tables, params);
        any |= changed;
        if (!changed) break;
    }
    return any;
}

namespace
{
// The charts (mesh.faceRegion) remeshed in parallel, each inside the frozen ring of triangles around it.
bool RemeshCharts(TriangleSurface &mesh, const RemeshParams &params)
{
    const size_t F0 = mesh.triangles.size(), V0 = mesh.vertexes.size();
    const std::vector<int> faceChart = mesh.faceRegion;
    const int charts = *std::max_element(faceChart.begin(), faceChart.end()) + 1;
    // Triangles per chart, and the ring of other triangles around each (sharing a vertex with it).
    std::vector<std::vector<uint32_t>> chartFaces(charts);
    for (uint32_t f = 0; f < F0; f++) chartFaces[faceChart[f]].push_back(f);
    Tables tables;
    tables.build(mesh, params.parallel);
    // Every chart is cut out of the mesh as it was: the charts write their results back while others are still
    // being cut out. (Chart borders are never changed, so the pieces still meet edge to edge.)
    const std::vector<std::array<uint32_t, 3>> sourceTriangles = mesh.triangles;
    const std::vector<int> sourceRegions = mesh.faceRegion;
    const Points3 sourceVertexes = mesh.vertexes;
    // Room for what the charts add (taken with atomic counters); a chart that would overflow it is left as it was.
    const size_t capacityV = V0 + F0 + 1024, capacityF = 3 * F0 + 1024;
    mesh.vertexes.resize(capacityV);
    mesh.triangles.resize(capacityF);
    mesh.faceRegion.resize(capacityF, 0);
    std::vector<uint8_t> removed(capacityF, 1);
    std::fill(removed.begin(), removed.begin() + static_cast<std::ptrdiff_t>(F0), 0);
    std::atomic<size_t> nextV(V0), nextF(F0);
    std::atomic<bool> changed(false);
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 1))
    for (int c = 0; c < charts; c++) {
        const std::vector<uint32_t> &own = chartFaces[c];
        if (own.empty()) continue;
        // The chart (free) and its ring (frozen), with local vertex numbers.
        std::vector<uint32_t> ring;
        for (uint32_t f : own) {
            for (uint32_t v : sourceTriangles[f]) {
                for (uint32_t g : tables.ring(v)) {
                    if (faceChart[g] != c) ring.push_back(g);
                }
            }
        }
        std::sort(ring.begin(), ring.end());
        ring.erase(std::unique(ring.begin(), ring.end()), ring.end());
        TriangleSurface local;
        std::vector<uint8_t> frozen;
        std::vector<uint32_t> toGlobal;
        std::unordered_map<uint32_t, uint32_t> toLocal;
        const auto localVertex = [&](uint32_t v) {
            const auto [it, inserted] = toLocal.try_emplace(v, static_cast<uint32_t>(toGlobal.size()));
            if (inserted) {
                toGlobal.push_back(v);
                local.vertexes.push_back(sourceVertexes[v]);
            }
            return it->second;
        };
        for (const std::vector<uint32_t> *list : {&own, static_cast<const std::vector<uint32_t> *>(&ring)}) {
            const uint8_t flag = (list == &ring) ? 1 : 0;
            for (uint32_t f : *list) {
                const auto &t = sourceTriangles[f];
                local.triangles.push_back({localVertex(t[0]), localVertex(t[1]), localVertex(t[2])});
                local.faceRegion.push_back(sourceRegions[f]);
                frozen.push_back(flag);
            }
        }
        const size_t localV0 = local.vertexes.size();
        RemeshParams p = params;
        p.parallel = false;
        p.frozen = &frozen;
        if (!RemeshSurface(local, p)) continue;
        // Write back: new vertexes and extra triangles into the reserved room, the chart's triangles over its
        // old ones (frozen ring triangles are unchanged); vertexes of frozen triangles never moved.
        const size_t newV = local.vertexes.size() - localV0;
        std::vector<uint32_t> freeFaces;
        for (size_t f = 0; f < frozen.size(); f++) {
            if (!frozen[f]) freeFaces.push_back(static_cast<uint32_t>(f));
        }
        const size_t extraF = freeFaces.size() > own.size() ? freeFaces.size() - own.size() : 0;
        const size_t firstV = nextV.fetch_add(newV), firstF = nextF.fetch_add(extraF);
        if (firstV + newV > capacityV || firstF + extraF > capacityF) continue;
        std::vector<uint8_t> touchesFrozen(local.vertexes.size(), 0);
        for (size_t f = 0; f < frozen.size(); f++) {
            if (frozen[f]) {
                for (uint32_t v : local.triangles[f]) touchesFrozen[v] = 1;
            }
        }
        for (size_t v = 0; v < localV0; v++) {
            if (!touchesFrozen[v]) mesh.vertexes[toGlobal[v]] = local.vertexes[v];
        }
        for (size_t v = 0; v < newV; v++) {
            toGlobal.push_back(static_cast<uint32_t>(firstV + v));
            mesh.vertexes[firstV + v] = local.vertexes[localV0 + v];
        }
        for (size_t i = 0; i < freeFaces.size(); i++) {
            const size_t slot = (i < own.size()) ? own[i] : firstF + (i - own.size());
            const auto &t = local.triangles[freeFaces[i]];
            mesh.triangles[slot] = {toGlobal[t[0]], toGlobal[t[1]], toGlobal[t[2]]};
            mesh.faceRegion[slot] = local.faceRegion[freeFaces[i]];
            removed[slot] = 0;
        }
        for (size_t i = freeFaces.size(); i < own.size(); i++) removed[own[i]] = 1;
        changed = true;
    }
    // Drop the unused room and the removed triangles.
    const size_t usedF = std::min(nextF.load(), capacityF);
    size_t out = 0;
    for (size_t f = 0; f < usedF; f++) {
        if (removed[f]) continue;
        mesh.triangles[out] = mesh.triangles[f];
        mesh.faceRegion[out] = mesh.faceRegion[f];
        out++;
    }
    mesh.triangles.resize(out);
    mesh.faceRegion.resize(out);
    mesh.vertexes.resize(std::min(nextV.load(), capacityV));
    return changed.load();
}

// The band along the chart borders: the triangles with a vertex on a border, the rest frozen.
std::vector<uint8_t> FrozenAwayFromChartBorders(const TriangleSurface &mesh)
{
    constexpr int kNone = std::numeric_limits<int>::min();
    std::vector<int> first(mesh.vertexes.size(), kNone);
    std::vector<uint8_t> border(mesh.vertexes.size(), 0);
    for (size_t f = 0; f < mesh.triangles.size(); f++) {
        for (uint32_t v : mesh.triangles[f]) {
            if (first[v] == kNone) {
                first[v] = mesh.faceRegion[f];
            } else if (first[v] != mesh.faceRegion[f]) {
                border[v] = 1;
            }
        }
    }
    std::vector<uint8_t> frozen(mesh.triangles.size(), 1);
    for (size_t f = 0; f < mesh.triangles.size(); f++) {
        const auto &t = mesh.triangles[f];
        if (border[t[0]] || border[t[1]] || border[t[2]]) frozen[f] = 0;
    }
    return frozen;
}

}  // namespace

bool RemeshSurfaceByCharts(TriangleSurface &mesh, const RemeshParams &params)
{
    const size_t F = mesh.triangles.size();
    if (mesh.faceRegion.size() != F) mesh.faceRegion.resize(F, 0);
    if (F == 0 || *std::min_element(mesh.faceRegion.begin(), mesh.faceRegion.end()) < 0) return RemeshSurface(mesh, params);
    bool changed = RemeshCharts(mesh, params);
    // Then the band along the chart borders, by independent sets over the whole mesh, borders remeshed like any
    // other edges: each triangle keeps the chart of the triangle it replaces, so the borders follow the remeshing.
    std::vector<uint8_t> frozen = FrozenAwayFromChartBorders(mesh);
    RemeshParams band = params;
    band.frozen = &frozen;
    band.keepRegionBorders = false;
    changed |= RemeshSurface(mesh, band);
    return changed;
}
}  // namespace springls
