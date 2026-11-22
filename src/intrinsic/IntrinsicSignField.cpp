#include <intrinsic/IntrinsicSignField.hpp>

#include <intrinsic/IntrinsicDistance.hpp>
#include <intrinsic/IntrinsicTrace.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>

namespace springls
{
namespace
{
double SegmentDistance(const Vec2d &a, const Vec2d &b, const Vec2d &x)
{
    const Vec2d ab = b - a;
    const double len2 = ab.squaredNorm();
    const double t = (len2 > 0.0) ? std::clamp((x - a).dot(ab) / len2, 0.0, 1.0) : 0.0;
    return (x - (a + t * ab)).norm();
}

template <typename Visit>
void ForNeighbors(const IntrinsicTriangulation &tri, uint32_t v, Visit &&visit)
{
    const uint32_t start = tri.vertexHalfedge(v);
    uint32_t he = start;
    do {
        visit(tri.head(he));
        he = tri.nextOutgoing(he);
    } while (he != start);
}
}  // namespace

SignField ReconstructSignField(const IntrinsicTriangulation &tri, const SpringlSegments &springls, const SignFieldParams &params)
{
    return SignFieldFromDistance(tri, SpringlDistance(tri, springls, params), params, {});
}

int SeedChartDepth(const IntrinsicTriangulation &tri, const SignFieldParams &params)
{
    const double h = tri.meanEdgeLength();
    const double seedRadius = params.capsule + h;
    return std::max(4, static_cast<int>(std::ceil(2.0 * seedRadius / h)) + 3);
}

std::vector<double> SpringlDistance(const IntrinsicTriangulation &tri, const SpringlSegments &springls, const SignFieldParams &params,
                                    const ChartProvider &chartFor)
{
    const size_t V = tri.vertexCount();
    if (springls.segments.empty() || V == 0) return std::vector<double>(V, std::numeric_limits<double>::infinity());

    // 1. U: exact chart distances within the seed radius, fast marching beyond. Each segment is measured in
    // the chart of its first point's face; segments are grouped by that face, so each chart is walked once.
    const double h = tri.meanEdgeLength();
    const double seedRadius = params.capsule + h;
    const int depth = SeedChartDepth(tri, params);
    std::vector<uint32_t> order(springls.segments.size());
    for (uint32_t s = 0; s < order.size(); s++) order[s] = s;
    std::sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) {
        const uint32_t fx = springls.segments[x].first.face, fy = springls.segments[y].first.face;
        return fx != fy ? fx < fy : x < y;
    });
    std::vector<size_t> groupStart;  // into order: one group per face
    for (size_t k = 0; k < order.size(); k++) {
        if (k == 0 || springls.segments[order[k]].first.face != springls.segments[order[k - 1]].first.face) groupStart.push_back(k);
    }
    groupStart.push_back(order.size());
    const int G = static_cast<int>(groupStart.size()) - 1;
    // The groups' charts: from the provider (serially: it may cache), else built in parallel.
    std::vector<const LocalChart *> charts(G, nullptr);
    std::vector<LocalChart> owned;
    if (chartFor) {
        for (int g = 0; g < G; g++) charts[g] = &chartFor(springls.segments[order[groupStart[g]]].first.face, depth);
    } else {
        owned.resize(G);
        SPRINGLS_OMP(omp parallel for schedule(dynamic, 4) if (G >= 16))
        for (int g = 0; g < G; g++) owned[g].build(tri, springls.segments[order[groupStart[g]]].first.face, depth);
        for (int g = 0; g < G; g++) charts[g] = &owned[g];
    }
    // Per group, the corners within the seed radius of its segments and their distances.
    std::vector<std::vector<std::pair<uint32_t, double>>> found(G);
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 4) if (G >= 16))
    for (int g = 0; g < G; g++) {
        const LocalChart &c = *charts[g];
        std::vector<std::pair<Vec2d, Vec2d>> segs;
        Vec2d center = Vec2d::Zero();
        for (size_t k = groupStart[g]; k < groupStart[g + 1]; k++) {
            const auto &[a, b] = springls.segments[order[k]];
            const Vec2d pa = c.position(a);
            const Vec2d pb = c.contains(b.face) ? c.position(b) : pa;
            segs.emplace_back(pa, pb);
            center += pa + pb;
        }
        center /= static_cast<double>(2 * segs.size());
        double radius = 0.0;
        for (const auto &[pa, pb] : segs) radius = std::max({radius, (pa - center).norm(), (pb - center).norm()});
        // A corner farther than this from the center is beyond the seed radius of every segment.
        const double reach = radius + seedRadius;
        for (uint32_t f : c.faces()) {
            const auto &C = c.corners(f);
            for (int k = 0; k < 3; k++) {
                if ((C[k] - center).norm() > reach) continue;
                double d = std::numeric_limits<double>::infinity();
                for (const auto &[pa, pb] : segs) d = std::min(d, SegmentDistance(pa, pb, C[k]));
                if (d > seedRadius) continue;
                found[g].emplace_back(tri.tail(3 * f + k), d);
            }
        }
    }
    // Nearest exact distance per vertex, and the vertices touched.
    std::vector<double> nearest(V, std::numeric_limits<double>::infinity());
    std::vector<uint32_t> touched;
    for (const auto &list : found) {
        for (const auto &[v, d] : list) {
            if (!std::isfinite(nearest[v])) touched.push_back(v);
            nearest[v] = std::min(nearest[v], d);
        }
    }
    std::sort(touched.begin(), touched.end());  // ascending vertex order
    std::vector<DistanceSeed> seeds;
    seeds.reserve(touched.size());
    for (uint32_t v : touched) seeds.push_back({v, nearest[v]});
    return FastMarch(tri, seeds, params.march);
}

SignField SignFieldFromDistance(const IntrinsicTriangulation &tri, const std::vector<double> &U, const SignFieldParams &params, const SpringlSide &side)
{
    const size_t V = tri.vertexCount();
    SignField out;
    out.phi.assign(V, params.clamp);
    out.leaked = true;  // nothing is left inside
    if (V == 0 || U.size() != V) return out;

    std::vector<uint8_t> wall(V, 0);
    for (size_t v = 0; v < V; v++) wall[v] = U[v] <= params.capsule;
    // The springl side of every wall vertex at any reach (the votes and the sweep ask for it), computed in
    // parallel up front: side must be safe to call concurrently.
    std::vector<int8_t> wallSide;
    if (side) {
        std::vector<uint32_t> walls;
        for (uint32_t v = 0; v < V; v++) {
            if (wall[v]) walls.push_back(v);
        }
        wallSide.assign(V, 0);
        const int n = static_cast<int>(walls.size());
        SPRINGLS_OMP(omp parallel for schedule(dynamic, 64) if (n >= kParallelMinHeavyItems))
        for (int i = 0; i < n; i++) wallSide[walls[i]] = static_cast<int8_t>(side(walls[i], std::numeric_limits<double>::infinity()));
    }
    const auto sideOfWall = [&](uint32_t v) { return static_cast<int>(wallSide[v]); };

    // Mesh components without a finite U (no springls in them) are exterior.
    std::vector<uint8_t> component(V, 0);  // 1: has a finite U
    {
        std::vector<uint8_t> seen(V, 0);
        std::vector<uint32_t> members;
        for (uint32_t s = 0; s < V; s++) {
            if (seen[s]) continue;
            members.assign(1, s);
            seen[s] = 1;
            bool finite = false;
            for (size_t k = 0; k < members.size(); k++) {
                finite = finite || std::isfinite(U[members[k]]);
                ForNeighbors(tri, members[k], [&](uint32_t m) {
                    if (!seen[m]) {
                        seen[m] = 1;
                        members.push_back(m);
                    }
                });
            }
            for (uint32_t v : members) component[v] = finite ? 1 : 0;
        }
    }

    // 2. Exterior flood from the non-wall vertex with the largest U, or from the largest unreached region.
    int64_t seed = -1;
    for (size_t v = 0; v < V; v++) {
        if (component[v] && !wall[v] && std::isfinite(U[v]) && (seed < 0 || U[v] > U[seed])) seed = static_cast<int64_t>(v);
    }
    std::vector<int32_t> region(V, -1);
    {
        // Regions of unreached vertices, connected through non-wall vertices; their areas (a third of each
        // incident face's).
        std::vector<double> vertexArea(V, 0.0);
        bool any = false;
        for (size_t v = 0; v < V && !any; v++) any = component[v] && !std::isfinite(U[v]);
        if (any) {
            for (uint32_t f = 0; f < tri.faceCount(); f++) {
                for (int c = 0; c < 3; c++) vertexArea[tri.tail(3 * f + c)] += tri.faceArea(f) / 3.0;
            }
            double bestArea = -1.0;
            int32_t regions = 0;
            std::vector<uint32_t> queue;
            for (uint32_t s = 0; s < V; s++) {
                if (!component[s] || std::isfinite(U[s]) || region[s] >= 0) continue;
                queue.assign(1, s);
                region[s] = regions;
                double area = 0.0;
                for (size_t k = 0; k < queue.size(); k++) {
                    area += vertexArea[queue[k]];
                    ForNeighbors(tri, queue[k], [&](uint32_t m) {
                        if (!wall[m] && region[m] < 0) {
                            region[m] = regions;
                            queue.push_back(m);
                        }
                    });
                }
                if (area > bestArea) {
                    bestArea = area;
                    seed = s;
                }
                regions++;
            }
        }
    }
    if (seed < 0) {
        for (size_t v = 0; v < V; v++) out.phi[v] = std::min(U[v], params.clamp);
        return out;
    }
    std::vector<int8_t> sign(V, 1);
    std::vector<uint8_t> exterior(V, 0);
    std::vector<uint32_t> queue{static_cast<uint32_t>(seed)};
    exterior[seed] = 1;
    for (size_t k = 0; k < queue.size(); k++) {
        ForNeighbors(tri, queue[k], [&](uint32_t m) {
            if (!wall[m] && !exterior[m]) {
                exterior[m] = 1;
                queue.push_back(m);
            }
        });
    }
    std::vector<uint8_t> assigned(V, 0);
    // With a springl side, the flooded region is outside unless the springls bordering it face it with
    // their inner side: then the object is the larger side of the surface (the largest U is inside) and the
    // roles swap.
    int8_t floodedSign = 1;
    if (side) {
        double vote = 0.0;
        for (uint32_t v : queue) {
            ForNeighbors(tri, v, [&](uint32_t m) {
                if (wall[m]) vote += sideOfWall(m);
            });
        }
        if (vote < 0.0) floodedSign = -1;
    }
    // Non-wall vertices: the exterior flood's take floodedSign. Each region it did not reach takes the
    // other sign, unless the springls next to it face it with the side of the flood's (side: e.g. a pocket
    // of the exterior behind a mouth the walls close).
    std::vector<uint8_t> regionDone(V, 0);
    for (uint32_t s = 0; s < V; s++) {
        if (wall[s] || regionDone[s]) continue;
        if (!component[s]) {
            sign[s] = 1;
            regionDone[s] = 1;
            continue;
        }
        if (exterior[s]) {
            sign[s] = floodedSign;
            regionDone[s] = 1;
            if (floodedSign < 0) out.leaked = false;
            continue;
        }
        std::vector<uint32_t> members{s};
        regionDone[s] = 1;
        double vote = 0.0;
        for (size_t k = 0; k < members.size(); k++) {
            ForNeighbors(tri, members[k], [&](uint32_t m) {
                if (wall[m]) {
                    if (side) vote += sideOfWall(m);
                } else if (!regionDone[m] && !exterior[m]) {
                    regionDone[m] = 1;
                    members.push_back(m);
                }
            });
        }
        const int8_t regionSign = (vote > 0.0) ? 1 : (vote < 0.0) ? -1 : static_cast<int8_t>(-floodedSign);
        for (uint32_t v : members) sign[v] = regionSign;
        if (regionSign < 0) out.leaked = false;
    }
    for (size_t v = 0; v < V; v++) assigned[v] = assigned[v] || !wall[v];

    // 3. Upwind sign into the (remaining) walls, in order of decreasing U (ties by index).
    std::vector<uint32_t> walls;
    for (uint32_t v = 0; v < V; v++) {
        if (wall[v] && !assigned[v]) walls.push_back(v);
    }
    std::sort(walls.begin(), walls.end(), [&](uint32_t a, uint32_t b) { return U[a] != U[b] ? U[a] > U[b] : a < b; });
    for (uint32_t v : walls) {
        double vote = 0.0;
        ForNeighbors(tri, v, [&](uint32_t m) {
            if (!assigned[m] || U[m] <= U[v]) return;
            vote += sign[m] * (U[m] - U[v]);
        });
        // No upwind neighbor (or a tie): the middle of a feature thinner than two capsules, inside, unless the
        // springls' side says otherwise (a thin exterior gap, e.g. a pocket's mouth).
        const int across = (vote == 0.0 && side) ? sideOfWall(v) : 0;
        sign[v] = (vote > 0.0 || across > 0) ? 1 : -1;
        assigned[v] = 1;
    }
    // 3b. Next to the springls, the side of the nearest one where it is unambiguous.
    if (side) {
        std::vector<uint32_t> near;
        for (uint32_t v = 0; v < V; v++) {
            if (U[v] <= params.correction) near.push_back(v);
        }
        const int n = static_cast<int>(near.size());
        std::vector<int8_t> sides(near.size());
        SPRINGLS_OMP(omp parallel for schedule(dynamic, 64) if (n >= kParallelMinHeavyItems))
        for (int i = 0; i < n; i++) sides[i] = static_cast<int8_t>(side(near[i], params.correction));
        for (int i = 0; i < n; i++) {
            if (sides[i] != 0) sign[near[i]] = sides[i];
        }
    }

    for (size_t v = 0; v < V; v++) {
        out.phi[v] = sign[v] * std::min(U[v], params.clamp);
        out.hasInside |= sign[v] < 0;
    }
    return out;
}
}  // namespace springls
