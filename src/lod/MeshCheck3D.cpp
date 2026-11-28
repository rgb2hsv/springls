#include <lod/MeshCheck3D.hpp>

#include <core/Common.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <execution>
#include <numbers>
#include <utility>

namespace springls
{
bool TrianglesIntersect(const Vec3 &a0, const Vec3 &a1, const Vec3 &a2, const Vec3 &b0, const Vec3 &b1, const Vec3 &b2)
{
    const std::array<Vec3, 3> A = {a0, a1, a2}, B = {b0, b1, b2};
    const std::array<Vec3, 3> ea = {a1 - a0, a2 - a1, a0 - a2}, eb = {b1 - b0, b2 - b1, b0 - b2};
    const Vec3 na = ea[0].cross(ea[1]), nb = eb[0].cross(eb[1]);
    const float scale = std::max({ea[0].norm(), ea[1].norm(), ea[2].norm(), eb[0].norm(), eb[1].norm(), eb[2].norm()});
    // Separated along an axis: projections strictly apart (triangles that only touch count as intersecting).
    const auto separated = [&](Vec3 axis) {
        const float length = axis.norm();
        if (length < 1e-12f * scale * scale) return false;  // degenerate axis: proves nothing
        axis /= length;
        float aLo = A[0].dot(axis), aHi = aLo, bLo = B[0].dot(axis), bHi = bLo;
        for (int i = 1; i < 3; i++) {
            const float pa = A[i].dot(axis), pb = B[i].dot(axis);
            aLo = std::min(aLo, pa);
            aHi = std::max(aHi, pa);
            bLo = std::min(bLo, pb);
            bHi = std::max(bHi, pb);
        }
        return aHi < bLo || bHi < aLo;
    };
    if (separated(na) || separated(nb)) return false;
    for (const Vec3 &u : ea) {
        for (const Vec3 &v : eb) {
            if (separated(u.cross(v))) return false;
        }
    }
    // Coplanar (or nearly): the in-plane edge normals of both triangles.
    for (const Vec3 &u : ea) {
        if (separated(na.cross(u))) return false;
    }
    for (const Vec3 &v : eb) {
        if (separated(nb.cross(v))) return false;
    }
    return true;
}

std::vector<std::pair<uint32_t, uint32_t>> FindIntersectingPairs(const TriangleSurface &mesh)
{
    const auto &tris = mesh.triangles;
    const auto &P = mesh.vertexes;
    const size_t F = tris.size();
    if (F == 0) return {};
    // Triangles binned by bounding box into cells of about twice the mean edge.
    double edgeSum = 0.0;
    for (const auto &t : tris) edgeSum += (P[t[1]] - P[t[0]]).norm() + (P[t[2]] - P[t[1]]).norm() + (P[t[0]] - P[t[2]]).norm();
    const float cellSize = std::max(1e-6f, static_cast<float>(2.0 * edgeSum / (3.0 * static_cast<double>(F))));
    const auto cellOf = [&](const Vec3 &p) {
        return Vec3i(static_cast<int>(std::floor(p.x() / cellSize)), static_cast<int>(std::floor(p.y() / cellSize)), static_cast<int>(std::floor(p.z() / cellSize)));
    };
    const auto key = [](const Vec3i &c) {
        constexpr int64_t kBias = 1 << 20, kMask = (int64_t(1) << 21) - 1;
        return (((c.x() + kBias) & kMask) << 42) | (((c.y() + kBias) & kMask) << 21) | ((c.z() + kBias) & kMask);
    };
    std::vector<Vec3i> lo(F), hi(F);
    std::vector<size_t> binCount(F);
    for (size_t f = 0; f < F; f++) {
        const Vec3 &a = P[tris[f][0]], &b = P[tris[f][1]], &c = P[tris[f][2]];
        lo[f] = cellOf(a.cwiseMin(b).cwiseMin(c));
        hi[f] = cellOf(a.cwiseMax(b).cwiseMax(c));
        const Vec3i d = hi[f] - lo[f] + Vec3i::Ones();
        binCount[f] = static_cast<size_t>(d.x()) * d.y() * d.z();
    }
    std::vector<size_t> binStart(F + 1, 0);
    for (size_t f = 0; f < F; f++) binStart[f + 1] = binStart[f] + binCount[f];
    std::vector<std::pair<int64_t, uint32_t>> bins(binStart[F]);
    for (size_t f = 0; f < F; f++) {
        size_t at = binStart[f];
        for (int z = lo[f].z(); z <= hi[f].z(); z++)
            for (int y = lo[f].y(); y <= hi[f].y(); y++)
                for (int x = lo[f].x(); x <= hi[f].x(); x++) bins[at++] = {key(Vec3i(x, y, z)), static_cast<uint32_t>(f)};
    }
    std::sort(std::execution::par, bins.begin(), bins.end());
    std::vector<size_t> cellStarts;
    for (size_t i = 0; i < bins.size(); i++) {
        if (i == 0 || bins[i].first != bins[i - 1].first) cellStarts.push_back(i);
    }
    cellStarts.push_back(bins.size());
    const int cells = static_cast<int>(cellStarts.size()) - 1;
    std::vector<std::pair<uint32_t, uint32_t>> overlaps;
    SPRINGLS_OMP(omp parallel)
    {
        std::vector<std::pair<uint32_t, uint32_t>> local;
        SPRINGLS_OMP(omp for schedule(dynamic, 64) nowait)
        for (int c = 0; c < cells; c++) {
            const size_t begin = cellStarts[c], end = cellStarts[c + 1];
            for (size_t i = begin; i < end; i++) {
                const uint32_t f = bins[i].second;
                for (size_t j = i + 1; j < end; j++) {
                    const uint32_t g = bins[j].second;
                    // Each pair once: in the lowest cell both boxes cover.
                    const Vec3i first = lo[f].cwiseMax(lo[g]);
                    if (first.x() > std::min(hi[f].x(), hi[g].x()) || first.y() > std::min(hi[f].y(), hi[g].y()) || first.z() > std::min(hi[f].z(), hi[g].z())) continue;
                    if (key(first) != bins[i].first) continue;
                    const auto &s = tris[f], &t = tris[g];
                    bool shared = false;
                    for (uint32_t u : s) shared |= (u == t[0] || u == t[1] || u == t[2]);
                    if (!shared && TrianglesIntersect(P[s[0]], P[s[1]], P[s[2]], P[t[0]], P[t[1]], P[t[2]])) local.push_back({std::min(f, g), std::max(f, g)});
                }
            }
        }
        SPRINGLS_OMP(omp critical)
        overlaps.insert(overlaps.end(), local.begin(), local.end());
    }
    std::sort(overlaps.begin(), overlaps.end());  // the same order whatever the threads did
    return overlaps;
}

MeshCheck CheckMesh(const TriangleSurface &mesh, const MeshCheckParams &params)
{
    MeshCheck check;
    const auto &tris = mesh.triangles;
    const auto &P = mesh.vertexes;
    const size_t F = tris.size();
    check.flags.assign(F, 0);
    if (F == 0) return check;

    // --- Fold-overs: each edge's two triangles, from the half-edges sorted by edge.
    std::vector<std::pair<uint64_t, uint32_t>> halves(3 * F);
    for (size_t f = 0; f < F; f++) {
        for (int e = 0; e < 3; e++) {
            const uint32_t a = tris[f][e], b = tris[f][(e + 1) % 3];
            halves[3 * f + e] = {(static_cast<uint64_t>(std::min(a, b)) << 32) | std::max(a, b), static_cast<uint32_t>(f)};
        }
    }
    std::sort(std::execution::par, halves.begin(), halves.end());
    std::vector<Vec3> normals(F);
    for (size_t f = 0; f < F; f++) {
        const Vec3 n = (P[tris[f][1]] - P[tris[f][0]]).cross(P[tris[f][2]] - P[tris[f][0]]);
        normals[f] = n.squaredNorm() > 0.0f ? Vec3(n.normalized()) : Vec3::Zero();
    }
    const float foldCos = std::cos(params.foldDegrees * std::numbers::pi_v<float> / 180.0f);
    for (size_t i = 0; i < halves.size();) {
        size_t j = i + 1;
        while (j < halves.size() && halves[j].first == halves[i].first) j++;
        if (j - i != 2) {
            check.nonManifoldEdges++;
        } else if (normals[halves[i].second].dot(normals[halves[i + 1].second]) < foldCos) {
            check.foldedEdges++;
            check.flags[halves[i].second] |= MeshCheck::Folded;
            check.flags[halves[i + 1].second] |= MeshCheck::Folded;
        }
        i = j;
    }

    const auto overlaps = FindIntersectingPairs(mesh);
    check.overlappingPairs = static_cast<int>(overlaps.size());
    for (const auto &[f, g] : overlaps) {
        check.flags[f] |= MeshCheck::Overlapping;
        check.flags[g] |= MeshCheck::Overlapping;
    }
    return check;
}
}  // namespace springls
