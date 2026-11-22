#pragma once
// 3D vector aliases and geometry helpers for the 3D springls (see Common.hpp for the 2D ones).
#include <core/Common.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>  // cross()

namespace springls
{
using Vec3 = Eigen::Vector3f;
using Vec3d = Eigen::Vector3d;
using Vec3i = Eigen::Vector3i;
using Points3 = std::vector<Vec3>;

// Polygon type of 3D springls and of the iso-surface they are seeded from; the value is the
// number of vertexes per element.
enum class ElementType : int {
    Triangle = 3,
    Quad = 4
};
constexpr int kMaxElementSize = 4;  // iso-surface faces
// Springls have any number of vertexes up to this (primal springls are triangles or quads; dual
// springls are polygons around a vertex of any valence).
constexpr int kMaxSpringlSize = 16;
inline int ElementSize(ElementType type) { return static_cast<int>(type); }

inline float distance(const Vec3 &a, const Vec3 &b) { return (a - b).norm(); }
inline float distanceSqr(const Vec3 &a, const Vec3 &b) { return (a - b).squaredNorm(); }
// Unit vector along v, or zero when v is (nearly) zero.
inline Vec3 normalizedOrZero(const Vec3 &v)
{
    const float n = v.norm();
    return (n > 1E-12f) ? Vec3(v / n) : Vec3::Zero();
}

// Squared distance from p to segment [a, b], and the closest point on it.
inline float SegmentDistanceSqr(const Vec3 &a, const Vec3 &b, const Vec3 &p, Vec3 *closest = nullptr)
{
    const Vec3 ab = b - a;
    const float len2 = ab.squaredNorm();
    const float t = (len2 > 0.0f) ? std::clamp((p - a).dot(ab) / len2, 0.0f, 1.0f) : 0.0f;
    const Vec3 q = a + t * ab;
    if (closest) *closest = q;
    return (p - q).squaredNorm();
}

// Feature of a triangle (a, b, c) that holds the closest point to a query.
enum TriangleRegion : int {
    kRegionFace = 0,
    kRegionVertexA,
    kRegionVertexB,
    kRegionVertexC,
    kRegionEdgeAB,
    kRegionEdgeBC,
    kRegionEdgeCA
};

// Squared distance from p to triangle (a, b, c), and the closest point on it (Ericson, "Real-Time
// Collision Detection", 5.1.5), plus the feature (TriangleRegion) it lies on. Degenerate triangles
// fall back to their edges.
inline float TriangleDistanceSqr(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3 &p, Vec3 *closest = nullptr, int *region = nullptr)
{
    int r = kRegionFace;
    const Vec3 ab = b - a, ac = c - a, ap = p - a;
    const float d1 = ab.dot(ap), d2 = ac.dot(ap);
    Vec3 q;
    if (d1 <= 0.0f && d2 <= 0.0f) {
        q = a;
        r = kRegionVertexA;
    } else {
        const Vec3 bp = p - b;
        const float d3 = ab.dot(bp), d4 = ac.dot(bp);
        if (d3 >= 0.0f && d4 <= d3) {
            q = b;
            r = kRegionVertexB;
        } else {
            const float vc = d1 * d4 - d3 * d2;
            const Vec3 cp = p - c;
            const float d5 = ab.dot(cp), d6 = ac.dot(cp);
            const float vb = d5 * d2 - d1 * d6;
            const float va = d3 * d6 - d5 * d4;
            if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
                q = a + (d1 / (d1 - d3)) * ab;
                r = kRegionEdgeAB;
            } else if (d6 >= 0.0f && d5 <= d6) {
                q = c;
                r = kRegionVertexC;
            } else if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
                q = a + (d2 / (d2 - d6)) * ac;
                r = kRegionEdgeCA;
            } else if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
                q = b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b);
                r = kRegionEdgeBC;
            } else {
                const float sum = va + vb + vc;
                if (!(std::abs(sum) > 1E-20f)) {
                    // Degenerate (collinear) triangle: closest of its three edges.
                    Vec3 q0, q1, q2;
                    const float e0 = SegmentDistanceSqr(a, b, p, &q0), e1 = SegmentDistanceSqr(b, c, p, &q1), e2 = SegmentDistanceSqr(c, a, p, &q2);
                    q = (e0 <= e1 && e0 <= e2) ? q0 : ((e1 <= e2) ? q1 : q2);
                    r = (e0 <= e1 && e0 <= e2) ? kRegionEdgeAB : ((e1 <= e2) ? kRegionEdgeBC : kRegionEdgeCA);
                } else {
                    const float denom = 1.0f / sum;
                    q = a + ab * (vb * denom) + ac * (vc * denom);
                }
            }
        }
    }
    if (closest) *closest = q;
    if (region) *region = r;
    return (p - q).squaredNorm();
}

// Area-weighted normal (sum of cross products) of the fan of triangles (center, v[k], v[k + 1]).
inline Vec3 FanNormal(const Vec3 &center, const Vec3 *v, int K)
{
    Vec3 n = Vec3::Zero();
    for (int k = 0; k < K; k++) n += (v[k] - center).cross(v[(k + 1) % K] - center);
    return n;
}
inline Vec3 Centroid(const Vec3 *v, int K)
{
    Vec3 c = Vec3::Zero();
    for (int k = 0; k < K; k++) c += v[k];
    return c / static_cast<float>(K);
}
}  // namespace springls
