#pragma once
#include <core/Common3D.hpp>

#include <map>
#include <optional>
#include <vector>

namespace springls
{
// Bounding-volume hierarchy over 3D triangles for closest-point queries (3D counterpart of
// SegmentTree2D). append() triangles, finish() to build, then query. Queries are const and
// allocation free, so they are safe to run concurrently.
class TriangleTree3D
{
public:
    struct Hit {
        float distanceSqr;
        Vec3 point;  // closest point on the triangle
        int id;      // id passed to append()
    };

    void reset();
    void reserve(size_t n) { mTriangles.reserve(n); }
    void append(const Vec3 &a, const Vec3 &b, const Vec3 &c, int id);
    void finish();
    bool empty() const { return mNodes.empty(); }
    size_t size() const { return mTriangles.size(); }

    // Closest triangle within maxDistance (strictly), if any.
    std::optional<Hit> nearest(const Vec3 &p, float maxDistance = std::numeric_limits<float>::infinity()) const;
    // Distance to the closest triangle, or maxDistance when none is closer.
    float distance(const Vec3 &p, float maxDistance) const;
    // Calls visit(id, closestPoint, distanceSqr) for every triangle within maxDistance of p.
    template <class Visit>
    void forEachWithin(const Vec3 &p, float maxDistance, Visit &&visit) const
    {
        if (mNodes.empty()) return;
        const float limit = maxDistance * maxDistance;
        int stack[kMaxStack];
        int top = 0;
        stack[top++] = 0;
        while (top > 0) {
            const Node &node = mNodes[stack[--top]];
            if (boxDistanceSqr(node, p) > limit) continue;
            if (node.count > 0) {
                for (int t = node.first; t < node.first + node.count; t++) {
                    const Triangle &tri = mTriangles[t];
                    Vec3 q;
                    const float d2 = TriangleDistanceSqr(tri.a, tri.b, tri.c, p, &q);
                    if (d2 <= limit) visit(tri.id, q, d2);
                }
                continue;
            }
            stack[top++] = node.first;
            stack[top++] = node.first + 1;
        }
    }

private:
    struct Triangle {
        Vec3 a, b, c;
        int id;
    };
    struct Node {
        Vec3 lo = Vec3::Zero(), hi = Vec3::Zero();
        int first = 0;  // leaf: first triangle; interior: left child (right child is first + 1)
        int count = 0;  // leaf: number of triangles; interior: 0
    };
    static constexpr int kLeafSize = 4;
    static constexpr int kMaxStack = 64;
    std::vector<Triangle> mTriangles;
    std::vector<Node> mNodes;

    // Number of child pairs below a node over `count` triangles at `depth` (memoized).
    static int pairCount(int count, int depth, std::map<std::pair<int, int>, int> &memo);
    // Builds node over [begin, end); its subtree's child pairs start at slot `next`.
    void build(int node, int begin, int end, int depth, int next, const std::map<std::pair<int, int>, int> &pairs);
    static float boxDistanceSqr(const Node &node, const Vec3 &p)
    {
        const Vec3 q = p.cwiseMax(node.lo).cwiseMin(node.hi);
        return (p - q).squaredNorm();
    }
};
}  // namespace springls
