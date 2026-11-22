#pragma once
#include <core/Common.hpp>

#include <optional>
#include <vector>

namespace springls
{
// Bounding-volume hierarchy over 2D line segments for closest-point queries. append() segments,
// finish() to build, then query. Queries are const and allocation free, so they are safe to run
// concurrently.
class SegmentTree2D
{
public:
    struct Hit {
        float distanceSqr;
        Vec2 point;  // closest point on the segment
        int id;      // id passed to append()
    };

    void reset();
    void append(const Vec2 &a, const Vec2 &b, int id);
    void finish();
    bool empty() const { return mNodes.empty(); }
    size_t size() const { return mSegments.size(); }

    // Closest segment within maxDistance (strictly), if any.
    std::optional<Hit> nearest(const Vec2 &p, float maxDistance = std::numeric_limits<float>::infinity()) const;
    // Distance to the closest segment, or maxDistance when none is closer.
    float distance(const Vec2 &p, float maxDistance) const;
    // Calls visit(id, closestPoint, distanceSqr) for every segment within maxDistance of p.
    template <class Visit>
    void forEachWithin(const Vec2 &p, float maxDistance, Visit &&visit) const
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
                for (int s = node.first; s < node.first + node.count; s++) {
                    Vec2 q;
                    const float d2 = SegmentDistanceSqr(mSegments[s].a, mSegments[s].b, p, &q);
                    if (d2 <= limit) visit(mSegments[s].id, q, d2);
                }
                continue;
            }
            stack[top++] = node.first;
            stack[top++] = node.first + 1;
        }
    }

private:
    struct Segment {
        Vec2 a, b;
        int id;
    };
    struct Node {
        Vec2 lo = Vec2::Zero(), hi = Vec2::Zero();
        int first = 0;  // leaf: first segment; interior: left child (right child is first + 1)
        int count = 0;  // leaf: number of segments; interior: 0
    };
    static constexpr int kLeafSize = 4;
    static constexpr int kMaxStack = 64;
    std::vector<Segment> mSegments;
    std::vector<Node> mNodes;

    void build(int node, int begin, int end, int depth);
    static float boxDistanceSqr(const Node &node, const Vec2 &p)
    {
        const Vec2 q = cwiseClamp(p, node.lo, node.hi);
        return (p - q).squaredNorm();
    }
};
}  // namespace springls
