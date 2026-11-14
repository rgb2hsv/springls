#include <springls/SegmentTree2D.hpp>

#include <algorithm>
#include <limits>

namespace springls
{
void SegmentTree2D::reset()
{
    mSegments.clear();
    mNodes.clear();
}

void SegmentTree2D::append(const Vec2 &a, const Vec2 &b, int id) { mSegments.push_back({a, b, id}); }

void SegmentTree2D::finish()
{
    mNodes.clear();
    if (mSegments.empty()) return;
    mNodes.reserve(4 * mSegments.size() / kLeafSize + 2);
    mNodes.push_back({});
    build(0, 0, static_cast<int>(mSegments.size()), 0);
}

void SegmentTree2D::build(int node, int begin, int end, int depth)
{
    Vec2 lo = mSegments[begin].a.cwiseMin(mSegments[begin].b);
    Vec2 hi = mSegments[begin].a.cwiseMax(mSegments[begin].b);
    for (int s = begin + 1; s < end; s++) {
        lo = lo.cwiseMin(mSegments[s].a.cwiseMin(mSegments[s].b));
        hi = hi.cwiseMax(mSegments[s].a.cwiseMax(mSegments[s].b));
    }
    mNodes[node].lo = lo;
    mNodes[node].hi = hi;
    if (end - begin <= kLeafSize || depth >= kMaxStack - 2) {
        mNodes[node].first = begin;
        mNodes[node].count = end - begin;
        return;
    }
    // Median split of segment midpoints along the longer axis.
    const int axis = (hi.x() - lo.x() >= hi.y() - lo.y()) ? 0 : 1;
    const int mid = (begin + end) / 2;
    std::nth_element(mSegments.begin() + begin, mSegments.begin() + mid, mSegments.begin() + end,
                     [axis](const Segment &l, const Segment &r) { return (l.a[axis] + l.b[axis]) < (r.a[axis] + r.b[axis]); });
    const int left = static_cast<int>(mNodes.size());
    mNodes.push_back({});
    mNodes.push_back({});
    mNodes[node].first = left;
    mNodes[node].count = 0;
    build(left, begin, mid, depth + 1);
    build(left + 1, mid, end, depth + 1);
}

std::optional<SegmentTree2D::Hit> SegmentTree2D::nearest(const Vec2 &p, float maxDistance) const
{
    if (mNodes.empty()) return std::nullopt;
    float best = std::isinf(maxDistance) ? std::numeric_limits<float>::infinity() : maxDistance * maxDistance;
    int bestSegment = -1;
    Vec2 bestPoint = p;
    int stack[kMaxStack];
    int top = 0;
    stack[top++] = 0;
    while (top > 0) {
        const Node &node = mNodes[stack[--top]];
        if (boxDistanceSqr(node, p) >= best) continue;
        if (node.count > 0) {
            for (int s = node.first; s < node.first + node.count; s++) {
                Vec2 q;
                const float d2 = SegmentDistanceSqr(mSegments[s].a, mSegments[s].b, p, &q);
                // Ties go to the smaller id so results do not depend on tree layout.
                if (d2 < best || (d2 == best && bestSegment >= 0 && mSegments[s].id < mSegments[bestSegment].id)) {
                    best = d2;
                    bestSegment = s;
                    bestPoint = q;
                }
            }
            continue;
        }
        // Visit the nearer child first (pushed last).
        const int l = node.first, r = node.first + 1;
        const float dl = boxDistanceSqr(mNodes[l], p), dr = boxDistanceSqr(mNodes[r], p);
        if (dl < dr) {
            if (dr < best) stack[top++] = r;
            if (dl < best) stack[top++] = l;
        } else {
            if (dl < best) stack[top++] = l;
            if (dr < best) stack[top++] = r;
        }
    }
    if (bestSegment < 0) return std::nullopt;
    return Hit{best, bestPoint, mSegments[bestSegment].id};
}

float SegmentTree2D::distance(const Vec2 &p, float maxDistance) const
{
    const auto hit = nearest(p, maxDistance);
    return hit ? std::sqrt(hit->distanceSqr) : maxDistance;
}
}  // namespace springls
