#include <springls/TriangleTree3D.hpp>

#include <algorithm>
#include <limits>
#include <map>

namespace springls
{
void TriangleTree3D::reset()
{
    mTriangles.clear();
    mNodes.clear();
}

void TriangleTree3D::append(const Vec3 &a, const Vec3 &b, const Vec3 &c, int id) { mTriangles.push_back({a, b, c, id}); }

void TriangleTree3D::finish()
{
    mNodes.clear();
    if (mTriangles.empty()) return;
    // Every subtree's node slots are known in advance (siblings are adjacent, and a node's pair of
    // children is followed by the left subtree's nodes, then the right's), so subtrees build in
    // parallel into the same layout a sequential build produces.
    std::map<std::pair<int, int>, int> pairs;
    mNodes.resize(1 + 2 * static_cast<size_t>(pairCount(static_cast<int>(mTriangles.size()), 0, pairs)));
    SPRINGLS_OMP(omp parallel)
    SPRINGLS_OMP(omp single)
    build(0, 0, static_cast<int>(mTriangles.size()), 0, 1, pairs);
}

int TriangleTree3D::pairCount(int count, int depth, std::map<std::pair<int, int>, int> &memo)
{
    if (count <= kLeafSize || depth >= kMaxStack - 2) return 0;
    const auto key = std::make_pair(count, depth);
    const auto it = memo.find(key);
    if (it != memo.end()) return it->second;
    const int half = count / 2;
    const int pairs = 1 + pairCount(half, depth + 1, memo) + pairCount(count - half, depth + 1, memo);
    memo.emplace(key, pairs);
    return pairs;
}

void TriangleTree3D::build(int node, int begin, int end, int depth, int next, const std::map<std::pair<int, int>, int> &pairs)
{
    Vec3 lo = mTriangles[begin].a, hi = lo;
    for (int t = begin; t < end; t++) {
        const Triangle &tri = mTriangles[t];
        lo = lo.cwiseMin(tri.a).cwiseMin(tri.b).cwiseMin(tri.c);
        hi = hi.cwiseMax(tri.a).cwiseMax(tri.b).cwiseMax(tri.c);
    }
    mNodes[node].lo = lo;
    mNodes[node].hi = hi;
    if (end - begin <= kLeafSize || depth >= kMaxStack - 2) {
        mNodes[node].first = begin;
        mNodes[node].count = end - begin;
        return;
    }
    // Median split of triangle centroids along the longest axis.
    const Vec3 extent = hi - lo;
    int axis = 0;
    if (extent.y() > extent[axis]) axis = 1;
    if (extent.z() > extent[axis]) axis = 2;
    const int mid = (begin + end) / 2;
    std::nth_element(mTriangles.begin() + begin, mTriangles.begin() + mid, mTriangles.begin() + end,
                     [axis](const Triangle &l, const Triangle &r) { return (l.a[axis] + l.b[axis] + l.c[axis]) < (r.a[axis] + r.b[axis] + r.c[axis]); });
    const int left = next;
    mNodes[node].first = left;
    mNodes[node].count = 0;
    const auto pairsOf = [&](int count, int d) {
        if (count <= kLeafSize || d >= kMaxStack - 2) return 0;
        return pairs.at(std::make_pair(count, d));
    };
    const int leftNext = left + 2;
    const int rightNext = leftNext + 2 * pairsOf(mid - begin, depth + 1);
    constexpr int kTaskMin = 16384;  // triangles; smaller subtrees build in their parent's task
    if (end - begin >= kTaskMin) {
        SPRINGLS_OMP(omp task)
        build(left, begin, mid, depth + 1, leftNext, pairs);
        build(left + 1, mid, end, depth + 1, rightNext, pairs);
        SPRINGLS_OMP(omp taskwait)
    } else {
        build(left, begin, mid, depth + 1, leftNext, pairs);
        build(left + 1, mid, end, depth + 1, rightNext, pairs);
    }
}

std::optional<TriangleTree3D::Hit> TriangleTree3D::nearest(const Vec3 &p, float maxDistance) const
{
    if (mNodes.empty()) return std::nullopt;
    float best = std::isinf(maxDistance) ? std::numeric_limits<float>::infinity() : maxDistance * maxDistance;
    int bestTriangle = -1;
    Vec3 bestPoint = p;
    int stack[kMaxStack];
    int top = 0;
    stack[top++] = 0;
    while (top > 0) {
        const Node &node = mNodes[stack[--top]];
        if (boxDistanceSqr(node, p) >= best) continue;
        if (node.count > 0) {
            for (int t = node.first; t < node.first + node.count; t++) {
                const Triangle &tri = mTriangles[t];
                Vec3 q;
                const float d2 = TriangleDistanceSqr(tri.a, tri.b, tri.c, p, &q);
                // Ties go to the smaller id so results do not depend on tree layout.
                if (d2 < best || (d2 == best && bestTriangle >= 0 && tri.id < mTriangles[bestTriangle].id)) {
                    best = d2;
                    bestTriangle = t;
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
    if (bestTriangle < 0) return std::nullopt;
    return Hit{best, bestPoint, mTriangles[bestTriangle].id};
}

float TriangleTree3D::distance(const Vec3 &p, float maxDistance) const
{
    const auto hit = nearest(p, maxDistance);
    return hit ? std::sqrt(hit->distanceSqr) : maxDistance;
}
}  // namespace springls
