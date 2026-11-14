#pragma once
#include <springls/Common.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace springls
{
// Uniform grid over springls (particle + two arm vertexes), for neighbor and closest-segment
// queries. Cells live in a dense table over the springls' bounding region (grown with a margin
// when a springl leaves it), so looking up a cell, empty or not, is one array read.
//
// Each springl is bucketed by its particle; queries widen their search by the longest arm so whole
// springls are always found. Unlike a kd-tree it is updated incrementally: update() only moves
// springls whose particle changed cell and appends new ones, and rebuilds from scratch only after
// invalidate() (when springl indexes were remapped, e.g. by compaction). Buckets are contiguous
// index arrays, and ties in every query are broken by index so results are deterministic.
class SpringlHashGrid2D
{
public:
    static constexpr float kDefaultCellSize = 1.0f;

    explicit SpringlHashGrid2D(float cellSize = kDefaultCellSize) : mCellSize(cellSize) {}

    float cellSize() const { return mCellSize; }
    size_t size() const { return mItemCx.size(); }
    // The next update() rebuilds every bucket (springl indexes changed).
    void invalidate() { mValid = false; }

    // Brings the buckets up to date with `particles` (one per springl) and `vertexes` (two per springl).
    void update(const Points2 &particles, const Points2 &vertexes);

    // Arm vertex index (2 * springl + end) of the vertex nearest p within `radius`, over springls
    // other than `exclude` for which accept(springl) holds; -1 if there is none.
    template <class Accept>
    int nearestVertex(const Points2 &vertexes, const Vec2 &p, float radius, size_t exclude, Accept &&accept) const
    {
        int best = -1;
        double bestDistanceSqr = static_cast<double>(radius) * radius;
        forEachInBox(p, radius + mReach, [&](uint32_t springl) {
            if (springl == exclude || !accept(springl)) return;
            for (uint32_t v = 2 * springl; v <= 2 * springl + 1; v++) {
                const double dx = static_cast<double>(vertexes[v].x()) - p.x(), dy = static_cast<double>(vertexes[v].y()) - p.y();
                const double d2 = dx * dx + dy * dy;
                if (d2 < bestDistanceSqr || (d2 == bestDistanceSqr && (best < 0 || static_cast<int>(v) < best))) {
                    bestDistanceSqr = d2;
                    best = static_cast<int>(v);
                }
            }
        });
        return best;
    }

    // Distance from p to the closest point on any accepted springl segment (vertex 0 to vertex 1),
    // if one lies within maxDistance. Searches rings of cells outward and stops once no farther
    // cell can hold anything closer.
    template <class Accept>
    std::optional<double> closestSegment(const Points2 &vertexes, const Vec2 &p, float maxDistance, Accept &&accept, Vec2 *closestPoint = nullptr,
                                         int *closestSpringl = nullptr) const
    {
        const int64_t cx = cellCoord(p.x()), cy = cellCoord(p.y());
        double bestDistanceSqr = static_cast<double>(maxDistance) * maxDistance;
        int best = -1;
        Vec2 bestPoint;
        // A point at distance D from p lies at most floor(D / cell) + 1 rings out (p may sit on its cell's edge).
        const int64_t rings = static_cast<int64_t>(std::ceil((maxDistance + mReach) / mCellSize)) + 1;
        for (int64_t k = 0; k <= rings; k++) {
            // Anything in ring k or beyond is at least (k - 1) cells minus one arm away.
            const double bound = (k - 1) * static_cast<double>(mCellSize) - mReach;
            if (best >= 0 && bound > 0.0 && bound * bound > bestDistanceSqr) break;
            forEachInRing(cx, cy, k, [&](uint32_t springl) {
                if (!accept(springl)) return;
                Vec2 q;
                const double d2 = segmentDistanceSqr(vertexes[2 * springl], vertexes[2 * springl + 1], p, q);
                if (d2 < bestDistanceSqr || (d2 == bestDistanceSqr && (best < 0 || static_cast<int>(springl) < best))) {
                    bestDistanceSqr = d2;
                    best = static_cast<int>(springl);
                    bestPoint = q;
                }
            });
        }
        if (best < 0) return std::nullopt;
        if (closestPoint) *closestPoint = bestPoint;
        if (closestSpringl) *closestSpringl = best;
        return std::sqrt(bestDistanceSqr);
    }

    // Approximate heap bytes held by the buckets and the index.
    size_t memoryBytes() const;

private:
    float mCellSize;
    float mReach = 0.0f;  // longest particle-to-vertex distance
    bool mValid = false;
    // Dense cell table: bucket of cell (cx, cy) at mTable[(cy - mTableY0) * mTableW + (cx - mTableX0)], or -1.
    std::vector<int> mTable;
    int64_t mTableX0 = 0, mTableY0 = 0, mTableW = 0, mTableH = 0;
    std::vector<std::vector<uint32_t>> mBuckets;
    std::vector<int> mFreeBuckets;
    std::vector<int32_t> mItemCx, mItemCy;            // cell of each springl
    std::vector<uint32_t> mItemSlot;                  // position of each springl in its bucket
    std::vector<std::array<int64_t, 2>> mBucketCell;  // cell of each bucket

    int64_t cellCoord(float x) const { return static_cast<int64_t>(std::floor(x / mCellSize)); }
    void rebuild(const Points2 &particles);
    // Makes the table cover cells [x0, x1] x [y0, y1] (growing it with a margin if needed).
    void ensureTable(int64_t x0, int64_t y0, int64_t x1, int64_t y1);
    void insert(uint32_t springl, int64_t cx, int64_t cy);
    void remove(uint32_t springl);

    int bucketAt(int64_t cx, int64_t cy) const
    {
        const int64_t x = cx - mTableX0, y = cy - mTableY0;
        if (x < 0 || y < 0 || x >= mTableW || y >= mTableH) return -1;
        return mTable[static_cast<size_t>(y * mTableW + x)];
    }
    bool inTable(int64_t cx, int64_t cy) const { return cx >= mTableX0 && cy >= mTableY0 && cx < mTableX0 + mTableW && cy < mTableY0 + mTableH; }
    template <class Fn>
    void forEachInCell(int64_t cx, int64_t cy, Fn &&fn) const
    {
        const int bucket = bucketAt(cx, cy);
        if (bucket < 0) return;
        for (uint32_t springl : mBuckets[bucket]) fn(springl);
    }
    template <class Fn>
    void forEachInBox(const Vec2 &p, float reach, Fn &&fn) const
    {
        const int64_t x0 = cellCoord(p.x() - reach), x1 = cellCoord(p.x() + reach);
        const int64_t y0 = cellCoord(p.y() - reach), y1 = cellCoord(p.y() + reach);
        for (int64_t cy = y0; cy <= y1; cy++) {
            for (int64_t cx = x0; cx <= x1; cx++) forEachInCell(cx, cy, fn);
        }
    }
    template <class Fn>
    void forEachInRing(int64_t cx, int64_t cy, int64_t k, Fn &&fn) const
    {
        if (k == 0) {
            forEachInCell(cx, cy, fn);
            return;
        }
        for (int64_t x = cx - k; x <= cx + k; x++) {
            forEachInCell(x, cy - k, fn);
            forEachInCell(x, cy + k, fn);
        }
        for (int64_t y = cy - k + 1; y <= cy + k - 1; y++) {
            forEachInCell(cx - k, y, fn);
            forEachInCell(cx + k, y, fn);
        }
    }
    static double segmentDistanceSqr(const Vec2 &a, const Vec2 &b, const Vec2 &p, Vec2 &closest)
    {
        const double abx = static_cast<double>(b.x()) - a.x(), aby = static_cast<double>(b.y()) - a.y();
        const double apx = static_cast<double>(p.x()) - a.x(), apy = static_cast<double>(p.y()) - a.y();
        const double len2 = abx * abx + aby * aby;
        const double t = (len2 > 0.0) ? std::clamp((apx * abx + apy * aby) / len2, 0.0, 1.0) : 0.0;
        const double qx = a.x() + t * abx, qy = a.y() + t * aby;
        closest = Vec2(static_cast<float>(qx), static_cast<float>(qy));
        const double dx = p.x() - qx, dy = p.y() - qy;
        return dx * dx + dy * dy;
    }
};
}  // namespace springls
