#pragma once
#include <springls/Common3D.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace springls
{
// Edge `edge` (vertex edge to vertex edge + 1, cyclic) of springl `springl`, and the squared
// distance from a query point to it.
struct SpringlEdge {
    uint32_t springl = 0;
    int8_t edge = -1;
    float distanceSqr = 0.0f;
    template <class Archive>
    void serialize(Archive &ar)
    {
        ar(springl, edge, distanceSqr);
    }
    friend bool operator<(const SpringlEdge &a, const SpringlEdge &b)
    {
        return (a.distanceSqr < b.distanceSqr) || (a.distanceSqr == b.distanceSqr && a.springl < b.springl);
    }
};

// Spatial index over 3D springls (particle + arm vertexes, any number per springl) for neighbor and closest-element
// queries; 3D counterpart of SpringlHashGrid2D.
//
// Springls are bucketed by particle into one of several levels by size (their longest arm), and
// each level is a sparse grid whose cell size matches its springls, like the per-level nodes of
// an OpenVDB tree. A query searches each level only as far as that level's springls reach, so on
// an adaptive grid tiny springls are found on a fine lattice and large ones on a coarse one
// instead of every query scanning a coarse cell crowded with tiny springls. Each level also keeps a
// tier of buckets kTierRatio times coarser (like a VDB internal node over its leaves), which wide
// queries (a coarse springl searching a fine level) walk instead of thousands of fine cells.
// update() rebuilds the
// index (a sort over the springls); cells within a level are sorted by springl index and ties are
// broken by index, so results are deterministic.
//
// A springl's surface element is the fan of triangles (particle, vertex k, vertex k + 1).
class SpringlHashGrid3D
{
public:
    // Kept for compatibility with incremental indexes: update() always rebuilds.
    void invalidate() {}
    // Rebuilds the index over `particles` (one per springl) and `vertexes`, springl n owning
    // vertexes [start[n], start[n + 1]). Queries must pass the same arrays.
    void update(const Points3 &particles, const Points3 &vertexes, const std::vector<uint32_t> &start);

    // Up to maxCount closest edges of distinct springls other than `exclude` (with accept(springl))
    // within `radius` of p, nearest first; each springl contributes its closest edge.
    template <class Accept>
    void nearestEdges(const Points3 &particles, const Points3 &vertexes, const Vec3 &p, float radius, size_t exclude, size_t maxCount, Accept &&accept,
                      std::vector<SpringlEdge> &out) const
    {
        out.clear();
        const float radiusSqr = radius * radius;
        for (const Level &level : mLevels) {
            // A springl's edges lie within its level's reach of its particle.
            const float rejectSqr = (radius + level.reach) * (radius + level.reach);
            level.forEachInBox(p, radius + level.reach, [&](uint32_t springl) {
                if (springl == exclude || distanceSqr(particles[springl], p) > rejectSqr || !accept(springl)) return true;
                float best = std::numeric_limits<float>::max();
                int bestEdge = -1;
                const uint32_t base = mStart[springl];
                const int K = static_cast<int>(mStart[springl + 1] - base);
                for (int k = 0; k < K; k++) {
                    const float d = SegmentDistanceSqr(vertexes[base + k], vertexes[base + (k + 1) % K], p);
                    if (d < best) {
                        best = d;
                        bestEdge = k;
                    }
                }
                if (bestEdge >= 0 && best <= radiusSqr) out.push_back({springl, static_cast<int8_t>(bestEdge), best});
                return true;
            });
        }
        if (out.size() > maxCount) {
            std::partial_sort(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(maxCount), out.end());
            out.resize(maxCount);
        } else {
            std::sort(out.begin(), out.end());
        }
    }

    // Whether any accepted springl element lies within maxDistance of p (stops at the first one).
    template <class Accept>
    bool anyElementWithin(const Points3 &particles, const Points3 &vertexes, const Vec3 &p, float maxDistance, Accept &&accept) const
    {
        const float maxDistanceSqr = maxDistance * maxDistance;
        for (const Level &level : mLevels) {
            const float rejectSqr = (maxDistance + level.reach) * (maxDistance + level.reach);
            bool found = false;
            level.forEachInBox(p, maxDistance + level.reach, [&](uint32_t springl) {
                if (distanceSqr(particles[springl], p) > rejectSqr || !accept(springl)) return true;
                found = ElementDistanceSqr(particles, vertexes, mStart[springl], static_cast<int>(mStart[springl + 1] - mStart[springl]), springl, p) <=
                        maxDistanceSqr;
                return !found;
            });
            if (found) return true;
        }
        return false;
    }

    // Squared distance from p to the triangle fan of springl `springl` (vertexes [base, base + K)).
    static float ElementDistanceSqr(const Points3 &particles, const Points3 &vertexes, uint32_t base, int K, size_t springl, const Vec3 &p)
    {
        const Vec3 &c = particles[springl];
        float best = std::numeric_limits<float>::max();
        for (int k = 0; k < K; k++) best = std::min(best, TriangleDistanceSqr(c, vertexes[base + k], vertexes[base + (k + 1) % K], p));
        return best;
    }

private:
    static constexpr int kTierRatio = 8;
    // A uniform bucket grid: springls sorted by (cell, index), and each occupied cell's range.
    struct Buckets {
        float cellSize = 1.0f;
        std::vector<uint32_t> items;
        std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> cells;

        int64_t coord(float x) const { return static_cast<int64_t>(std::floor(x / cellSize)); }
        void build(const Points3 &particles, const std::vector<uint32_t> &springls);
        static uint64_t Key(int64_t x, int64_t y, int64_t z)
        {
            constexpr int64_t kBias = 1 << 20;
            return (static_cast<uint64_t>(x + kBias) << 42) | (static_cast<uint64_t>(y + kBias) << 21) | static_cast<uint64_t>(z + kBias);
        }
        // Calls fn(springl) for the springls in the cells overlapping the box p +- reach, until fn
        // returns false; returns false if fn stopped the walk.
        template <class Fn>
        bool forEachInBox(const Vec3 &p, float reach, Fn &&fn) const
        {
            const int64_t x0 = coord(p.x() - reach), x1 = coord(p.x() + reach);
            const int64_t y0 = coord(p.y() - reach), y1 = coord(p.y() + reach);
            const int64_t z0 = coord(p.z() - reach), z1 = coord(p.z() + reach);
            for (int64_t z = z0; z <= z1; z++) {
                for (int64_t y = y0; y <= y1; y++) {
                    for (int64_t x = x0; x <= x1; x++) {
                        const auto it = cells.find(Key(x, y, z));
                        if (it == cells.end()) continue;
                        for (uint32_t s = it->second.first; s < it->second.second; s++) {
                            if (!fn(items[s])) return false;
                        }
                    }
                }
            }
            return true;
        }
    };
    struct Level {
        float reach = 0.0f;    // longest particle-to-vertex distance of this level's springls
        Buckets fine, coarse;  // cells sized to the springls, and kTierRatio times larger

        // Calls fn(springl) for the springls in cells overlapping the box p +- reach (a superset),
        // until fn returns false; uses whichever tier visits fewer cells.
        template <class Fn>
        void forEachInBox(const Vec3 &p, float box, Fn &&fn) const
        {
            const float span = 2.0f * box / fine.cellSize + 1.0f;
            if (span * span * span > 64.0f) {
                coarse.forEachInBox(p, box, fn);
            } else {
                fine.forEachInBox(p, box, fn);
            }
        }
    };
    std::vector<uint32_t> mStart;  // copy of the springl offsets the index was built with
    std::vector<Level> mLevels;    // non-empty levels, finest first
};
}  // namespace springls
