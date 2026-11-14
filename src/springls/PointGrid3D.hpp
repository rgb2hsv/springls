#pragma once
#include <springls/Common3D.hpp>

#include <vector>

namespace springls
{
// Static uniform bucket grid over 3D points for fixed-radius queries (3D counterpart of
// PointGrid2D). Points keep their index in the input array (or the id given to build()).
class PointGrid3D
{
public:
    struct Neighbor {
        float distanceSqr;
        int id;
    };

    // Builds over `points`; ids default to the point indexes. cellSize should be about the
    // typical query radius.
    void build(const Points3 &points, float cellSize, const std::vector<int> *ids = nullptr);
    bool empty() const { return mPoints.empty(); }

    // Every point within `radius` of p (inclusive), sorted by distance, ties by id.
    void within(const Vec3 &p, float radius, std::vector<Neighbor> &out) const;
    // Whether any point lies in the box [lo, hi] (stops at the first).
    bool anyInBox(const Vec3 &lo, const Vec3 &hi) const;

private:
    float mCellSize = 1.0f;
    Vec3i mOrigin = Vec3i::Zero();
    Vec3i mDims = Vec3i::Zero();
    Points3 mPoints;              // sorted by cell
    std::vector<int> mIds;        // parallel to mPoints
    std::vector<int> mCellStart;  // CSR offsets, cell count + 1

    int cellCoord(float v) const { return static_cast<int>(std::floor(v / mCellSize)); }
    Vec3i cellOf(const Vec3 &p) const { return Vec3i(cellCoord(p.x()), cellCoord(p.y()), cellCoord(p.z())); }
    size_t cellIndex(const Vec3i &c) const { return (static_cast<size_t>(c.z()) * mDims.y() + c.y()) * mDims.x() + c.x(); }
};
}  // namespace springls
