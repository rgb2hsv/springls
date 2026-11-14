#pragma once
#include <springls/Common.hpp>

#include <vector>

namespace springls
{
// Static uniform bucket grid over points for fixed-radius queries. Points keep their index in
// the input array (or the id given to build()).
class PointGrid2D
{
public:
    struct Neighbor {
        float distanceSqr;
        int id;
    };

    // Builds over `points`; ids default to the point indexes. cellSize should be about the
    // typical query radius.
    void build(const Points2 &points, float cellSize, const std::vector<int> *ids = nullptr);
    bool empty() const { return mPoints.empty(); }

    // Every point within `radius` of p (inclusive), sorted by distance, ties by id.
    void within(const Vec2 &p, float radius, std::vector<Neighbor> &out) const;
    // Whether any point lies in the box [lo, hi] (stops at the first).
    bool anyInBox(const Vec2 &lo, const Vec2 &hi) const;

private:
    float mCellSize = 1.0f;
    Vec2i mOrigin = Vec2i::Zero();
    int mWidth = 0, mHeight = 0;
    Points2 mPoints;              // sorted by cell
    std::vector<int> mIds;        // parallel to mPoints
    std::vector<int> mCellStart;  // CSR offsets, mWidth * mHeight + 1

    int cellCoord(float v) const { return static_cast<int>(std::floor(v / mCellSize)); }
};
}  // namespace springls
