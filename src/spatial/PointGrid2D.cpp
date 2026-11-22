#include <spatial/PointGrid2D.hpp>

#include <algorithm>

namespace springls
{
void PointGrid2D::build(const Points2 &points, float cellSize, const std::vector<int> *ids)
{
    mCellSize = std::max(cellSize, 1E-6f);
    mPoints.clear();
    mIds.clear();
    mCellStart.clear();
    mWidth = mHeight = 0;
    const size_t N = points.size();
    if (N == 0) return;
    Vec2i lo(cellCoord(points[0].x()), cellCoord(points[0].y())), hi = lo;
    for (const Vec2 &p : points) {
        const Vec2i c(cellCoord(p.x()), cellCoord(p.y()));
        lo = lo.cwiseMin(c);
        hi = hi.cwiseMax(c);
    }
    mOrigin = lo;
    mWidth = hi.x() - lo.x() + 1;
    mHeight = hi.y() - lo.y() + 1;
    const size_t cells = static_cast<size_t>(mWidth) * static_cast<size_t>(mHeight);
    std::vector<int> cellOf(N);
    mCellStart.assign(cells + 1, 0);
    for (size_t n = 0; n < N; n++) {
        const int cx = cellCoord(points[n].x()) - mOrigin.x();
        const int cy = cellCoord(points[n].y()) - mOrigin.y();
        cellOf[n] = cy * mWidth + cx;
        mCellStart[cellOf[n] + 1]++;
    }
    for (size_t c = 0; c < cells; c++) mCellStart[c + 1] += mCellStart[c];
    std::vector<int> fill(mCellStart.begin(), mCellStart.end() - 1);
    mPoints.resize(N);
    mIds.resize(N);
    for (size_t n = 0; n < N; n++) {
        const int slot = fill[cellOf[n]]++;
        mPoints[slot] = points[n];
        mIds[slot] = ids ? (*ids)[n] : static_cast<int>(n);
    }
}

bool PointGrid2D::anyInBox(const Vec2 &lo, const Vec2 &hi) const
{
    if (mPoints.empty()) return false;
    const int x0 = std::max(cellCoord(lo.x()) - mOrigin.x(), 0), x1 = std::min(cellCoord(hi.x()) - mOrigin.x(), mWidth - 1);
    const int y0 = std::max(cellCoord(lo.y()) - mOrigin.y(), 0), y1 = std::min(cellCoord(hi.y()) - mOrigin.y(), mHeight - 1);
    for (int cy = y0; cy <= y1; cy++) {
        for (int cx = x0; cx <= x1; cx++) {
            const int c = cy * mWidth + cx;
            for (int s = mCellStart[c]; s < mCellStart[c + 1]; s++) {
                if ((mPoints[s].array() >= lo.array()).all() && (mPoints[s].array() <= hi.array()).all()) return true;
            }
        }
    }
    return false;
}

void PointGrid2D::within(const Vec2 &p, float radius, std::vector<Neighbor> &out) const
{
    out.clear();
    if (mPoints.empty()) return;
    const float r2 = radius * radius;
    const int x0 = std::max(cellCoord(p.x() - radius) - mOrigin.x(), 0);
    const int x1 = std::min(cellCoord(p.x() + radius) - mOrigin.x(), mWidth - 1);
    const int y0 = std::max(cellCoord(p.y() - radius) - mOrigin.y(), 0);
    const int y1 = std::min(cellCoord(p.y() + radius) - mOrigin.y(), mHeight - 1);
    for (int cy = y0; cy <= y1; cy++) {
        for (int cx = x0; cx <= x1; cx++) {
            const int c = cy * mWidth + cx;
            for (int s = mCellStart[c]; s < mCellStart[c + 1]; s++) {
                const float d2 = distanceSqr(mPoints[s], p);
                if (d2 <= r2) out.push_back({d2, mIds[s]});
            }
        }
    }
    std::sort(out.begin(), out.end(),
              [](const Neighbor &a, const Neighbor &b) { return (a.distanceSqr != b.distanceSqr) ? a.distanceSqr < b.distanceSqr : a.id < b.id; });
}
}  // namespace springls
