#include <springls/PointGrid3D.hpp>

#include <algorithm>

namespace springls
{
void PointGrid3D::build(const Points3 &points, float cellSize, const std::vector<int> *ids)
{
    mCellSize = std::max(cellSize, 1E-6f);
    mPoints.clear();
    mIds.clear();
    mCellStart.clear();
    mDims = Vec3i::Zero();
    const size_t N = points.size();
    if (N == 0) return;
    Vec3i lo = cellOf(points[0]), hi = lo;
    for (const Vec3 &p : points) {
        const Vec3i c = cellOf(p);
        lo = lo.cwiseMin(c);
        hi = hi.cwiseMax(c);
    }
    mOrigin = lo;
    mDims = hi - lo + Vec3i::Ones();
    const size_t cells = static_cast<size_t>(mDims.x()) * mDims.y() * mDims.z();
    std::vector<size_t> cellOfPoint(N);
    mCellStart.assign(cells + 1, 0);
    for (size_t n = 0; n < N; n++) {
        cellOfPoint[n] = cellIndex(cellOf(points[n]) - mOrigin);
        mCellStart[cellOfPoint[n] + 1]++;
    }
    for (size_t c = 0; c < cells; c++) mCellStart[c + 1] += mCellStart[c];
    std::vector<int> fill(mCellStart.begin(), mCellStart.end() - 1);
    mPoints.resize(N);
    mIds.resize(N);
    for (size_t n = 0; n < N; n++) {
        const int slot = fill[cellOfPoint[n]]++;
        mPoints[slot] = points[n];
        mIds[slot] = ids ? (*ids)[n] : static_cast<int>(n);
    }
}

bool PointGrid3D::anyInBox(const Vec3 &lo, const Vec3 &hi) const
{
    if (mPoints.empty()) return false;
    const Vec3i a = (cellOf(lo) - mOrigin).cwiseMax(Vec3i::Zero());
    const Vec3i b = (cellOf(hi) - mOrigin).cwiseMin(mDims - Vec3i::Ones());
    for (int cz = a.z(); cz <= b.z(); cz++) {
        for (int cy = a.y(); cy <= b.y(); cy++) {
            for (int cx = a.x(); cx <= b.x(); cx++) {
                const size_t c = cellIndex(Vec3i(cx, cy, cz));
                for (int s = mCellStart[c]; s < mCellStart[c + 1]; s++) {
                    if ((mPoints[s].array() >= lo.array()).all() && (mPoints[s].array() <= hi.array()).all()) return true;
                }
            }
        }
    }
    return false;
}

void PointGrid3D::within(const Vec3 &p, float radius, std::vector<Neighbor> &out) const
{
    out.clear();
    if (mPoints.empty()) return;
    const float r2 = radius * radius;
    const Vec3i a = (cellOf(p - Vec3::Constant(radius)) - mOrigin).cwiseMax(Vec3i::Zero());
    const Vec3i b = (cellOf(p + Vec3::Constant(radius)) - mOrigin).cwiseMin(mDims - Vec3i::Ones());
    for (int cz = a.z(); cz <= b.z(); cz++) {
        for (int cy = a.y(); cy <= b.y(); cy++) {
            for (int cx = a.x(); cx <= b.x(); cx++) {
                const size_t c = cellIndex(Vec3i(cx, cy, cz));
                for (int s = mCellStart[c]; s < mCellStart[c + 1]; s++) {
                    const float d2 = distanceSqr(mPoints[s], p);
                    if (d2 <= r2) out.push_back({d2, mIds[s]});
                }
            }
        }
    }
    std::sort(out.begin(), out.end(),
              [](const Neighbor &l, const Neighbor &r) { return (l.distanceSqr != r.distanceSqr) ? l.distanceSqr < r.distanceSqr : l.id < r.id; });
}
}  // namespace springls
