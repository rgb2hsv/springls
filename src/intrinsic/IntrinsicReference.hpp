#pragma once
// The initial contour's points advected by the flow alone (no relaxation, filling, contraction or level
// set): the comparison for the springls.
#include <intrinsic/IntrinsicContour.hpp>
#include <intrinsic/IntrinsicTriangulation.hpp>

#include <functional>
#include <vector>

namespace springls
{
class IntrinsicReference
{
public:
    // One closed polyline per initial contour curve: its crossing points plus `subdivisions - 1` points
    // inside each segment (in the segment's face).
    void initialize(const IntrinsicTriangulation &tri, const std::vector<IntrinsicCurve> &contour, int subdivisions = 4);
    // RK4 with stage points by Trace and velocities transported back, sub-steps so no point moves more
    // than maxStep (absolute length) per sub-step, never stepping across a time in `breaks` (the
    // schedule's sign change): the velocity is evaluated at a sub-step's end as the limit from before it.
    void advance(const IntrinsicTriangulation &tri, const std::function<Vec2d(const SurfacePoint &, double)> &velocity, double fromTime, double toTime,
                 double maxStep, const std::vector<double> &breaks);
    const std::vector<std::vector<SurfacePoint>> &curves() const { return mCurves; }
    const std::vector<int> &labels() const { return mLabels; }
    // Every point, for relocation through a refinement of the triangulation they live on.
    std::vector<SurfacePoint *> points();
    // Largest LocalLog distance between consecutive points (+inf if any pair is out of LocalLog's reach).
    double maxGap(const IntrinsicTriangulation &tri) const;
    // Geodesic distance field of the reference (fast marching from its segments, consecutive points joined
    // by LocalLog; a pair out of reach seeds its points only), capped at maxDistance.
    std::vector<double> distanceField(const IntrinsicTriangulation &tri, double maxDistance) const;

    // The distance from each point to the reference, capped: exact (to the straight segments, in a local
    // chart of the point's face) when a reference segment has both ends within the chart; otherwise
    // the fast-marching field's interpolated value (computed only if some point needs it).
    std::vector<double> distancesAt(const IntrinsicTriangulation &tri, const std::vector<SurfacePoint> &points, double cap) const;

private:
    std::vector<std::vector<SurfacePoint>> mCurves;
    std::vector<int> mLabels;
    std::vector<bool> mClosed;
};
}  // namespace springls
