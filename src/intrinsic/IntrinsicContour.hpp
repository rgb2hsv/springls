#pragma once
// Labeled contours on an intrinsic triangulation: marching triangles per object label on the label's
// signed vertex values (-d inside, +d outside), with per-label copies of crossings on shared edges
// (as ExtractMultiLabelContour in 2D). Every crossing lies on an intrinsic edge and every contour
// segment inside one face.
#include <intrinsic/IntrinsicTriangulation.hpp>

#include <functional>
#include <vector>

namespace springls
{
// A curve of region `label`, the region on its left. Point k lies on edge edges[k] at parameter t[k]
// along that edge's canonical halfedge (edgeHalfedge), expressed in that halfedge's face. Segment k
// joins point k to point k + 1 (to point 0 after the last when closed) inside face faces[k].
struct IntrinsicCurve {
    int label = 0;
    std::vector<SurfacePoint> points;
    std::vector<uint32_t> edges;
    std::vector<double> t;
    std::vector<uint32_t> faces;
    bool closed = false;
};
// Point k of the curve expressed in face f, which must contain its edge.
SurfacePoint ContourPointInFace(const IntrinsicTriangulation &tri, const IntrinsicCurve &curve, size_t k, uint32_t f);

// Inside a region: signedValue < 0.
std::vector<IntrinsicCurve> ExtractIntrinsicContour(const IntrinsicTriangulation &tri, const std::function<double(uint32_t v, int label)> &signedValue,
                                                    const std::vector<int> &objectLabels);
// 1D refinement: each crossing slides along its edge toward the nearest same-label particle within
// `proximity` (absolute length): t moves by stepSize * w toward the particle's projection on the edge,
// w = 1 - distance / proximity (the 2D refineContour weights). Crossings stay inside their edge.
void RefineIntrinsicContour(const IntrinsicTriangulation &tri, std::vector<IntrinsicCurve> &curves, const std::vector<SurfacePoint> &particles,
                            const std::vector<int> &particleLabels, int iterations, double proximity, double stepSize);
// Exact area of {φ < 0} for the piecewise-linear φ of vertex values (per-face clipping).
double RegionArea(const IntrinsicTriangulation &tri, const std::function<double(uint32_t v)> &signedValue);
// The part of {φ < 0} inside one face (the same per-face clipping as RegionArea): its area and the
// centroid of the clipped polygon, as a point of the face.
struct RegionPiece {
    uint32_t face = 0;
    double area = 0.0;
    SurfacePoint centroid;
};
// The non-empty pieces of {φ < 0}, in face order; their areas sum to RegionArea.
std::vector<RegionPiece> RegionPieces(const IntrinsicTriangulation &tri, const std::function<double(uint32_t v)> &signedValue);
// Area of region `label` bounded by its extracted (and refined) curves: as RegionArea on the label's
// signed values, with each crossing where the curves put it.
double ContourArea(const IntrinsicTriangulation &tri, const std::vector<IntrinsicCurve> &curves, int label,
                   const std::function<double(uint32_t v, int label)> &signedValue);
}  // namespace springls
