#pragma once
// Moving and measuring along an intrinsic triangulation: the discrete exponential map with parallel
// transport (Trace / TransportBack), local charts (faces unfolded into one face's layout) and the
// short-range log map (LocalLog) [Sharp 2021, Sections 2.4 and 3.4].
#include <intrinsic/IntrinsicTriangulation.hpp>

#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>

namespace springls
{
Vec2d ToLayout(const IntrinsicTriangulation &tri, const SurfacePoint &p);  // in layout(p.face)

// direction: unit direction at the end in end.face's layout. rotation: transport angle (end frame
// minus start frame): a vector w at the start is R(rotation) w at the end.
struct TraceResult {
    SurfacePoint end;
    Vec2d direction = Vec2d(1, 0);
    double rotation = 0.0;
};
// Discrete exponential map: `displacement` is in start.face's layout.
TraceResult Trace(const IntrinsicTriangulation &tri, const SurfacePoint &start, const Vec2d &displacement, std::vector<EdgeCrossing> *crossings = nullptr);
// A vector at the trace's end (end.face's layout) carried back into the start face's layout.
Vec2d TransportBack(const TraceResult &trace, const Vec2d &vectorAtEnd);
Vec2d Rotate(const Vec2d &v, double angle);

// Faces around a center face unfolded into the center face's layout (breadth first over shared edges,
// up to `depth` edges away). Each face is placed by its shortest face path; with maxPlacements > 1
// a face reached around a vertex from both sides (where the surface is curved, the two unfoldings
// differ) keeps up to that many distinct placements. Cheap positions of nearby points in one plane.
class LocalChart
{
public:
    LocalChart() = default;
    LocalChart(const IntrinsicTriangulation &tri, uint32_t centerFace, int depth, int maxPlacements = 1) { build(tri, centerFace, depth, maxPlacements); }
    // With stopAt (maxPlacements 1 only): the unfolding stops as soon as that face is placed, so the chart holds
    // the faces placed up to then (stopAt included, if within depth), each where the full build places it.
    void build(const IntrinsicTriangulation &tri, uint32_t centerFace, int depth, int maxPlacements = 1, uint32_t stopAt = kNoFace);
    uint32_t center() const { return mCenter; }
    bool contains(uint32_t face) const { return find(face) != kNoEntry; }
    const std::vector<uint32_t> &faces() const { return mFaces; }
    // Position of a point in a contained face / a vector given in that face's layout, in chart coordinates.
    Vec2d position(const SurfacePoint &p) const
    {
        const auto &C = mCorners[at(p.face)];
        return p.bary[0] * C[0] + p.bary[1] * C[1] + p.bary[2] * C[2];
    }
    Vec2d vector(uint32_t face, const Vec2d &w) const;
    // Inverse of vector(): a chart vector expressed in `face`'s layout.
    Vec2d vectorInFace(uint32_t face, const Vec2d &w) const;
    // Chart positions of a contained face's corners.
    const std::array<Vec2d, 3> &corners(uint32_t face) const { return mCorners[at(face)]; }
    // Every placement of a contained face (the first is the one used above).
    std::vector<std::array<Vec2d, 3>> placements(uint32_t face) const;

    static constexpr uint32_t kNoFace = std::numeric_limits<uint32_t>::max();

private:
    static constexpr uint32_t kNoEntry = std::numeric_limits<uint32_t>::max();
    // First placement of a face: open addressing (linear probing) over mSlots, a power of two in size.
    uint32_t find(uint32_t face) const
    {
        if (mSlots.empty()) return kNoEntry;
        const size_t mask = mSlots.size() - 1;
        for (size_t i = Hash(face) & mask;; i = (i + 1) & mask) {
            const uint32_t e = mSlots[i];
            if (e == kNoEntry || mFaces[e] == face) return e;
        }
    }
    // As find(), but a face not in the chart is an error (std::out_of_range).
    uint32_t at(uint32_t face) const;
    static size_t Hash(uint32_t face) { return static_cast<size_t>(face * 0x9E3779B1u); }
    void insertFirst(uint32_t face, uint32_t entry);

    uint32_t mCenter = 0;
    std::vector<uint32_t> mFaces;
    std::vector<std::array<Vec2d, 3>> mCorners;
    std::vector<Vec2d> mRotation;  // rotation from face layout to chart, as (cos, sin)
    std::vector<uint32_t> mNext;   // next placement of the same face (kNoEntry: none)
    std::vector<uint32_t> mSlots;  // face -> first placement
};

// Displacement in from.face's layout whose trace ends at `to`, if `to` lies in the faces within
// maxRings rings of `from` (refined by tracing until it lands on `to`).
std::optional<Vec2d> LocalLog(const IntrinsicTriangulation &tri, const SurfacePoint &from, const SurfacePoint &to, int maxRings = 3);
// |LocalLog|, +inf if unreached.
double GeodesicDistanceLocal(const IntrinsicTriangulation &tri, const SurfacePoint &a, const SurfacePoint &b, int maxRings = 3);
}  // namespace springls
