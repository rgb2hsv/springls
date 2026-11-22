#pragma once
// The closed-loop scenario for intrinsic springls: an object is carried around the z axis and comes
// back. The velocity is defined on the input mesh, so refinement does not change the flow. Two flows:
//  - Rotation: omega ẑ × (x - c) at the exact 3D position, projected onto the input face's plane (on a
//    sphere centred at c: a Killing field, tangent, divergence-free, its orbits the latitude circles;
//    on any other surface the projection compresses and stretches, so areas are not kept).
//  - Hodge: the divergence-free part of that projected rotation by the Helmholtz-Hodge decomposition
//    (HodgeDecomposition.hpp): n × ∇ψ per input face, ψ the least-squares stream function. Exactly
//    divergence-free on any closed surface (flux continuous across every edge), its streamlines the
//    level sets of ψ, which wind around the axis much as the rotation's orbits do. Scaled so the
//    object's center moves at `speed`; one revolution is the traced time for the center to come back.
//    By default the face velocities are averaged to the vertices and interpolated (LoopConfig::smoothVelocity).
// The schedule scales the flow by cos(πt/T), as the Enright tests do: it slows smoothly to rest at T/2,
// one revolution on, and runs back to the start at T, so the exact solution at the end is the initial
// object on any surface.
#include <intrinsic/IntrinsicSpringls.hpp>

#include <string>
#include <vector>

namespace springls
{
enum class LoopObject {
    PuzzlePiece,
    Star
};
enum class LoopFlow {
    Rotation,
    Hodge
};
struct LoopConfig {
    double speed = 1.0;
    double loopDistance = 0.0;  // ρ: geodesic distance of the object's center from c; 0 picks it
    double loopFraction = 1.0;  // with loopDistance 0: ρ = loopFraction × half the distance from c to the vertex nearest -z
    double radius = 0.0;        // r: object size; 0 uses ρ / 6
    LoopObject object = LoopObject::PuzzlePiece;
    LoopFlow flow = LoopFlow::Rotation;
    double lengthScale = 0.0;  // edge length where the object is (outline sampling); 0: the mean edge length
    // Hodge flow: interpolate per-vertex velocities (area-weighted averages of the incident faces') across
    // each input face instead of the piecewise-constant face velocity, so the field is continuous across
    // input edges. RK4 then keeps its order across edges: on the stone loop at sub-steps of 1 h_t, the shape
    // error after the loop falls from 0.34 to 0.03 h_t. The smoothed field is no longer exactly
    // divergence-free: the region's area swells by up to about 0.5% along the way and comes back by the
    // end. false: the exact piecewise-constant field. No effect on the rotation flow (already linear in
    // the position).
    bool smoothVelocity = true;
};

// 2D outlines in the object's tangent frame (closed, counter-clockwise, first point not repeated).
// Puzzle piece: square body of side 2r; round tabs (disks of radius 0.35 r centred 0.25 r beyond the
// side's midpoint) on the front (+x) and top (+y) sides, matching round sockets cut into the back (-x)
// and bottom (-y) sides. samplesPerSide sets the sampling density (about 2r / samplesPerSide apart).
std::vector<Vec2d> PuzzlePieceOutline(double r, int samplesPerSide);
// Star R(θ) = r (1 + 0.35 cos 5θ).
std::vector<Vec2d> StarOutline(double r, int samples);

// The object as one region from an outline mapped onto the surface by the exponential map from
// `center` (xAxis: the local x direction in center.face's layout; y is x turned +90°), consecutive
// samples joined by short traces: phi per vertex, negative inside, |phi| the distance to the mapped
// outline (fast marching from it). Inside by flood fill from a vertex of center's face, never across
// an edge the outline crosses; the outline must contain its origin. The object may be the larger side
// of the surface: a leaked flood, or an outline overlapping itself on the surface, is told by the
// outline's orientation (inside on the left of a counter-clockwise outline; either orientation works:
// every edge the outline crosses once has exactly its inside endpoint flooded), not by the region's size.
bool ObjectFromOutline(const IntrinsicTriangulation &tri, const SurfacePoint &center, const Vec2d &xAxis, const std::vector<Vec2d> &outline,
                       std::vector<double> &phi, std::string &error);

class LoopScenario
{
public:
    // The object is placed at geodesic distance ρ from c, the vertex nearest +z, toward +x (loopDistance
    // 0: loopFraction × half the distance from c to the vertex nearest -z); r = radius or ρ / 6. False with a message if
    // the speed is not positive, the object's center lies on the rotation axis, or the outline does not
    // enclose a region on the mesh.
    bool setup(const IntrinsicTriangulation &tri, const LoopConfig &config, std::string &error);
    uint32_t center() const { return mCenter; }  // the vertex nearest the +z pole
    double loopDistance() const { return mRho; }
    double radius() const { return mRadius; }
    // The rotation about the z axis through axisCenter() (the input mesh's bounding-box center) at
    // angularSpeed() = speed / (the object's center's distance from the axis): the object's center moves at
    // `speed` (at the schedule's full weight). T = π × the time of one revolution at full speed, so that
    // ∫ cos(πt/T) over [0, T/2] = T/π is one revolution: rotation flow, T = π · 2π / omega; Hodge flow, π ×
    // the traced time for the object's center to go once around its streamline.
    double period() const { return mPeriod; }
    double angularSpeed() const { return mOmega; }
    const Vec3d &axisCenter() const { return mAxisCenter; }
    const SurfacePoint &objectCenter() const { return mObjectCenter; }
    // Geodesic distance from c per vertex (places the object; the toybox's arrows band).
    const std::vector<double> &loopDistanceField() const { return mD; }
    // The object as one region: negative inside, |phi| the distance to its boundary, per vertex.
    const std::vector<double> &initialPhi() const { return mPhi; }
    LoopFlow flow() const { return mConfig.flow; }
    // Hodge flow: the velocity per input face (3D, tangent to it; n × ∇ψ scaled to the speed), the stream
    // function ψ per input vertex (same scale), and the decomposition's relative norms |∇φ| / |X| (how far
    // the projected rotation X is from divergence-free) and |r| / |X| (the remainder; small on a genus-0
    // surface). Empty / zero for the rotation flow.
    const std::vector<Vec3d> &faceVelocity() const { return mFaceVelocity; }
    const std::vector<double> &streamFunction() const { return mStream; }
    double exactFraction() const { return mExactFraction; }
    double remainderFraction() const { return mRemainderFraction; }
    // Rotation flow: omega ẑ × (x - c) at p's 3D position x (toExtrinsic), projected onto the plane of the
    // input face p lies on (Hodge flow: that face's faceVelocity()), in p.face's layout (IntrinsicTriangulation::extrinsicToLayout); unsigned (the
    // schedule's weight gives the direction). `tri` is the triangulation p lives on: the scenario's own, or
    // any triangulation derived from the same input mesh (e.g. the simulation's refined copy). Independent
    // of the mesh resolution. The two-argument form uses the scenario's triangulation.
    Vec2d velocity(const IntrinsicTriangulation &tri, const SurfacePoint &p, double t) const;
    Vec2d velocity(const SurfacePoint &p, double t) const { return velocity(*mTri, p, t); }
    // Weight cos(πt/T): forward for T/2 (one revolution, slowing to rest), reversed for T/2.
    IntrinsicSpringls::Schedule schedule() const;

    struct Metrics {
        double time = 0.0;
        std::vector<double> areas, relativeAreaErrors;     // one entry: the object as one region
        double meanShapeError = 0.0, maxShapeError = 0.0;  // h_t
        size_t springls = 0;
        // The Lagrangian reference's distance to the contour (mean and max over the contour points) and the
        // largest gap between neighboring reference points, in length-scale units. measure() does not know
        // the reference: the caller (Session) fills these after it.
        double referenceMeanDistance = 0.0, referenceMaxDistance = 0.0, referenceMaxGap = 0.0;
    };
    // On the simulation's triangulation: the area of the region bounded by the contour; shape error, the
    // initial boundary's distance at every contour point (through the input mesh).
    Metrics measure(const IntrinsicSpringls &sim) const;

private:
    const IntrinsicTriangulation *mTri = nullptr;
    LoopConfig mConfig;
    uint32_t mCenter = 0;
    double mRho = 0.0, mRadius = 0.0, mPeriod = 0.0, mOmega = 0.0;
    Vec3d mAxisCenter = Vec3d::Zero();
    SurfacePoint mObjectCenter;
    std::vector<double> mD;
    std::vector<double> mPhi;
    double mInitialArea = 0.0;
    std::vector<Vec3d> mFaceVelocity;
    std::vector<Vec3d> mVertexVelocity;  // smoothVelocity: per input vertex
    std::vector<double> mStream;
    double mExactFraction = 0.0, mRemainderFraction = 0.0;
    bool setupHodge(const IntrinsicTriangulation &tri, std::string &error);
};
}  // namespace springls
