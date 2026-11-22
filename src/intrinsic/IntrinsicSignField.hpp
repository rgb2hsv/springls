#pragma once
// Level set reconstructed from springls on an intrinsic triangulation (the surface counterpart of
// SpringlSignField2D). By default the springls define only an unsigned distance and inside and outside
// come from topology, not from springl normals (SignFieldFromDistance with a springl side also uses
// their orientation next to them):
//
// 1. U is the distance to the springl segments: exact in charts within the capsule radius (and a
//    little beyond, to seed), fast marching from there on. The vertices with U <= capsule are walls.
// 2. The exterior floods over mesh edges around the walls from the non-wall vertex with the largest
//    U (ties: smallest index); non-wall vertices it does not reach are inside (with a springl side:
//    unless the side votes a region outside; and if the side votes the flooded region itself inside,
//    the object is the larger side of the surface and the roles swap). With unreached vertices
//    (U = +inf: a march capped at params.march), the flood starts from the largest region (by area) of
//    unreached vertices connected through non-wall vertices: far from the springls, the exterior is the
//    larger side (as "the largest U" assumes).
// 3. The sign propagates upwind into the walls: wall vertices in order of decreasing U, each taking
//    the sign of its one-ring neighbors with larger U weighted by how much larger. A wall vertex
//    with no larger neighbor (a local maximum of U: the middle of a feature thinner than two
//    capsules) is inside, unless a springl side (SignFieldFromDistance) says outside.
// 4. phi = sign * min(U, clamp).
//
// Without springls phi is positive everywhere. `leaked` reports that no non-wall vertex ended inside:
// without a springl side, that the exterior flood reached every non-wall vertex (no springls, or a gap
// wider than two capsules); with one, also that the flooded region and every region it did not reach
// were voted outside. phi
// can still be negative then (wall local maxima, vertices corrected by the side), so hasInside (some
// phi < 0) does not tell a leak. Mesh components with no finite U (no springls in them) are exterior.
// Deterministic.
#include <intrinsic/IntrinsicTrace.hpp>
#include <intrinsic/IntrinsicTriangulation.hpp>

#include <functional>
#include <limits>

#include <utility>
#include <vector>

namespace springls
{
struct SpringlSegments {
    std::vector<std::pair<SurfacePoint, SurfacePoint>> segments;  // each springl's two arms
};
struct SignFieldParams {
    double capsule = 0.0;  // absolute lengths (1.5 h)
    double clamp = 0.0;    // band width
    // Fast marching stops beyond this distance: farther vertices are left unreached (U = +inf), on both
    // sides of the springls. A value of at least the clamp leaves phi unchanged.
    double march = std::numeric_limits<double>::infinity();
    // With a springl side (SignFieldFromDistance): vertices this near the springls take their side.
    double correction = 0.0;
};
struct SignField {
    std::vector<double> phi;
    bool hasInside = false;  // some phi < 0
    // No non-wall vertex ended inside: the exterior flood reached them all, or (with a springl side) every
    // region was voted outside. See the top of the file.
    bool leaked = false;
};

SignField ReconstructSignField(const IntrinsicTriangulation &tri, const SpringlSegments &springls, const SignFieldParams &params);
// Step 1 alone: U (+inf everywhere without springls). chartFor, if set, supplies the local chart of a
// face at a depth (e.g. from the caller's cache, valid for tri); otherwise charts are built per call.
using ChartProvider = std::function<const LocalChart &(uint32_t face, int depth)>;
// The depth of the charts SpringlDistance measures segments in (a provider is asked for that depth).
int SeedChartDepth(const IntrinsicTriangulation &tri, const SignFieldParams &params);
std::vector<double> SpringlDistance(const IntrinsicTriangulation &tri, const SpringlSegments &springls, const SignFieldParams &params,
                                    const ChartProvider &chartFor = {});
// The side of vertex v of the springls (oriented: the region on their left) within `reach` of it: +1
// outside, -1 inside, 0 when no springl is that near or the side is ambiguous (v nearest a free end).
using SpringlSide = std::function<int(uint32_t v, double reach)>;
// Steps 2-4 from a given U (per vertex, uncapped: +inf where no springl reaches), e.g. one the caller
// already computed by fast marching from its own exact seeds. All-infinite U: phi = clamp, leaked.
//
// With `side`, the springls' orientation corrects what topology gets wrong next to them: (0) when the
// side of the walls bordering the flooded region is more often inside than outside, the flooded region
// is inside and the regions it did not reach default to outside (the object is the larger side of the
// surface, so the largest U lies inside); (a) a non-wall region the exterior flood does not reach takes
// the side of its bordering wall vertices (at any reach) when more of them say one side than the other,
// e.g. outside for an exterior pocket whose mouth the walls close; (b) a wall vertex with no upwind neighbor takes its side (at any reach) when it has one, instead
// of inside: a thin exterior gap, such as that mouth; (c) after the upwind sweep, every vertex with
// U <= params.correction takes its side when it is unambiguous (as the 2D springl correction: e.g. the
// tip of an exterior notch at a concave corner, which the sweep signs from the deeper interior across the
// springls). `leaked`: no non-wall vertex ended inside.
SignField SignFieldFromDistance(const IntrinsicTriangulation &tri, const std::vector<double> &U, const SignFieldParams &params, const SpringlSide &side = {});
}  // namespace springls
