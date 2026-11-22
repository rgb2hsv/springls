#pragma once
// Intrinsic triangulations in signpost form [N. Sharp, "Intrinsic Triangulations in Geometry
// Processing", PhD thesis, CMU-CS-21-132, 2021]: the connectivity of a closed triangle mesh plus one
// length per edge and, per halfedge, its direction at its tail vertex (the signpost). Everything
// along the surface (angles, layouts, traces) is computed from lengths alone; the input mesh's 3D
// positions are used once, for the lengths, and again only to map points back to the input mesh.
#include <core/Common.hpp>
#include <core/Common3D.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace springls
{
// A point on the surface: barycentric coordinates in a face (corner order of the face).
struct SurfacePoint {
    uint32_t face = 0;
    Vec3d bary = Vec3d(1, 0, 0);
};

// One edge crossing of a trace: the halfedge of the face being left, and the parameter t along it
// (0 at its tail, 1 at its head) where the path crosses.
struct EdgeCrossing {
    uint32_t halfedge = 0;
    double t = 0.0;
};

class IntrinsicTriangulation
{
public:
    // Input mesh: vertices + triangles (closed manifold, consistently oriented). False with a
    // message otherwise.
    bool build(const Points3 &vertices, const std::vector<std::array<uint32_t, 3>> &triangles, std::string &error);
    bool load(const std::filesystem::path &ply, std::string &error);  // LoadPly + Triangulate + build

    // Lawson flips until every edge is Delaunay (cotan weight >= 0 up to rounding), except the edges of
    // bisected faces (see refine), which are never flipped; returns the flip count. Points in `points` are
    // relocated through every flip (see flipEdge).
    size_t flipToDelaunay(const std::vector<SurfacePoint *> &points = {});
    // Flips edge e (between faces ijk and jil, to kl). False, and nothing changes, if the two faces do
    // not form a strictly convex quadrilateral or an endpoint would be left with fewer than two edges.
    // The two faces keep their indices (as kil and ljk); points in them are relocated by the exact map
    // of the flip (the hinge laid out in one plane), points on an outer edge stay on it exactly. Both
    // faces get the larger of the two levels.
    bool flipEdge(uint32_t e, const std::vector<SurfacePoint *> &points = {});

    // Adaptive refinement [S, vertex insertion]. The surface never changes: inserted vertices lie on
    // it (their 3D positions through the input mesh), and every SurfacePoint passed in is relocated to
    // the same surface location by the exact barycentric map of each operation. Null pointers and
    // points with face >= faceCount() are ignored (left unchanged).
    //
    // Inserts a vertex at edge e's intrinsic midpoint (flat: angle sum 2π), splitting its two faces;
    // returns the vertex. Face indices: with he = edgeHalfedge(e) in face A and its twin in face B, A
    // and B keep their indices for the halves at tail(he) and appended faces F, F + 1 (F = the old
    // face count) are the halves of A and B at head(he). Edge e becomes the half at tail(he); three edges
    // are appended (the half at head(he), then the edges to A's and B's opposite corners). The halves
    // keep their parent's level (a bisection).
    uint32_t splitEdge(uint32_t e, const std::vector<SurfacePoint *> &points = {});
    // Loop 1:4 split of the marked faces (no smoothing), red-green: neighbors with two or three split edges
    // split too, with one split edge are bisected (green). A bisected face is never bisected or split as
    // such later: when one of its halves is marked or must split an edge, the bisection is undone into the
    // parent's 1:4 split (its other two edges split, the bisecting edge flipped), so the mesh stays regular
    // across repeated calls. Then Lawson flips to intrinsic Delaunay, except the bisected faces' edges
    // (flipping them would lose the bisection). Every point in `points` is relocated exactly (same surface location). Returns the number
    // of faces split 1:4 (an undone bisection counts once). Faces of `faceMarks` beyond its size are
    // unmarked. Face indices are reused and appended.
    size_t refine(const std::vector<uint8_t> &faceMarks, const std::vector<SurfacePoint *> &points);
    // Restores the input mesh in place: its connectivity, lengths, signposts and angle sums (face f is
    // input face f, vertex v input vertex v), no inserted vertices, no green pairs, all levels 0. Keeps
    // the input, so toExtrinsic / extrinsicFrame keep tracing through the correspondence. A no-op on a
    // triangulation that is its own input.
    void resetToInput();
    // How many 1:4 splits from an input face (bisections count as the parent's level). A flipped face
    // takes the larger level of the two faces of the flip, so the level is an upper bound (a cap on
    // refinement depth), not the exact split depth of the face: a refinement criterion should combine
    // it with the face's edge lengths.
    int faceLevel(uint32_t face) const { return mFaceLevel[face]; }
    // A half of a face refine bisected (green), its bisection still remembered.
    bool bisected(uint32_t face) const { return mGreenPartner[face] != kNoFace; }
    // 3D position: input vertices their position; inserted ones through the input mesh, cached.
    Vec3d vertexPosition(uint32_t v) const;
    size_t inputVertexCount() const { return mVertices.size(); }

    size_t vertexCount() const { return mVertexHalfedge.size(); }
    size_t faceCount() const { return mTail.size() / 3; }
    size_t edgeCount() const { return mLength.size(); }
    double meanEdgeLength() const;  // h̄ of the current (intrinsic) triangulation

    // Halfedges: he = 3 * face + corner runs from corner `corner` to corner `corner + 1` of the face.
    uint32_t next(uint32_t he) const { return (he % 3 == 2) ? he - 2 : he + 1; }
    uint32_t prev(uint32_t he) const { return (he % 3 == 0) ? he + 2 : he - 1; }
    uint32_t twin(uint32_t he) const { return mTwin[he]; }
    uint32_t tail(uint32_t he) const { return mTail[he]; }
    uint32_t head(uint32_t he) const { return mTail[next(he)]; }
    uint32_t face(uint32_t he) const { return he / 3; }
    uint32_t edge(uint32_t he) const { return mEdge[he]; }
    uint32_t edgeHalfedge(uint32_t e) const { return mEdgeHalfedge[e]; }
    uint32_t vertexHalfedge(uint32_t v) const { return mVertexHalfedge[v]; }
    // The outgoing halfedge after `he` counter-clockwise around its tail.
    uint32_t nextOutgoing(uint32_t he) const { return twin(prev(he)); }
    size_t degree(uint32_t v) const;

    double length(uint32_t e) const { return mLength[e]; }
    double halfedgeLength(uint32_t he) const { return mLength[mEdge[he]]; }
    // Direction of he at its tail: angle in [0, 2π) of the tail's normalized polar frame.
    double signpost(uint32_t he) const { return mSignpost[he]; }
    double angleSum(uint32_t v) const { return mAngleSum[v]; }
    // Interior angle of face(he) at tail(he).
    double cornerAngle(uint32_t he) const;
    double faceArea(uint32_t face) const;
    // 2D positions of the face's corners: corner 0 at the origin, corner 1 on +x, corner 2 above.
    std::array<Vec2d, 3> layout(uint32_t face) const;
    double cotanWeight(uint32_t e) const;  // ½(cot α + cot β)

    // Layout position of a surface point.
    Vec2d toLayout(const SurfacePoint &p) const;
    // The surface point at corner `corner` of `face` / at vertex v.
    static SurfacePoint cornerPoint(uint32_t face, int corner);
    SurfacePoint vertexPoint(uint32_t v) const { return cornerPoint(face(mVertexHalfedge[v]), static_cast<int>(mVertexHalfedge[v] % 3)); }
    // The point at parameter t along halfedge he (0 at its tail), in face(he).
    static SurfacePoint halfedgePoint(uint32_t he, double t);

    // Discrete exponential map (thesis Section 3.4.1): from `start`, along the straight line of
    // `displacement` (in start.face's layout), across edges and through vertices. `direction` is the
    // unit direction at the end in end.face's layout (that of the start if the displacement is zero).
    struct Trace {
        SurfacePoint end;
        Vec2d startDirection = Vec2d(1, 0);
        Vec2d direction = Vec2d(1, 0);
    };
    Trace trace(const SurfacePoint &start, const Vec2d &displacement, std::vector<EdgeCrossing> *crossings = nullptr) const;
    // Trace from vertex v along the direction at normalized polar angle `angle`, over `distance`.
    Trace traceFromVertex(uint32_t v, double angle, double distance, std::vector<EdgeCrossing> *crossings = nullptr) const;

    // Extrinsic correspondence (thesis Section 3.4.1): input face, barycentric coords, 3D position.
    struct ExtrinsicPoint {
        uint32_t face;
        Vec3d bary;
        Vec3d position;
    };
    ExtrinsicPoint toExtrinsic(uint32_t intrinsicFace, const Vec3d &bary) const;
    ExtrinsicPoint toExtrinsic(const SurfacePoint &p) const { return toExtrinsic(p.face, p.bary); }
    // The extrinsic point of vertex v (an input vertex, or an inserted one as cached at insertion).
    ExtrinsicPoint vertexExtrinsic(uint32_t v) const;
    // Tangent frames through the correspondence: the geodesic toExtrinsic traces (from the corner of
    // p.face nearest p) is straight in both triangulations, so the angle between its directions at p in
    // the input face's layout and in p.face's layout is the rotation between the two frames. `point` is
    // p's extrinsic point (as toExtrinsic) and `rotation` turns a vector in the layout of the input face
    // point.face into p.face's layout. At a vertex, the geodesic runs along the bisector of p's corner
    // of p.face (an input vertex: the input face that contains that direction). On an input edge,
    // point.face is the input face on p.face's side of it (toward the corner's bisector at a vertex, the
    // inward normal on an edge of p.face), so the face of the input mesh matches the face of p: on the
    // input mesh itself, a point of face f, corners and edges included, is framed in f. point.face may
    // therefore differ from toExtrinsic's (same position).
    struct ExtrinsicFrame {
        ExtrinsicPoint point;
        double rotation = 0.0;
    };
    ExtrinsicFrame extrinsicFrame(const SurfacePoint &p) const;
    // A 3D vector at p into p's face layout: projected onto the plane of the input face the frame names,
    // expressed in that face's layout and turned by the frame's rotation. Exact on the input mesh: a
    // vector Q - X (X p's position, Q a point of that input face) maps to the displacement whose trace
    // from p ends at Q.
    Vec2d extrinsicToLayout(const SurfacePoint &p, const Vec3d &v) const { return extrinsicToLayout(extrinsicFrame(p), v); }
    Vec2d extrinsicToLayout(const ExtrinsicFrame &frame, const Vec3d &v) const;
    const Points3 &inputVertices() const { return mVertices; }
    // The input mesh as an (unflipped) intrinsic triangulation.
    const IntrinsicTriangulation &input() const { return mInput ? *mInput : *this; }

private:
    Points3 mVertices;
    std::vector<uint32_t> mTail, mTwin, mEdge;  // per halfedge
    std::vector<double> mSignpost;              // per halfedge
    std::vector<double> mLength;                // per edge
    std::vector<uint32_t> mEdgeHalfedge;        // per edge
    std::vector<uint32_t> mVertexHalfedge;      // per vertex
    std::vector<double> mAngleSum;              // per vertex
    std::vector<int> mFaceLevel;                // per face
    // Green pairs (refine): the two halves of a bisected face, each the other's partner (kNoFace: none), and
    // the bisected edge's midpoint. Any split or flip of a half dissolves its pair.
    static constexpr uint32_t kNoFace = UINT32_MAX;
    std::vector<uint32_t> mGreenPartner, mGreenMid;  // per face
    // Inserted vertices (index v - mVertices.size()): the extrinsic point, and the angle in the layout
    // of the input face `point.face` of the direction at signpost 0 (inserted vertices are flat, so
    // their normalized polar angles are true angles).
    struct Inserted {
        ExtrinsicPoint point;
        double frame = 0.0;
    };
    std::vector<Inserted> mInserted;
    std::shared_ptr<const IntrinsicTriangulation> mInput;

    // Surface points being relocated, bucketed by face.
    struct PointSet;
    uint32_t splitEdgeImpl(uint32_t e, PointSet *points);
    void dissolveGreen(uint32_t f);
    // Green half f's halfedge shared with its partner (inner) or opposite the midpoint (outer: an edge of the parent).
    uint32_t greenHalfedge(uint32_t f, bool inner) const;
    bool flipEdgeImpl(uint32_t e, PointSet *points);
    size_t flipToDelaunayImpl(PointSet *points);
    // Trace on the input mesh from the tail of halfedge he along w (in face(he)'s layout).
    Trace inputTrace(uint32_t he, const Vec2d &w) const;

    // The trace loop. With `along` set, the trace starts on halfedge `along` at parameter `alongStart`
    // and walks it exactly (direction along the edge); `moved` says whether it has travelled yet.
    Trace traceLoop(SurfacePoint point, Vec2d direction, double remaining, const Vec2d &startDirection, std::vector<EdgeCrossing> *crossings,
                    uint32_t along = UINT32_MAX, double alongStart = 0.0, bool moved = false) const;
    // Start of a trace from vertex v at normalized angle `angle`: the face and unit direction. A
    // direction along an edge (up to rounding) sets *along to that outgoing halfedge (the point and
    // direction are then in its face), and UINT32_MAX otherwise.
    void vertexDirection(uint32_t v, double angle, SurfacePoint &point, Vec2d &direction, uint32_t *along = nullptr) const;
    // Normalized polar angle at the tail of halfedge `he` of a vector w given in face(he)'s layout.
    double polarAngle(uint32_t he, const Vec2d &w) const;
};
}  // namespace springls
