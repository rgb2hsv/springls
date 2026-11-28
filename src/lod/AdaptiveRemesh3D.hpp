#pragma once
// Triangle quality of a deforming closed mesh (Wojtan, Thürey, Gross and Turk, "Deforming Meshes that Split and
// Merge", 2009, section 6.2): long edges split, short edges and the shortest edge of a sliver collapsed, and edges
// flipped where that raises the triangles' smallest angle. Regions (faceRegion) are respected: flips stay inside
// one region, and a vertex on a region border only moves along a border edge.
//
// In parallel, by independent sets: each pass finds every candidate operation and checks it (in parallel), the
// candidates then claim what they would change (the faces of a split edge; the vertexes of the one-rings of a
// collapse's two ends; the four vertexes of a flip) in order of priority (longest split first, shortest collapse
// first, largest flip gain first) with atomic minimums, and every candidate that holds all its claims is applied
// at once, in parallel. Edges and vertex rings come from tables rebuilt by sorting (no hash maps).
#include <lod/TriangleSurface3D.hpp>

#include <functional>

namespace springls
{
struct RemeshParams {
    float targetEdge = 1.0f;        // the edge length the mesh keeps
    float splitRatio = 1.5f;        // edges longer than splitRatio * targetEdge are split at their midpoint
    float collapseRatio = 0.5f;     // edges shorter than collapseRatio * targetEdge are collapsed
    float minAngleDegrees = 15.0f;  // ... as is the shortest edge of a triangle with a smaller angle
    int passes = 4;                 // split, collapse and flip passes (fewer when nothing changes)
    bool parallel = true;           // false: one thread (for remeshing many pieces at once)
    // true: region borders are kept (no flip across one; a border vertex only moves along its border).
    // false: regions are only carried along (every triangle keeps the region of the triangle it replaces), and
    // borders are remeshed like any other edge.
    bool keepRegionBorders = true;
    // When set (one flag per triangle, kept in step with the triangles): operations that would change a frozen
    // triangle, or move or remove a vertex of one, are skipped.
    std::vector<uint8_t> *frozen = nullptr;
};

// Hooks run while operations are applied in parallel: each call touches only its own vertexes.
struct RemeshHooks {
    // Called (serially) before vertexes are appended: the mesh will have `count` vertexes.
    std::function<void(size_t count)> grow;
    // Vertex m was appended at the midpoint of a and b.
    std::function<void(uint32_t m, uint32_t a, uint32_t b)> split;
    // Vertex b was merged into a; when midpoint, a moved to the midpoint of the two (otherwise a stayed).
    std::function<void(uint32_t a, uint32_t b, bool midpoint)> merge;
};

// Returns whether anything changed. Vertexes are only appended (a collapsed vertex is left unused).
bool RemeshSurface(TriangleSurface &mesh, const RemeshParams &params, const RemeshHooks &hooks = {});

// The same, chart by chart. The charts are the regions (mesh.faceRegion, e.g. from SegmentCvtCharts) and are
// kept: triangles made from a chart's triangles join it. First every chart is cut out with the ring of triangles
// around it (frozen, so the chart's border stays as it is), remeshed on its own thread and written back, all
// charts at once; then the band of triangles along the chart borders is remeshed by independent sets, the rest
// frozen, with the borders remeshed like any other edge (keepRegionBorders false): every triangle keeps the chart
// of the triangle it replaces, so the borders move with the remeshing. No hooks; returns whether anything changed.
bool RemeshSurfaceByCharts(TriangleSurface &mesh, const RemeshParams &params);
}  // namespace springls
