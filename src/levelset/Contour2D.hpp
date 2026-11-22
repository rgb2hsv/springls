#pragma once
#include <core/Image.hpp>

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace springls
{
// Marching squares on one cell. Corners are counter-clockwise: c0 (0,0), c1 (1,0), c2 (1,1),
// c3 (0,1); edge k runs from corner k to corner k + 1. `val` must be nonzero (nudged away from
// the iso level). Calls emit(e0, e1) for every contour segment, oriented from the crossing on
// edge e0 to the crossing on edge e1 with the inside (val < 0) on its right. `cross` receives
// the crossing positions.
//
// Saddles (diagonal corners share a sign) are resolved by `rule`:
//   CenterAverage: the cell center takes the sign of the corner average (adaptive grid).
//   EdgeOverlap:   corners sharing c0's sign stay connected iff their spans along the bottom and
//                  top edges overlap in x (the rule of D. Eberly's Geometric Tools level-curve
//                  extraction, used by the uniform grid).
enum class SaddleRule {
    CenterAverage,
    EdgeOverlap
};
template <class Emit>
void MarchSquare(const float val[4], const Vec2 pos[4], Vec2 cross[4], Emit &&emit, SaddleRule rule = SaddleRule::CenterAverage)
{
    int insideMask = 0;
    for (int k = 0; k < 4; k++) {
        if (val[k] < 0.0f) insideMask |= (1 << k);
    }
    if (insideMask == 0 || insideMask == 15) return;
    bool has[4];
    for (int k = 0; k < 4; k++) {
        const int k1 = (k + 1) & 3;
        has[k] = (val[k] < 0.0f) != (val[k1] < 0.0f);
        if (has[k]) {
            const float t = val[k] / (val[k] - val[k1]);
            cross[k] = pos[k] + t * (pos[k1] - pos[k]);
        }
    }
    const auto oriented = [&](int e0, int e1, const Vec2 &insidePoint) {
        const Vec2 d = cross[e1] - cross[e0];
        const Vec2 w = insidePoint - cross[e0];
        if (cross2(d, w) > 0.0f) {
            emit(e1, e0);
        } else {
            emit(e0, e1);
        }
    };
    const int count = has[0] + has[1] + has[2] + has[3];
    if (count == 2) {
        int e[2] = {0, 0}, m = 0;
        for (int k = 0; k < 4; k++) {
            if (has[k]) e[m++] = k;
        }
        Vec2 insidePoint = Vec2::Zero();
        int n = 0;
        for (int k = 0; k < 4; k++) {
            if (insideMask & (1 << k)) {
                insidePoint += pos[k];
                n++;
            }
        }
        oriented(e[0], e[1], insidePoint / static_cast<float>(n));
    } else {
        // Saddle: corners whose sign differs from the cell center are cut off on their own.
        bool centerInside;
        if (rule == SaddleRule::CenterAverage) {
            centerInside = 0.25f * (val[0] + val[1] + val[2] + val[3]) < 0.0f;
        } else {
            // Corner 0's region spans the bottom edge up to cross[0]; corner 2's spans the top edge
            // from cross[2]. They are separate when those spans do not overlap.
            const bool separate = cross[2].x() > cross[0].x();
            const bool c0Inside = val[0] < 0.0f;
            centerInside = separate ? !c0Inside : c0Inside;
        }
        const Vec2 centerPos = 0.25f * (pos[0] + pos[1] + pos[2] + pos[3]);
        for (int k = 0; k < 4; k++) {
            const bool cornerInside = (insideMask >> k) & 1;
            if (cornerInside == centerInside) continue;
            oriented((k + 3) & 3, k, cornerInside ? pos[k] : centerPos);
        }
    }
}

// Successor of each vertex along oriented segments (first segment wins), and in-degrees.
void LinkSegments(size_t vertexCount, const std::vector<std::pair<uint32_t, uint32_t>> &segments, std::vector<int> &nextOf, std::vector<int> &inDegree);
// Walks successor links into curves: open curves (starting at a vertex with no predecessor)
// first, then closed loops, which repeat their first index at the end.
void WalkCurves(const std::vector<int> &nextOf, const std::vector<int> &inDegree, Curves &curves);

// Iso-contour (phi = 0) of a uniform grid, sample (i, j) at position (i, j). Values within
// `tolerance` of zero are nudged off it (zero counts as outside). Inside (phi < 0) lies on the
// right of each curve, i.e. closed curves around an inside region run clockwise.
void ExtractIsoContour(const ImageF &phi, Points2 &vertices, Curves &curves, float tolerance = 1E-3f);

// Labeled contour of a multi-object level set on a width x height grid: for each label of
// objectLabels (in order), the iso-contour of signedValue(i, j, label) as ExtractIsoContour
// computes it. Each label gets its own vertexes, so a boundary shared by two regions has two
// coincident copies, one per side, and every vertex belongs to one curve of one label. Where three
// or more regions meet in a cell, each label's curve passes through the cell's junction vertex
// (FindJunction; one copy per label).
// curveLabels[c] is the label of curves[c].
// junctionVertices, when given, flags those junction vertexes (1) among all others (0).
void ExtractMultiLabelContour(const std::function<float(int i, int j, int label)> &signedValue, int width, int height, const std::vector<int> &objectLabels,
                              Points2 &vertices, Curves &curves, std::vector<int> &curveLabels, float tolerance = 1E-3f,
                              std::vector<uint8_t> *junctionVertices = nullptr);
// Steiner vertex of cell (i, j)-(i+1, j+1) of a width x height labeled node lattice (lattice
// coordinates): false unless three or more regions own its corners (owner(i, j): the node's label,
// background 0). Then J is where the region boundaries meet, and every label's contour through the
// cell passes through J instead of cutting its own corner (which leaves a gap no region covers).
// signedValue(i, j, l): < 0 inside region l, its crossings place the boundaries.
bool FindJunction(const std::function<int(int, int)> &owner, const std::function<float(int, int, int)> &signedValue, int width, int height, int i, int j,
                  Vec2 &J);
// Sorted distinct labels above the background (label 0) present in `labels`.
std::vector<int> ObjectLabels(const ImageI &labels);
}  // namespace springls
