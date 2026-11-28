#pragma once
// Charts of a triangle mesh from meshlet-optimizer. By default they are its meshlet pipeline's base level (what
// MeshletBuilder makes with a single level), with every setting at meshlet-optimizer's default:
//  1. buildBaseCharts(): the restricted centroidal Voronoi tessellation (CvtSegmentation) of MeshletParams::cvt:
//     blue-noise seeds, as many as charts of at most maxTrianglesPerChart (256) need, 4 Lloyd iterations then
//     L-BFGS on the CVT energy, every vertex labeled by its nearest seed, and the labels regularized (an opening
//     removes thin bands, region competition smooths the borders, charts that are not disks or have more than
//     maxTrianglesPerChart triangles are split);
//  2. buildBaseLevel(): every face takes the label most of its corners have (ties: the lower id), faces with no
//     edge neighbor in their own chart join a neighbor, and charts left empty are dropped;
//  3. straightenBaseSeams(): each border between two junctions is replaced by the cheapest path along mesh edges
//     near the bisector of the two charts (StraightenRegionSeams), junctions kept.
// Only the seed differs: a different randomSeed gives different chart borders.
// Only src/lod/mopt/CvtCharts.cpp sees meshlet-optimizer itself.
#include <core/Common3D.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace springls
{
struct CvtChartParams {
    // true: the meshlet pipeline's base level, as above. false: the raw CVT (chartCount blue-noise seeds, no
    // Lloyd, L-BFGS, regularization or straightening; faces by majority label), much cheaper.
    bool meshletBaseLevel = true;
    int maxTrianglesPerChart = 256;  // meshlet-optimizer's default (meshletBaseLevel)
    double chartFill = 0.375;        // share of that limit each chart aims at; sets the seed count (its default)
    int chartCount = 64;             // raw CVT only
    uint32_t randomSeed = 1;
};

// One chart label per triangle (0 .. count - 1). Empty on failure. Vertexes no triangle uses are ignored.
std::vector<int> SegmentCvtCharts(const Points3 &vertexes, const std::vector<std::array<uint32_t, 3>> &triangles, const CvtChartParams &params);
}  // namespace springls
