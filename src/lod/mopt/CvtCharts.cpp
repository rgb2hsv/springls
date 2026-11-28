// meshlet-optimizer's charts for springls (see CvtCharts.hpp); built into springls_lod.
#include <lod/CvtCharts.hpp>

#include <charts/CvtSegmentation.hpp>
#include <hierarchy/MeshletBuilder.hpp>
#include <hierarchy/RegionSplit.hpp>
#include <hierarchy/SeamStraightening.hpp>
#include <mesh/Mesh.hpp>

#include <algorithm>
#include <exception>

namespace springls
{
std::vector<int> SegmentCvtCharts(const Points3 &vertexes, const std::vector<std::array<uint32_t, 3>> &triangles, const CvtChartParams &params)
{
    if (triangles.empty()) return {};
    try {
        // The mesh of the vertexes the triangles use.
        mopt::Mesh mesh;
        std::vector<int> map(vertexes.size(), -1);
        mesh.triIndexes.reserve(triangles.size());
        for (const auto &t : triangles) {
            mopt::Vector3u face;
            for (int c = 0; c < 3; c++) {
                int &m = map[t[c]];
                if (m < 0) {
                    m = static_cast<int>(mesh.vertexLocations.size());
                    mesh.vertexLocations.push_back(vertexes[t[c]]);
                }
                face[c] = static_cast<uint32_t>(m);
            }
            mesh.triIndexes.push_back(face);
        }

        // MeshletBuilder::buildBaseCharts(), with MeshletParams' CVT settings (or the raw CVT).
        mopt::CvtSegmentationParams cvtParams = mopt::MeshletParams{}.cvt;
        if (params.meshletBaseLevel) {
            cvtParams.regularization.maxTrianglesPerChart = std::max(1, params.maxTrianglesPerChart);
            cvtParams.chartFill = params.chartFill;
        } else {
            cvtParams.seedCount = std::max(1, params.chartCount);
            cvtParams.lloydIterations = cvtParams.lbfgsIterations = 0;
            cvtParams.regularize = false;
            cvtParams.minChartVertices = 1;
        }
        cvtParams.randomSeed = params.randomSeed;
        mopt::CvtSegmentation cvt(mesh);
        std::vector<int> vertexLabels;
        cvt.solve(cvtParams, vertexLabels);

        // MeshletBuilder::buildBaseLevel(): each face goes to the chart holding most of its corners (ties: the
        // lower id); then isolated faces join a neighbor.
        std::vector<int> labels(mesh.triIndexes.size(), -1);
        for (size_t f = 0; f < labels.size(); f++) {
            const mopt::Vector3u t = mesh.triIndexes[f];
            const int l[3] = {vertexLabels[t.x()], vertexLabels[t.y()], vertexLabels[t.z()]};
            int best = -1, bestCount = 0;
            for (int k = 0; k < 3; k++) {
                if (l[k] < 0) continue;
                const int count = (l[0] == l[k]) + (l[1] == l[k]) + (l[2] == l[k]);
                if (count > bestCount || (count == bestCount && l[k] < best)) {
                    best = l[k];
                    bestCount = count;
                }
            }
            labels[f] = best;
        }
        int next = 0;
        for (int l : labels) next = std::max(next, l + 1);
        for (int &l : labels) {
            if (l < 0) l = next;  // no labeled corner (the builder leaves these to RelabelIsolatedFaces as well)
        }
        if (params.meshletBaseLevel) {
            mopt::RelabelIsolatedFaces(mesh, labels);
            // MeshletBuilder::straightenBaseSeams() (one UV island: no input texture coordinates).
            mopt::StraightenRegionSeams(mesh, labels);
        }
        // Dense labels 0 .. count - 1 (charts left empty dropped), in order of first appearance.
        std::vector<int> remap;
        int count = 0;
        for (int &l : labels) {
            if (l >= static_cast<int>(remap.size())) remap.resize(l + 1, -1);
            if (remap[l] < 0) remap[l] = count++;
            l = remap[l];
        }
        return labels;
    } catch (const std::exception &) {
        return {};
    }
}
}  // namespace springls
