#include <mesh/LoopSubdivision.hpp>

#include <cmath>
#include <map>
#include <numbers>
#include <utility>

namespace springls
{
LoopMesh LoopSubdivide(const LoopMesh &mesh, const std::vector<uint8_t> &refineFace)
{
    const size_t F = mesh.triangles.size(), V = mesh.vertices.size();
    // Edges, with the vertex opposite them in each of their (up to two) faces.
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> edgeIndex;
    std::vector<std::array<uint32_t, 2>> edgeEnds;
    std::vector<std::vector<uint32_t>> edgeOpposite;
    std::vector<std::array<uint32_t, 3>> faceEdges(F);
    for (size_t f = 0; f < F; f++) {
        for (int k = 0; k < 3; k++) {
            const uint32_t a = mesh.triangles[f][k], b = mesh.triangles[f][(k + 1) % 3], c = mesh.triangles[f][(k + 2) % 3];
            const auto key = std::make_pair(std::min(a, b), std::max(a, b));
            auto it = edgeIndex.find(key);
            if (it == edgeIndex.end()) {
                it = edgeIndex.emplace(key, static_cast<uint32_t>(edgeEnds.size())).first;
                edgeEnds.push_back({key.first, key.second});
                edgeOpposite.emplace_back();
            }
            edgeOpposite[it->second].push_back(c);
            faceEdges[f][k] = it->second;  // edge k joins corners k and k + 1
        }
    }
    // Split faces, closed under "two or more split edges".
    std::vector<uint8_t> split(F, 0), edgeSplit(edgeEnds.size(), 0);
    for (size_t f = 0; f < F; f++) {
        if (f < refineFace.size() && refineFace[f]) split[f] = 1;
    }
    for (bool changed = true; changed;) {
        changed = false;
        for (size_t f = 0; f < F; f++) {
            if (split[f]) {
                for (uint32_t e : faceEdges[f]) edgeSplit[e] = 1;
            }
        }
        for (size_t f = 0; f < F; f++) {
            if (split[f]) continue;
            int n = 0;
            for (uint32_t e : faceEdges[f]) n += edgeSplit[e];
            if (n >= 2) {
                split[f] = 1;
                changed = true;
            }
        }
    }
    LoopMesh out;
    out.vertices = mesh.vertices;
    // Old vertexes inside the refined region: Loop's vertex rule.
    std::vector<std::vector<uint32_t>> neighbors(V);
    std::vector<uint8_t> interior(V, 1);
    for (size_t e = 0; e < edgeEnds.size(); e++) {
        neighbors[edgeEnds[e][0]].push_back(edgeEnds[e][1]);
        neighbors[edgeEnds[e][1]].push_back(edgeEnds[e][0]);
    }
    for (size_t f = 0; f < F; f++) {
        if (!split[f]) {
            for (uint32_t v : mesh.triangles[f]) interior[v] = 0;
        }
    }
    for (uint32_t v = 0; v < V; v++) {
        if (!interior[v] || neighbors[v].empty()) continue;
        const double n = static_cast<double>(neighbors[v].size());
        const double c = 3.0 / 8.0 + 0.25 * std::cos(2.0 * std::numbers::pi / n);
        const double beta = (5.0 / 8.0 - c * c) / n;
        Vec3d p = (1.0 - n * beta) * mesh.vertices[v].cast<double>();
        for (uint32_t w : neighbors[v]) p += beta * mesh.vertices[w].cast<double>();
        out.vertices[v] = p.cast<float>();
    }
    // Edge vertexes: 3/8 (a + b) + 1/8 (c + d) for interior edges, the midpoint on a boundary.
    std::vector<uint32_t> edgeVertex(edgeEnds.size(), UINT32_MAX);
    for (size_t e = 0; e < edgeEnds.size(); e++) {
        if (!edgeSplit[e]) continue;
        const Vec3d a = mesh.vertices[edgeEnds[e][0]].cast<double>(), b = mesh.vertices[edgeEnds[e][1]].cast<double>();
        Vec3d p = 0.5 * (a + b);
        if (edgeOpposite[e].size() == 2) {
            p = 0.375 * (a + b) + 0.125 * (mesh.vertices[edgeOpposite[e][0]].cast<double>() + mesh.vertices[edgeOpposite[e][1]].cast<double>());
        }
        edgeVertex[e] = static_cast<uint32_t>(out.vertices.size());
        out.vertices.push_back(p.cast<float>());
    }
    for (size_t f = 0; f < F; f++) {
        const auto &t = mesh.triangles[f];
        if (split[f]) {
            const uint32_t m01 = edgeVertex[faceEdges[f][0]], m12 = edgeVertex[faceEdges[f][1]], m20 = edgeVertex[faceEdges[f][2]];
            out.triangles.push_back({t[0], m01, m20});
            out.triangles.push_back({m01, t[1], m12});
            out.triangles.push_back({m20, m12, t[2]});
            out.triangles.push_back({m01, m12, m20});
            continue;
        }
        int k = -1;
        for (int j = 0; j < 3; j++) {
            if (edgeSplit[faceEdges[f][j]]) k = j;
        }
        if (k < 0) {
            out.triangles.push_back(t);
            continue;
        }
        // One split edge (corners k, k + 1): bisect toward the opposite corner.
        const uint32_t m = edgeVertex[faceEdges[f][k]];
        const uint32_t x = t[k], y = t[(k + 1) % 3], z = t[(k + 2) % 3];
        out.triangles.push_back({x, m, z});
        out.triangles.push_back({m, y, z});
    }
    return out;
}
}  // namespace springls
