#include <springls/Contour2D.hpp>

namespace springls
{
void LinkSegments(size_t vertexCount, const std::vector<std::pair<uint32_t, uint32_t>> &segments, std::vector<int> &nextOf, std::vector<int> &inDegree)
{
    nextOf.assign(vertexCount, -1);
    inDegree.assign(vertexCount, 0);
    for (const auto &s : segments) {
        if (nextOf[s.first] < 0) {
            nextOf[s.first] = static_cast<int>(s.second);
            inDegree[s.second]++;
        }
    }
}

void WalkCurves(const std::vector<int> &nextOf, const std::vector<int> &inDegree, Curves &curves)
{
    curves.clear();
    const size_t V = nextOf.size();
    std::vector<bool> visited(V, false);
    const auto walk = [&](uint32_t start) {
        Curve curve;
        uint32_t cur = start;
        while (true) {
            curve.push_back(cur);
            visited[cur] = true;
            const int nxt = nextOf[cur];
            if (nxt < 0) break;
            if (static_cast<uint32_t>(nxt) == start) {
                curve.push_back(start);  // closed
                break;
            }
            if (visited[nxt]) break;
            cur = static_cast<uint32_t>(nxt);
        }
        if (curve.size() > 1) curves.push_back(std::move(curve));
    };
    for (uint32_t v = 0; v < V; v++) {
        if (!visited[v] && inDegree[v] == 0 && nextOf[v] >= 0) walk(v);  // open curves (domain boundary)
    }
    for (uint32_t v = 0; v < V; v++) {
        if (!visited[v] && nextOf[v] >= 0) walk(v);
    }
}

void ExtractIsoContour(const ImageF &phi, Points2 &vertices, Curves &curves, float tolerance)
{
    vertices.clear();
    curves.clear();
    const int W = phi.width(), H = phi.height();
    if (W < 2 || H < 2) return;
    const auto value = [&](int i, int j) {
        const float v = phi(i, j);
        return (v < 0.0f) ? std::min(v, -tolerance) : std::max(v, tolerance);
    };
    // One vertex per crossed grid edge: horizontal edge (i,j)-(i+1,j) and vertical edge (i,j)-(i,j+1).
    std::vector<int> horizontal(static_cast<size_t>(W) * H, -1), vertical(static_cast<size_t>(W) * H, -1);
    std::vector<std::pair<uint32_t, uint32_t>> segments;
    for (int i = 0; i + 1 < W; i++) {
        for (int j = 0; j + 1 < H; j++) {
            const float val[4] = {value(i, j), value(i + 1, j), value(i + 1, j + 1), value(i, j + 1)};
            const Vec2 pos[4] = {Vec2(i, j), Vec2(i + 1, j), Vec2(i + 1, j + 1), Vec2(i, j + 1)};
            Vec2 cross[4];
            // Edge k of the cell -> its slot in the shared edge tables.
            int *slots[4] = {&horizontal[phi.indexOf(i, j)], &vertical[phi.indexOf(i + 1, j)], &horizontal[phi.indexOf(i, j + 1)],
                             &vertical[phi.indexOf(i, j)]};
            const auto vertexOf = [&](int e) {
                int &slot = *slots[e];
                if (slot < 0) {
                    slot = static_cast<int>(vertices.size());
                    vertices.push_back(cross[e]);
                }
                return static_cast<uint32_t>(slot);
            };
            MarchSquare(
                val, pos, cross,
                [&](int e0, int e1) {
                    const uint32_t u = vertexOf(e0), v = vertexOf(e1);
                    if (u != v) segments.emplace_back(u, v);
                },
                SaddleRule::EdgeOverlap);
        }
    }
    std::vector<int> nextOf, inDegree;
    LinkSegments(vertices.size(), segments, nextOf, inDegree);
    WalkCurves(nextOf, inDegree, curves);
}
}  // namespace springls
