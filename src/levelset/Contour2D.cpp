#include <levelset/Contour2D.hpp>

#include <algorithm>

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

namespace
{
// Iso-contour of phi; junction(i, j, J), when given, names cells (i, j)-(i+1, j+1) whose single
// segment a -> b is routed through J as a -> J -> b (J gets its own vertex).
void ExtractIsoContourThrough(const ImageF &phi, Points2 &vertices, Curves &curves, float tolerance, const std::function<bool(int, int, Vec2 &)> &junction,
                              std::vector<uint32_t> *junctionVertices = nullptr)
{
    if (junctionVertices) junctionVertices->clear();
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
            std::pair<uint32_t, uint32_t> cell[2];
            int cellCount = 0;
            MarchSquare(
                val, pos, cross,
                [&](int e0, int e1) {
                    const uint32_t u = vertexOf(e0), v = vertexOf(e1);
                    if (u != v && cellCount < 2) cell[cellCount++] = {u, v};
                },
                SaddleRule::EdgeOverlap);
            Vec2 J;
            if (cellCount == 1 && junction && junction(i, j, J)) {
                const uint32_t k = static_cast<uint32_t>(vertices.size());
                vertices.push_back(J);
                if (junctionVertices) junctionVertices->push_back(k);
                segments.emplace_back(cell[0].first, k);
                segments.emplace_back(k, cell[0].second);
            } else {
                for (int n = 0; n < cellCount; n++) segments.push_back(cell[n]);
            }
        }
    }
    std::vector<int> nextOf, inDegree;
    LinkSegments(vertices.size(), segments, nextOf, inDegree);
    WalkCurves(nextOf, inDegree, curves);
}
}  // namespace

void ExtractIsoContour(const ImageF &phi, Points2 &vertices, Curves &curves, float tolerance)
{
    ExtractIsoContourThrough(phi, vertices, curves, tolerance, nullptr);
}

bool FindJunction(const std::function<int(int, int)> &owner, const std::function<float(int, int, int)> &signedValue, int width, int height, int i, int j,
                  Vec2 &J)
{
    // Corner k of cell (i, j), counterclockwise from (i, j); edge k runs from corner k to k + 1.
    const auto corner = [](int ci, int cj, int k) { return Vec2i(ci + (k == 1 || k == 2), cj + (k >= 2)); };
    const auto ownerOf = [&](const Vec2i &p) { return owner(p.x(), p.y()); };
    int distinct[4], count = 0;
    for (int k = 0; k < 4; k++) {
        const int o = ownerOf(corner(i, j, k));
        if (std::find(distinct, distinct + count, o) == distinct + count) distinct[count++] = o;
    }
    if (count < 3) return false;
    // Boundary crossing on edge p-q (owners differ): the zero of the inside region's signed value.
    const auto crossing = [&](const Vec2i &p, const Vec2i &q) {
        const int l = ownerOf(p) > 0 ? ownerOf(p) : ownerOf(q);
        const float a = signedValue(p.x(), p.y(), l), b = signedValue(q.x(), q.y(), l);
        const float t = (a != b) ? std::clamp(a / (a - b), 0.0f, 1.0f) : 0.5f;
        return Vec2(p.x() + t * (q.x() - p.x()), p.y() + t * (q.y() - p.y()));
    };
    const auto pairOf = [&](const Vec2i &p, const Vec2i &q) { return std::make_pair(std::min(ownerOf(p), ownerOf(q)), std::max(ownerOf(p), ownerOf(q))); };
    // Each region boundary crossing an edge of the cell continues through the neighbors across it.
    // Crossings on the junction cell's own edges are biased (there a node's distance is to the
    // nearest other region, not to this boundary), so the boundary's line runs through its crossings
    // one and two cells out. J is the least-squares meeting point of those lines (exact for straight
    // boundaries) when it lies in the cell, else the mean of the cell's crossings.
    static constexpr int kNeighbor[4][2] = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
    // The boundary leaving cell (ci, cj) through edge k (owner pair `pair`): the next cell and the
    // edge through which it leaves that one, if it is the only one with that pair.
    const auto follow = [&](int ci, int cj, int k, std::pair<int, int> pair, int &ni, int &nj, int &nk) {
        ni = ci + kNeighbor[k][0];
        nj = cj + kNeighbor[k][1];
        if (ni < 0 || nj < 0 || ni + 1 >= width || nj + 1 >= height) return false;
        const int entry = (k + 2) % 4;  // the shared edge, seen from the next cell
        int matches = 0;
        for (int m = 0; m < 4; m++) {
            if (m == entry) continue;
            const Vec2i a = corner(ni, nj, m), b = corner(ni, nj, (m + 1) % 4);
            if (ownerOf(a) == ownerOf(b) || pairOf(a, b) != pair) continue;
            nk = m;
            matches++;
        }
        return matches == 1;
    };
    // Per boundary: crossings one, two and (if the boundary goes on) three cells out.
    struct Boundary {
        Vec2 x[3];
        int count = 0;
    };
    std::vector<Boundary> boundaries;
    Vec2 mean = Vec2::Zero();
    int n = 0;
    for (int k = 0; k < 4; k++) {
        const Vec2i p = corner(i, j, k), q = corner(i, j, (k + 1) % 4);
        if (ownerOf(p) == ownerOf(q)) continue;
        mean += crossing(p, q);
        n++;
        const auto pair = pairOf(p, q);
        Boundary boundary;
        int ci = i, cj = j, ck = k;
        for (int step = 0; step < 3; step++) {
            int ni, nj, nk;
            if (!follow(ci, cj, ck, pair, ni, nj, nk)) break;
            boundary.x[boundary.count++] = crossing(corner(ni, nj, nk), corner(ni, nj, (nk + 1) % 4));
            ci = ni;
            cj = nj;
            ck = nk;
        }
        // Too short to continue: the junction cell's own crossing gives the second point.
        if (boundary.count == 1) boundary.x[boundary.count++] = crossing(p, q);
        if (boundary.count >= 2 && (boundary.x[1] - boundary.x[0]).norm() > 1e-4f) boundaries.push_back(boundary);
    }
    J = mean / static_cast<float>(n);
    // Least-squares meeting point of lines (point, direction).
    const auto meet = [&](const std::vector<std::pair<Vec2, Vec2>> &lines, Vec2 &out) {
        Vec2 rhs = Vec2::Zero();
        float axx = 0.0f, axy = 0.0f, ayy = 0.0f;  // sum of normal normal^T
        for (const auto &[point, direction] : lines) {
            const Vec2 normal(-direction.y(), direction.x());
            axx += normal.x() * normal.x();
            axy += normal.x() * normal.y();
            ayy += normal.y() * normal.y();
            rhs += normal * normal.dot(point);
        }
        const float det = axx * ayy - axy * axy;
        if (det <= 0.05f) return false;
        out = Vec2(ayy * rhs.x() - axy * rhs.y(), axx * rhs.y() - axy * rhs.x()) / det;
        return out.x() >= i && out.x() <= i + 1 && out.y() >= j && out.y() <= j + 1;
    };
    // First the chords through the two nearest crossings; then, where a boundary has three, the
    // tangent of the parabola through them at the first estimate (curved boundaries such as the
    // rim: the extended chord overshoots outward), which leaves straight boundaries unchanged.
    std::vector<std::pair<Vec2, Vec2>> lines;
    for (const Boundary &b : boundaries) lines.emplace_back(b.x[0], (b.x[1] - b.x[0]).normalized());
    Vec2 estimate;
    if (!meet(lines, estimate)) return true;
    J = estimate;
    lines.clear();
    for (const Boundary &b : boundaries) {
        if (b.count < 3) {
            lines.emplace_back(b.x[0], (b.x[1] - b.x[0]).normalized());
            continue;
        }
        // Parabola P(s) through (0, x0), (s1, x1), (s2, x2) in chord length s, evaluated at the
        // (negative) distance of the estimate before x0.
        const float s1 = (b.x[1] - b.x[0]).norm(), s2 = s1 + (b.x[2] - b.x[1]).norm();
        const float s = -(estimate - b.x[0]).norm();
        const Vec2 d01 = (b.x[1] - b.x[0]) / s1, d12 = (b.x[2] - b.x[1]) / (s2 - s1);
        const Vec2 second = (d12 - d01) / s2;  // P = x0 + d01 (s) + second s (s - s1)
        const Vec2 point = b.x[0] + d01 * s + second * s * (s - s1);
        const Vec2 tangent = d01 + second * (2.0f * s - s1);
        if (tangent.norm() < 1e-6f) {
            lines.emplace_back(b.x[0], d01.normalized());
            continue;
        }
        lines.emplace_back(point, tangent.normalized());
    }
    if (meet(lines, estimate)) J = estimate;
    return true;
}

void ExtractMultiLabelContour(const std::function<float(int, int, int)> &signedValue, int width, int height, const std::vector<int> &objectLabels,
                              Points2 &vertices, Curves &curves, std::vector<int> &curveLabels, float tolerance, std::vector<uint8_t> *junctionVertices)
{
    vertices.clear();
    curves.clear();
    curveLabels.clear();
    if (junctionVertices) junctionVertices->clear();
    if (width < 2 || height < 2) return;
    // The region of each node: the label inside which it lies (signedValue < 0), else background.
    ImageI owner(width, height);
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            owner(i, j) = 0;
            for (int label : objectLabels) {
                if (signedValue(i, j, label) < 0.0f) {
                    owner(i, j) = label;
                    break;
                }
            }
        }
    }
    // Junction cells: every label's contour passes through one Steiner vertex (see FindJunction).
    const auto junction = [&](int i, int j, Vec2 &J) { return FindJunction([&](int a, int b) { return owner(a, b); }, signedValue, width, height, i, j, J); };
    ImageF phi(width, height);
    Points2 labelVertices;
    Curves labelCurves;
    std::vector<uint32_t> labelJunctions;
    for (int label : objectLabels) {
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) phi(i, j) = signedValue(i, j, label);
        }
        ExtractIsoContourThrough(phi, labelVertices, labelCurves, tolerance, junction, &labelJunctions);
        const uint32_t offset = static_cast<uint32_t>(vertices.size());
        vertices.insert(vertices.end(), labelVertices.begin(), labelVertices.end());
        if (junctionVertices) {
            junctionVertices->resize(vertices.size(), 0);
            for (uint32_t v : labelJunctions) (*junctionVertices)[offset + v] = 1;
        }
        for (Curve &curve : labelCurves) {
            for (uint32_t &v : curve) v += offset;
            curves.push_back(std::move(curve));
            curveLabels.push_back(label);
        }
    }
}

std::vector<int> ObjectLabels(const ImageI &labels)
{
    std::vector<int> found;
    for (size_t n = 0; n < labels.size(); n++) {
        if (labels[n] > 0) found.push_back(labels[n]);
    }
    std::sort(found.begin(), found.end());
    found.erase(std::unique(found.begin(), found.end()), found.end());
    return found;
}
}  // namespace springls
