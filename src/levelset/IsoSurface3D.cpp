#include <levelset/IsoSurface3D.hpp>

#include <algorithm>

namespace springls
{
void SurfaceMesh3D::updateVertexNormals()
{
    normals.assign(vertexes.size(), Vec3::Zero());
    const int K = elementSize();
    const size_t F = faceCount();
    for (size_t f = 0; f < F; f++) {
        const uint32_t *idx = face(f);
        Vec3 v[kMaxElementSize];
        for (int k = 0; k < K; k++) v[k] = vertexes[idx[k]];
        const Vec3 n = FanNormal(Centroid(v, K), v, K);
        for (int k = 0; k < K; k++) normals[idx[k]] += n;
    }
    for (Vec3 &n : normals) n = normalizedOrZero(n);
}

namespace
{
struct Crossing {
    uint64_t key;  // 3 * voxel index + axis
    Vec3 point;
    bool pivotNegative;  // phi < 0 at the lower end of the edge
    bool operator<(const Crossing &o) const { return key < o.key; }
};
}  // namespace

void ExtractIsoSurface(const VolumeF &phi, const std::vector<Vec3i> &active, ElementType type, SurfaceMesh3D &mesh)
{
    mesh.clear();
    mesh.type = type;
    const Vec3i dims = phi.dimensions();
    // Cells around the edge from voxel p along axis a: p minus these offsets, in the cyclic axes
    // (u, v) = (a + 1, a + 2): 0, e_v, e_u, e_u + e_v.
    const auto cellOffset = [](int a, int corner) {
        Vec3i o = Vec3i::Zero();
        if (corner & 1) o[(a + 2) % 3] = 1;
        if (corner & 2) o[(a + 1) % 3] = 1;
        return o;
    };
    const auto cellInside = [&](const Vec3i &c) { return (c.array() >= 0).all() && (c.array() <= dims.array() - 2).all(); };
    std::vector<Crossing> crossings;
    for (const Vec3i &p : active) {
        const float f1 = phi(p);
        for (int a = 0; a < 3; a++) {
            Vec3i q = p;
            q[a]++;
            if (q[a] >= dims[a]) continue;
            const float f2 = phi(q);
            if (!(f1 * f2 < 0.0f)) continue;
            bool inside = true;
            for (int c = 0; c < 4 && inside; c++) inside = cellInside(p - cellOffset(a, c));
            if (!inside) continue;
            Crossing x;
            x.key = 3 * phi.indexOf(p.x(), p.y(), p.z()) + a;
            const float delta = f2 - f1;
            const float t = (std::abs(delta) < 1E-3f) ? 0.5f : -f1 / delta;
            x.point = p.cast<float>();
            x.point[a] += t;
            // The pivot's side, from the larger-magnitude end so near-zero values are stable.
            x.pivotNegative = (std::abs(f1) >= std::abs(f2)) ? (f1 < 0.0f) : (f2 > 0.0f);
            crossings.push_back(x);
        }
    }
    std::sort(crossings.begin(), crossings.end());
    // One vertex per cell touched by a crossing, numbered in cell order.
    const auto voxelOf = [&](uint64_t key) {
        const size_t index = key / 3;
        const int i = static_cast<int>(index % dims.x());
        const int j = static_cast<int>((index / dims.x()) % dims.y());
        const int k = static_cast<int>(index / (static_cast<size_t>(dims.x()) * dims.y()));
        return Vec3i(i, j, k);
    };
    const auto cellKey = [&](const Vec3i &c) { return static_cast<uint64_t>(phi.indexOf(c.x(), c.y(), c.z())); };
    std::vector<uint64_t> cells;
    cells.reserve(crossings.size() * 4);
    for (const Crossing &x : crossings) {
        const Vec3i p = voxelOf(x.key);
        const int a = static_cast<int>(x.key % 3);
        for (int c = 0; c < 4; c++) cells.push_back(cellKey(p - cellOffset(a, c)));
    }
    std::sort(cells.begin(), cells.end());
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    const auto vertexOf = [&](const Vec3i &c) { return static_cast<uint32_t>(std::lower_bound(cells.begin(), cells.end(), cellKey(c)) - cells.begin()); };
    Points3 sums(cells.size(), Vec3::Zero());
    std::vector<int> counts(cells.size(), 0);
    std::vector<uint32_t> quads;
    quads.reserve(crossings.size() * 4);
    for (const Crossing &x : crossings) {
        const Vec3i p = voxelOf(x.key);
        const int a = static_cast<int>(x.key % 3);
        uint32_t v[4];
        for (int c = 0; c < 4; c++) {
            v[c] = vertexOf(p - cellOffset(a, c));
            sums[v[c]] += x.point;
            counts[v[c]]++;
        }
        // (v0, v2, v3, v1) circles the edge counterclockwise about +e_a; the outward normal points
        // from phi < 0 to phi > 0.
        if (x.pivotNegative) {
            quads.insert(quads.end(), {v[0], v[2], v[3], v[1]});
        } else {
            quads.insert(quads.end(), {v[0], v[1], v[3], v[2]});
        }
    }
    mesh.vertexes.resize(cells.size());
    for (size_t c = 0; c < cells.size(); c++) mesh.vertexes[c] = sums[c] / static_cast<float>(counts[c]);
    if (type == ElementType::Quad) {
        mesh.indexes = std::move(quads);
    } else {
        mesh.indexes.reserve(quads.size() / 4 * 6);
        for (size_t q = 0; q < quads.size(); q += 4) {
            const uint32_t a = quads[q], b = quads[q + 1], c = quads[q + 2], d = quads[q + 3];
            const Vec3 &pa = mesh.vertexes[a], &pb = mesh.vertexes[b], &pc = mesh.vertexes[c], &pd = mesh.vertexes[d];
            if (distanceSqr(pa, pc) <= distanceSqr(pb, pd)) {
                mesh.indexes.insert(mesh.indexes.end(), {a, b, c, a, c, d});
            } else {
                mesh.indexes.insert(mesh.indexes.end(), {a, b, d, b, c, d});
            }
        }
    }
    mesh.updateVertexNormals();
}
}  // namespace springls
