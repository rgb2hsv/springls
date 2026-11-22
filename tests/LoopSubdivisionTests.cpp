// Adaptive Loop subdivision of closed triangle meshes.
#include "Check.hpp"
#include "IntrinsicTestMeshes.hpp"

#include <intrinsic/IntrinsicTriangulation.hpp>
#include <mesh/LoopSubdivision.hpp>

#include <cmath>
#include <set>
#include <string>

using namespace springls;

namespace
{
bool Closed(const LoopMesh &mesh)
{
    IntrinsicTriangulation tri;
    std::string error;
    const bool ok = tri.build(mesh.vertices, mesh.triangles, error);
    if (!ok) std::printf("  build: %s\n", error.c_str());
    return ok;
}
}  // namespace

TEST(loop_uniform_counts_and_rules)
{
    const springls_test::TestMesh ico = springls_test::Icosphere(0);
    const LoopMesh mesh{ico.vertices, ico.triangles};
    const LoopMesh fine = LoopSubdivide(mesh, std::vector<uint8_t>(mesh.triangles.size(), 1));
    CHECK(fine.triangles.size() == 4 * mesh.triangles.size());
    CHECK(fine.vertices.size() == 12 + 30);
    CHECK(Closed(fine));
    // Old vertex (valence 5): (1 - 5 β) v + β Σ neighbors, β = (5/8 - (3/8 + cos(2π/5)/4)²) / 5.
    const double c = 3.0 / 8.0 + 0.25 * std::cos(2.0 * kPi / 5.0);
    const double beta = (5.0 / 8.0 - c * c) / 5.0;
    std::set<uint32_t> ring;
    for (const auto &t : mesh.triangles) {
        for (int k = 0; k < 3; k++) {
            if (t[k] == 0) {
                ring.insert(t[(k + 1) % 3]);
                ring.insert(t[(k + 2) % 3]);
            }
        }
    }
    Vec3d expected = (1.0 - 5.0 * beta) * mesh.vertices[0].cast<double>();
    for (uint32_t n : ring) expected += beta * mesh.vertices[n].cast<double>();
    CHECK((fine.vertices[0].cast<double>() - expected).norm() < 1e-6);
    // Edge vertices: 3/8 (a + b) + 1/8 (c + d); on a regular icosahedron all are at the same radius.
    double lo = 1e30, hi = 0.0;
    for (size_t v = 12; v < fine.vertices.size(); v++) {
        lo = std::min(lo, static_cast<double>(fine.vertices[v].norm()));
        hi = std::max(hi, static_cast<double>(fine.vertices[v].norm()));
    }
    CHECK(hi - lo < 1e-6);
}

TEST(loop_adaptive_region_stays_closed)
{
    const springls_test::TestMesh ico = springls_test::Icosphere(2);
    const LoopMesh mesh{ico.vertices, ico.triangles};
    // Refine the cap z > 0.5 twice.
    LoopMesh current = mesh;
    for (int level = 0; level < 2; level++) {
        std::vector<uint8_t> refine(current.triangles.size(), 0);
        for (size_t f = 0; f < current.triangles.size(); f++) {
            for (uint32_t v : current.triangles[f]) refine[f] |= current.vertices[v].z() > 0.5f;
        }
        const LoopMesh next = LoopSubdivide(current, refine);
        CHECK(Closed(next));
        CHECK(next.triangles.size() > current.triangles.size());
        current = next;
    }
    // Faces far from the cap are untouched; the cap's edges are about a quarter as long.
    const auto meanEdge = [&](const LoopMesh &m, bool cap) {
        double sum = 0.0;
        int n = 0;
        for (const auto &t : m.triangles) {
            const bool inCap = m.vertices[t[0]].z() > 0.8f && m.vertices[t[1]].z() > 0.8f && m.vertices[t[2]].z() > 0.8f;
            const bool far = m.vertices[t[0]].z() < 0.0f && m.vertices[t[1]].z() < 0.0f && m.vertices[t[2]].z() < 0.0f;
            if (cap ? !inCap : !far) continue;
            for (int k = 0; k < 3; k++) sum += (m.vertices[t[k]] - m.vertices[t[(k + 1) % 3]]).norm();
            n += 3;
        }
        return sum / n;
    };
    CHECK(std::abs(meanEdge(current, false) - meanEdge(mesh, false)) < 1e-6);
    CHECK(meanEdge(current, true) < 0.3 * meanEdge(mesh, true));
    // Old vertices keep their indices.
    for (size_t v = 0; v < mesh.vertices.size(); v++) {
        if (mesh.vertices[v].z() < 0.0f) CHECK(current.vertices[v] == mesh.vertices[v]);
    }
}
