// Helmholtz-Hodge decomposition on triangle meshes and the loop scenario's Hodge flow.
#include "Check.hpp"
#include "IntrinsicTestMeshes.hpp"
#include "IntrinsicTestRegions.hpp"

#include <intrinsic/IntrinsicLoopScenario.hpp>
#include <intrinsic/IntrinsicTrace.hpp>
#include <mesh/HodgeDecomposition.hpp>

#include <algorithm>
#include <cmath>
#include <string>

using namespace springls;

namespace
{
// The rigid rotation ẑ × x at each triangle's centroid (DecomposeHodge takes its tangential part).
std::vector<Vec3d> Rotation(const springls_test::TestMesh &mesh)
{
    std::vector<Vec3d> X;
    for (const auto &t : mesh.triangles) {
        const Vec3d c = (mesh.vertices[t[0]] + mesh.vertices[t[1]] + mesh.vertices[t[2]]).cast<double>() / 3.0;
        X.push_back(Vec3d::UnitZ().cross(c));
    }
    return X;
}

HodgeDecomposition Decompose(const springls_test::TestMesh &mesh, const std::vector<Vec3d> &X)
{
    HodgeDecomposition h;
    std::string error;
    CHECK(DecomposeHodge(mesh.vertices, mesh.triangles, X, h, error));
    if (!error.empty()) std::printf("  decompose: %s\n", error.c_str());
    return h;
}

double Norm(const springls_test::TestMesh &mesh, const std::vector<Vec3d> &field) { return FaceFieldNorm(mesh.vertices, mesh.triangles, field); }

std::vector<Vec3d> Minus(const std::vector<Vec3d> &a, const std::vector<Vec3d> &b)
{
    std::vector<Vec3d> d(a.size());
    for (size_t k = 0; k < a.size(); k++) d[k] = a[k] - b[k];
    return d;
}

// An ellipsoid (the icosphere stretched along x): the rotation about z projected onto it is not
// divergence-free.
springls_test::TestMesh Ellipsoid(int level) { return springls_test::SkewedIcosphere(level, 1.6); }
}  // namespace

TEST(hodge_rotation_on_sphere_is_coexact)
{
    // On the sphere the rotation is divergence-free: nearly all of it is co-exact (ψ = -z, up to the
    // centroid sampling and the mesh's flat faces).
    const springls_test::TestMesh mesh = springls_test::Icosphere(4);
    const std::vector<Vec3d> X = Rotation(mesh);
    const HodgeDecomposition h = Decompose(mesh, X);
    const double norm = Norm(mesh, X);
    const double coexactError = Norm(mesh, Minus(h.coexact, X)) / norm, exact = Norm(mesh, h.exact) / norm;
    std::printf("  sphere: |X - n x grad psi| / |X| %.3g, |grad phi| / |X| %.3g\n", coexactError, exact);
    CHECK(coexactError < 0.02 && exact < 0.005);
    // ψ ≈ -z (n × ∇(-z) = ẑ × x on the unit sphere), mean zero.
    double worst = 0.0;
    for (size_t v = 0; v < mesh.vertices.size(); v++) worst = std::max(worst, std::abs(h.psi[v] + mesh.vertices[v].z()));
    CHECK(worst < 0.02);
}

TEST(hodge_parts_on_ellipsoid)
{
    const springls_test::TestMesh mesh = Ellipsoid(4);
    const std::vector<Vec3d> X = Rotation(mesh);
    const HodgeDecomposition h = Decompose(mesh, X);
    const double norm = Norm(mesh, X);
    // X has a real curl-free part here; the co-exact part carries most of the field.
    const double exact = Norm(mesh, h.exact) / norm, coexact = Norm(mesh, h.coexact) / norm, remainder = Norm(mesh, h.remainder) / norm;
    std::printf("  ellipsoid: exact %.3g, co-exact %.3g, remainder %.3g of |X|\n", exact, coexact, remainder);
    CHECK(exact > 0.05 && coexact > 0.9 && remainder < 0.05);
    // The parts are orthogonal: |X|² = |∇φ|² + |n × ∇ψ|² + |r|².
    CHECK_NEAR(exact * exact + coexact * coexact + remainder * remainder, 1.0, 1e-9);
    // X is not divergence-free; its co-exact part is, exactly, at every vertex.
    const std::vector<double> divX = WeakDivergence(mesh.vertices, mesh.triangles, X);
    const std::vector<double> divV = WeakDivergence(mesh.vertices, mesh.triangles, h.coexact);
    double worstX = 0.0, worstV = 0.0;
    for (size_t v = 0; v < divX.size(); v++) {
        worstX = std::max(worstX, std::abs(divX[v]));
        worstV = std::max(worstV, std::abs(divV[v]));
    }
    std::printf("  ellipsoid: worst weak divergence of X %.3g, of n x grad psi %.3g\n", worstX, worstV);
    CHECK(worstX > 1e-4 && worstV < 1e-8 * worstX);
    // The exact part carries all of X's divergence.
    const std::vector<double> divE = WeakDivergence(mesh.vertices, mesh.triangles, h.exact);
    double worstDiff = 0.0;
    for (size_t v = 0; v < divX.size(); v++) worstDiff = std::max(worstDiff, std::abs(divE[v] - divX[v]));
    CHECK(worstDiff < 1e-9 * worstX);
}

TEST(hodge_coexact_flux_is_continuous)
{
    // Through each edge a→b of each triangle, the in-plane normal flux of n × ∇ψ is ψ(a) - ψ(b): the same
    // from both triangles, so no flux appears or vanishes across an edge.
    const springls_test::TestMesh mesh = Ellipsoid(3);
    const HodgeDecomposition h = Decompose(mesh, Rotation(mesh));
    double worst = 0.0, scale = 0.0;
    for (size_t f = 0; f < mesh.triangles.size(); f++) {
        const auto &t = mesh.triangles[f];
        const Vec3d x0 = mesh.vertices[t[0]].cast<double>(), x1 = mesh.vertices[t[1]].cast<double>(), x2 = mesh.vertices[t[2]].cast<double>();
        const Vec3d n = (x1 - x0).cross(x2 - x0).normalized();
        for (int k = 0; k < 3; k++) {
            const uint32_t a = t[k], b = t[(k + 1) % 3];
            const Vec3d e = mesh.vertices[b].cast<double>() - mesh.vertices[a].cast<double>();
            const double flux = h.coexact[f].dot(e.cross(n));  // through a→b, toward the outside of the triangle
            worst = std::max(worst, std::abs(flux - (h.psi[a] - h.psi[b])));
            scale = std::max(scale, std::abs(flux));
        }
    }
    std::printf("  ellipsoid: worst flux mismatch %.3g, largest edge flux %.3g\n", worst, scale);
    CHECK(scale > 0.0 && worst < 1e-9 * scale);
}

TEST(hodge_remainder_shrinks_with_refinement)
{
    // Genus 0: no harmonic fields, so the remainder is discretization only and falls with h.
    double previous = 1.0;
    for (int level = 3; level <= 5; level++) {
        const springls_test::TestMesh mesh = Ellipsoid(level);
        const std::vector<Vec3d> X = Rotation(mesh);
        const HodgeDecomposition h = Decompose(mesh, X);
        const double remainder = Norm(mesh, h.remainder) / Norm(mesh, X);
        std::printf("  ellipsoid level %d: remainder %.3g of |X|\n", level, remainder);
        CHECK(remainder < 0.6 * previous);
        previous = remainder;
    }
}

TEST(loop_hodge_flow_on_sphere_matches_rotation)
{
    // On the sphere the Hodge flow is (nearly) the rotation itself: same speed at the object, same period.
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    LoopConfig config;
    config.object = LoopObject::Star;
    config.speed = 2.0;
    LoopScenario rotation, hodge;
    std::string error;
    CHECK(rotation.setup(tri, config, error));
    config.flow = LoopFlow::Hodge;
    CHECK(hodge.setup(tri, config, error));
    if (!error.empty()) std::printf("  setup: %s\n", error.c_str());
    CHECK(rotation.flow() == LoopFlow::Rotation && rotation.faceVelocity().empty());
    CHECK(hodge.flow() == LoopFlow::Hodge && hodge.faceVelocity().size() == tri.input().faceCount());
    CHECK_NEAR(hodge.velocity(hodge.objectCenter(), 0.0).norm(), config.speed, 1e-9 * config.speed);
    std::printf("  sphere: period rotation %.5f, Hodge %.5f; exact %.3g, remainder %.3g\n", rotation.period(), hodge.period(), hodge.exactFraction(),
                hodge.remainderFraction());
    CHECK(std::abs(hodge.period() - rotation.period()) < 0.02 * rotation.period());
    CHECK(hodge.exactFraction() < 0.005);
    // Same direction at the object (the outline's orientation follows it).
    const Vec2d a = rotation.velocity(rotation.objectCenter(), 0.0), b = hodge.velocity(hodge.objectCenter(), 0.0);
    CHECK(a.normalized().dot(b.normalized()) > 0.99);
}

TEST(loop_hodge_flow_on_ellipsoid)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(Ellipsoid(4));
    LoopConfig config;
    config.object = LoopObject::Star;
    config.flow = LoopFlow::Hodge;
    LoopScenario scenario;
    std::string error;
    CHECK(scenario.setup(tri, config, error));
    if (!error.empty()) std::printf("  setup: %s\n", error.c_str());
    CHECK(scenario.exactFraction() > 0.05 && scenario.remainderFraction() < 0.05);
    CHECK_NEAR(scenario.velocity(scenario.objectCenter(), 0.0).norm(), config.speed, 1e-9);
    // The center comes back after one revolution at full speed (the period over π: the schedule's
    // cos(πt/T) covers one revolution in T/2), forward along the streamline.
    const double half = scenario.period() / std::acos(-1.0);
    CHECK(half > 0.0 && std::isfinite(half));
    const int steps = 2000;
    const double dt = half / steps;
    SurfacePoint x = scenario.objectCenter();
    double farthest = 0.0;
    const Vec3d start = tri.toExtrinsic(x).position;
    for (int k = 0; k < steps; k++) {
        const Vec2d v1 = scenario.velocity(x, 0.0);
        const TraceResult mid = Trace(tri, x, (0.5 * dt) * v1);
        x = Trace(tri, x, dt * TransportBack(mid, scenario.velocity(mid.end, 0.0))).end;
        farthest = std::max(farthest, (tri.toExtrinsic(x).position - start).norm());
    }
    const double back = (tri.toExtrinsic(x).position - start).norm();
    std::printf("  ellipsoid: period %.4f, farthest %.3f, back within %.4f (h %.4f)\n", scenario.period(), farthest, back, tri.meanEdgeLength());
    CHECK(farthest > 1.0 && back < 0.5 * tri.meanEdgeLength());
    CHECK(scenario.streamFunction().size() == tri.inputVertices().size());
}
