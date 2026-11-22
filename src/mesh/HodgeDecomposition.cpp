#include <mesh/HodgeDecomposition.hpp>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include <cmath>

namespace springls
{
namespace
{
// One triangle: area, unit normal, and the hat-function gradients ∇λ_k (constant on the triangle).
struct FaceGeometry {
    double area = 0.0;
    Vec3d normal = Vec3d::Zero();
    std::array<Vec3d, 3> grad;
};

FaceGeometry Geometry(const Points3 &V, const std::array<uint32_t, 3> &t)
{
    FaceGeometry g;
    const Vec3d x[3] = {V[t[0]].cast<double>(), V[t[1]].cast<double>(), V[t[2]].cast<double>()};
    const Vec3d n = (x[1] - x[0]).cross(x[2] - x[0]);
    const double twiceArea = n.norm();
    g.area = 0.5 * twiceArea;
    if (!(twiceArea > 0.0)) return g;
    g.normal = n / twiceArea;
    // ∇λ_k = n × (x_{k+2} - x_{k+1}) / 2A: perpendicular to the opposite edge, pointing toward corner k.
    for (int k = 0; k < 3; k++) g.grad[k] = g.normal.cross(x[(k + 2) % 3] - x[(k + 1) % 3]) / twiceArea;
    return g;
}

Vec3d Tangential(const FaceGeometry &g, const Vec3d &v) { return v - g.normal.dot(v) * g.normal; }

// Solves K u = b with u[0] pinned to 0 (K is singular by the constants; b sums to zero), then shifts u
// to mean zero.
bool SolvePoisson(const Eigen::SparseMatrix<double> &K, const Eigen::VectorXd &b, std::vector<double> &u, std::string &error)
{
    const Eigen::Index n = K.rows();
    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(K.nonZeros());
    for (Eigen::Index c = 0; c < K.outerSize(); c++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(K, c); it; ++it) {
            if (it.row() == 0 || it.col() == 0) continue;
            entries.emplace_back(it.row(), it.col(), it.value());
        }
    }
    entries.emplace_back(0, 0, 1.0);
    Eigen::SparseMatrix<double> A(n, n);
    A.setFromTriplets(entries.begin(), entries.end());
    Eigen::VectorXd rhs = b;
    rhs[0] = 0.0;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(A);
    if (solver.info() != Eigen::Success) {
        error = "the cotan Laplacian could not be factored (degenerate or disconnected mesh)";
        return false;
    }
    const Eigen::VectorXd x = solver.solve(rhs);
    if (solver.info() != Eigen::Success || !x.allFinite()) {
        error = "the Poisson solve failed";
        return false;
    }
    const double mean = x.mean();
    u.resize(n);
    for (Eigen::Index i = 0; i < n; i++) u[i] = x[i] - mean;
    return true;
}
}  // namespace

bool DecomposeHodge(const Points3 &vertices, const Triangles &triangles, const std::vector<Vec3d> &field, HodgeDecomposition &out, std::string &error)
{
    if (field.size() != triangles.size()) {
        error = "one field vector per triangle expected";
        return false;
    }
    const Eigen::Index n = static_cast<Eigen::Index>(vertices.size());
    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(9 * triangles.size());
    Eigen::VectorXd bPhi = Eigen::VectorXd::Zero(n), bPsi = Eigen::VectorXd::Zero(n);
    for (size_t f = 0; f < triangles.size(); f++) {
        const auto &t = triangles[f];
        const FaceGeometry g = Geometry(vertices, t);
        if (!(g.area > 0.0)) {
            error = "degenerate triangle " + std::to_string(f);
            return false;
        }
        const Vec3d X = Tangential(g, field[f]);
        const Vec3d rotated = X.cross(g.normal);
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) entries.emplace_back(t[a], t[b], g.area * g.grad[a].dot(g.grad[b]));
            bPhi[t[a]] += g.area * g.grad[a].dot(X);
            bPsi[t[a]] += g.area * g.grad[a].dot(rotated);
        }
    }
    Eigen::SparseMatrix<double> K(n, n);
    K.setFromTriplets(entries.begin(), entries.end());
    if (!SolvePoisson(K, bPhi, out.phi, error) || !SolvePoisson(K, bPsi, out.psi, error)) return false;
    out.exact = FaceGradient(vertices, triangles, out.phi);
    out.coexact = FaceCoGradient(vertices, triangles, out.psi);
    out.remainder.resize(triangles.size());
    for (size_t f = 0; f < triangles.size(); f++) {
        const FaceGeometry g = Geometry(vertices, triangles[f]);
        out.remainder[f] = Tangential(g, field[f]) - out.exact[f] - out.coexact[f];
    }
    return true;
}

std::vector<Vec3d> FaceGradient(const Points3 &vertices, const Triangles &triangles, const std::vector<double> &f)
{
    std::vector<Vec3d> out(triangles.size(), Vec3d::Zero());
    for (size_t k = 0; k < triangles.size(); k++) {
        const auto &t = triangles[k];
        const FaceGeometry g = Geometry(vertices, t);
        if (!(g.area > 0.0)) continue;
        out[k] = f[t[0]] * g.grad[0] + f[t[1]] * g.grad[1] + f[t[2]] * g.grad[2];
    }
    return out;
}

std::vector<Vec3d> FaceCoGradient(const Points3 &vertices, const Triangles &triangles, const std::vector<double> &f)
{
    std::vector<Vec3d> out = FaceGradient(vertices, triangles, f);
    for (size_t k = 0; k < triangles.size(); k++) out[k] = Geometry(vertices, triangles[k]).normal.cross(out[k]);
    return out;
}

std::vector<double> WeakDivergence(const Points3 &vertices, const Triangles &triangles, const std::vector<Vec3d> &field)
{
    std::vector<double> div(vertices.size(), 0.0);
    for (size_t f = 0; f < triangles.size(); f++) {
        const auto &t = triangles[f];
        const FaceGeometry g = Geometry(vertices, t);
        const Vec3d X = Tangential(g, field[f]);
        for (int a = 0; a < 3; a++) div[t[a]] -= g.area * g.grad[a].dot(X);
    }
    return div;
}

double FaceFieldNorm(const Points3 &vertices, const Triangles &triangles, const std::vector<Vec3d> &field)
{
    double sum = 0.0;
    for (size_t f = 0; f < triangles.size(); f++) {
        const FaceGeometry g = Geometry(vertices, triangles[f]);
        sum += g.area * Tangential(g, field[f]).squaredNorm();
    }
    return std::sqrt(sum);
}
}  // namespace springls
