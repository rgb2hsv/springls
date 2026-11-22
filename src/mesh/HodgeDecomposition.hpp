#pragma once
// Helmholtz-Hodge decomposition of a tangent vector field on a closed triangle mesh.
//
// A field X, constant on each face and tangent to it, splits L2-orthogonally (area-weighted) into
//     X = ∇φ + n × ∇ψ + r
// with φ, ψ piecewise linear (one value per vertex): the exact part ∇φ (curl-free), the co-exact part
// n × ∇ψ (divergence-free, ψ its stream function) and the remainder r: the harmonic part (2g dimensions;
// none on a genus-0 surface) plus what piecewise-constant fields hold beyond the gradients of
// piecewise-linear functions (it shrinks with h for a smooth X). φ and ψ are the least-squares fits min |∇φ - X| and min |n × ∇ψ - X| (= |∇ψ - X × n|),
// each one cotan Poisson solve:
//     K φ = Σ_f A_f ∇λ_i · X_f,    K ψ = Σ_f A_f ∇λ_i · (X_f × n_f),
// K the P1 stiffness (cotan) matrix and λ_i the hat functions.
//
// The co-exact part is divergence-free in the strongest discrete sense: its weak divergence vanishes at
// every vertex, and its flux through each edge is the same from both faces (the jump of ψ along the
// edge), so its streamlines are the level sets of ψ and it carries area exactly.
#include <core/Common3D.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace springls
{
using Triangles = std::vector<std::array<uint32_t, 3>>;

struct HodgeDecomposition {
    std::vector<double> phi, psi;                  // per vertex, mean zero
    std::vector<Vec3d> exact, coexact, remainder;  // per face: ∇φ, n × ∇ψ, X - ∇φ - n × ∇ψ
};

// `field`: one 3D vector per triangle (its component normal to the triangle is ignored). False with a
// message if the mesh has a degenerate triangle or the Poisson solve fails (e.g. a disconnected mesh).
bool DecomposeHodge(const Points3 &vertices, const Triangles &triangles, const std::vector<Vec3d> &field, HodgeDecomposition &out, std::string &error);

// Per-face gradient ∇f and rotated gradient n × ∇f of a function with one value per vertex (n the
// unit normal of the counter-clockwise triangle).
std::vector<Vec3d> FaceGradient(const Points3 &vertices, const Triangles &triangles, const std::vector<double> &f);
std::vector<Vec3d> FaceCoGradient(const Points3 &vertices, const Triangles &triangles, const std::vector<double> &f);
// Weak divergence at each vertex i: -Σ_f A_f ∇λ_i · X_f (the net flux of X out of i's dual cell).
std::vector<double> WeakDivergence(const Points3 &vertices, const Triangles &triangles, const std::vector<Vec3d> &field);
// Area-weighted L2 norm sqrt(Σ_f A_f |X_f|²) of a per-face field (its tangential part).
double FaceFieldNorm(const Points3 &vertices, const Triangles &triangles, const std::vector<Vec3d> &field);
}  // namespace springls
