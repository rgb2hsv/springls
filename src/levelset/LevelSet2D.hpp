#pragma once
#include <levelset/NarrowBand2D.hpp>

#include <cmath>

namespace springls
{
// Narrow-band level set on a uniform 2D grid (phi < 0 inside, in pixels) evolved by
//   phi_t + v . grad(phi) = w kappa
// with first-order upwind advection [Osher and Sethian, "Fronts Propagating with
// Curvature-Dependent Speed", J. Comput. Phys. 1988] on the sparse field of NarrowBand2D: only the
// zero layer (|phi| <= 1/2) moves, the distance layers around it are rebuilt.
//
// The spring level set evolves it toward its springls (v is the pull toward the nearest springl);
// LevelSetMethod2D evolves it with the flow itself (the basic level set method).
class LevelSet2D : public NarrowBand2D
{
public:
    // One update: every zero-layer pixel (i, j) moves by timeStep * clamp(-upwind(v) + w kappa, -1, 1)
    // with v = velocity(i, j) (pixels per unit of timeStep; evaluated only on the zero layer), w =
    // curvatureWeight and the stencils on previous().
    template <class Velocity>
    void evolve(const Velocity &velocity, float curvatureWeight, float timeStep)
    {
        const std::vector<Vec2i> &band = active();
        std::vector<float> &d = delta();
        const int N = static_cast<int>(band.size());
        SPRINGLS_OMP(omp parallel for if (N >= kParallelMinHeavyItems))
        for (int n = 0; n < N; n++) {
            const int i = band[n].x(), j = band[n].y();
            d[n] = isZeroLayer(i, j) ? speed(i, j, velocity(i, j), curvatureWeight) : 0.0f;
        }
        update(timeStep);
    }

    // Whether (i, j) is on the zero layer of the previous update.
    bool isZeroLayer(int i, int j) const { return std::abs(previous()(i, j)) <= 0.5f; }
    // -upwind(v) + curvatureWeight * kappa at (i, j), from previous().
    float speed(int i, int j, const Vec2 &v, float curvatureWeight) const;
};
}  // namespace springls
