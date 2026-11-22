#pragma once
#include <levelset/NarrowBand3D.hpp>

#include <cmath>

namespace springls
{
// Narrow-band level set on a uniform 3D grid (phi < 0 inside, in voxels) evolved by
//   phi_t + v . grad(phi) = w kappa
// with first-order upwind advection [Osher and Sethian, J. Comput. Phys. 1988] on the sparse field
// of NarrowBand3D; 3D counterpart of LevelSet2D.
//
// The spring level set evolves it toward its springls (v is the pull toward the nearest springl);
// LevelSetMethod3D evolves it with the flow itself (the basic level set method).
class LevelSet3D : public NarrowBand3D
{
public:
    // One update: every zero-layer voxel (i, j, k) moves by timeStep * clamp(-upwind(v) + w kappa,
    // -1, 1) with v = velocity(i, j, k) (voxels per unit of timeStep; evaluated only on the zero
    // layer), w = curvatureWeight and the stencils on previous().
    template <class Velocity>
    void evolve(const Velocity &velocity, float curvatureWeight, float timeStep)
    {
        const std::vector<Vec3i> &band = active();
        std::vector<float> &d = delta();
        const int N = static_cast<int>(band.size());
        SPRINGLS_OMP(omp parallel for if (N >= kParallelMinHeavyItems))
        for (int n = 0; n < N; n++) {
            const int i = band[n].x(), j = band[n].y(), k = band[n].z();
            d[n] = isZeroLayer(i, j, k) ? speed(i, j, k, velocity(i, j, k), curvatureWeight) : 0.0f;
        }
        update(timeStep);
    }

    // Whether (i, j, k) is on the zero layer of the previous update.
    bool isZeroLayer(int i, int j, int k) const { return std::abs(previous()(i, j, k)) <= 0.5f; }
    // -upwind(v) + curvatureWeight * kappa at (i, j, k), from previous().
    float speed(int i, int j, int k, const Vec3 &v, float curvatureWeight) const;
};
}  // namespace springls
