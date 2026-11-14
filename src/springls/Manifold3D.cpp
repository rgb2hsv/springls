#include <springls/Manifold3D.hpp>

namespace springls
{
void Manifold3D::updateNormals()
{
    const int N = static_cast<int>(springlParticles.size());
    springlParticleNormals.resize(N);
    SPRINGLS_OMP(omp parallel for if (N >= kParallelMinLightItems))
    for (int i = 0; i < N; i++) {
        springlParticleNormals[i] = normalizedOrZero(FanNormal(springlParticles[i], springl(i), springlSize(i)));
    }
}
}  // namespace springls
