#include <springls/Manifold2D.hpp>

#ifdef SPRINGLS_OPENMP
#include <omp.h>
#endif

namespace springls
{
void Manifold2D::updateNormals()
{
    springlParticleNormals.resize(springlParticles.size());
    const int N = static_cast<int>(springlParticles.size());
    SPRINGLS_OMP(omp parallel for if (N >= kParallelMinLightItems))
    for (int i = 0; i < N; i++) {
        const Vec2 t = normalizedOrZero(springlVertexes[2 * i + 1] - springlVertexes[2 * i]);
        springlParticleNormals[i] = Vec2(-t.y(), t.x());
    }
}

int MaxThreads()
{
#ifdef SPRINGLS_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void SetThreads(int n)
{
#ifdef SPRINGLS_OPENMP
    if (n > 0) omp_set_num_threads(n);
#else
    (void)n;
#endif
}
}  // namespace springls
