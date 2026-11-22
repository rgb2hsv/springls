#include <levelset/LevelSet3D.hpp>

#include <algorithm>

namespace springls
{
namespace
{
constexpr float kMaxCurvatureForce = 10.0f;
}  // namespace

float LevelSet3D::speed(int i, int j, int k, const Vec3 &g, float curvatureWeight) const
{
    const VolumeF &s = previous();
    const float v111 = s(i, j, k);
    const float v011 = s.at(i - 1, j, k), v211 = s.at(i + 1, j, k);
    const float v101 = s.at(i, j - 1, k), v121 = s.at(i, j + 1, k);
    const float v110 = s.at(i, j, k - 1), v112 = s.at(i, j, k + 1);
    const float DxNeg = v111 - v011, DxPos = v211 - v111;
    const float DyNeg = v111 - v101, DyPos = v121 - v111;
    const float DzNeg = v111 - v110, DzPos = v112 - v111;
    float kappa = 0.0f;
    if (curvatureWeight != 0.0f) {
        const float v010 = s.at(i - 1, j, k - 1), v120 = s.at(i, j + 1, k - 1), v100 = s.at(i, j - 1, k - 1), v210 = s.at(i + 1, j, k - 1);
        const float v001 = s.at(i - 1, j - 1, k), v201 = s.at(i + 1, j - 1, k), v221 = s.at(i + 1, j + 1, k), v021 = s.at(i - 1, j + 1, k);
        const float v012 = s.at(i - 1, j, k + 1), v122 = s.at(i, j + 1, k + 1), v102 = s.at(i, j - 1, k + 1), v212 = s.at(i + 1, j, k + 1);
        const float DxCtr = 0.5f * (v211 - v011), DyCtr = 0.5f * (v121 - v101), DzCtr = 0.5f * (v112 - v110);
        const float DxxCtr = v211 - v111 - v111 + v011, DyyCtr = v121 - v111 - v111 + v101, DzzCtr = v112 - v111 - v111 + v110;
        const float DxyCtr = (v221 - v021 - v201 + v001) * 0.25f;
        const float DxzCtr = (v212 - v012 - v210 + v010) * 0.25f;
        const float DyzCtr = (v122 - v102 - v120 + v100) * 0.25f;
        const float numer = 0.5f * ((DyyCtr + DzzCtr) * DxCtr * DxCtr + (DxxCtr + DzzCtr) * DyCtr * DyCtr + (DxxCtr + DyyCtr) * DzCtr * DzCtr -
                                    2 * DxCtr * DyCtr * DxyCtr - 2 * DxCtr * DzCtr * DxzCtr - 2 * DyCtr * DzCtr * DyzCtr);
        const float denom = DxCtr * DxCtr + DyCtr * DyCtr + DzCtr * DzCtr;
        kappa = (std::abs(denom) > 1E-5f) ? curvatureWeight * numer / denom : curvatureWeight * numer * signOf(denom) * 1E5f;
        kappa = std::clamp(kappa, -kMaxCurvatureForce, kMaxCurvatureForce);
    }
    // Upwind advection by v.
    const float neg[3] = {DxNeg, DyNeg, DzNeg}, pos[3] = {DxPos, DyPos, DzPos};
    float advection = 0.0f;
    for (int a = 0; a < 3; a++) {
        if (g[a] > 0) advection += g[a] * neg[a];
        else if (g[a] < 0) advection += g[a] * pos[a];
    }
    return -advection + kappa;
}
}  // namespace springls
