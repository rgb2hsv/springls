#include <levelset/LevelSet2D.hpp>

#include <algorithm>
#include <cmath>

namespace springls
{
namespace
{
constexpr float kMaxCurvatureForce = 10.0f;
}  // namespace

float LevelSet2D::speed(int i, int j, const Vec2 &g, float curvatureWeight) const
{
    const ImageF &s = previous();
    const float v11 = s(i, j);
    const float v01 = s.at(i - 1, j), v10 = s.at(i, j - 1), v21 = s.at(i + 1, j), v12 = s.at(i, j + 1);
    const float DxNeg = v11 - v01, DxPos = v21 - v11;
    const float DyNeg = v11 - v10, DyPos = v12 - v11;
    float kappa = 0.0f;
    if (curvatureWeight != 0.0f) {
        const float v00 = s.at(i - 1, j - 1), v20 = s.at(i + 1, j - 1), v22 = s.at(i + 1, j + 1), v02 = s.at(i - 1, j + 1);
        const float DxCtr = 0.5f * (v21 - v01), DyCtr = 0.5f * (v12 - v10);
        const float DxxCtr = v21 - v11 - v11 + v01, DyyCtr = v12 - v11 - v11 + v10;
        const float DxyCtr = (v22 - v02 - v20 + v00) * 0.25f;
        const float numer = 0.5f * (DyCtr * DyCtr * DxxCtr - 2 * DxCtr * DyCtr * DxyCtr + DxCtr * DxCtr * DyyCtr);
        const float denom = DxCtr * DxCtr + DyCtr * DyCtr;
        kappa = (std::abs(denom) > 1E-5f) ? curvatureWeight * numer / denom : curvatureWeight * numer * signOf(denom) * 1E5f;
        kappa = std::clamp(kappa, -kMaxCurvatureForce, kMaxCurvatureForce);
    }
    // Upwind advection by v.
    float advection = 0.0f;
    if (g.x() > 0) advection = g.x() * DxNeg;
    else if (g.x() < 0) advection = g.x() * DxPos;
    if (g.y() > 0) advection += g.y() * DyNeg;
    else if (g.y() < 0) advection += g.y() * DyPos;
    return -advection + kappa;
}
}  // namespace springls
