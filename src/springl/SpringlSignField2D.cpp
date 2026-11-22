#include <springl/SpringlSignField2D.hpp>

#include <cmath>
#include <limits>

namespace springls
{
float SpringlSignField2D::bandRadius(const Vec2 &p) const
{
    return std::max({mParams.capsuleCells * mInput.cellSize(p), kMinBandRadius * mSpacing, mParams.minBandPixels});
}

int SpringlSignField2D::pixelOf(const Vec2 &p) const
{
    const int i = std::clamp(static_cast<int>(std::lround(p.x() / mSpacing)), 0, mRasterWidth - 1);
    const int j = std::clamp(static_cast<int>(std::lround(p.y() / mSpacing)), 0, mRasterHeight - 1);
    return j * mRasterWidth + i;
}

void SpringlSignField2D::build(const SignFieldInput &input, const SignFieldParams &params)
{
    mInput = input;
    mParams = params;
    mTree.reset();
    for (uint32_t n : mInput.springls) mTree.append((*mInput.vertexes)[2 * n], (*mInput.vertexes)[2 * n + 1], static_cast<int>(n));
    mTree.finish();
    // Raster samples at (i, j) * spacing, covering [0, width-1] x [0, height-1].
    mSpacing = mInput.rasterSpacing > 0.0f ? mInput.rasterSpacing : 1.0f;
    mRasterWidth = std::max(1, static_cast<int>(std::lround((mInput.width - 1) / mSpacing)) + 1);
    mRasterHeight = std::max(1, static_cast<int>(std::lround((mInput.height - 1) / mSpacing)) + 1);
    const int W = mRasterWidth, H = mRasterHeight;
    const float S = mSpacing;
    const int N = W * H;
    mPixelSign.assign(N, 1);
    mWallCount = 0;
    // Explicit Vec2 return: an Eigen product would otherwise outlive the temporary it refers to.
    const auto position = [W, S](int n) -> Vec2 { return S * Vec2(static_cast<float>(n % W), static_cast<float>(n / W)); };
    if (mTree.empty()) return;  // no springls: no surface, outside everywhere

    // 1. Unsigned distance U (the min over the springls), out to twice the capsule radius; the
    // capsules (U <= r) are walls. Samples farther out keep U = their reach.
    std::vector<float> U(N), reach(N);
    std::vector<uint8_t> wall(N, 0);
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 1024) if (N >= kParallelMinHeavyItems))
    for (int n = 0; n < N; n++) {
        const Vec2 q = position(n);
        const float r = bandRadius(q);
        reach[n] = 2.0f * r;
        U[n] = mTree.distance(q, reach[n]);
        wall[n] = U[n] <= r;
    }
    for (int n = 0; n < N; n++) mWallCount += wall[n];
    const auto forNeighbors = [W, H](int n, auto &&visit) {
        const int i = n % W, j = n / W;
        if (i > 0) visit(n - 1);
        if (i + 1 < W) visit(n + 1);
        if (j > 0) visit(n - W);
        if (j + 1 < H) visit(n + W);
    };

    // 2. The exterior floods in from the raster border around the walls; non-wall samples it does
    // not reach are inside.
    std::vector<uint8_t> exterior(N, 0);
    std::vector<int> queue;
    const auto seed = [&](int n) {
        if (!wall[n] && !exterior[n]) {
            exterior[n] = 1;
            queue.push_back(n);
        }
    };
    for (int i = 0; i < W; i++) {
        seed(i);
        seed((H - 1) * W + i);
    }
    for (int j = 0; j < H; j++) {
        seed(j * W);
        seed(j * W + W - 1);
    }
    for (size_t k = 0; k < queue.size(); k++) forNeighbors(queue[k], seed);
    for (int n = 0; n < N; n++) {
        if (!wall[n]) mPixelSign[n] = exterior[n] ? 1 : -1;
    }

    // 3. Upwind sign into the walls, in order of decreasing U (ties by index).
    std::vector<int> walls;
    walls.reserve(mWallCount);
    for (int n = 0; n < N; n++) {
        if (wall[n]) walls.push_back(n);
    }
    std::sort(walls.begin(), walls.end(), [&](int a, int b) { return U[a] != U[b] ? U[a] > U[b] : a < b; });
    std::vector<uint8_t> assigned(N, 0);
    for (int n = 0; n < N; n++) assigned[n] = !wall[n];
    for (int n : walls) {
        float vote = 0.0f;
        bool any = false;
        forNeighbors(n, [&](int m) {
            if (!assigned[m] || U[m] <= U[n]) return;
            vote += static_cast<float>(mPixelSign[m]) * (U[m] - U[n]);
            any = true;
        });
        if (any && vote != 0.0f) {
            mPixelSign[n] = vote < 0.0f ? -1 : 1;
        } else {
            mPixelSign[n] = -1;  // the middle of a feature thinner than two capsules: inside
        }
        assigned[n] = 1;
    }
}

float SpringlSignField2D::distance(const Vec2 &p, float maxDistance) const
{
    if (mTree.empty()) return maxDistance;
    return std::min(mTree.distance(p, maxDistance), maxDistance);
}

float SpringlSignField2D::sign(const Vec2 &p) const { return mPixelSign.empty() ? 1.0f : mPixelSign[pixelOf(p)]; }
}  // namespace springls
