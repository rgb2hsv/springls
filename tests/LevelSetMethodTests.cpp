// The basic level set method: LevelSet2D / LevelSet3D and the Enright runs of the level set methods.
#include "Check.hpp"

#include <levelset/LevelSet2D.hpp>
#include <levelset/LevelSet3D.hpp>
#include <scenarios/EnrightTest2D.hpp>
#include <scenarios/EnrightTest3D.hpp>

#include <cmath>

using namespace springls;

namespace
{
// x of the zero crossing of phi along row j, searching from i0 in direction step.
float CrossingAlongRow(const ImageF &phi, int j, int i0, int step)
{
    for (int i = i0; i >= 0 && i + step >= 0 && i + step < phi.width(); i += step) {
        const float a = phi(i, j), b = phi(i + step, j);
        if ((a < 0.0f) != (b < 0.0f)) return static_cast<float>(i) + step * a / (a - b);
    }
    return -1.0f;
}

EnrightMetrics2D RunToEnd2D(Method method, int grid, double period, int depth = 2)
{
    EnrightConfig2D config;
    config.gridSize = grid;
    config.period = period;
    auto sim = MakeSimulation(method, depth);
    ConfigureEnright(*sim, config);
    CHECK(sim->init());
    int steps = 0;
    while (sim->step() && steps < 100000) steps++;
    return MeasureEnright(*sim, config);
}

EnrightMetrics3D RunToEnd3D(Method3D method, int grid, double period, int depth = 2)
{
    EnrightConfig3D config;
    config.gridSize = grid;
    config.period = period;
    auto sim = MakeSimulation(method, ElementType::Quad, depth);
    ConfigureEnright(*sim, config);
    CHECK(sim->init());
    int steps = 0;
    while (sim->step() && steps < 100000) steps++;
    return MeasureEnright(*sim, config);
}
}  // namespace

TEST(level_set_2d_translates_a_circle)
{
    // A circle of radius 8 at (24, 32) moves 4 pixels in +x: 16 updates of a quarter pixel.
    ImageF phi;
    phi.resize(64, 64);
    for (int j = 0; j < 64; j++) {
        for (int i = 0; i < 64; i++) phi(i, j) = std::hypot(i - 24.0f, j - 32.0f) - 8.0f;
    }
    LevelSet2D levelSet;
    levelSet.initialize(phi);
    for (int n = 0; n < 16; n++) levelSet.evolve([](int, int) { return Vec2(0.25f, 0.0f); }, 0.0f, 1.0f);
    const float right = CrossingAlongRow(levelSet.phi(), 32, 28, 1), left = CrossingAlongRow(levelSet.phi(), 32, 28, -1);
    CHECK_NEAR(right, 36.0f, 0.3f);
    CHECK_NEAR(left, 20.0f, 0.3f);
}

TEST(level_set_3d_translates_a_sphere)
{
    VolumeF phi;
    phi.resize(32, 32, 32);
    for (int k = 0; k < 32; k++) {
        for (int j = 0; j < 32; j++) {
            for (int i = 0; i < 32; i++) phi(i, j, k) = std::sqrt((i - 12.0f) * (i - 12.0f) + (j - 16.0f) * (j - 16.0f) + (k - 16.0f) * (k - 16.0f)) - 6.0f;
        }
    }
    LevelSet3D levelSet;
    levelSet.initialize(phi);
    for (int n = 0; n < 12; n++) levelSet.evolve([](int, int, int) { return Vec3(0.25f, 0.0f, 0.0f); }, 0.0f, 1.0f);
    // The +x crossing through the center: from 18 to 21.
    float crossing = -1.0f;
    for (int i = 16; i < 31; i++) {
        const float a = levelSet.phi()(i, 16, 16), b = levelSet.phi()(i + 1, 16, 16);
        if (a < 0.0f && b >= 0.0f) {
            crossing = static_cast<float>(i) + a / (a - b);
            break;
        }
    }
    CHECK_NEAR(crossing, 21.0f, 0.3f);
}

TEST(level_set_methods_enright_2d)
{
    // First-order upwinding loses area, and halves the loss when the grid doubles.
    const EnrightMetrics2D coarse = RunToEnd2D(Method::LevelSet, 64, 0.5);
    const EnrightMetrics2D fine = RunToEnd2D(Method::LevelSet, 128, 0.5);
    std::printf("  2D level set area error: g64 %.4f, g128 %.4f\n", coarse.relativeAreaError, fine.relativeAreaError);
    CHECK(coarse.springlCount == 0 && fine.springlCount == 0);
    CHECK(coarse.relativeAreaError < 0.0 && coarse.relativeAreaError > -0.25);
    CHECK(fine.relativeAreaError < 0.0 && std::abs(coarse.relativeAreaError) > 1.6 * std::abs(fine.relativeAreaError));
    // The adaptive grid at depth 2 (64 x 64 equivalent) is about as accurate as the uniform 64 x 64 grid.
    const EnrightMetrics2D adaptive = RunToEnd2D(Method::AdaptiveLevelSet, 16, 0.5, 2);
    std::printf("  2D adaptive level set area error (16, depth 2): %.4f\n", adaptive.relativeAreaError);
    CHECK(adaptive.springlCount == 0);
    CHECK(adaptive.relativeAreaError < 0.0 && adaptive.relativeAreaError > -0.3);
}

TEST(level_set_methods_enright_3d)
{
    const EnrightMetrics3D uniform = RunToEnd3D(Method3D::LevelSet, 48, 0.25);
    const EnrightMetrics3D adaptive = RunToEnd3D(Method3D::AdaptiveLevelSet, 12, 0.25, 2);
    std::printf("  3D level set volume error: g48 %.4f, adaptive g12 d2 %.4f\n", uniform.relativeVolumeError, adaptive.relativeVolumeError);
    CHECK(uniform.springlCount == 0 && adaptive.springlCount == 0);
    CHECK(uniform.relativeVolumeError < 0.0 && uniform.relativeVolumeError > -0.6);
    CHECK(adaptive.relativeVolumeError < 0.0 && adaptive.relativeVolumeError > -0.6);
}
