// Particle-sample correction for single-object springls on uniform grids (2D and 3D): band
// samples beside a springl but on the wrong side of it take its signed distance.
#include "Check.hpp"

#include <levelset/NarrowBand2D.hpp>
#include <levelset/NarrowBand3D.hpp>
#include <scenarios/EnrightTest2D.hpp>
#include <scenarios/EnrightTest3D.hpp>
#include <springl/SpringLevelSet2D.hpp>
#include <springl/SpringLevelSet3D.hpp>

#include <cmath>
#include <cstdio>

using namespace springls;

namespace
{
// Mean |phi| at the springl particles over every step: how far the level set trails the springls.
double MeanTracking2D(bool correction)
{
    EnrightConfig2D config;
    config.gridSize = 64;
    config.period = 1.0;
    SpringLevelSet2D sim;
    sim.tuning.springlCorrection = correction;
    ConfigureEnright(sim, config);
    CHECK(sim.init());
    double sum = 0.0;
    size_t count = 0;
    while (sim.step()) {
        for (const Vec2 &p : sim.getManifold().springlParticles) {
            sum += std::abs(sim.getLevelSet().interpolate(p));
            count++;
        }
    }
    return sum / std::max<size_t>(count, 1);
}

double MeanTracking3D(bool correction)
{
    EnrightConfig3D config;
    config.gridSize = 32;
    config.period = 1.0;
    SpringLevelSet3D sim;
    sim.tuning.springlCorrection = correction;
    ConfigureEnright(sim, config);
    CHECK(sim.init());
    double sum = 0.0;
    size_t count = 0;
    for (int step = 0; step < 40 && sim.step(); step++) {
        for (const Vec3 &p : sim.getManifold().springlParticles) {
            sum += std::abs(sim.getLevelSet().interpolate(p));
            count++;
        }
    }
    return sum / std::max<size_t>(count, 1);
}
}  // namespace

TEST(narrow_band_2d_correct_signs_flips_wrong_side)
{
    constexpr int kSize = 32;
    ImageF phi(kSize, kSize);
    for (int j = 0; j < kSize; j++) {
        for (int i = 0; i < kSize; i++) phi(i, j) = i - 15.5f;
    }
    NarrowBand2D band;
    band.initialize(phi);
    const int changed = band.correctSigns([](int i, int j, float &d) {
        if (i == 16 && j == 10) {
            d = -0.3f;  // inside, though phi says outside
            return true;
        }
        if (i == 14 && j == 10) {
            d = -1.4f;  // inside and phi agrees: unchanged
            return true;
        }
        return false;
    });
    CHECK(changed == 1);
    CHECK_NEAR(band.phi()(16, 10), -0.3f, 1E-6f);
    CHECK_NEAR(band.previous()(16, 10), -0.3f, 1E-6f);
    CHECK_NEAR(band.phi()(14, 10), -1.5f, 1E-6f);
}

TEST(narrow_band_3d_correct_signs_flips_wrong_side)
{
    constexpr int kSize = 16;
    VolumeF phi(kSize, kSize, kSize);
    for (int k = 0; k < kSize; k++) {
        for (int j = 0; j < kSize; j++) {
            for (int i = 0; i < kSize; i++) phi(i, j, k) = i - 7.5f;
        }
    }
    NarrowBand3D band;
    band.initialize(phi);
    const int changed = band.correctSigns([](int i, int j, int k, float &d) {
        if (i != 8 || j != 5 || k != 5) return false;
        d = -0.2f;
        return true;
    });
    CHECK(changed == 1);
    CHECK_NEAR(band.phi()(8, 5, 5), -0.2f, 1E-6f);
    CHECK_NEAR(band.previous()(8, 5, 5), -0.2f, 1E-6f);
}

TEST(springls_2d_correction_tightens_tracking)
{
    const double with = MeanTracking2D(true);
    const double without = MeanTracking2D(false);
    std::printf("  2D mean |phi| at springl particles: with correction %.4f, without %.4f\n", with, without);
    CHECK(with < without);
}

TEST(springls_3d_correction_tightens_tracking)
{
    const double with = MeanTracking3D(true);
    const double without = MeanTracking3D(false);
    std::printf("  3D mean |phi| at springl particles: with correction %.4f, without %.4f\n", with, without);
    CHECK(with < without);
}
