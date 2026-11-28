// MultiActiveContour2D: plain MOGAC (no springls) moved by a velocity field.
#include "Check.hpp"

#include <levelset/MultiActiveContour2D.hpp>
#include <memory>

#include <cmath>
#include <numbers>

using namespace springls;

namespace
{
constexpr int kSize = 64;
const Vec2 kCenter(32.0f, 32.0f);
constexpr float kRadius = 12.0f;

// A disk split by its vertical diameter: label 1 on the left, 2 on the right, background 0.
void SplitDisk(ImageI &labels, ImageF &distance)
{
    labels = ImageI(kSize, kSize);
    distance = ImageF(kSize, kSize);
    for (int j = 0; j < kSize; j++) {
        for (int i = 0; i < kSize; i++) {
            const Vec2 p(i, j);
            const float r = (p - kCenter).norm();
            const bool inside = r < kRadius;
            labels(i, j) = inside ? ((p.x() < kCenter.x()) ? 1 : 2) : 0;
            // Distance to the circle or to the diameter segment, whichever is nearer.
            const float dy = std::clamp(p.y() - kCenter.y(), -kRadius, kRadius);
            const float toDiameter = (p - Vec2(kCenter.x(), kCenter.y() + dy)).norm();
            distance(i, j) = std::min(std::abs(r - kRadius), toDiameter);
        }
    }
}

struct LabelStats {
    double count[3] = {0, 0, 0};
    double cx[3] = {0, 0, 0};
};

LabelStats Stats(const ImageI &labels)
{
    LabelStats s;
    for (int j = 0; j < labels.height(); j++) {
        for (int i = 0; i < labels.width(); i++) {
            const int l = labels(i, j);
            if (l < 0 || l > 2) continue;
            s.count[l] += 1.0;
            s.cx[l] += i;
        }
    }
    for (int l = 1; l <= 2; l++) s.cx[l] /= std::max(1.0, s.count[l]);
    return s;
}

std::unique_ptr<MultiActiveContour2D> MakeMogac(Vec2 velocity, double endTime)
{
    auto sim = std::make_unique<MultiActiveContour2D>();
    ImageI labels;
    ImageF distance;
    SplitDisk(labels, distance);
    sim->setInitialLabels(labels, distance);
    sim->setCurvatureWeight(0.0f);
    sim->backgroundCurvatureWeight = 0.0f;
    sim->setVelocityFunction([velocity](const Vec2 &) { return velocity; }, 1.0f);
    SpringLevelSet2D::AdvectionSchedule schedule;
    schedule.weight = [](double) { return 1.0f; };
    schedule.endTime = endTime;
    sim->setAdvectionSchedule(schedule);
    return sim;
}
}  // namespace

TEST(mogac_zero_velocity_keeps_labels)
{
    auto sim = MakeMogac(Vec2::Zero(), 1.0);
    sim->setMaxIterations(10);
    CHECK(sim->init());
    const ImageI labels0 = sim->getBand().labels();
    while (sim->step()) {
    }
    int changed = 0;
    for (size_t n = 0; n < labels0.size(); n++) changed += (sim->getBand().labels()[n] != labels0[n]) ? 1 : 0;
    CHECK(changed == 0);
}

TEST(mogac_translation_moves_centroids)
{
    auto sim = MakeMogac(Vec2(4.0f, 0.0f), 1.0);
    CHECK(sim->init());
    const LabelStats before = Stats(sim->getBand().labels());
    int steps = 0;
    while (sim->step() && steps < 1000) steps++;
    CHECK_NEAR(sim->getSimulationTime(), 1.0, 1E-9);
    const LabelStats after = Stats(sim->getBand().labels());
    for (int l = 1; l <= 2; l++) {
        CHECK_NEAR(after.cx[l] - before.cx[l], 4.0, 0.5);
        // Against the exact half disk, not the initial pixel count: the diameter falls on a pixel
        // column, so the initial counts are uneven (207 vs 230) by one column.
        CHECK_NEAR(after.count[l] / (0.5 * std::numbers::pi * kRadius * kRadius), 1.0, 0.05);
    }
}

TEST(mogac_curves_labeled)
{
    auto sim = MakeMogac(Vec2::Zero(), 1.0);
    CHECK(sim->init());
    const Manifold2D &m = sim->getManifold();
    CHECK(!m.vertexIndexes.empty());
    CHECK(m.curveLabels.size() == m.vertexIndexes.size());
    for (int l : m.curveLabels) CHECK(l == 1 || l == 2);
    CHECK(m.springlCount() == 0);
    CHECK(sim->isLabeled());
}
