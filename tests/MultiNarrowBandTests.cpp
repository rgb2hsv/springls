// MultiNarrowBand2D: the MOGAC label + unsigned distance narrow band.
#include "Check.hpp"

#include <levelset/MultiNarrowBand2D.hpp>

#include <cmath>

using namespace springls;

namespace
{
constexpr int kSize = 32;

// Label 1 left of x = 15.5, label 2 right of it.
MultiNarrowBand2D HalfPlanes()
{
    ImageI labels(kSize, kSize);
    ImageF distance(kSize, kSize);
    for (int j = 0; j < kSize; j++) {
        for (int i = 0; i < kSize; i++) {
            labels(i, j) = (i < 16) ? 1 : 2;
            distance(i, j) = std::abs(i - 15.5f);
        }
    }
    MultiNarrowBand2D band;
    band.initialize(labels, distance);
    return band;
}

void ZeroSpeed(MultiNarrowBand2D &band)
{
    for (auto &speeds : band.delta()) speeds.fill(0.0f);
}

// First column of row j with label 2.
int BoundaryColumn(const MultiNarrowBand2D &band, int j)
{
    for (int i = 0; i < kSize; i++) {
        if (band.labels()(i, j) == 2) return i;
    }
    return kSize;
}
}  // namespace

TEST(multi_band_zero_speed_is_stable)
{
    MultiNarrowBand2D band = HalfPlanes();
    const ImageI labels0 = band.labels();
    const ImageF distance0 = band.distance();
    for (int k = 0; k < 5; k++) {
        ZeroSpeed(band);
        band.update(0.5f);
    }
    int labelChanges = 0;
    float maxDistanceChange = 0.0f;
    for (const Vec2i &p : band.active()) {
        if (band.labels()(p) != labels0(p)) labelChanges++;
        maxDistanceChange = std::max(maxDistanceChange, std::abs(band.distance()(p) - distance0(p)));
    }
    CHECK(labelChanges == 0);
    CHECK(maxDistanceChange <= 1E-4f);
}

TEST(multi_band_uniform_push_moves_boundary)
{
    MultiNarrowBand2D band = HalfPlanes();
    CHECK(BoundaryColumn(band, 16) == 16);
    for (int k = 0; k < 4; k++) {
        // Label 1 recedes, label 2 advances: one cell per unit time.
        for (size_t n = 0; n < band.active().size(); n++) {
            for (int c = 0; c < MultiNarrowBand2D::kCandidates; c++) {
                const int label = band.candidates()[n][c];
                band.delta()[n][c] = (label == 1) ? 1.0f : (label == 2) ? -1.0f : 0.0f;
            }
        }
        band.update(0.5f);
    }
    CHECK(std::abs(BoundaryColumn(band, 16) - 14) <= 1);
    int labeled = 0;
    for (int j = 0; j < kSize; j++) {
        for (int i = 0; i < kSize; i++) labeled += (band.labels()(i, j) == 1 || band.labels()(i, j) == 2) ? 1 : 0;
    }
    CHECK(labeled == kSize * kSize);
}

TEST(multi_band_plugs_single_pixel_hole)
{
    ImageI labels(kSize, kSize, 1);
    ImageF distance(kSize, kSize, 3.0f);
    labels(10, 10) = 2;
    distance(10, 10) = 0.5f;
    distance(9, 10) = distance(11, 10) = distance(10, 9) = distance(10, 11) = 0.5f;
    MultiNarrowBand2D band;
    band.initialize(labels, distance);
    ZeroSpeed(band);
    band.update(0.5f);
    CHECK(band.labels()(10, 10) == 1);
}

TEST(multi_band_signed_value_sign)
{
    const MultiNarrowBand2D band = HalfPlanes();
    int mismatches = 0;
    for (int j = 0; j < kSize; j++) {
        for (int i = 0; i < kSize; i++) {
            for (int label = 1; label <= 2; label++) {
                const bool inside = band.signedValue(i, j, label) < 0.0f;
                if (inside != (band.labels()(i, j) == label)) mismatches++;
            }
        }
    }
    CHECK(mismatches == 0);
    CHECK_NEAR(band.signedValue(Vec2(15.5f, 8.0f), 1), 0.0f, 1E-5f);
}

TEST(multi_band_three_labels_no_vacuum)
{
    ImageI labels(kSize, kSize);
    ImageF distance(kSize, kSize);
    for (int j = 0; j < kSize; j++) {
        for (int i = 0; i < kSize; i++) {
            labels(i, j) = 1 + i / 11;
            distance(i, j) = std::min(std::abs(i - 10.5f), std::abs(i - 21.5f));
        }
    }
    MultiNarrowBand2D band;
    band.initialize(labels, distance);
    for (int k = 0; k < 10; k++) {
        for (size_t n = 0; n < band.active().size(); n++) {
            for (int c = 0; c < MultiNarrowBand2D::kCandidates; c++) band.delta()[n][c] = ((n + c + k) % 2) ? 1.0f : -1.0f;
        }
        band.update(0.5f);
    }
    int bad = 0;
    for (int j = 0; j < kSize; j++) {
        for (int i = 0; i < kSize; i++) bad += (band.labels()(i, j) < 1 || band.labels()(i, j) > 3) ? 1 : 0;
    }
    CHECK(bad == 0);
}

TEST(multi_band_plug_rebuilds_neighbor_distances)
{
    // After the hole at (10, 10) is plugged, no pixel around it lies on an interface any more:
    // none may keep a zero-layer distance.
    ImageI labels(kSize, kSize, 1);
    ImageF distance(kSize, kSize, 3.0f);
    labels(10, 10) = 2;
    distance(10, 10) = 0.5f;
    distance(9, 10) = distance(11, 10) = distance(10, 9) = distance(10, 11) = 0.5f;
    MultiNarrowBand2D band;
    band.initialize(labels, distance);
    ZeroSpeed(band);
    band.update(0.5f);
    CHECK(band.labels()(10, 10) == 1);
    CHECK(band.distance()(10, 10) > 0.5f);
    CHECK(band.distance()(9, 10) > 0.5f);
    CHECK(band.distance()(11, 10) > 0.5f);
    CHECK(band.distance()(10, 9) > 0.5f);
    CHECK(band.distance()(10, 11) > 0.5f);
}
