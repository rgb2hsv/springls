// ExtractMultiLabelContour: one set of closed curves per object label.
#include "Check.hpp"

#include <levelset/Contour2D.hpp>

#include <algorithm>
#include <cmath>

using namespace springls;

namespace
{
constexpr int kSize = 32;

// Unsigned distance good enough for contouring: 0.5 next to another label, 1.5 elsewhere.
ImageF BoundaryDistance(const ImageI &labels)
{
    ImageF d(labels.width(), labels.height(), 1.5f);
    for (int j = 0; j < labels.height(); j++) {
        for (int i = 0; i < labels.width(); i++) {
            const int l = labels(i, j);
            if (labels.at(i - 1, j) != l || labels.at(i + 1, j) != l || labels.at(i, j - 1) != l || labels.at(i, j + 1) != l) d(i, j) = 0.5f;
        }
    }
    return d;
}

struct Extracted {
    Points2 vertices;
    Curves curves;
    std::vector<int> curveLabels;
};

Extracted Extract(const ImageI &labels, const std::vector<int> &objectLabels)
{
    const ImageF d = BoundaryDistance(labels);
    Extracted e;
    ExtractMultiLabelContour([&](int i, int j, int l) { return (labels(i, j) == l) ? -d(i, j) : d(i, j); }, labels.width(), labels.height(), objectLabels,
                             e.vertices, e.curves, e.curveLabels);
    return e;
}

bool Closed(const Curve &c) { return c.size() > 2 && c.front() == c.back(); }

// Proper intersection of segments ab and cd (shared endpoints do not count).
bool Cross(const Vec2 &a, const Vec2 &b, const Vec2 &c, const Vec2 &d)
{
    const auto side = [](const Vec2 &p, const Vec2 &q, const Vec2 &r) { return cross2(q - p, r - p); };
    const float d1 = side(a, b, c), d2 = side(a, b, d), d3 = side(c, d, a), d4 = side(c, d, b);
    constexpr float kEps = 1E-5f;
    return ((d1 > kEps && d2 < -kEps) || (d1 < -kEps && d2 > kEps)) && ((d3 > kEps && d4 < -kEps) || (d3 < -kEps && d4 > kEps));
}
}  // namespace

TEST(multi_contour_two_labels_shared_edge)
{
    ImageI labels(kSize, kSize, 0);
    for (int j = 6; j < 26; j++) {
        for (int i = 6; i < 26; i++) labels(i, j) = (i < 16) ? 1 : 2;
    }
    const Extracted e = Extract(labels, ObjectLabels(labels));
    CHECK(e.curves.size() == 2);
    CHECK(e.curveLabels.size() == e.curves.size());
    for (const Curve &c : e.curves) CHECK(Closed(c));
    // Interior shared-edge vertexes (x = 15.5) of label 1 have a coincident, distinct label-2 copy.
    int shared = 0, matched = 0;
    for (size_t a = 0; a < e.curves.size(); a++) {
        if (e.curveLabels[a] != 1) continue;
        for (uint32_t v : e.curves[a]) {
            const Vec2 p = e.vertices[v];
            if (std::abs(p.x() - 15.5f) > 1E-4f || p.y() < 7.0f || p.y() > 24.0f) continue;
            shared++;
            for (size_t b = 0; b < e.curves.size(); b++) {
                if (e.curveLabels[b] != 2) continue;
                for (uint32_t w : e.curves[b]) {
                    if (w != v && (e.vertices[w] - p).norm() < 1E-5f) {
                        matched++;
                        goto next;
                    }
                }
            }
        next:;
        }
    }
    CHECK(shared > 0);
    CHECK(matched == shared);
}

TEST(multi_contour_t_junction_closed)
{
    ImageI labels(kSize, kSize, 0);
    for (int j = 6; j < 26; j++) {
        for (int i = 6; i < 26; i++) labels(i, j) = (i < 16) ? 1 : (j >= 16) ? 2 : 3;
    }
    const Extracted e = Extract(labels, ObjectLabels(labels));
    CHECK(e.curves.size() == 3);
    CHECK((e.curveLabels == std::vector<int>{1, 2, 3}));
    for (const Curve &c : e.curves) CHECK(Closed(c));
    int crossings = 0;
    for (size_t a = 0; a < e.curves.size(); a++) {
        for (size_t b = a + 1; b < e.curves.size(); b++) {
            const Curve &ca = e.curves[a], &cb = e.curves[b];
            for (size_t s = 0; s + 1 < ca.size(); s++) {
                for (size_t t = 0; t + 1 < cb.size(); t++) {
                    if (Cross(e.vertices[ca[s]], e.vertices[ca[s + 1]], e.vertices[cb[t]], e.vertices[cb[t + 1]])) crossings++;
                }
            }
        }
    }
    CHECK(crossings == 0);
}

TEST(multi_contour_skips_absent_label)
{
    ImageI labels(kSize, kSize, 0);
    for (int j = 8; j < 20; j++) {
        for (int i = 8; i < 20; i++) labels(i, j) = 1;
    }
    const Extracted e = Extract(labels, {1, 5});
    CHECK(e.curves.size() == 1);
    CHECK((e.curveLabels == std::vector<int>{1}));
}

TEST(multi_contour_matches_single_label)
{
    ImageI labels(kSize, kSize, 0);
    ImageF phi(kSize, kSize);
    for (int j = 0; j < kSize; j++) {
        for (int i = 0; i < kSize; i++) {
            phi(i, j) = std::hypot(i - 15.3f, j - 16.1f) - 9.0f;
            labels(i, j) = (phi(i, j) < 0.0f) ? 1 : 0;
        }
    }
    Points2 vertices;
    Curves curves;
    std::vector<int> curveLabels;
    ExtractMultiLabelContour([&](int i, int j, int l) { return (l == 1) ? phi(i, j) : -phi(i, j); }, kSize, kSize, {1}, vertices, curves, curveLabels);
    Points2 expectedVertices;
    Curves expectedCurves;
    ExtractIsoContour(phi, expectedVertices, expectedCurves);
    CHECK(vertices.size() == expectedVertices.size());
    float maxError = 0.0f;
    for (size_t v = 0; v < std::min(vertices.size(), expectedVertices.size()); v++) maxError = std::max(maxError, (vertices[v] - expectedVertices[v]).norm());
    CHECK(maxError < 1E-5f);
    CHECK(curves == expectedCurves);
}

TEST(multi_contour_curves_meet_at_junctions)
{
    // Four quadrant regions meeting at c (between nodes), exact distances to the region boundaries:
    // every label's curve passes through c, rather than cutting its corner of the junction cell.
    const Vec2 c(15.3f, 16.6f);
    const auto regionOf = [&](int i, int j) { return (i < c.x() ? 1 : 2) + (j < c.y() ? 0 : 2); };
    const auto signedValue = [&](int i, int j, int l) {
        const float d = std::min(std::abs(i - c.x()), std::abs(j - c.y()));
        return regionOf(i, j) == l ? -d : d;
    };
    Points2 vertices;
    Curves curves;
    std::vector<int> curveLabels;
    ExtractMultiLabelContour(signedValue, kSize, kSize, {1, 2, 3, 4}, vertices, curves, curveLabels);
    CHECK(curves.size() == 4);
    for (const Curve &curve : curves) {
        float nearest = 1e9f;
        for (uint32_t v : curve) nearest = std::min(nearest, distance(vertices[v], c));
        CHECK(nearest < 1e-3f);
    }
    // A T-junction with the background: an object above y = 16.6 split at x = 15.3.
    const auto tLabel = [&](int i, int j) { return j < c.y() ? 0 : (i < c.x() ? 1 : 2); };
    const auto tValue = [&](int i, int j, int l) {
        const float d = (j < c.y()) ? c.y() - j : std::min(j - c.y(), std::abs(i - c.x()));
        return tLabel(i, j) == l ? -d : d;
    };
    ExtractMultiLabelContour(tValue, kSize, kSize, {1, 2}, vertices, curves, curveLabels);
    for (const Curve &curve : curves) {
        float nearest = 1e9f;
        for (uint32_t v : curve) nearest = std::min(nearest, distance(vertices[v], c));
        CHECK(nearest < 1e-3f);
    }
}
