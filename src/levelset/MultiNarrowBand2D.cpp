#include <levelset/MultiNarrowBand2D.hpp>

#include <core/Common.hpp>

#include <algorithm>
#include <cmath>

namespace springls
{
namespace
{
constexpr int kShiftX[4] = {-1, 1, 0, 0};
constexpr int kShiftY[4] = {0, 0, -1, 1};
constexpr float kTieEpsilon = std::numeric_limits<float>::epsilon();
}  // namespace

MogacResult MogacUpdate(int currentLabel, const MultiNarrowBand2D::Candidates &labels, const MultiNarrowBand2D::Speeds &updatedValues)
{
    constexpr float kInvalid = 1E30f;
    float min1 = kInvalid, min2 = kInvalid;
    int label1 = MultiNarrowBand2D::kNoLabel, label2 = MultiNarrowBand2D::kNoLabel;
    for (int c = 0; c < MultiNarrowBand2D::kCandidates; c++) {
        const int label = labels[c];
        if (label == MultiNarrowBand2D::kNoLabel) continue;
        const float u = updatedValues[c];
        if (u < min1) {
            min2 = min1;
            label2 = label1;
            min1 = u;
            label1 = label;
        } else if (u < min2) {
            min2 = u;
            label2 = label;
        }
    }
    MogacResult result;
    if (label2 == MultiNarrowBand2D::kNoLabel) {
        result.label = label1;
        result.distance = std::abs(min1);
        return result;
    }
    // Equal values leave the interface on the node: keep the current label if it is one of the two,
    // else the smaller label (deterministic, unlike the random choice of the original).
    if (std::abs(min1 - min2) < kTieEpsilon) {
        result.label = (label1 == currentLabel || label2 == currentLabel) ? currentLabel : std::min(label1, label2);
    } else {
        result.label = label1;
    }
    result.distance = std::abs(0.5f * (min1 - min2));
    return result;
}

void MultiNarrowBand2D::initialize(const ImageI &labels, const ImageF &distance)
{
    mLabels = labels;
    mDistance.clear();
    mDistance.resize(labels.width(), labels.height());
    const float limit = kMaxDistance + 0.5f;
    for (size_t n = 0; n < distance.size(); n++) mDistance[n] = std::min(std::abs(distance[n]), limit);
    mPrevLabels = mLabels;
    mPrevDistance = mDistance;
    rebuild();
}

float MultiNarrowBand2D::signedValue(const Vec2 &p, int label) const
{
    const float x = std::clamp(p.x(), 0.0f, static_cast<float>(mLabels.width() - 1));
    const float y = std::clamp(p.y(), 0.0f, static_cast<float>(mLabels.height() - 1));
    const int i = std::min(static_cast<int>(x), std::max(0, mLabels.width() - 2));
    const int j = std::min(static_cast<int>(y), std::max(0, mLabels.height() - 2));
    const float fx = x - i, fy = y - j;
    return (1.0f - fy) * ((1.0f - fx) * signedValue(i, j, label) + fx * signedValue(i + 1, j, label)) +
           fy * ((1.0f - fx) * signedValue(i, j + 1, label) + fx * signedValue(i + 1, j + 1, label));
}

float MultiNarrowBand2D::previousUpwindAdvection(int i, int j, int label, const Vec2 &g) const
{
    const float v11 = previousSignedValue(i, j, label);
    float advection = 0.0f;
    if (g.x() > 0) advection = g.x() * (v11 - previousSignedValue(i - 1, j, label));
    else if (g.x() < 0) advection = g.x() * (previousSignedValue(i + 1, j, label) - v11);
    if (g.y() > 0) advection += g.y() * (v11 - previousSignedValue(i, j - 1, label));
    else if (g.y() < 0) advection += g.y() * (previousSignedValue(i, j + 1, label) - v11);
    return advection;
}

float MultiNarrowBand2D::previousCurvatureTerm(int i, int j, int label, float weight) const
{
    constexpr float kMaxCurvatureForce = 10.0f;
    if (weight == 0.0f) return 0.0f;
    const auto s = [&](int a, int b) { return previousSignedValue(a, b, label); };
    const float v11 = s(i, j), v01 = s(i - 1, j), v21 = s(i + 1, j), v10 = s(i, j - 1), v12 = s(i, j + 1);
    const float v00 = s(i - 1, j - 1), v20 = s(i + 1, j - 1), v22 = s(i + 1, j + 1), v02 = s(i - 1, j + 1);
    const float DxCtr = 0.5f * (v21 - v01), DyCtr = 0.5f * (v12 - v10);
    const float DxxCtr = v21 - v11 - v11 + v01, DyyCtr = v12 - v11 - v11 + v10;
    const float DxyCtr = (v22 - v02 - v20 + v00) * 0.25f;
    const float numer = 0.5f * (DyCtr * DyCtr * DxxCtr - 2 * DxCtr * DyCtr * DxyCtr + DxCtr * DxCtr * DyyCtr);
    const float denom = DxCtr * DxCtr + DyCtr * DyCtr;
    const float kappa = (std::abs(denom) > 1E-5f) ? weight * numer / denom : weight * numer * signOf(denom) * 1E5f;
    return std::clamp(kappa, -kMaxCurvatureForce, kMaxCurvatureForce);
}

void MultiNarrowBand2D::rebuild()
{
    mActive.clear();
    mInBand.assign(mLabels.size(), 0);
    for (int j = 0; j < mLabels.height(); j++) {
        for (int i = 0; i < mLabels.width(); i++) {
            if (mPrevDistance(i, j) <= kMaxDistance) {
                mActive.emplace_back(i, j);
                mInBand[mLabels.indexOf(i, j)] = 1;
            }
        }
    }
    refreshCandidates();
}

void MultiNarrowBand2D::refreshCandidates()
{
    const int N = static_cast<int>(mActive.size());
    mCandidates.resize(N);
    SPRINGLS_OMP(omp parallel for if (N >= kParallelMinLightItems))
    for (int n = 0; n < N; n++) {
        const Vec2i p = mActive[n];
        Candidates &c = mCandidates[n];
        c.fill(kNoLabel);
        int count = 0;
        const auto add = [&](int label) {
            for (int k = 0; k < count; k++) {
                if (c[k] == label) return;
            }
            c[count++] = label;
        };
        add(mPrevLabels(p));
        for (int k = 0; k < 4; k++) add(mPrevLabels.at(p.x() + kShiftX[k], p.y() + kShiftY[k]));
    }
    mDelta.assign(mActive.size(), Speeds{});
}

void MultiNarrowBand2D::update(float timeStep)
{
    const int AS = static_cast<int>(mActive.size());
    // Zero layer: the two-smallest rule on the previous values.
    SPRINGLS_OMP(omp parallel for if (AS >= kParallelMinLightItems))
    for (int n = 0; n < AS; n++) {
        const Vec2i p = mActive[n];
        if (mPrevDistance(p) > 0.5f) continue;
        Speeds updated;
        for (int c = 0; c < kCandidates; c++) {
            const int label = mCandidates[n][c];
            updated[c] = (label == kNoLabel) ? 0.0f : previousSignedValue(p.x(), p.y(), label) + timeStep * std::clamp(mDelta[n][c], -1.0f, 1.0f);
        }
        const MogacResult r = MogacUpdate(mPrevLabels(p), mCandidates[n], updated);
        mLabels(p) = r.label;
        mDistance(p) = r.distance;
    }
    // Plug: a pixel none of whose 8 neighbors shares its label is not a region; it joins the most
    // common neighboring label (ties to the smaller label).
    std::vector<int> plugged(AS, kNoLabel);
    SPRINGLS_OMP(omp parallel for if (AS >= kParallelMinLightItems))
    for (int n = 0; n < AS; n++) {
        const Vec2i p = mActive[n];
        const int label = mLabels(p);
        std::array<int, 8> around;
        int m = 0;
        bool alone = true;
        for (int dj = -1; dj <= 1; dj++) {
            for (int di = -1; di <= 1; di++) {
                if (di == 0 && dj == 0) continue;
                const int qi = p.x() + di, qj = p.y() + dj;
                if (!mLabels.contains(qi, qj)) continue;
                const int q = mLabels(qi, qj);
                if (q == label) alone = false;
                around[m++] = q;
            }
        }
        if (!alone || m == 0) continue;
        std::sort(around.begin(), around.begin() + m);
        int bestLabel = around[0], bestCount = 0;
        for (int a = 0; a < m;) {
            int b = a;
            while (b < m && around[b] == around[a]) b++;
            if (b - a > bestCount) {
                bestCount = b - a;
                bestLabel = around[a];
            }
            a = b;
        }
        plugged[n] = bestLabel;
    }
    for (int n = 0; n < AS; n++) {
        if (plugged[n] == kNoLabel) continue;
        mLabels(mActive[n]) = plugged[n];
    }
    // The zero layer is what lies on an interface: a pixel with a 4-neighbor of another label.
    // Any other pixel (one a plug or a moving front left behind) is rebuilt from its neighbors.
    std::vector<uint8_t> zeroLayer(AS, 0);
    for (int n = 0; n < AS; n++) {
        const Vec2i p = mActive[n];
        bool interface = false;
        for (int k = 0; k < 4; k++) {
            const int qi = p.x() + kShiftX[k], qj = p.y() + kShiftY[k];
            interface |= mLabels.contains(qi, qj) && mLabels(qi, qj) != mLabels(p);
        }
        if (interface && mDistance(p) <= 0.5f) {
            zeroLayer[n] = 1;
        } else if (mDistance(p) <= 0.5f) {
            mDistance(p) = kMaxDistance + 0.5f;
        }
    }
    // Distance layers, one layer at a time from the layer inside it. Each layer reads the current
    // values and writes into a separate buffer, so the result does not depend on loop order.
    std::vector<float> layer(AS);
    for (int band = 1; band <= kLayers; band++) {
        SPRINGLS_OMP(omp parallel for if (AS >= kParallelMinLightItems))
        for (int n = 0; n < AS; n++) {
            const Vec2i p = mActive[n];
            float d = mDistance(p);
            if (!zeroLayer[n] && d > band - 0.5f) {
                const int label = mLabels(p);
                float best = 1E30f;
                for (int k = 0; k < 4; k++) best = std::min(best, std::abs(signedValue(p.x() + kShiftX[k], p.y() + kShiftY[k], label) - 1.0f));
                d = best;
            }
            layer[n] = d;
        }
        for (int n = 0; n < AS; n++) mDistance(mActive[n]) = layer[n];
    }
    for (int n = 0; n < AS; n++) {
        mPrevLabels(mActive[n]) = mLabels(mActive[n]);
        mPrevDistance(mActive[n]) = mDistance(mActive[n]);
    }
    deleteElements();
    addElements();
    refreshCandidates();
}

int MultiNarrowBand2D::correctLabels(const std::function<bool(int, int, int &, float &)> &measure)
{
    const int AS = static_cast<int>(mActive.size());
    // Measured into a buffer first, so the result does not depend on loop order.
    std::vector<int> labels(AS, kNoLabel);
    std::vector<float> distances(AS, 0.0f);
    SPRINGLS_OMP(omp parallel for if (AS >= kParallelMinHeavyItems))
    for (int n = 0; n < AS; n++) {
        const Vec2i p = mActive[n];
        int label;
        float d;
        if (!measure(p.x(), p.y(), label, d) || label == mLabels(p)) continue;
        labels[n] = label;
        distances[n] = std::min(std::abs(d), kMaxDistance);
    }
    int corrected = 0;
    for (int n = 0; n < AS; n++) {
        if (labels[n] == kNoLabel) continue;
        const Vec2i p = mActive[n];
        mLabels(p) = mPrevLabels(p) = labels[n];
        mDistance(p) = mPrevDistance(p) = distances[n];
        corrected++;
    }
    if (corrected > 0) refreshCandidates();
    return corrected;
}

void MultiNarrowBand2D::deleteElements()
{
    size_t out = 0;
    for (size_t n = 0; n < mActive.size(); n++) {
        const Vec2i p = mActive[n];
        if (mPrevDistance(p) <= kMaxDistance) {
            mActive[out++] = p;
        } else {
            mDistance(p) = mPrevDistance(p) = kMaxDistance + 0.5f;
            mInBand[mLabels.indexOf(p.x(), p.y())] = 0;
        }
    }
    mActive.resize(out);
}

void MultiNarrowBand2D::addElements()
{
    // Pixels outside the band next to the inner band join it at the band edge; they keep their label.
    const size_t count = mActive.size();
    for (int offset = 0; offset < 4; offset++) {
        for (size_t n = 0; n < count; n++) {
            const Vec2i p = mActive[n];
            const Vec2i q(p.x() + kShiftX[offset], p.y() + kShiftY[offset]);
            if (!mLabels.contains(q.x(), q.y())) continue;
            const size_t qi = mLabels.indexOf(q.x(), q.y());
            if (mInBand[qi]) continue;
            if (mDistance(p) <= kMaxDistance - 1.0f && mDistance(q) >= kMaxDistance) {
                mDistance(q) = mPrevDistance(q) = kMaxDistance;
                mInBand[qi] = 1;
                mActive.push_back(q);
            }
        }
    }
}
}  // namespace springls
