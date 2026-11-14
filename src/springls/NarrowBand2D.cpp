#include <springls/NarrowBand2D.hpp>

namespace springls
{
void NarrowBand2D::initialize(const ImageF &signedDistance)
{
    const int W = signedDistance.width(), H = signedDistance.height();
    mPhi.clear();
    mPhi.resize(W, H);
    const float limit = kLayers + 1.0f;
    const int N = static_cast<int>(signedDistance.size());
    SPRINGLS_OMP(omp parallel for schedule(static) if (N >= kParallelMinLightItems))
    for (int n = 0; n < N; n++) {
        mPhi[n] = std::clamp(signedDistance[n], -limit, limit);
    }
    mSwap = mPhi;
    rebuild();
}

void NarrowBand2D::rebuild()
{
    mActive.clear();
    mInBand.assign(mPhi.size(), 0);
    for (int j = 0; j < mSwap.height(); j++) {
        for (int i = 0; i < mSwap.width(); i++) {
            if (std::abs(mSwap(i, j)) <= kMaxDistance) {
                mActive.emplace_back(i, j);
                mInBand[mSwap.indexOf(i, j)] = 1;
            }
        }
    }
    mDelta.assign(mActive.size(), 0.0f);
}

void NarrowBand2D::updateDistanceLayer(int i, int j, int band)
{
    if (std::abs(mSwap(i, j)) <= 0.5f) return;  // zero layer
    float v11 = mPhi(i, j);
    const float old = v11;
    const float v01 = mPhi.at(i - 1, j);
    const float v12 = mPhi.at(i, j + 1);
    const float v10 = mPhi.at(i, j - 1);
    const float v21 = mPhi.at(i + 1, j);
    if (v11 < -band + 0.5f) {
        v11 = -1E30f;
        v11 = (v01 > 1) ? v11 : std::max(v01, v11);
        v11 = (v12 > 1) ? v11 : std::max(v12, v11);
        v11 = (v10 > 1) ? v11 : std::max(v10, v11);
        v11 = (v21 > 1) ? v11 : std::max(v21, v11);
        v11 -= 1.0f;
    } else if (v11 > band - 0.5f) {
        v11 = 1E30f;
        v11 = (v01 < -1) ? v11 : std::min(v01, v11);
        v11 = (v12 < -1) ? v11 : std::min(v12, v11);
        v11 = (v10 < -1) ? v11 : std::min(v10, v11);
        v11 = (v21 < -1) ? v11 : std::min(v21, v11);
        v11 += 1.0f;
    }
    mPhi(i, j) = (old * v11 > 0) ? v11 : old;
}

void NarrowBand2D::plug(int i, int j)
{
    // A zero-layer pixel whose four neighbors all lie on its side of the interface is not on the
    // interface: push it out to the band edge.
    float v11 = mPhi(i, j);
    const float sgn = signOf(v11);
    v11 *= sgn;
    const float v01 = sgn * mPhi.at(i - 1, j);
    const float v12 = sgn * mPhi.at(i, j + 1);
    const float v10 = sgn * mPhi.at(i, j - 1);
    const float v21 = sgn * mPhi.at(i + 1, j);
    if (v11 > 0 && v11 < 0.5f && v01 > 0 && v12 > 0 && v10 > 0 && v21 > 0) {
        mPhi(i, j) = sgn * kMaxDistance;
    }
}

void NarrowBand2D::update(float timeStep)
{
    const int AS = static_cast<int>(mActive.size());
    SPRINGLS_OMP(omp parallel for if (AS >= kParallelMinLightItems))
    for (int n = 0; n < AS; n++) {
        const Vec2i p = mActive[n];
        const float old = mSwap(p);
        if (std::abs(old) > 0.5f) continue;
        mPhi(p) = old + timeStep * std::clamp(mDelta[n], -1.0f, 1.0f);
    }
    for (int band = 1; band <= kLayers; band++) {
        SPRINGLS_OMP(omp parallel for if (AS >= kParallelMinLightItems))
        for (int n = 0; n < AS; n++) {
            updateDistanceLayer(mActive[n].x(), mActive[n].y(), band);
        }
    }
    SPRINGLS_OMP(omp parallel for if (AS >= kParallelMinLightItems))
    for (int n = 0; n < AS; n++) {
        plug(mActive[n].x(), mActive[n].y());
    }
    SPRINGLS_OMP(omp parallel for if (AS >= kParallelMinLightItems))
    for (int n = 0; n < AS; n++) {
        mSwap(mActive[n]) = mPhi(mActive[n]);
    }
    deleteElements();
    addElements();
    mDelta.assign(mActive.size(), 0.0f);
}

void NarrowBand2D::deleteElements()
{
    size_t out = 0;
    for (size_t n = 0; n < mActive.size(); n++) {
        const Vec2i p = mActive[n];
        const float val = mSwap(p);
        if (std::abs(val) <= kMaxDistance) {
            mActive[out++] = p;
        } else {
            const float clamped = signOf(val) * (kMaxDistance + 0.5f);
            mPhi(p) = clamped;
            mSwap(p) = clamped;
            mInBand[mPhi.indexOf(p.x(), p.y())] = 0;
        }
    }
    mActive.resize(out);
}

void NarrowBand2D::addElements()
{
    constexpr int kShiftX[4] = {-1, 1, 0, 0};
    constexpr int kShiftY[4] = {0, 0, -1, 1};
    // Pixels outside the band next to the inner band (|phi| <= kMaxDistance - 1) join it at the
    // band edge, keeping their side of the interface.
    const size_t count = mActive.size();
    for (int offset = 0; offset < 4; offset++) {
        for (size_t n = 0; n < count; n++) {
            const Vec2i p = mActive[n];
            const Vec2i q(p.x() + kShiftX[offset], p.y() + kShiftY[offset]);
            if (!mPhi.contains(q.x(), q.y())) continue;
            const size_t qi = mPhi.indexOf(q.x(), q.y());
            if (mInBand[qi]) continue;
            if (std::abs(mPhi(p)) <= kMaxDistance - 1.0f && std::abs(mPhi(q)) >= kMaxDistance) {
                const float value = signOf(mSwap(q)) * kMaxDistance;
                mSwap(q) = value;
                mPhi(q) = value;
                mInBand[qi] = 1;
                mActive.push_back(q);
            }
        }
    }
}
}  // namespace springls
