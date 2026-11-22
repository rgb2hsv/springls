#include <io/Vdb.hpp>
#include <levelset/NarrowBand3D.hpp>

namespace springls
{
namespace
{
constexpr int kShift[6][3] = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
}

void NarrowBand3D::initialize(const VolumeF &signedDistance)
{
    mPhi.clear();
    mPhi.resize(signedDistance.width(), signedDistance.height(), signedDistance.depth());
    const float limit = kLayers + 1.0f;
    const int N = static_cast<int>(signedDistance.size());
    SPRINGLS_OMP(omp parallel for schedule(static) if (N >= kParallelMinLightItems))
    for (int n = 0; n < N; n++) {
        mPhi[n] = std::clamp(signedDistance[n], -limit, limit);
    }
    mSwap = mPhi;
    rebuild();
}

void NarrowBand3D::rebuild()
{
    mActive.clear();
    mInBand.assign(mPhi.size(), 0);
    for (int k = 0; k < mSwap.depth(); k++) {
        for (int j = 0; j < mSwap.height(); j++) {
            for (int i = 0; i < mSwap.width(); i++) {
                if (std::abs(mSwap(i, j, k)) <= kMaxDistance) {
                    mActive.emplace_back(i, j, k);
                    mInBand[mSwap.indexOf(i, j, k)] = 1;
                }
            }
        }
    }
    mDelta.assign(mActive.size(), 0.0f);
}

void NarrowBand3D::updateDistanceLayer(int i, int j, int k, int band)
{
    if (std::abs(mSwap(i, j, k)) <= 0.5f) return;  // zero layer
    float v = mPhi(i, j, k);
    const float old = v;
    float n[6];
    for (int s = 0; s < 6; s++) n[s] = mPhi.at(i + kShift[s][0], j + kShift[s][1], k + kShift[s][2]);
    if (v < -band + 0.5f) {
        v = -(kMaxDistance + 0.5f);
        for (float u : n) v = (u > 1) ? v : std::max(u, v);
        v -= 1.0f;
    } else if (v > band - 0.5f) {
        v = kMaxDistance + 0.5f;
        for (float u : n) v = (u < -1) ? v : std::min(u, v);
        v += 1.0f;
    }
    mPhi(i, j, k) = (old * v > 0) ? v : old;
}

void NarrowBand3D::plug(int i, int j, int k)
{
    // A zero-layer voxel whose six neighbors all lie on its side of the interface is not on the
    // interface: push it out to the band edge.
    const float v = mPhi(i, j, k);
    const float sgn = signOf(v);
    if (!(sgn * v > 0 && sgn * v < 0.5f)) return;
    for (int s = 0; s < 6; s++) {
        if (!(sgn * mPhi.at(i + kShift[s][0], j + kShift[s][1], k + kShift[s][2]) > 0)) return;
    }
    mPhi(i, j, k) = sgn * kMaxDistance;
}

void NarrowBand3D::update(float timeStep)
{
    const int AS = static_cast<int>(mActive.size());
    SPRINGLS_OMP(omp parallel for if (AS >= kParallelMinLightItems))
    for (int n = 0; n < AS; n++) {
        const Vec3i p = mActive[n];
        const float old = mSwap(p);
        if (std::abs(old) > 0.5f) continue;
        mPhi(p) = old + timeStep * std::clamp(mDelta[n], -1.0f, 1.0f);
    }
    for (int band = 1; band <= kLayers; band++) {
        SPRINGLS_OMP(omp parallel for if (AS >= kParallelMinLightItems))
        for (int n = 0; n < AS; n++) {
            updateDistanceLayer(mActive[n].x(), mActive[n].y(), mActive[n].z(), band);
        }
    }
    SPRINGLS_OMP(omp parallel for if (AS >= kParallelMinLightItems))
    for (int n = 0; n < AS; n++) {
        plug(mActive[n].x(), mActive[n].y(), mActive[n].z());
    }
    SPRINGLS_OMP(omp parallel for if (AS >= kParallelMinLightItems))
    for (int n = 0; n < AS; n++) {
        mSwap(mActive[n]) = mPhi(mActive[n]);
    }
    deleteElements();
    addElements();
    mDelta.assign(mActive.size(), 0.0f);
}

int NarrowBand3D::correctSigns(const std::function<bool(int, int, int, float &)> &measure)
{
    const int AS = static_cast<int>(mActive.size());
    const float limit = kLayers + 1.0f;
    // Measured into a buffer first, so the result does not depend on loop order.
    std::vector<float> corrected(AS, 0.0f);
    std::vector<uint8_t> change(AS, 0);
    SPRINGLS_OMP(omp parallel for if (AS >= kParallelMinHeavyItems))
    for (int n = 0; n < AS; n++) {
        const Vec3i p = mActive[n];
        float d;
        if (!measure(p.x(), p.y(), p.z(), d) || (d < 0.0f) == (mPhi(p.x(), p.y(), p.z()) < 0.0f)) continue;
        corrected[n] = std::clamp(d, -limit, limit);
        change[n] = 1;
    }
    int count = 0;
    for (int n = 0; n < AS; n++) {
        if (!change[n]) continue;
        const Vec3i p = mActive[n];
        mPhi(p.x(), p.y(), p.z()) = mSwap(p.x(), p.y(), p.z()) = corrected[n];
        count++;
    }
    return count;
}

void NarrowBand3D::deleteElements()
{
    size_t out = 0;
    for (size_t n = 0; n < mActive.size(); n++) {
        const Vec3i p = mActive[n];
        const float val = mSwap(p);
        if (std::abs(val) <= kMaxDistance) {
            mActive[out++] = p;
        } else {
            const float clamped = signOf(val) * (kMaxDistance + 0.5f);
            mPhi(p) = clamped;
            mSwap(p) = clamped;
            mInBand[mPhi.indexOf(p.x(), p.y(), p.z())] = 0;
        }
    }
    mActive.resize(out);
}

void NarrowBand3D::addElements()
{
    // Voxels outside the band next to the inner band (|phi| <= kMaxDistance - 1) join it at the
    // band edge, keeping their side of the interface.
    const size_t count = mActive.size();
    for (int offset = 0; offset < 6; offset++) {
        for (size_t n = 0; n < count; n++) {
            const Vec3i p = mActive[n];
            const Vec3i q(p.x() + kShift[offset][0], p.y() + kShift[offset][1], p.z() + kShift[offset][2]);
            if (!mPhi.contains(q.x(), q.y(), q.z())) continue;
            const size_t qi = mPhi.indexOf(q.x(), q.y(), q.z());
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
bool NarrowBand3D::writeVdb(VdbWriter &writer, const std::string &prefix, std::string &error) const
{
    return WriteVolumeVdb(writer, prefix + "phi", mPhi, kMaxDistance, true, error) &&
           WriteVolumeVdb(writer, prefix + "previous", mSwap, kMaxDistance, true, error);
}

bool NarrowBand3D::readVdb(const VdbReader &reader, const std::string &prefix, std::string &error)
{
    VolumeF phi, previous;
    if (!ReadVolumeVdb(reader, prefix + "phi", phi, error) || !ReadVolumeVdb(reader, prefix + "previous", previous, error)) return false;
    mPhi = std::move(phi);
    mSwap = std::move(previous);
    return true;
}
}  // namespace springls
