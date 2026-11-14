#pragma once
#include <springls/Common3D.hpp>
#include <springls/Image.hpp>

#include <cassert>
#include <vector>

namespace springls
{
// 3D grid of values; voxel (i, j, k) is stored x fastest. at() clamps to the border and
// interpolate() is trilinear with border clamping.
template <class T>
class Volume
{
public:
    Volume() = default;
    Volume(int width, int height, int depth, const T &value = ZeroValue<T>()) { resize(width, height, depth, value); }

    // Resizes and fills with `value`. Keeps the contents when the size does not change.
    void resize(int width, int height, int depth, const T &value = ZeroValue<T>())
    {
        if (width == mW && height == mH && depth == mD) return;
        mW = width;
        mH = height;
        mD = depth;
        mData.assign(static_cast<size_t>(width) * height * depth, value);
    }
    void fill(const T &value) { std::fill(mData.begin(), mData.end(), value); }
    void clear()
    {
        mW = mH = mD = 0;
        mData.clear();
    }

    int width() const { return mW; }
    int height() const { return mH; }
    int depth() const { return mD; }
    Vec3i dimensions() const { return Vec3i(mW, mH, mD); }
    size_t size() const { return mData.size(); }
    bool empty() const { return mData.empty(); }
    bool contains(int i, int j, int k) const { return i >= 0 && j >= 0 && k >= 0 && i < mW && j < mH && k < mD; }
    size_t indexOf(int i, int j, int k) const { return (static_cast<size_t>(k) * mH + static_cast<size_t>(j)) * mW + static_cast<size_t>(i); }

    T &operator()(int i, int j, int k)
    {
        assert(contains(i, j, k));
        return mData[indexOf(i, j, k)];
    }
    const T &operator()(int i, int j, int k) const
    {
        assert(contains(i, j, k));
        return mData[indexOf(i, j, k)];
    }
    T &operator()(const Vec3i &p) { return (*this)(p.x(), p.y(), p.z()); }
    const T &operator()(const Vec3i &p) const { return (*this)(p.x(), p.y(), p.z()); }
    T &operator[](size_t n) { return mData[n]; }
    const T &operator[](size_t n) const { return mData[n]; }

    // Clamped access.
    const T &at(int i, int j, int k) const { return mData[indexOf(std::clamp(i, 0, mW - 1), std::clamp(j, 0, mH - 1), std::clamp(k, 0, mD - 1))]; }

    T interpolate(float x, float y, float z) const
    {
        const int i = static_cast<int>(std::floor(x));
        const int j = static_cast<int>(std::floor(y));
        const int k = static_cast<int>(std::floor(z));
        const float dx = x - static_cast<float>(i), dy = y - static_cast<float>(j), dz = z - static_cast<float>(k);
        const auto lerpX = [&](int jj, int kk) -> T { return at(i, jj, kk) * (1.0f - dx) + at(i + 1, jj, kk) * dx; };
        const T c0 = lerpX(j, k) * (1.0f - dy) + lerpX(j + 1, k) * dy;
        const T c1 = lerpX(j, k + 1) * (1.0f - dy) + lerpX(j + 1, k + 1) * dy;
        const T result = c0 * (1.0f - dz) + c1 * dz;
        return result;
    }
    T interpolate(const Vec3 &p) const { return interpolate(p.x(), p.y(), p.z()); }

    T *data() { return mData.data(); }
    const T *data() const { return mData.data(); }

private:
    int mW = 0, mH = 0, mD = 0;
    std::vector<T> mData;

public:
    // cereal serialization (checkpoints).
    template <class Archive>
    void serialize(Archive &ar)
    {
        ar(mW, mH, mD, mData);
    }
};

using VolumeF = Volume<float>;
using VolumeV3 = Volume<Vec3>;
}  // namespace springls
