#pragma once
#include <core/Common.hpp>

#include <cassert>
#include <type_traits>
#include <vector>

namespace springls
{
// Zero of a scalar or an Eigen fixed-size type (Eigen's default constructor leaves values uninitialized).
template <class T>
T ZeroValue()
{
    if constexpr (std::is_arithmetic_v<T>) {
        return T(0);
    } else {
        return T::Zero();
    }
}

// Row-major 2D grid of values; pixel (i, j) is column i, row j. at() clamps to the border and
// interpolate() is bilinear with border clamping.
template <class T>
class Image
{
public:
    Image() = default;
    Image(int width, int height, const T &value = ZeroValue<T>()) { resize(width, height, value); }

    // Resizes and fills with `value`. Keeps the contents when the size does not change.
    void resize(int width, int height, const T &value = ZeroValue<T>())
    {
        if (width == mWidth && height == mHeight) return;
        mWidth = width;
        mHeight = height;
        mData.assign(static_cast<size_t>(width) * static_cast<size_t>(height), value);
    }
    void fill(const T &value) { std::fill(mData.begin(), mData.end(), value); }
    void clear()
    {
        mWidth = mHeight = 0;
        mData.clear();
    }

    int width() const { return mWidth; }
    int height() const { return mHeight; }
    size_t size() const { return mData.size(); }
    bool empty() const { return mData.empty(); }
    bool contains(int i, int j) const { return i >= 0 && j >= 0 && i < mWidth && j < mHeight; }
    size_t indexOf(int i, int j) const { return static_cast<size_t>(j) * static_cast<size_t>(mWidth) + static_cast<size_t>(i); }

    T &operator()(int i, int j)
    {
        assert(contains(i, j));
        return mData[indexOf(i, j)];
    }
    const T &operator()(int i, int j) const
    {
        assert(contains(i, j));
        return mData[indexOf(i, j)];
    }
    T &operator()(const Vec2i &p) { return (*this)(p.x(), p.y()); }
    const T &operator()(const Vec2i &p) const { return (*this)(p.x(), p.y()); }
    T &operator[](size_t n) { return mData[n]; }
    const T &operator[](size_t n) const { return mData[n]; }

    // Clamped access.
    T &at(int i, int j) { return mData[indexOf(std::clamp(i, 0, mWidth - 1), std::clamp(j, 0, mHeight - 1))]; }
    const T &at(int i, int j) const { return mData[indexOf(std::clamp(i, 0, mWidth - 1), std::clamp(j, 0, mHeight - 1))]; }

    T interpolate(float x, float y) const
    {
        const int i = static_cast<int>(std::floor(x));
        const int j = static_cast<int>(std::floor(y));
        const float dx = x - static_cast<float>(i);
        const float dy = y - static_cast<float>(j);
        const T v00 = at(i, j), v10 = at(i + 1, j), v01 = at(i, j + 1), v11 = at(i + 1, j + 1);
        const T result = (v00 * (1.0f - dx) + v10 * dx) * (1.0f - dy) + (v01 * (1.0f - dx) + v11 * dx) * dy;
        return result;
    }
    T interpolate(const Vec2 &p) const { return interpolate(p.x(), p.y()); }

    T *data() { return mData.data(); }
    const T *data() const { return mData.data(); }
    auto begin() { return mData.begin(); }
    auto end() { return mData.end(); }
    auto begin() const { return mData.begin(); }
    auto end() const { return mData.end(); }

    // cereal serialization (checkpoints).
    template <class Archive>
    void serialize(Archive &ar)
    {
        ar(mWidth, mHeight, mData);
    }

private:
    int mWidth = 0;
    int mHeight = 0;
    std::vector<T> mData;
};

using ImageF = Image<float>;
using ImageV2 = Image<Vec2>;
using ImageI = Image<int>;
}  // namespace springls
