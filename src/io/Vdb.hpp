#pragma once
#include <core/Volume.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace springls
{
struct VdbCoord {
    int x = 0, y = 0, z = 0;
};

// Grids collected for one OpenVDB file, written with tinyvdb: standard 5-4-3 trees (8^3 leaf nodes) of float or
// int32 values with a uniform voxel size, ZIP-compressed. Values round-trip bit for bit.
class VdbWriter
{
public:
    VdbWriter();
    ~VdbWriter();
    VdbWriter(const VdbWriter &) = delete;
    VdbWriter &operator=(const VdbWriter &) = delete;

    // A grid whose active voxels are coords[i] = values[i] (no duplicates); background fills the rest. levelSet
    // marks it class "level set" for VDB tools.
    bool addFloatGrid(const std::string &name, double voxelSize, const std::vector<VdbCoord> &coords, const std::vector<float> &values, float background,
                      bool levelSet, std::string &error);
    bool addInt32Grid(const std::string &name, double voxelSize, const std::vector<VdbCoord> &coords, const std::vector<int32_t> &values, std::string &error);
    bool hasGrid(const std::string &name) const;
    // Metadata on a grid added before.
    bool setMetadata(const std::string &grid, const std::string &key, const std::string &text, std::string &error);
    bool setMetadata(const std::string &grid, const std::string &key, int32_t value, std::string &error);
    bool setMetadata(const std::string &grid, const std::string &key, float value, std::string &error);
    bool setMetadata(const std::string &grid, const std::string &key, const std::array<int32_t, 3> &value, std::string &error);
    bool setMetadata(const std::string &grid, const std::string &key, const std::array<float, 3> &value, std::string &error);

    bool write(std::ostream &out, std::string &error) const;

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

// One grid of a file read by VdbReader; valid while the reader is.
class VdbGrid
{
public:
    ~VdbGrid();
    bool isFloat() const;
    bool isInt32() const;
    // The value of active voxel p; false when p is inactive or the grid is of the other type.
    bool get(const VdbCoord &p, float &value) const;
    bool get(const VdbCoord &p, int32_t &value) const;
    // Every active voxel (in no particular order); nothing for a grid of the other type.
    void forEachFloat(const std::function<void(const VdbCoord &, float)> &f) const;
    void forEachInt32(const std::function<void(const VdbCoord &, int32_t)> &f) const;
    // Metadata by key; false when it is missing or of another type.
    bool metadata(const std::string &key, std::string &value) const;
    bool metadata(const std::string &key, int32_t &value) const;
    bool metadata(const std::string &key, float &value) const;
    bool metadata(const std::string &key, std::array<int32_t, 3> &value) const;
    bool metadata(const std::string &key, std::array<float, 3> &value) const;
    // Every string-valued metadata entry.
    std::vector<std::pair<std::string, std::string>> stringMetadata() const;

private:
    friend class VdbReader;
    VdbGrid();
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

// An OpenVDB file read with tinyvdb (float and int32 grids with 8^3 leaf nodes; other grids are left out).
class VdbReader
{
public:
    VdbReader();
    ~VdbReader();
    VdbReader(const VdbReader &) = delete;
    VdbReader &operator=(const VdbReader &) = delete;

    bool read(std::istream &in, std::string &error);
    const VdbGrid *find(const std::string &name) const;

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

// A dense volume as a float grid of voxel size 1 (voxel (i, j, k) at index (i, j, k)): every voxel whose value
// differs (bitwise) from `background` is active, and the dimensions and background are metadata, so the
// volume round-trips exactly (an empty volume too). levelSet marks the grid class "level set".
bool WriteVolumeVdb(VdbWriter &writer, const std::string &name, const VolumeF &volume, float background, bool levelSet, std::string &error);
bool ReadVolumeVdb(const VdbReader &reader, const std::string &name, VolumeF &volume, std::string &error);
}  // namespace springls
