// VdbWriter / VdbReader: OpenVDB files through tinyvdb.
#include <io/Vdb.hpp>

#include <tinyvdb_io.h>
#include <tinyvdb_sparse_tree.h>

#include <bit>
#include <cstdlib>
#include <cstring>
#include <istream>
#include <iterator>
#include <map>
#include <ostream>
#include <unordered_map>

namespace springls
{
namespace
{
constexpr int kLeafValues = 512;  // 8^3

struct CoordHash {
    size_t operator()(const VdbCoord &c) const
    {
        return (static_cast<size_t>(static_cast<uint32_t>(c.x)) * 73856093u) ^ (static_cast<size_t>(static_cast<uint32_t>(c.y)) * 19349663u) ^
               (static_cast<size_t>(static_cast<uint32_t>(c.z)) * 83492791u);
    }
};
struct CoordEqual {
    bool operator()(const VdbCoord &a, const VdbCoord &b) const { return a.x == b.x && a.y == b.y && a.z == b.z; }
};

// Leaf node containing voxel p (in leaf units), and p's slot in it (OpenVDB order: x major).
VdbCoord LeafOf(const VdbCoord &p) { return {p.x >> 3, p.y >> 3, p.z >> 3}; }
int SlotOf(const VdbCoord &p) { return ((p.x & 7) << 6) | ((p.y & 7) << 3) | (p.z & 7); }

std::string Message(const std::string &what, const tvdb_error_t &err)
{
    std::string s = what;
    if (err.message[0] != '\0') s += std::string(": ") + err.message;
    return s;
}
}  // namespace

// ---------------------------------------------------------------------------------------------
// VdbWriter
// ---------------------------------------------------------------------------------------------

struct VdbWriter::Impl {
    std::vector<tvdb_grid_t> grids;  // built with the sparse-tree builder; destroyed with its allocator
    std::map<std::string, size_t> byName;
    ~Impl()
    {
        for (tvdb_grid_t &g : grids) tvdb_grid_destroy_owned(&g);
    }

    bool build(const std::string &name, tvdb_value_type_t type, double voxelSize, const std::vector<VdbCoord> &coords, const void *values,
               const void *background, std::string &error)
    {
        if (byName.count(name)) {
            error = "duplicate VDB grid " + name;
            return false;
        }
        tvdb_grid_t tmpl;
        std::memset(&tmpl, 0, sizeof(tmpl));
        tmpl.descriptor.grid_type = const_cast<char *>(type == TVDB_VALUE_INT32 ? "Tree_int32_5_4_3" : "Tree_float_5_4_3");
        tmpl.transform.type = TVDB_TRANSFORM_UNIFORM_SCALE;
        for (int a = 0; a < 3; a++) {
            tmpl.transform.scale_values[a] = voxelSize;
            tmpl.transform.voxel_size[a] = voxelSize;
            tmpl.transform.matrix[a][a] = voxelSize;
        }
        tmpl.transform.matrix[3][3] = 1.0;
        tmpl.tree.layout.num_levels = 4;
        const int dims[4] = {0, 5, 4, 3};
        for (int l = 0; l < 4; l++) {
            tmpl.tree.layout.levels[l].log2dim = dims[l];
            tmpl.tree.layout.levels[l].value_type = type;
            tmpl.tree.layout.levels[l].node_type = l == 0 ? TVDB_NODE_ROOT : (l == 3 ? TVDB_NODE_LEAF : TVDB_NODE_INTERNAL);
        }
        static_assert(sizeof(VdbCoord) == sizeof(tvdb_vec3i), "VdbCoord must match tvdb_vec3i");
        tvdb_grid_t out;
        std::memset(&out, 0, sizeof(out));
        if (!tvdb_grid_from_sparse_typed_using_template(&tmpl, reinterpret_cast<const tvdb_vec3i *>(coords.data()), values, coords.size(), type, background,
                                                        name.c_str(), &out)) {
            error = "could not build the VDB grid " + name;
            return false;
        }
        byName[name] = grids.size();
        grids.push_back(out);
        return true;
    }

    tvdb_grid_t *find(const std::string &name, std::string &error)
    {
        const auto it = byName.find(name);
        if (it == byName.end()) {
            error = "no VDB grid " + name;
            return nullptr;
        }
        return &grids[it->second];
    }

    bool set(const std::string &grid, const std::string &key, const tvdb_value_t &value, std::string &error)
    {
        tvdb_grid_t *g = find(grid, error);
        if (!g) return false;
        tvdb_error_t err{};
        if (tvdb_grid_set_metadata(g, key.c_str(), &value, &err) != TVDB_OK) {
            error = Message("could not store " + key, err);
            return false;
        }
        return true;
    }
};

VdbWriter::VdbWriter() : mImpl(std::make_unique<Impl>()) {}
VdbWriter::~VdbWriter() = default;

bool VdbWriter::addFloatGrid(const std::string &name, double voxelSize, const std::vector<VdbCoord> &coords, const std::vector<float> &values, float background,
                             bool levelSet, std::string &error)
{
    if (coords.size() != values.size()) {
        error = "VDB grid " + name + " has a value count unlike its voxel count";
        return false;
    }
    if (!mImpl->build(name, TVDB_VALUE_FLOAT, voxelSize, coords, values.data(), &background, error)) return false;
    return !levelSet || setMetadata(name, "class", std::string("level set"), error);
}

bool VdbWriter::addInt32Grid(const std::string &name, double voxelSize, const std::vector<VdbCoord> &coords, const std::vector<int32_t> &values,
                             std::string &error)
{
    if (coords.size() != values.size()) {
        error = "VDB grid " + name + " has a value count unlike its voxel count";
        return false;
    }
    const int32_t background = 0;
    return mImpl->build(name, TVDB_VALUE_INT32, voxelSize, coords, values.data(), &background, error);
}

bool VdbWriter::hasGrid(const std::string &name) const { return mImpl->byName.count(name) != 0; }

bool VdbWriter::setMetadata(const std::string &grid, const std::string &key, const std::string &text, std::string &error)
{
    tvdb_grid_t *g = mImpl->find(grid, error);
    if (!g) return false;
    tvdb_error_t err{};
    if (tvdb_grid_set_metadata_string(g, key.c_str(), text.c_str(), &err) != TVDB_OK) {
        error = Message("could not store " + key, err);
        return false;
    }
    return true;
}

bool VdbWriter::setMetadata(const std::string &grid, const std::string &key, int32_t value, std::string &error)
{
    tvdb_value_t v{};
    v.type = TVDB_VALUE_INT32;
    v.u.i32 = value;
    return mImpl->set(grid, key, v, error);
}

bool VdbWriter::setMetadata(const std::string &grid, const std::string &key, float value, std::string &error)
{
    tvdb_value_t v{};
    v.type = TVDB_VALUE_FLOAT;
    v.u.f = value;
    return mImpl->set(grid, key, v, error);
}

bool VdbWriter::setMetadata(const std::string &grid, const std::string &key, const std::array<int32_t, 3> &value, std::string &error)
{
    tvdb_value_t v{};
    v.type = TVDB_VALUE_VEC3I;
    for (int a = 0; a < 3; a++) v.u.vec3i[a] = value[a];
    return mImpl->set(grid, key, v, error);
}

bool VdbWriter::setMetadata(const std::string &grid, const std::string &key, const std::array<float, 3> &value, std::string &error)
{
    tvdb_value_t v{};
    v.type = TVDB_VALUE_VEC3F;
    for (int a = 0; a < 3; a++) v.u.vec3f[a] = value[a];
    return mImpl->set(grid, key, v, error);
}

bool VdbWriter::write(std::ostream &out, std::string &error) const
{
    if (mImpl->grids.empty()) {
        error = "no VDB grids to write";
        return false;
    }
    tvdb_file_t file;
    tvdb_file_init_for_write(&file, nullptr);
    file.grids = mImpl->grids.data();
    file.num_grids = mImpl->grids.size();
    uint8_t *data = nullptr;
    size_t size = 0;
    tvdb_error_t err{};
    if (tvdb_write_to_memory(&file, TVDB_COMPRESS_ZIP | TVDB_COMPRESS_ACTIVE_MASK, 0, &data, &size, &err) != TVDB_OK) {
        error = Message("could not write the VDB file", err);
        return false;
    }
    out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(size));
    std::free(data);
    if (!out) {
        error = "could not write the VDB file";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// VdbGrid / VdbReader
// ---------------------------------------------------------------------------------------------

struct VdbGrid::Impl {
    const tvdb_grid_t *grid = nullptr;
    tvdb_value_type_t type = TVDB_VALUE_FLOAT;
    std::unordered_map<VdbCoord, tvdb_leaf_view_t, CoordHash, CoordEqual> leaves;

    static int visit(const tvdb_leaf_view_t *leaf, void *user)
    {
        if (leaf->log2dim == 3 && leaf->data && leaf->value_mask) {
            static_cast<Impl *>(user)->leaves[LeafOf(VdbCoord{leaf->origin[0], leaf->origin[1], leaf->origin[2]})] = *leaf;
        }
        return 0;
    }
    template <class T>
    bool get(tvdb_value_type_t want, const VdbCoord &p, T &value) const
    {
        if (type != want) return false;
        const auto it = leaves.find(LeafOf(p));
        if (it == leaves.end()) return false;
        const int slot = SlotOf(p);
        if (!tvdb_nodemask_is_on(it->second.value_mask, slot)) return false;
        std::memcpy(&value, reinterpret_cast<const uint8_t *>(it->second.data) + static_cast<size_t>(slot) * sizeof(T), sizeof(T));
        return true;
    }
    template <class T, class F>
    void forEach(tvdb_value_type_t want, const F &f) const
    {
        if (type != want) return;
        for (const auto &[leaf, view] : leaves) {
            for (int slot = 0; slot < kLeafValues; slot++) {
                if (!tvdb_nodemask_is_on(view.value_mask, slot)) continue;
                T value;
                std::memcpy(&value, reinterpret_cast<const uint8_t *>(view.data) + static_cast<size_t>(slot) * sizeof(T), sizeof(T));
                f(VdbCoord{view.origin[0] + (slot >> 6), view.origin[1] + ((slot >> 3) & 7), view.origin[2] + (slot & 7)}, value);
            }
        }
    }
    const tvdb_value_t *meta(const std::string &key, tvdb_value_type_t want) const
    {
        const tvdb_meta_entry_t *entry = tvdb_grid_find_metadata(grid, key.c_str());
        return entry && entry->value.type == want ? &entry->value : nullptr;
    }
};

VdbGrid::VdbGrid() : mImpl(std::make_unique<Impl>()) {}
VdbGrid::~VdbGrid() = default;
bool VdbGrid::isFloat() const { return mImpl->type == TVDB_VALUE_FLOAT; }
bool VdbGrid::isInt32() const { return mImpl->type == TVDB_VALUE_INT32; }
bool VdbGrid::get(const VdbCoord &p, float &value) const { return mImpl->get(TVDB_VALUE_FLOAT, p, value); }
bool VdbGrid::get(const VdbCoord &p, int32_t &value) const { return mImpl->get(TVDB_VALUE_INT32, p, value); }
void VdbGrid::forEachFloat(const std::function<void(const VdbCoord &, float)> &f) const { mImpl->forEach<float>(TVDB_VALUE_FLOAT, f); }
void VdbGrid::forEachInt32(const std::function<void(const VdbCoord &, int32_t)> &f) const { mImpl->forEach<int32_t>(TVDB_VALUE_INT32, f); }

bool VdbGrid::metadata(const std::string &key, std::string &value) const
{
    const tvdb_value_t *v = mImpl->meta(key, TVDB_VALUE_STRING);
    if (!v) return false;
    value.assign(v->u.s.str ? v->u.s.str : "", v->u.s.len);
    return true;
}

bool VdbGrid::metadata(const std::string &key, int32_t &value) const
{
    const tvdb_value_t *v = mImpl->meta(key, TVDB_VALUE_INT32);
    if (!v) return false;
    value = v->u.i32;
    return true;
}

bool VdbGrid::metadata(const std::string &key, float &value) const
{
    const tvdb_value_t *v = mImpl->meta(key, TVDB_VALUE_FLOAT);
    if (!v) return false;
    value = v->u.f;
    return true;
}

bool VdbGrid::metadata(const std::string &key, std::array<int32_t, 3> &value) const
{
    const tvdb_value_t *v = mImpl->meta(key, TVDB_VALUE_VEC3I);
    if (!v) return false;
    for (int a = 0; a < 3; a++) value[a] = v->u.vec3i[a];
    return true;
}

bool VdbGrid::metadata(const std::string &key, std::array<float, 3> &value) const
{
    const tvdb_value_t *v = mImpl->meta(key, TVDB_VALUE_VEC3F);
    if (!v) return false;
    for (int a = 0; a < 3; a++) value[a] = v->u.vec3f[a];
    return true;
}

std::vector<std::pair<std::string, std::string>> VdbGrid::stringMetadata() const
{
    std::vector<std::pair<std::string, std::string>> out;
    const tvdb_metadata_t &m = mImpl->grid->metadata;
    for (size_t e = 0; e < m.count; e++) {
        const tvdb_meta_entry_t &entry = m.entries[e];
        if (!entry.name || entry.value.type != TVDB_VALUE_STRING) continue;
        out.emplace_back(entry.name, std::string(entry.value.u.s.str ? entry.value.u.s.str : "", entry.value.u.s.len));
    }
    return out;
}

struct VdbReader::Impl {
    std::vector<uint8_t> bytes;  // the file, kept while tinyvdb may refer to it
    tvdb_file_t file{};
    bool open = false;
    std::map<std::string, std::unique_ptr<VdbGrid>> grids;
    void close()
    {
        grids.clear();
        if (open) tvdb_file_close(&file);
        file = tvdb_file_t{};
        open = false;
    }
    ~Impl() { close(); }
};

VdbReader::VdbReader() : mImpl(std::make_unique<Impl>()) {}
VdbReader::~VdbReader() = default;

bool VdbReader::read(std::istream &in, std::string &error)
{
    mImpl->close();
    mImpl->bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    if (in.bad()) {
        error = "could not read the VDB file";
        return false;
    }
    tvdb_error_t err{};
    if (tvdb_file_open_memory(&mImpl->file, mImpl->bytes.data(), mImpl->bytes.size(), nullptr, &err) != TVDB_OK) {
        error = Message("not a readable VDB file", err);
        return false;
    }
    mImpl->open = true;
    if (tvdb_read_all_grids(&mImpl->file, &err) != TVDB_OK) {
        error = Message("not a readable VDB file", err);
        mImpl->close();
        return false;
    }
    for (size_t g = 0; g < mImpl->file.num_grids; g++) {
        const tvdb_grid_t &grid = mImpl->file.grids[g];
        if (!grid.descriptor.grid_name) continue;
        const tvdb_grid_layout_t &layout = grid.tree.layout;
        if (layout.num_levels < 1) continue;
        const tvdb_node_info_t &leaf = layout.levels[layout.num_levels - 1];
        if (leaf.log2dim != 3 || (leaf.value_type != TVDB_VALUE_FLOAT && leaf.value_type != TVDB_VALUE_INT32)) continue;
        std::unique_ptr<VdbGrid> view(new VdbGrid());
        view->mImpl->grid = &grid;
        view->mImpl->type = leaf.value_type;
        tvdb_grid_visit_leaves(&grid, &VdbGrid::Impl::visit, view->mImpl.get());
        mImpl->grids[grid.descriptor.grid_name] = std::move(view);
    }
    return true;
}

const VdbGrid *VdbReader::find(const std::string &name) const
{
    const auto it = mImpl->grids.find(name);
    return it == mImpl->grids.end() ? nullptr : it->second.get();
}

bool WriteVolumeVdb(VdbWriter &writer, const std::string &name, const VolumeF &volume, float background, bool levelSet, std::string &error)
{
    const Vec3i dims = volume.dimensions();
    std::vector<VdbCoord> coords;
    std::vector<float> values;
    const uint32_t backgroundBits = std::bit_cast<uint32_t>(background);
    const float *data = volume.data();
    size_t index = 0;
    for (int k = 0; k < dims.z(); k++) {
        for (int j = 0; j < dims.y(); j++) {
            for (int i = 0; i < dims.x(); i++, index++) {
                if (std::bit_cast<uint32_t>(data[index]) == backgroundBits) continue;
                coords.push_back(VdbCoord{i, j, k});
                values.push_back(data[index]);
            }
        }
    }
    return writer.addFloatGrid(name, 1.0, coords, values, background, levelSet, error) &&
           writer.setMetadata(name, "springls.dimensions", std::array<int32_t, 3>{dims.x(), dims.y(), dims.z()}, error) &&
           writer.setMetadata(name, "springls.background", background, error);
}

bool ReadVolumeVdb(const VdbReader &reader, const std::string &name, VolumeF &volume, std::string &error)
{
    const VdbGrid *grid = reader.find(name);
    std::array<int32_t, 3> dims{};
    float background = 0.0f;
    if (!grid || !grid->isFloat() || !grid->metadata("springls.dimensions", dims) || !grid->metadata("springls.background", background) || dims[0] < 0 ||
        dims[1] < 0 || dims[2] < 0 || static_cast<double>(dims[0]) * dims[1] * dims[2] > 4e9) {
        error = "the VDB file has no volume " + name + " (or it is damaged)";
        return false;
    }
    VolumeF result;
    if (dims[0] > 0 && dims[1] > 0 && dims[2] > 0) result.resize(dims[0], dims[1], dims[2], background);
    bool inside = true;
    grid->forEachFloat([&](const VdbCoord &p, float v) {
        if (p.x < 0 || p.y < 0 || p.z < 0 || p.x >= dims[0] || p.y >= dims[1] || p.z >= dims[2]) {
            inside = false;
            return;
        }
        result.data()[(static_cast<size_t>(p.z) * dims[1] + p.y) * dims[0] + p.x] = v;
    });
    if (!inside) {
        error = "the VDB volume " + name + " has voxels outside its dimensions";
        return false;
    }
    volume = std::move(result);
    return true;
}
}  // namespace springls
