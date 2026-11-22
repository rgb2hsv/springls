#include <io/Ply.hpp>

#include <happly.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <unordered_map>

namespace springls
{
namespace
{
bool Fail(std::string &error, const std::string &message)
{
    error = message;
    return false;
}

std::vector<std::string> Words(const std::string &line)
{
    std::istringstream in(line);
    return {std::istream_iterator<std::string>(in), std::istream_iterator<std::string>()};
}

// happly trusts its input: a header without end_header, or ASCII data with too few rows or values, makes it read
// out of bounds or loop forever. This checks those before happly parses `bytes` (binary data running short is
// caught after parsing, from the stream state).
bool CheckPly(const std::string &bytes, std::string &error)
{
    struct Element {
        size_t count = 0;
        std::vector<bool> isList;
    };
    std::vector<Element> elements;
    bool ascii = false;
    size_t pos = 0;
    bool hadCR = false;
    const auto next = [&](std::string &line) {
        if (pos >= bytes.size()) return false;
        size_t end = bytes.find('\n', pos);
        if (end == std::string::npos) end = bytes.size();
        line = bytes.substr(pos, end - pos);
        hadCR = !line.empty() && line.back() == '\r';
        if (hadCR) line.pop_back();
        pos = end + 1;
        return true;
    };
    std::string line;
    if (!next(line) || line != "ply") return Fail(error, "not a PLY file (missing 'ply' magic)");
    bool ended = false;
    while (!ended && next(line)) {
        const std::vector<std::string> words = Words(line);
        if (words.empty()) continue;
        if (words[0] == "end_header") ended = true;
        else if (words[0] == "format" && words.size() >= 2) ascii = words[1] == "ascii";
        else if (words[0] == "element" && words.size() == 3) elements.push_back({std::strtoull(words[2].c_str(), nullptr, 10), {}});
        else if (words[0] == "property" && !elements.empty()) elements.back().isList.push_back(words.size() >= 2 && words[1] == "list");
    }
    if (!ended) return Fail(error, "PLY header ended without end_header");
    if (!ascii) return true;
    for (const Element &element : elements) {
        if (element.isList.empty()) {
            // happly reads one (ignored) line per row of an element without properties.
            for (size_t row = 0; row < element.count && next(line); row++) {
            }
            continue;
        }
        for (size_t row = 0; row < element.count; row++) {
            do {
                if (!next(line)) return Fail(error, "unexpected end of PLY data");
            } while (line.empty() && !hadCR);
            // Values as happly splits them: on spaces only.
            std::vector<std::string> words;
            for (size_t begin = 0; begin < line.size();) {
                size_t end = line.find(' ', begin);
                if (end == std::string::npos) end = line.size();
                if (end > begin) words.push_back(line.substr(begin, end - begin));
                begin = end + 1;
            }
            size_t word = 0;
            for (bool isList : element.isList) {
                if (word >= words.size()) return Fail(error, "PLY data row has too few values: " + line);
                if (isList) {
                    char *end = nullptr;
                    const unsigned long long n = std::strtoull(words[word].c_str(), &end, 10);
                    if (*end != '\0' || n > words.size()) return Fail(error, "malformed PLY list in row: " + line);
                    word += static_cast<size_t>(n);
                }
                word++;
            }
            if (word > words.size()) return Fail(error, "PLY data row has too few values: " + line);
        }
    }
    return true;
}

// Copies `property` into `out` when its values are of type T (one of happly's eight PLY types).
template <class T>
bool CopyIf(happly::Property *property, PlyType type, PlyProperty &out)
{
    if (auto *scalar = dynamic_cast<happly::TypedProperty<T> *>(property)) {
        out.isList = false;
        out.values.assign(scalar->data.begin(), scalar->data.end());
    } else if (auto *list = dynamic_cast<happly::TypedListProperty<T> *>(property)) {
        if (list->flattenedData.size() > UINT32_MAX) return false;
        out.isList = true;
        out.values.assign(list->flattenedData.begin(), list->flattenedData.end());
        out.listStart.resize(list->flattenedIndexStart.size());
        std::transform(list->flattenedIndexStart.begin(), list->flattenedIndexStart.end(), out.listStart.begin(),
                       [](size_t i) { return static_cast<uint32_t>(i); });
    } else {
        return false;
    }
    out.type = type;
    return true;
}

bool Copy(happly::Property *property, PlyProperty &out)
{
    return CopyIf<int8_t>(property, PlyType::Int8, out) || CopyIf<uint8_t>(property, PlyType::UInt8, out) || CopyIf<int16_t>(property, PlyType::Int16, out) ||
           CopyIf<uint16_t>(property, PlyType::UInt16, out) || CopyIf<int32_t>(property, PlyType::Int32, out) ||
           CopyIf<uint32_t>(property, PlyType::UInt32, out) || CopyIf<float>(property, PlyType::Float32, out) ||
           CopyIf<double>(property, PlyType::Float64, out);
}

// `property` as a happly property of value type T.
template <class T>
std::unique_ptr<happly::Property> Make(const PlyProperty &property)
{
    std::vector<T> values(property.values.size());
    std::transform(property.values.begin(), property.values.end(), values.begin(), [](double v) { return static_cast<T>(v); });
    if (!property.isList) return std::make_unique<happly::TypedProperty<T>>(property.name, values);
    auto list = std::make_unique<happly::TypedListProperty<T>>(property.name, 1);
    list->flattenedData = std::move(values);
    list->flattenedIndexStart.assign(property.listStart.begin(), property.listStart.end());
    return list;
}

std::unique_ptr<happly::Property> Make(const PlyProperty &property)
{
    switch (property.type) {
        case PlyType::Int8: return Make<int8_t>(property);
        case PlyType::UInt8: return Make<uint8_t>(property);
        case PlyType::Int16: return Make<int16_t>(property);
        case PlyType::UInt16: return Make<uint16_t>(property);
        case PlyType::Int32: return Make<int32_t>(property);
        case PlyType::UInt32: return Make<uint32_t>(property);
        case PlyType::Float32: return Make<float>(property);
        case PlyType::Float64: return Make<double>(property);
    }
    return Make<float>(property);
}

uint64_t EdgeKey(uint32_t a, uint32_t b)
{
    if (a > b) std::swap(a, b);
    return (static_cast<uint64_t>(a) << 32) | b;
}

// Number of polygons using each undirected edge.
std::unordered_map<uint64_t, int> EdgeUses(const PolygonMesh &mesh)
{
    std::unordered_map<uint64_t, int> uses;
    for (size_t f = 0; f < mesh.faceCount(); ++f) {
        const uint32_t begin = mesh.faceStart[f], end = mesh.faceStart[f + 1];
        for (uint32_t i = begin; i < end; ++i) {
            const uint32_t a = mesh.faceVertexes[i];
            const uint32_t b = mesh.faceVertexes[i + 1 < end ? i + 1 : begin];
            if (a != b) ++uses[EdgeKey(a, b)];
        }
    }
    return uses;
}
}  // namespace

bool LoadPly(const std::filesystem::path &file, PolygonMesh &mesh, std::string &error)
{
    error.clear();
    mesh = PolygonMesh();
    std::ifstream in(file, std::ios::binary);
    if (!in) return Fail(error, "cannot open " + file.string());
    PlyTable table;
    if (!ReadPly(in, table, error)) return false;

    const PlyElement *vertex = table.find("vertex");
    if (!vertex) return Fail(error, "PLY file has no vertex element");
    const PlyProperty *x = vertex->find("x"), *y = vertex->find("y"), *z = vertex->find("z");
    if (!x || !y || !z || x->isList || y->isList || z->isList) return Fail(error, "PLY vertex element lacks x, y, z");
    PolygonMesh result;
    result.positions.reserve(vertex->count);
    for (size_t i = 0; i < vertex->count; i++)
        result.positions.emplace_back(static_cast<float>(x->values[i]), static_cast<float>(y->values[i]), static_cast<float>(z->values[i]));

    if (const PlyElement *face = table.find("face")) {
        const PlyProperty *indices = face->find("vertex_indices");
        if (!indices) indices = face->find("vertex_index");
        if (!indices || !indices->isList) return Fail(error, "PLY face element lacks a vertex_indices list");
        for (size_t f = 0; f < face->count; f++) {
            if (indices->listSize(f) < 3) continue;
            for (uint32_t i = indices->listStart[f]; i < indices->listStart[f + 1]; i++) {
                const double index = indices->values[i];
                if (index < 0 || index >= static_cast<double>(result.positions.size())) return Fail(error, "PLY face references a missing vertex");
                result.faceVertexes.push_back(static_cast<uint32_t>(index));
            }
            result.faceStart.push_back(static_cast<uint32_t>(result.faceVertexes.size()));
        }
    }
    mesh = std::move(result);
    return true;
}

std::vector<std::array<uint32_t, 3>> Triangulate(const PolygonMesh &mesh)
{
    std::vector<std::array<uint32_t, 3>> triangles;
    for (size_t f = 0; f < mesh.faceCount(); ++f) {
        const uint32_t begin = mesh.faceStart[f], end = mesh.faceStart[f + 1];
        for (uint32_t i = begin + 1; i + 1 < end; ++i) triangles.push_back({mesh.faceVertexes[begin], mesh.faceVertexes[i], mesh.faceVertexes[i + 1]});
    }
    return triangles;
}

float MeanEdgeLength(const PolygonMesh &mesh)
{
    const auto uses = EdgeUses(mesh);
    if (uses.empty()) return 0.0f;
    double sum = 0.0;
    for (const auto &edge : uses) {
        const uint32_t a = static_cast<uint32_t>(edge.first >> 32), b = static_cast<uint32_t>(edge.first);
        sum += (mesh.positions[a] - mesh.positions[b]).norm();
    }
    return static_cast<float>(sum / static_cast<double>(uses.size()));
}

MeshTopology AnalyzeTopology(const PolygonMesh &mesh)
{
    MeshTopology topology;
    for (const auto &edge : EdgeUses(mesh)) {
        if (edge.second == 1) ++topology.boundaryEdges;
        else if (edge.second > 2) ++topology.nonManifoldEdges;
    }
    return topology;
}

const PlyProperty *PlyElement::find(const std::string &property) const
{
    for (const PlyProperty &p : properties) {
        if (p.name == property) return &p;
    }
    return nullptr;
}

const PlyElement *PlyTable::find(const std::string &element) const
{
    for (const PlyElement &e : elements) {
        if (e.name == element) return &e;
    }
    return nullptr;
}

bool WritePly(std::ostream &out, const PlyTable &table, std::string &error)
{
    happly::PLYData ply;
    ply.comments = table.comments;
    for (const std::string &comment : table.comments) {
        if (comment.find_first_of("\r\n") != std::string::npos) return Fail(error, "a PLY comment cannot hold a line break");
    }
    for (const PlyElement &element : table.elements) {
        ply.addElement(element.name, element.count);
        happly::Element &e = ply.getElement(element.name);
        for (const PlyProperty &property : element.properties) {
            const bool sized = property.isList ? property.listStart.size() == element.count + 1 && property.listStart.back() == property.values.size()
                                               : property.values.size() == element.count;
            if (!sized) return Fail(error, "PLY property " + element.name + "." + property.name + " does not have one value per row");
            if (property.isList) {
                for (size_t row = 0; row < element.count; row++) {
                    if (property.listSize(row) > 255)
                        return Fail(error, "PLY list " + element.name + "." + property.name + " has a row of more than 255 items");
                }
            }
            e.properties.push_back(Make(property));
        }
    }
    std::ostringstream bytes;
    try {
        ply.write(bytes, happly::DataFormat::Binary);
    } catch (const std::exception &e) {
        return Fail(error, std::string("could not write the PLY data: ") + e.what());
    }
    // happly always signs the header with a comment; leave it out.
    std::string data = std::move(bytes).str();
    const std::string signature = "comment Written with hapPLY (https://github.com/nmwsharp/happly)\n";
    const size_t at = data.find(signature);
    if (at != std::string::npos && at < data.find("end_header\n")) data.erase(at, signature.size());
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    if (!out) return Fail(error, "could not write the PLY data");
    return true;
}

bool ReadPly(std::istream &in, PlyTable &table, std::string &error)
{
    std::string bytes(std::istreambuf_iterator<char>(in), {});
    if (!CheckPly(bytes, error)) return false;
    std::istringstream stream(std::move(bytes));
    PlyTable result;
    try {
        happly::PLYData ply(stream);
        if (stream.fail()) return Fail(error, "unexpected end of PLY data");
        for (std::string comment : ply.comments) {
            if (!comment.empty() && comment.back() == '\r') comment.pop_back();
            result.comments.push_back(comment);
        }
        for (const std::string &name : ply.getElementNames()) {
            happly::Element &element = ply.getElement(name);
            PlyElement &out = result.elements.emplace_back();
            out.name = element.name;
            out.count = element.count;
            for (const auto &property : element.properties) {
                PlyProperty &p = out.properties.emplace_back();
                p.name = property->name;
                if (!Copy(property.get(), p)) return Fail(error, "PLY property " + element.name + "." + p.name + " could not be read");
            }
        }
    } catch (const std::exception &e) {
        return Fail(error, e.what());
    }
    table = std::move(result);
    return true;
}
}  // namespace springls
