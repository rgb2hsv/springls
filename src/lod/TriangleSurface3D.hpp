#pragma once
// A closed triangle mesh with a region label per triangle (mesh tracking's surface; the remesher's input).
#include <core/Common3D.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace springls
{
struct TriangleSurface {
    Points3 vertexes;
    std::vector<std::array<uint32_t, 3>> triangles;
    std::vector<int> faceRegion;  // carried along; new triangles take the region of the triangle they replace
};
}  // namespace springls
