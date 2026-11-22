#pragma once
// Adaptive Loop subdivision of a closed triangle mesh [C. Loop, "Smooth Subdivision Surfaces Based on
// Triangles", MSc thesis, University of Utah, 1987]: marked faces split 1:4; a face with two or three
// split edges is split too, and a face with one split edge is bisected, so the result has no hanging
// vertexes. New edge vertexes follow Loop's edge rule; an old vertex moves by Loop's vertex rule only
// when every face around it is split (vertexes on the border of the refined region stay put).
#include <core/Common3D.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace springls
{
struct LoopMesh {
    Points3 vertices;
    std::vector<std::array<uint32_t, 3>> triangles;
};
// One level. refineFace: one flag per triangle. Old vertexes keep their indexes; new ones follow.
LoopMesh LoopSubdivide(const LoopMesh &mesh, const std::vector<uint8_t> &refineFace);
}  // namespace springls
