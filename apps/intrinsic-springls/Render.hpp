#pragma once
// Offline image of an intrinsic-springls state: the input mesh (vertices colored by region, Lambert
// shaded) and the springls drawn as dark lines, seen orthographically from outside the surface.
#include <intrinsic/IntrinsicSpringls.hpp>

#include <filesystem>
#include <string>

namespace springls::intrinsic
{
struct RenderView {
    Vec3d center = Vec3d::Zero();   // looked at
    Vec3d normal = Vec3d(0, 0, 1);  // toward the camera
    Vec3d up = Vec3d(0, 1, 0);      // projected onto the image plane
    double halfWidth = 1.0;         // half the image's width in mesh units
    int size = 512;                 // pixels (square)
};
// Region color (RGB 0-255) of a label: background gray, then a fixed palette.
void LabelColor(int label, double rgb[3]);
bool RenderPng(const std::filesystem::path &file, const IntrinsicSpringls &sim, const RenderView &view, std::string &error);
}  // namespace springls::intrinsic
