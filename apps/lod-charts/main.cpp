// lod-charts: builds the meshlet hierarchy of the LOD Spring Level Sets sphere (BuildMeshletHierarchy, a unit
// icosphere) and saves, per level, its isometric UV charts:
//   level_<k>_initial.png  the atlas UVs the isometric UVs start from, and
//   level_<k>_final.png    the final isometric UVs, both drawn with NanoVG (offscreen): triangles in gray,
//                          each region's atlas chart box in green (the box it is fitted into), a line
//                          between every pair of corresponding corners, a circle at every corner;
//   report.txt             the UV checks of every level, initial and final (flips, overlaps, angles, areas, corner gaps).
// Needs an OpenGL 3.3 context (a hidden window).
#include <common/GL.hpp>
#include <common/Offscreen.hpp>
#include <lod/MeshletHierarchy.hpp>
#include <lod/MeshletUvChecks.hpp>
#include <mesh/Icosphere.hpp>

#include "nanovg.hpp"
#include "nanovg_gl.hpp"

#include <argparse/argparse.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <tuple>

using namespace springls;
namespace fs = std::filesystem;

namespace
{
// Draws the triangles of `level` at the per-corner UVs `uv` (3 per triangle) into the bound framebuffer (size x
// size pixels), fitted to their UV bounds and the atlas chart boxes.
void DrawCharts(nvg::Context &vg, int size, const MeshletLevel &level, const std::vector<Vec2d> &uv, const std::vector<uint32_t> &overlapping)
{
    Vec2d lo = Vec2d::Constant(std::numeric_limits<double>::infinity()), hi = -lo;
    double edgeSum = 0.0;
    size_t edges = 0;
    for (size_t f = 0; f < level.triangles.size(); f++) {
        for (int c = 0; c < 3; c++) {
            lo = lo.cwiseMin(uv[3 * f + c]);
            hi = hi.cwiseMax(uv[3 * f + c]);
            if (level.atlasTexcoords.size() == uv.size()) {
                lo = lo.cwiseMin(level.atlasTexcoords[3 * f + c]);
                hi = hi.cwiseMax(level.atlasTexcoords[3 * f + c]);
            }
            edgeSum += (uv[3 * f + (c + 1) % 3] - uv[3 * f + c]).norm();
            edges++;
        }
    }
    const float margin = 24.0f;
    const double extent = std::max(hi.x() - lo.x(), hi.y() - lo.y());
    const double scale = extent > 0.0 ? (size - 2.0 * margin) / extent : 1.0;
    const auto toPixel = [&](const Vec2d &uv) {
        return Eigen::Vector2f(static_cast<float>(margin + (uv.x() - lo.x()) * scale), static_cast<float>(size - margin - (uv.y() - lo.y()) * scale));
    };
    const float meanEdgePixels = edges ? static_cast<float>(edgeSum / edges * scale) : 10.0f;

    vg.beginFrame(static_cast<float>(size), static_cast<float>(size), 1.0f);
    vg.beginPath();
    vg.rect(0.0f, 0.0f, static_cast<float>(size), static_cast<float>(size));
    vg.fillColor(nvg::rgbaf(1.0f, 1.0f, 1.0f, 1.0f));
    vg.fill();
    // Triangles in gray, outlined.
    const nvg::Color fill = nvg::rgbaf(0.78f, 0.78f, 0.78f, 1.0f);
    const float outlineWidth = std::clamp(0.06f * meanEdgePixels, 0.4f, 1.5f);
    for (size_t f = 0; f < level.triangles.size(); f++) {
        const Eigen::Vector2f a = toPixel(uv[3 * f]), b = toPixel(uv[3 * f + 1]), c = toPixel(uv[3 * f + 2]);
        vg.beginPath();
        vg.moveTo(a.x(), a.y());
        vg.lineTo(b.x(), b.y());
        vg.lineTo(c.x(), c.y());
        vg.closePath();
        vg.fillColor(fill);
        vg.fill();
        vg.strokeColor(nvg::rgbaf(0.25f, 0.25f, 0.25f, 1.0f));
        vg.strokeWidth(outlineWidth);
        vg.stroke();
    }
    // Each region's atlas chart box (the box its relaxed chart is fitted into), in green.
    if (level.atlasTexcoords.size() == uv.size()) {
        std::map<int, std::pair<Vec2d, Vec2d>> boxes;
        for (size_t f = 0; f < level.triangles.size(); f++) {
            auto [it, inserted] = boxes.try_emplace(level.faceRegion[f], Vec2d::Constant(std::numeric_limits<double>::infinity()),
                                                    Vec2d::Constant(-std::numeric_limits<double>::infinity()));
            for (int c = 0; c < 3; c++) {
                it->second.first = it->second.first.cwiseMin(level.atlasTexcoords[3 * f + c]);
                it->second.second = it->second.second.cwiseMax(level.atlasTexcoords[3 * f + c]);
            }
        }
        vg.beginPath();
        for (const auto &[region, box] : boxes) {
            const Eigen::Vector2f a = toPixel(box.first), b = toPixel(box.second);
            vg.rect(a.x(), b.y(), b.x() - a.x(), a.y() - b.y());
        }
        vg.strokeColor(nvg::rgbaf(0.1f, 0.6f, 0.2f, 1.0f));
        vg.strokeWidth(1.5f);
        vg.stroke();
    }
    // A line between every pair of corresponding corners.
    vg.beginPath();
    for (const MeshletCornerPair &p : MeshletCornerPairs(level)) {
        const Eigen::Vector2f a = toPixel(uv[3 * p.faceA + p.cornerA]), b = toPixel(uv[3 * p.faceB + p.cornerB]);
        vg.moveTo(a.x(), a.y());
        vg.lineTo(b.x(), b.y());
    }
    vg.strokeColor(nvg::rgbaf(0.85f, 0.1f, 0.15f, 1.0f));
    vg.strokeWidth(std::clamp(0.08f * meanEdgePixels, 0.6f, 2.0f));
    vg.stroke();
    // A circle at every corner.
    const float radius = std::clamp(0.08f * meanEdgePixels, 0.6f, 4.0f);
    vg.beginPath();
    for (size_t f = 0; f < level.triangles.size(); f++) {
        for (int c = 0; c < 3; c++) {
            const Eigen::Vector2f p = toPixel(uv[3 * f + c]);
            vg.circle(p.x(), p.y(), radius);
        }
    }
    vg.fillColor(nvg::rgbaf(0.1f, 0.2f, 0.55f, 0.85f));
    vg.fill();
    // A heavy red outline around every triangle that overlaps another by more than one texel.
    vg.beginPath();
    for (uint32_t f : overlapping) {
        const Eigen::Vector2f a = toPixel(uv[3 * f]), b = toPixel(uv[3 * f + 1]), c = toPixel(uv[3 * f + 2]);
        vg.moveTo(a.x(), a.y());
        vg.lineTo(b.x(), b.y());
        vg.lineTo(c.x(), c.y());
        vg.closePath();
    }
    vg.strokeColor(nvg::rgbaf(0.9f, 0.0f, 0.0f, 1.0f));
    vg.strokeWidth(4.0f);
    vg.stroke();
    vg.endFrame();
}

std::string Describe(const MeshletUvChecks &c)
{
    std::ostringstream s;
    s << c.triangles << " triangles: " << c.flips << " flipped, " << c.overlaps << " overlapping pairs over one texel at " << kOverlapCheckTextureSize
      << "^2 (" << c.crossRegionOverlaps << " across regions, deepest overlap " << std::setprecision(3) << c.maxOverlapTexels << " texels), "
      << "worst angle error " << c.maxAngleError << " rad, area ratio " << c.minAreaRatio << ".." << c.maxAreaRatio
      << " of the region median, corner gap mean " << c.meanCornerGap << " max " << c.maxCornerGap << " UV units";
    return s.str();
}
}  // namespace

int main(int argc, char *argv[])
{
    argparse::ArgumentParser p("lod-charts");
    p.add_description("Saves the isometric UV charts of every level of the LOD springls meshlet hierarchy (PNG images and a checks report).");
    p.add_argument("--out").required().help("output directory");
    p.add_argument("--sphere-levels").scan<'i', int>().default_value(5).help("icosphere subdivisions of the sphere");
    p.add_argument("--max-depth").scan<'i', int>().default_value(MeshletHierarchyParams().maxDepth).help("clustering passes above the base");
    p.add_argument("--uv-relax-steps").scan<'i', int>().default_value(MeshletHierarchyParams().uvRelaxSteps).help("UV rigid-body relax steps");
    p.add_argument("--size").scan<'i', int>().default_value(2048).help("image size in pixels");
    try {
        p.parse_args(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n' << p;
        return 2;
    }
    const fs::path out = p.get<std::string>("--out");
    const int size = p.get<int>("--size");
    fs::create_directories(out);

    MeshletHierarchyParams params;
    params.maxDepth = p.get<int>("--max-depth");
    params.uvRelaxSteps = p.get<int>("--uv-relax-steps");
    const std::vector<MeshletLevel> levels =
        BuildMeshletHierarchy(MakeIcosphere(p.get<int>("--sphere-levels")), params, [](const Vec3 &v) { return Vec3(v.normalized()); });
    if (levels.empty()) {
        std::cerr << "error: the meshlet hierarchy could not be built\n";
        return 1;
    }

    if (!glfwInit()) {
        std::cerr << "error: could not initialize GLFW\n";
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_STENCIL_BITS, 8);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow *window = glfwCreateWindow(64, 64, "lod-charts", nullptr, nullptr);
    if (!window) {
        std::cerr << "error: could not create an OpenGL 3.3 context\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    int status = 0;
    if (!toybox::LoadGL()) {
        std::cerr << "error: could not load OpenGL\n";
        status = 1;
    }
    std::ofstream report(out / "report.txt");
    report << "LOD springls meshlet hierarchy: icosphere " << p.get<int>("--sphere-levels") << ", max depth " << params.maxDepth << ", UV relax steps "
           << params.uvRelaxSteps << '\n';
    {
        springls_gui::OffscreenTarget target;
        std::unique_ptr<nvg::Context> vg(status == 0 ? nvg::createGL(static_cast<int>(nvg::CreateFlags::Antialias | nvg::CreateFlags::StencilStrokes))
                                                     : nullptr);
        if (status == 0 && (!vg || !target.init(size, size))) {
            std::cerr << "error: could not set up offscreen NanoVG rendering\n";
            status = 1;
        }
        for (size_t k = 0; status == 0 && k < levels.size(); k++) {
            const MeshletLevel &level = levels[k];
            const std::string name = "level_" + std::to_string(k);
            const MeshletUvChecks checks = CheckMeshletUvs(level);
            MeshletLevel initial = level;
            initial.texcoords = level.atlasTexcoords;
            const MeshletUvChecks initialChecks = CheckMeshletUvs(initial);
            for (std::ostream *s : {static_cast<std::ostream *>(&report), static_cast<std::ostream *>(&std::cout)}) {
                *s << "level " << k << " initial (atlas): " << Describe(initialChecks) << '\n';
                *s << "level " << k << " final:           " << Describe(checks) << std::endl;
            }
            const std::tuple<const char *, const std::vector<Vec2d> *, const std::vector<uint32_t> *> images[] = {
                {"_initial.png", &level.atlasTexcoords, &initialChecks.overlappingTriangles}, {"_final.png", &level.texcoords, &checks.overlappingTriangles}};
            for (const auto &[suffix, uv, overlapping] : images) {
                if (uv->size() != 3 * level.triangles.size()) continue;
                target.bind();
                glViewport(0, 0, size, size);
                DrawCharts(*vg, size, level, *uv, *overlapping);
                if (!target.writePng(out / (name + suffix))) status = 1;
            }
        }
        if (vg) nvg::deleteGL(vg);
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    if (status != 0) std::cerr << "error: could not write every file to " << out.string() << '\n';
    return status;
}
