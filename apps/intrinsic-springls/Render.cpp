#include <intrinsic-springls/Render.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include <stb_image_write.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace springls::intrinsic
{
void LabelColor(int label, double rgb[3])
{
    static const double palette[][3] = {{170, 170, 170}, {220, 80, 60}, {60, 120, 220}, {80, 180, 90}, {230, 180, 50}, {150, 90, 200}};
    const int n = static_cast<int>(sizeof(palette) / sizeof(palette[0]));
    const int k = (label <= 0) ? 0 : 1 + (label - 1) % (n - 1);
    for (int c = 0; c < 3; c++) rgb[c] = palette[k][c];
}

bool RenderPng(const std::filesystem::path &file, const IntrinsicSpringls &sim, const RenderView &view, std::string &error)
{
    const IntrinsicTriangulation &tri = sim.triangulation();
    const IntrinsicTriangulation &input = tri.input();
    const int W = view.size, H = view.size;
    const Vec3d n = view.normal.normalized();
    const Vec3d right = view.up.cross(n).normalized(), up = n.cross(right);
    const double scale = 0.5 * W / view.halfWidth;
    // Screen position (pixels) and depth (toward the camera) of a 3D point.
    const auto project = [&](const Vec3d &p) {
        const Vec3d d = p - view.center;
        return Vec3d(0.5 * W + scale * d.dot(right), 0.5 * H - scale * d.dot(up), d.dot(n));
    };
    std::vector<double> depth(static_cast<size_t>(W) * H, -std::numeric_limits<double>::infinity());
    std::vector<unsigned char> rgb(static_cast<size_t>(W) * H * 3, 255);
    const auto &P = tri.inputVertices();
    std::vector<Vec3d> screen(P.size());
    for (size_t v = 0; v < P.size(); v++) screen[v] = project(P[v].cast<double>());
    for (uint32_t f = 0; f < input.faceCount(); f++) {
        const uint32_t a = input.tail(3 * f), b = input.tail(3 * f + 1), c = input.tail(3 * f + 2);
        const Vec3d A = P[a].cast<double>(), B = P[b].cast<double>(), C = P[c].cast<double>();
        const double shade = std::abs((B - A).cross(C - A).normalized().dot(n));
        if ((B - A).cross(C - A).dot(n) <= 0.0) continue;  // facing away
        const Vec3d &sa = screen[a], &sb = screen[b], &sc = screen[c];
        const int x0 = std::max(0, static_cast<int>(std::floor(std::min({sa.x(), sb.x(), sc.x()}))));
        const int x1 = std::min(W - 1, static_cast<int>(std::ceil(std::max({sa.x(), sb.x(), sc.x()}))));
        const int y0 = std::max(0, static_cast<int>(std::floor(std::min({sa.y(), sb.y(), sc.y()}))));
        const int y1 = std::min(H - 1, static_cast<int>(std::ceil(std::max({sa.y(), sb.y(), sc.y()}))));
        if (x0 > x1 || y0 > y1) continue;
        const double area = (sb.x() - sa.x()) * (sc.y() - sa.y()) - (sb.y() - sa.y()) * (sc.x() - sa.x());
        if (std::abs(area) < 1e-12) continue;
        double ca[3], cb[3], cc[3];
        // Input vertices keep their indices in the refined triangulation.
        const auto region = [&](uint32_t v) { return sim.phi()[v] < 0.0 ? IntrinsicSpringls::kRegionLabel : 0; };
        LabelColor(region(a), ca);
        LabelColor(region(b), cb);
        LabelColor(region(c), cc);
        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                const double px = x + 0.5, py = y + 0.5;
                const double wa = ((sb.x() - px) * (sc.y() - py) - (sb.y() - py) * (sc.x() - px)) / area;
                const double wb = ((sc.x() - px) * (sa.y() - py) - (sc.y() - py) * (sa.x() - px)) / area;
                const double wc = 1.0 - wa - wb;
                if (wa < 0.0 || wb < 0.0 || wc < 0.0) continue;
                const double z = wa * sa.z() + wb * sb.z() + wc * sc.z();
                const size_t i = static_cast<size_t>(y) * W + x;
                if (z <= depth[i]) continue;
                depth[i] = z;
                for (int k = 0; k < 3; k++)
                    rgb[3 * i + k] = static_cast<unsigned char>(std::clamp((0.35 + 0.65 * shade) * (wa * ca[k] + wb * cb[k] + wc * cc[k]), 0.0, 255.0));
            }
        }
    }
    // Springls: dark lines, drawn where they are not hidden (a small depth bias).
    const double bias = 2.0 * sim.meanEdgeLength();
    const auto line = [&](const Vec3d &a, const Vec3d &b, const unsigned char color[3]) {
        const int steps = std::max(1, static_cast<int>(std::ceil(std::max(std::abs(b.x() - a.x()), std::abs(b.y() - a.y())))));
        for (int s = 0; s <= steps; s++) {
            const Vec3d p = a + (b - a) * (static_cast<double>(s) / steps);
            for (int dy = -1; dy <= 0; dy++) {
                for (int dx = -1; dx <= 0; dx++) {
                    const int x = static_cast<int>(p.x()) + dx, y = static_cast<int>(p.y()) + dy;
                    if (x < 0 || y < 0 || x >= W || y >= H) continue;
                    const size_t i = static_cast<size_t>(y) * W + x;
                    if (p.z() + bias < depth[i]) continue;
                    for (int k = 0; k < 3; k++) rgb[3 * i + k] = color[k];
                }
            }
        }
    };
    for (const IntrinsicSpringl &s : sim.springls()) {
        double c[3];
        LabelColor(IntrinsicSpringls::kRegionLabel, c);
        const unsigned char dark[3] = {static_cast<unsigned char>(0.35 * c[0]), static_cast<unsigned char>(0.35 * c[1]),
                                       static_cast<unsigned char>(0.35 * c[2])};
        const Vec3d a = project(tri.toExtrinsic(s.v0).position), p = project(tri.toExtrinsic(s.particle).position), b = project(tri.toExtrinsic(s.v1).position);
        line(a, p, dark);
        line(p, b, dark);
    }
    if (!stbi_write_png(file.string().c_str(), W, H, 3, rgb.data(), W * 3)) {
        error = "cannot write " + file.string();
        return false;
    }
    return true;
}
}  // namespace springls::intrinsic
