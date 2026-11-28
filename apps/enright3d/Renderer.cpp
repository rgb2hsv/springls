#include "Renderer.hpp"

#include "../common/Palette.hpp"

#include "GL.hpp"

#include <dweave/DweaveLevelSet3D.hpp>

#include "nanovg.hpp"
#define NANOVG_GL3 1  // the implementation is in common/NanovgGl.cpp
#include "nanovg_gl.hpp"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#include "stb_image_write.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <utility>
#include <cstdio>
#include <iomanip>
#include <sstream>

namespace enright3d
{
using namespace springls;
namespace
{
constexpr float kTitleFontSize = 26.0f;
constexpr float kTextFontSize = 18.0f;
constexpr float kTextLineHeight = 22.0f;
constexpr int kMetricsPrecision = 5;

// Orthographic view: the unit cube about the origin spans [-kViewHalfExtent, kViewHalfExtent] at
// zoom 1, and depth covers view z in [-kDepthHalfRange, kDepthHalfRange].
constexpr float kViewHalfExtent = 0.75f;
constexpr float kDepthHalfRange = 2.0f;
constexpr float kMinPitch = -1.55f, kMaxPitch = 1.55f;

// Layer priorities, as depth offsets in voxels: the layers stack iso-surface, then dual, then primal.
// Where both constellations lie on the same surface the primal layer wins, and springls within a
// small tolerance of the iso-surface are drawn over it.
constexpr float kLayerEpsilonVoxels = 0.64f;
constexpr float kCapsuleSurfaceToleranceVoxels = 2.56f;
// Capsule overshoot (outside the springl element) sits behind its plane, more toward its rim:
// neighboring capsules lie in their own tangent planes, which rise in front of a curved surface
// away from their particles.
constexpr float kOvershootDepthVoxels = 0.128f;
constexpr float kOvershootSlopeVoxels = 1.92f;
// Particles are spheres of this radius (voxels), at least kMinParticlePixels across on screen.
constexpr float kParticleDrawRadius = 2.0f * SpringLevelSet3D::kParticleRadius;
constexpr float kMinParticlePixels = 3.0f;
constexpr float kMaxArcStep = kPi / 8.0f;
// Wireframes sit this far in front of their surface (they lie on it); widths in device pixels.
constexpr float kLineBiasVoxels = 0.05f;
const Vec3 kWireColor(0.5f, 0.5f, 0.5f);
const Vec3 kIsoWireColor(0.2f, 0.2f, 0.22f);  // the iso-surface's wireframe, over its matcap shading
const Vec3 kBadColor(0.9f, 0.1f, 0.1f);       // tracked mesh triangles that fold over or overlap
const Vec3 kTrackedTint(0.36f, 0.72f, 0.42f);  // the tracked mesh's surface (matcap tinted green)
const Vec3 kTrackedWireColor(0.66f, 0.68f, 0.70f);  // its wireframe, lighter than the iso-surface's
constexpr float kWirePixels = 1.0f;


// Colors from the toybox palette (common/Palette.hpp), shared by every window: springls (tints of the
// matcap) and particles as in the 2D window, primal then dual; every capsule's overshoot (the region outside
// its springl element) the primal capsule color.
Vec3 ToVec3(const toybox::palette::Color &c) { return Vec3(c.r, c.g, c.b); }
const Vec3 kBackground = ToVec3(toybox::palette::kBackground);
const Vec3 kPanelColor = ToVec3(toybox::palette::kPanel);
const Vec3 kIsoTint = ToVec3(toybox::palette::kIsoSurface);
const std::array<Vec3, 2> kSpringlColors = {ToVec3(toybox::palette::kPrimalSpringl), ToVec3(toybox::palette::kDualSpringl)};
const std::array<Vec3, 2> kParticleColors = {ToVec3(toybox::palette::kPrimalParticle), ToVec3(toybox::palette::kDualParticle)};
const Vec3 kCapsuleColor = ToVec3(toybox::palette::kPrimalCapsule);

nvg::Color Paint(const toybox::palette::Color &c) { return nvg::rgbaf(c.r, c.g, c.b, 1.0f); }
const nvg::Color kTextColor = Paint(toybox::palette::kText);
const nvg::Color kStatusColor = Paint(toybox::palette::kStatus);
const nvg::Color kTextShadow = Paint(toybox::palette::kTextShadow);

int Columns(size_t count, int columns = 0) { return (columns > 0) ? columns : ((count > 1) ? 2 : 1); }
int Rows(size_t count, int columns = 0) { return static_cast<int>((count + Columns(count, columns) - 1) / Columns(count, columns)); }

// ---------------------------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------------------------

// Grayscale matcap of a view-space unit normal (z toward the viewer), tinted per layer: a soft key
// light, a highlight and a little rim light, approximating a brushed silver matcap. GAMMA > 1 lifts
// the midtones so the shaded iso-surface reads close to white.
const char *kMatcapGlsl = R"(
const float GAMMA = 2.2;
float matcap(vec3 n) {
    const vec3 L = normalize(vec3(-0.40, 0.55, 0.75));
    const vec3 H = normalize(L + vec3(0.0, 0.0, 1.0));
    float diffuse = max(dot(n, L), 0.0);
    float highlight = pow(max(dot(n, H), 0.0), 48.0);
    float rim = pow(1.0 - clamp(n.z, 0.0, 1.0), 2.0);
    float shade = 0.16 + 0.66 * diffuse + 0.32 * highlight + 0.14 * rim;
    return pow(clamp(shade, 0.0, 1.0), 1.0 / GAMMA);
}
)";

// Iso-surface, springls and capsules. aInlay is the capsule inlay coordinate: 0 on the springl
// element, 1 on the capsule boundary (always 0 for the iso-surface and springls).
const char *kSurfaceVertex = R"(#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in float aInlay;
uniform mat4 uMvp;
uniform mat3 uNormalMatrix;
out vec3 vNormal;
out float vInlay;
void main() {
    vNormal = uNormalMatrix * aNormal;
    vInlay = aInlay;
    gl_Position = uMvp * vec4(aPosition, 1.0);
}
)";
const char *kSurfaceFragment = R"(
in vec3 vNormal;
in float vInlay;
uniform vec3 uTint;
uniform int uShaded;        // 1: matcap-shaded tint, 0: solid tint
uniform float uDepthBias;   // window-depth offset of the whole layer
uniform vec2 uOvershoot;    // capsules: window-depth offset of the overshoot, and its slope over the inlay coordinate
uniform int uCapsules;
uniform vec3 uCapsuleColor;
out vec4 fragColor;
void main() {
    vec3 color = uTint;
    float depth = gl_FragCoord.z + uDepthBias;
    if (uCapsules != 0 && vInlay > 1E-5) {
        // Overshoot in the capsule color; the inlay keeps the constellation color.
        color = uCapsuleColor;
        depth += uOvershoot.x + uOvershoot.y * vInlay;
    }
    if (uShaded != 0) {
        vec3 n = normalize(vNormal);
        if (n.z < 0.0) n = -n;  // two-sided
        color *= matcap(n);
    }
    gl_FragDepth = depth;
    fragColor = vec4(color, 1.0);
}
)";

// Particles as solid point-sprite discs, with sphere-impostor depth.
const char *kParticleVertex = R"(#version 330 core
layout(location = 0) in vec3 aPosition;
uniform mat4 uMvp;
uniform float uPointSize;
void main() {
    gl_Position = uMvp * vec4(aPosition, 1.0);
    gl_PointSize = uPointSize;
}
)";
const char *kParticleFragment = R"(
uniform vec3 uTint;
uniform float uRadiusDepth;  // sphere radius in window depth
out vec4 fragColor;
void main() {
    vec2 p = 2.0 * gl_PointCoord - 1.0;
    p.y = -p.y;
    float r2 = dot(p, p);
    if (r2 > 1.0) discard;
    vec3 n = vec3(p, sqrt(1.0 - r2));
    gl_FragDepth = gl_FragCoord.z - uRadiusDepth * n.z;
    fragColor = vec4(uTint, 1.0);
}
)";

// Lines of any width (core-profile OpenGL draws only 1-pixel lines): the geometry shader turns each segment
// into a screen-space quad uWidth device pixels wide, extended by half that at both ends so joints close.
const char *kLineVertex = R"(#version 330 core
layout(location = 0) in vec3 aPosition;
uniform mat4 uMvp;
void main() { gl_Position = uMvp * vec4(aPosition, 1.0); }
)";
const char *kLineGeometry = R"(#version 330 core
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;
uniform vec2 uViewport;  // device pixels
uniform float uWidth;    // device pixels
void main() {
    vec4 a = gl_in[0].gl_Position, b = gl_in[1].gl_Position;
    vec2 pa = a.xy / a.w * 0.5 * uViewport, pb = b.xy / b.w * 0.5 * uViewport;
    vec2 d = pb - pa;
    float len = length(d);
    vec2 t = len > 0.0 ? d / len : vec2(1.0, 0.0);
    vec2 n = vec2(-t.y, t.x);
    vec2 toNdc = 2.0 / uViewport;
    vec2 side = n * 0.5 * uWidth * toNdc, along = t * 0.5 * uWidth * toNdc;
    gl_Position = vec4((a.xy / a.w - along + side) * a.w, a.z, a.w); EmitVertex();
    gl_Position = vec4((a.xy / a.w - along - side) * a.w, a.z, a.w); EmitVertex();
    gl_Position = vec4((b.xy / b.w + along + side) * b.w, b.z, b.w); EmitVertex();
    gl_Position = vec4((b.xy / b.w + along - side) * b.w, b.z, b.w); EmitVertex();
    EndPrimitive();
}
)";
const char *kLineFragment = R"(
uniform vec3 uTint;
uniform float uDepthBias;
out vec4 fragColor;
void main() {
    gl_FragDepth = gl_FragCoord.z + uDepthBias;
    fragColor = vec4(uTint, 1.0);
}
)";

GLuint CompileShader(GLenum type, const std::string &source, std::string &error)
{
    const GLuint shader = glCreateShader(type);
    const char *text = source.c_str();
    glShaderSource(shader, 1, &text, nullptr);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        error = std::string("shader compile error: ") + log;
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

struct Program {
    GLuint id = 0;
    bool build(const std::string &vertex, const std::string &fragment, std::string &error, const std::string &geometry = {})
    {
        const GLuint vs = CompileShader(GL_VERTEX_SHADER, vertex, error);
        if (!vs) return false;
        const GLuint fs = CompileShader(GL_FRAGMENT_SHADER, fragment, error);
        if (!fs) {
            glDeleteShader(vs);
            return false;
        }
        GLuint gs = 0;
        if (!geometry.empty() && !(gs = CompileShader(GL_GEOMETRY_SHADER, geometry, error))) {
            glDeleteShader(vs);
            glDeleteShader(fs);
            return false;
        }
        id = glCreateProgram();
        glAttachShader(id, vs);
        glAttachShader(id, fs);
        if (gs) glAttachShader(id, gs);
        glLinkProgram(id);
        glDeleteShader(vs);
        glDeleteShader(fs);
        if (gs) glDeleteShader(gs);
        GLint ok = 0;
        glGetProgramiv(id, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[2048];
            glGetProgramInfoLog(id, sizeof(log), nullptr, log);
            error = std::string("shader link error: ") + log;
            return false;
        }
        return true;
    }
    void release()
    {
        if (id) glDeleteProgram(id);
        id = 0;
    }
    GLint at(const char *name) const { return glGetUniformLocation(id, name); }
};

// ---------------------------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------------------------

struct Vertex {
    float position[3];
    float normal[3];
    float inlay;
};
using Vertexes = std::vector<Vertex>;

void Push(Vertexes &out, const Vec3 &p, const Vec3 &n, float inlay = 0.0f) { out.push_back({{p.x(), p.y(), p.z()}, {n.x(), n.y(), n.z()}, inlay}); }

// A vertex array of non-indexed primitives with the Vertex layout.
struct GpuMesh {
    GLuint vao = 0, vbo = 0;
    GLsizei count = 0;
    void upload(const Vertexes &vertexes)
    {
        if (!vao) {
            glGenVertexArrays(1, &vao);
            glGenBuffers(1, &vbo);
            glBindVertexArray(vao);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void *>(offsetof(Vertex, position)));
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void *>(offsetof(Vertex, normal)));
            glEnableVertexAttribArray(2);
            glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void *>(offsetof(Vertex, inlay)));
            glBindVertexArray(0);
        }
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertexes.size() * sizeof(Vertex)), vertexes.data(), GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        count = static_cast<GLsizei>(vertexes.size());
    }
    void draw(GLenum mode) const
    {
        if (count == 0) return;
        glBindVertexArray(vao);
        glDrawArrays(mode, 0, count);
        glBindVertexArray(0);
    }
    void release()
    {
        if (vbo) glDeleteBuffers(1, &vbo);
        if (vao) glDeleteVertexArrays(1, &vao);
        vao = vbo = 0;
        count = 0;
    }
};

// Appends, as triangles, one springl's capsule: its element (the polygon projected onto the plane
// through the particle) offset outward by `radius`, with circular arcs around each vertex. Each
// corner's inlay coordinate is 0 on the springl element and 1 on the capsule boundary.
void AppendCapsule(const Vec3 &particle, const Vec3 *vertexes, int K, const Vec3 &normal, float radius, Vertexes &out)
{
    const auto triangle = [&](const Vec3 &a, float ia, const Vec3 &b, float ib, const Vec3 &c, float ic) {
        Push(out, a, normal, ia);
        Push(out, b, normal, ib);
        Push(out, c, normal, ic);
    };
    std::array<Vec3, kMaxSpringlSize> v, outward;
    for (int k = 0; k < K; k++) v[k] = vertexes[k] - (vertexes[k] - particle).dot(normal) * normal;
    // Outward in-plane normal of edge k (v[k] -> v[k + 1]).
    for (int k = 0; k < K; k++) {
        const Vec3 a = v[k], b = v[(k + 1) % K];
        Vec3 o = (b - a).cross(normal);
        const float len = o.norm();
        o = (len > 1E-6f) ? Vec3(o / len) : normalizedOrZero(Vec3(0.5f * (a + b) - particle));
        if (o.dot(0.5f * (a + b) - particle) < 0.0f) o = -o;
        outward[k] = o;
    }
    for (int k = 0; k < K; k++) {
        const Vec3 a = v[k], b = v[(k + 1) % K];
        triangle(particle, 0.0f, a, 0.0f, b, 0.0f);  // inlay
        // Overshoot band along the edge.
        const Vec3 ao = a + radius * outward[k], bo = b + radius * outward[k];
        triangle(a, 0.0f, ao, 1.0f, bo, 1.0f);
        triangle(a, 0.0f, bo, 1.0f, b, 0.0f);
        // Rounded corner at b, from this edge's normal to the next edge's.
        const Vec3 n0 = outward[k], n1 = outward[(k + 1) % K];
        const float angle = std::atan2(n0.cross(n1).dot(normal), n0.dot(n1));
        const int steps = std::max(1, static_cast<int>(std::ceil(std::abs(angle) / kMaxArcStep)));
        Vec3 prev = b + radius * n0;
        for (int i = 1; i <= steps; i++) {
            const float t = angle * i / steps;
            const Vec3 next = b + radius * (std::cos(t) * n0 + std::sin(t) * normal.cross(n0));
            triangle(b, 0.0f, prev, 1.0f, next, 1.0f);
            prev = next;
        }
    }
}

// GPU meshes of one run, rebuilt when the run or a geometry setting changes.
struct RunGeometry {
    uint64_t generation = 0, iteration = 0;
    float springlScale = -1.0f, capsuleRadius = -1.0f;
    bool capsules = false, valid = false;
    GpuMesh iso, isoWire, tracked, trackedWire, trackedBad, trackedBadWire;
    std::array<GpuMesh, kConstellationCount> springls, capsuleMeshes, particles;

    void release()
    {
        iso.release();
        isoWire.release();
        tracked.release();
        trackedWire.release();
        trackedBad.release();
        trackedBadWire.release();
        for (size_t w = 0; w < kConstellationCount; w++) {
            springls[w].release();
            capsuleMeshes[w].release();
            particles[w].release();
        }
        valid = false;
    }

    void update(const ExperimentSnapshot &experiment, const RunSnapshot &run, const Visibility &visibility)
    {
        const Manifold3D &m = run.manifold;
        if (valid && generation == experiment.generation && iteration == run.iteration && springlScale == visibility.springlScale &&
            capsules == visibility.capsules && (!capsules || capsuleRadius == visibility.capsuleRadius)) {
            return;
        }
        generation = experiment.generation;
        iteration = run.iteration;
        springlScale = visibility.springlScale;
        capsules = visibility.capsules;
        capsuleRadius = visibility.capsuleRadius;
        valid = true;

        // Iso-surface (quads split in two), smooth shaded, and its edges for the wireframe.
        const SurfaceMesh3D &surface = m.isoSurface;
        Vertexes buffer, edges;
        const int FK = surface.elementSize();
        buffer.reserve(surface.faceCount() * 3 * (FK - 2));
        for (size_t f = 0; f < surface.faceCount(); f++) {
            const uint32_t *idx = surface.face(f);
            for (int t = 1; t + 1 < FK; t++) {
                for (uint32_t i : {idx[0], idx[t], idx[t + 1]}) Push(buffer, surface.vertexes[i], surface.normals[i]);
            }
            for (int k = 0; k < FK; k++) {
                Push(edges, surface.vertexes[idx[k]], Vec3::UnitZ());
                Push(edges, surface.vertexes[idx[(k + 1) % FK]], Vec3::UnitZ());
            }
        }
        iso.upload(buffer);
        isoWire.upload(edges);
        buffer.clear();
        edges.clear();
        if (run.trackedMesh) {
            // The tracked mesh (smooth shaded: area-weighted vertex normals) and its edges.
            const springls::TriangleSurface &mesh = *run.trackedMesh;
            Points3 normals(mesh.vertexes.size(), Vec3::Zero());
            for (const auto &t : mesh.triangles) {
                const Vec3 n = (mesh.vertexes[t[1]] - mesh.vertexes[t[0]]).cross(mesh.vertexes[t[2]] - mesh.vertexes[t[0]]);
                for (uint32_t v : t) normals[v] += n;
            }
            for (Vec3 &n : normals) n = (n.squaredNorm() > 0.0f) ? Vec3(n.normalized()) : Vec3::UnitZ();
            for (const auto &t : mesh.triangles) {
                for (uint32_t i : t) Push(buffer, mesh.vertexes[i], normals[i]);
                for (int k = 0; k < 3; k++) {
                    Push(edges, mesh.vertexes[t[k]], Vec3::UnitZ());
                    Push(edges, mesh.vertexes[t[(k + 1) % 3]], Vec3::UnitZ());
                }
            }
        }
        tracked.upload(buffer);
        trackedWire.upload(edges);
        buffer.clear();
        edges.clear();
        if (run.trackedMesh && run.trackedCheck && run.trackedCheck->flags.size() == run.trackedMesh->triangles.size()) {
            // Triangles the check flagged, drawn again over the mesh in red with heavy outlines.
            const springls::TriangleSurface &mesh = *run.trackedMesh;
            for (size_t f = 0; f < mesh.triangles.size(); f++) {
                if (!run.trackedCheck->flags[f]) continue;
                const auto &t = mesh.triangles[f];
                const Vec3 n = (mesh.vertexes[t[1]] - mesh.vertexes[t[0]]).cross(mesh.vertexes[t[2]] - mesh.vertexes[t[0]]);
                const Vec3 normal = n.squaredNorm() > 0.0f ? Vec3(n.normalized()) : Vec3::UnitZ();
                for (uint32_t i : t) Push(buffer, mesh.vertexes[i], normal);
                for (int k = 0; k < 3; k++) {
                    Push(edges, mesh.vertexes[t[k]], Vec3::UnitZ());
                    Push(edges, mesh.vertexes[t[(k + 1) % 3]], Vec3::UnitZ());
                }
            }
        }
        trackedBad.upload(buffer);
        trackedBadWire.upload(edges);

        // Springls (polygons of any size, fanned from the particle since dual cells need not be
        // convex), capsules and particles per constellation.
        const size_t N = m.springlCount();
        const bool tagged = m.springlConstellations.size() == N;
        const bool hasNormals = m.springlParticleNormals.size() == N;
        std::array<Vertexes, kConstellationCount> elements, caps, points;
        for (size_t n = 0; n < N; n++) {
            const size_t w = tagged ? std::min<size_t>(m.springlConstellations[n], kConstellationCount - 1) : 0;
            const Vec3 &p = m.springlParticles[n];
            const Vec3 normal = hasNormals ? m.springlParticleNormals[n] : Vec3::UnitZ();
            const int K = m.springlSize(n);
            const Vec3 *corners = m.springl(n);
            for (int k = 0; k < K; k++) {
                Push(elements[w], p, normal);
                Push(elements[w], p + springlScale * (corners[k] - p), normal);
                Push(elements[w], p + springlScale * (corners[(k + 1) % K] - p), normal);
            }
            Push(points[w], p, normal);
            if (capsules) AppendCapsule(p, corners, K, normal, capsuleRadius, caps[w]);
        }
        for (size_t w = 0; w < kConstellationCount; w++) {
            springls[w].upload(elements[w]);
            capsuleMeshes[w].upload(caps[w]);
            particles[w].upload(points[w]);
        }
    }
};

// Metric lines shown under a run's title.
std::vector<std::pair<std::string, nvg::Color>> HeaderLines(const ExperimentSnapshot &experiment, const RunSnapshot &run, const std::string &status)
{
    const EnrightConfig3D &config = run.config;
    const EnrightMetrics3D &mt = run.metrics;
    const Manifold3D &m = run.manifold;
    const bool hasConstellations = m.springlConstellations.size() == m.springlCount();
    std::vector<std::pair<std::string, nvg::Color>> lines;
    std::ostringstream s;
    const auto push = [&](const nvg::Color &color) {
        lines.emplace_back(s.str(), color);
        s.str("");
    };
    s << std::fixed << std::setprecision(3) << "t = " << mt.time << " / " << experiment.period << (run.done ? "  (done)" : status);
    push((!run.done && !status.empty()) ? kStatusColor : kTextColor);
    s << std::fixed << std::setprecision(kMetricsPrecision) << "volume error " << 100.0 * mt.relativeVolumeError << " %";
    push(kTextColor);
    s << std::fixed << std::setprecision(kMetricsPrecision) << "shape error vs sphere: mean " << mt.meanShapeError << "  max " << mt.maxShapeError;
    push(kTextColor);
    s << "advection sub-steps " << run.substeps << std::fixed << std::setprecision(5) << "   mean dt "
      << (run.substeps > 0 ? mt.time / static_cast<double>(run.substeps) : 0.0) << std::setprecision(1) << "   compute " << run.seconds << " s";
    push(kTextColor);
    if (run.trackedMesh) {
        s << "tracked mesh: " << run.trackedMesh->triangles.size() << " triangles";
        if (run.trackedCheck) {
            s << ", " << run.trackedCheck->foldedEdges << " fold-overs, " << run.trackedCheck->overlappingPairs << " overlaps";
            if (run.trackedCheck->nonManifoldEdges) s << ", " << run.trackedCheck->nonManifoldEdges << " non-manifold edges";
        }
    } else {
        s << "springls " << mt.springlCount;
        if (hasConstellations) s << "  (primal " << mt.constellationSpringlCounts[0] << ", dual " << mt.constellationSpringlCounts[1] << ")";
    }
    push(kTextColor);
    if (run.maxDepth > 0) {
        s << "grid " << config.gridSize << "^3 to " << (config.gridSize << run.finestDepth) << "^3 equivalent (depths 0-" << run.finestDepth << " of "
          << run.maxDepth << ", " << run.leafCount << " blocks), ";
    } else {
        s << "grid " << config.gridSize << "^3, ";
    }
    if (run.trackedMesh) {
        s << m.isoSurface.faceCount() << " iso-surface faces";
    } else {
        s << ElementTypeKey(m.elementType) << " springls, " << m.isoSurface.faceCount() << " faces";
    }
    push(kTextColor);
    return lines;
}

void Shadowed(nvg::Context &vg, float x, float y, const std::string &text, const nvg::Color &color)
{
    vg.fillColor(kTextShadow);
    vg.text(x + 1.0f, y + 1.0f, text.c_str(), nullptr);
    vg.fillColor(color);
    vg.text(x, y, text.c_str(), nullptr);
}

void SetVec3(GLint location, const Vec3 &v) { glUniform3f(location, v.x(), v.y(), v.z()); }
}  // namespace

void OrbitCamera::rotate(float dx, float dy)
{
    constexpr float kRadiansPerPixel = 1.0f / 200.0f;
    yaw += dx * kRadiansPerPixel;
    pitch = std::clamp(pitch + dy * kRadiansPerPixel, kMinPitch, kMaxPitch);
}

// ---------------------------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------------------------

struct Renderer::Impl {
    std::unique_ptr<nvg::Context> vg;
    int font = -1;
    Program surface, particle, line;
    float devicePixelRatio = 1.0f;
    std::vector<RunGeometry> geometry;

    ~Impl()
    {
        for (RunGeometry &g : geometry) g.release();
        surface.release();
        particle.release();
        line.release();
        if (vg) nvg::deleteGL(vg);
    }

    void drawPanel(const ExperimentSnapshot &experiment, const RunSnapshot &run, RunGeometry &g, const Visibility &visibility, const OrbitCamera &camera,
                   float panelPixels)
    {
        // Voxel coordinates -> unit cube about the origin -> rotated, panned view -> orthographic clip.
        const float D = run.config.voxelScale();
        const Eigen::Matrix3f rotation =
            (Eigen::AngleAxisf(camera.pitch, Vec3::UnitX()) * Eigen::AngleAxisf(camera.yaw, Vec3::UnitY()) * Eigen::AngleAxisf(-0.5f * kPi, Vec3::UnitX()))
                .toRotationMatrix();
        Eigen::Affine3f model = Eigen::Affine3f::Identity();
        model.translate(Vec3(camera.panX, camera.panY, 0.0f));
        model.rotate(rotation);
        model.scale(1.0f / D);
        model.translate(Vec3::Constant(-0.5f * D));
        const float halfExtent = kViewHalfExtent / camera.zoom;
        Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
        projection(0, 0) = 1.0f / halfExtent;
        projection(1, 1) = 1.0f / halfExtent;
        projection(2, 2) = -1.0f / kDepthHalfRange;
        const Eigen::Matrix4f mvp = projection * model.matrix();
        const float depthPerVoxel = 1.0f / (2.0f * kDepthHalfRange * D);  // window depth per voxel
        const float pixelsPerVoxel = panelPixels / (2.0f * halfExtent * D);

        g.update(experiment, run, visibility);
        const bool capsules = visibility.capsules;
        const std::array<bool, kConstellationCount> show = {visibility.primal, visibility.dual};
        // An iso-surface with nothing else to show gets its wireframe drawn over it. Mesh tracking always shows one
        // surface, the tracked mesh or the iso-surface (in the iso-surface buffers), with its wireframe.
        const bool trackedRun = run.trackedMesh != nullptr;
        const bool showTracked = trackedRun && visibility.trackedSurface;
        const bool drawIso = trackedRun ? !showTracked : visibility.isoSurface;
        const bool isoOnly = trackedRun ? !showTracked : (visibility.isoSurface && !show[0] && !show[1] && !visibility.particles);

        glUseProgram(surface.id);
        glUniformMatrix4fv(surface.at("uMvp"), 1, GL_FALSE, mvp.data());
        glUniformMatrix3fv(surface.at("uNormalMatrix"), 1, GL_FALSE, rotation.data());
        glUniform2f(surface.at("uOvershoot"), kOvershootDepthVoxels * depthPerVoxel, kOvershootSlopeVoxels * depthPerVoxel);
        if (drawIso) {
            const float tolerance = capsules ? kCapsuleSurfaceToleranceVoxels : kLayerEpsilonVoxels;
            SetVec3(surface.at("uTint"), kIsoTint);
            glUniform1i(surface.at("uShaded"), 1);
            glUniform1f(surface.at("uDepthBias"), (tolerance + kLayerEpsilonVoxels) * depthPerVoxel);
            glUniform1i(surface.at("uCapsules"), 0);
            g.iso.draw(GL_TRIANGLES);
        }
        if (showTracked) {
            SetVec3(surface.at("uTint"), kTrackedTint);
            glUniform1i(surface.at("uShaded"), 1);
            glUniform1f(surface.at("uDepthBias"), 0.0f);
            glUniform1i(surface.at("uCapsules"), 0);
            g.tracked.draw(GL_TRIANGLES);
        }
        // Dual, then primal on top of it (the dual layer sits kLayerEpsilonVoxels behind). Springls
        // and capsules are matcap shaded too, the matcap tinted so their color shows through.
        glUniform1i(surface.at("uShaded"), 1);
        glUniform1i(surface.at("uCapsules"), capsules ? 1 : 0);
        SetVec3(surface.at("uCapsuleColor"), kCapsuleColor);
        constexpr std::array<size_t, kConstellationCount> kLayerOrder = {static_cast<size_t>(Constellation::Dual), static_cast<size_t>(Constellation::Primal)};
        for (size_t w : kLayerOrder) {
            if (!show[w]) continue;
            SetVec3(surface.at("uTint"), kSpringlColors[w]);
            glUniform1f(surface.at("uDepthBias"), (w == static_cast<size_t>(Constellation::Dual) ? kLayerEpsilonVoxels : 0.0f) * depthPerVoxel);
            (capsules ? g.capsuleMeshes[w] : g.springls[w]).draw(GL_TRIANGLES);
        }
        if (isoOnly) {
            GLint viewport[4] = {0, 0, 1, 1};
            glGetIntegerv(GL_VIEWPORT, viewport);
            glUseProgram(line.id);
            glUniformMatrix4fv(line.at("uMvp"), 1, GL_FALSE, mvp.data());
            glUniform2f(line.at("uViewport"), static_cast<float>(viewport[2]), static_cast<float>(viewport[3]));
            glUniform1f(line.at("uWidth"), kWirePixels * devicePixelRatio);
            glUniform1f(line.at("uDepthBias"), -kLineBiasVoxels * depthPerVoxel);
            SetVec3(line.at("uTint"), kIsoWireColor);
            g.isoWire.draw(GL_LINES);
            glUseProgram(surface.id);
        }
        if (showTracked) {
            GLint viewport[4] = {0, 0, 1, 1};
            glGetIntegerv(GL_VIEWPORT, viewport);
            glUseProgram(line.id);
            glUniformMatrix4fv(line.at("uMvp"), 1, GL_FALSE, mvp.data());
            glUniform2f(line.at("uViewport"), static_cast<float>(viewport[2]), static_cast<float>(viewport[3]));
            glUniform1f(line.at("uWidth"), kWirePixels * devicePixelRatio);
            glUniform1f(line.at("uDepthBias"), -kLineBiasVoxels * depthPerVoxel);
            SetVec3(line.at("uTint"), kTrackedWireColor);
            g.trackedWire.draw(GL_LINES);
            glUseProgram(surface.id);
        }
        if (trackedRun && visibility.trackedSurface) {
            // Fold-overs and overlaps (when checked): red, in front of the mesh and its wireframe, outlined.
            SetVec3(surface.at("uTint"), kBadColor);
            glUniform1i(surface.at("uShaded"), 1);
            glUniform1i(surface.at("uCapsules"), 0);
            glUniform1f(surface.at("uDepthBias"), -kLineBiasVoxels * depthPerVoxel);
            g.trackedBad.draw(GL_TRIANGLES);
            GLint viewport[4] = {0, 0, 1, 1};
            glGetIntegerv(GL_VIEWPORT, viewport);
            glUseProgram(line.id);
            glUniformMatrix4fv(line.at("uMvp"), 1, GL_FALSE, mvp.data());
            glUniform2f(line.at("uViewport"), static_cast<float>(viewport[2]), static_cast<float>(viewport[3]));
            glUniform1f(line.at("uWidth"), 3.0f * kWirePixels * devicePixelRatio);
            glUniform1f(line.at("uDepthBias"), -2.0f * kLineBiasVoxels * depthPerVoxel);
            SetVec3(line.at("uTint"), kBadColor);
            g.trackedBadWire.draw(GL_LINES);
            glUseProgram(surface.id);
        }
        if (visibility.particles) {
            glUseProgram(particle.id);
            glUniformMatrix4fv(particle.at("uMvp"), 1, GL_FALSE, mvp.data());
            glUniform1f(particle.at("uPointSize"), std::max(kMinParticlePixels, 2.0f * kParticleDrawRadius * pixelsPerVoxel));
            glUniform1f(particle.at("uRadiusDepth"), kParticleDrawRadius * depthPerVoxel);
            for (size_t w : kLayerOrder) {
                if (!show[w]) continue;
                SetVec3(particle.at("uTint"), kParticleColors[w]);
                g.particles[w].draw(GL_POINTS);
            }
        }
        glUseProgram(0);
    }
};

Renderer::Renderer() = default;
Renderer::~Renderer() { shutdown(); }

bool Renderer::init(const std::string &fontFile, std::string &error)
{
    mImpl = std::make_unique<Impl>();
    const std::string header = "#version 330 core\n";
    if (!mImpl->surface.build(kSurfaceVertex, header + kMatcapGlsl + kSurfaceFragment, error) ||
        !mImpl->particle.build(kParticleVertex, header + kMatcapGlsl + kParticleFragment, error) ||
        !mImpl->line.build(kLineVertex, header + kLineFragment, error, kLineGeometry)) {
        return false;
    }
    mImpl->vg = nvg::createGL(static_cast<int>(nvg::CreateFlags::Antialias | nvg::CreateFlags::StencilStrokes));
    if (!mImpl->vg) {
        error = "could not create the NanoVG context";
        return false;
    }
    mImpl->font = mImpl->vg->createFont("sans", fontFile.c_str());
    if (mImpl->font < 0) {
        error = "could not load font " + fontFile;
        return false;
    }
    return true;
}

void Renderer::shutdown() { mImpl.reset(); }

std::vector<Renderer::PanelView> Renderer::Views(const ExperimentSnapshot &experiment, const Visibility &visibility)
{
    std::vector<PanelView> views;
    for (size_t i = 0; i < experiment.runs.size(); i++) views.push_back({i});
    return views;
}

std::vector<Rect> Renderer::Layout(size_t count, const Rect &bounds, int requestedColumns)
{
    std::vector<Rect> panels(count);
    if (count == 0) return panels;
    const int columns = Columns(count, requestedColumns), rows = Rows(count, requestedColumns);
    const float size =
        std::max(0.0f, std::min((bounds.w - (columns + 1) * kPanelMargin) / columns, (bounds.h - rows * kHeaderHeight - (rows + 1) * kPanelMargin) / rows));
    // Center the block of panels horizontally.
    const float used = columns * size + (columns + 1) * kPanelMargin;
    const float x0 = bounds.x + 0.5f * std::max(0.0f, bounds.w - used);
    for (size_t i = 0; i < count; i++) {
        const int column = static_cast<int>(i) % columns, row = static_cast<int>(i) / columns;
        panels[i] = {x0 + kPanelMargin + column * (size + kPanelMargin), bounds.y + kPanelMargin + kHeaderHeight + row * (size + kHeaderHeight + kPanelMargin),
                     size, size};
    }
    return panels;
}

void Renderer::ImageSize(size_t count, float panelSize, int &width, int &height, int requestedColumns)
{
    const int columns = Columns(count, requestedColumns), rows = Rows(count, requestedColumns);
    width = static_cast<int>(std::ceil(columns * panelSize + (columns + 1) * kPanelMargin));
    height = static_cast<int>(std::ceil(rows * (panelSize + kHeaderHeight) + (rows + 1) * kPanelMargin));
    width += width & 1;
    height += height & 1;
}

void Renderer::draw(const ExperimentSnapshot &experiment, const Visibility &visibility, const OrbitCamera &camera, const Rect &bounds, float viewWidth,
                    float viewHeight, float devicePixelRatio, const std::string &status)
{
    Impl &impl = *mImpl;
    impl.devicePixelRatio = devicePixelRatio;
    const std::vector<RunSnapshot> &runs = experiment.runs;
    const std::vector<PanelView> views = Views(experiment, visibility);
    const std::vector<Rect> panels = Layout(views.size(), bounds);
    if (impl.geometry.size() != views.size()) {
        for (RunGeometry &g : impl.geometry) g.release();
        impl.geometry.assign(views.size(), RunGeometry());
    }
    const float r = devicePixelRatio;
    const float fbHeight = viewHeight * r;
    // Device-pixel scissor / viewport rectangle (GL's origin is the bottom left).
    const auto setRect = [&](const Rect &rect) {
        const GLint x = static_cast<GLint>(std::lround(rect.x * r));
        const GLint y = static_cast<GLint>(std::lround(fbHeight - (rect.y + rect.h) * r));
        const GLsizei w = static_cast<GLsizei>(std::lround(rect.w * r)), h = static_cast<GLsizei>(std::lround(rect.h * r));
        glViewport(x, y, w, h);
        glScissor(x, y, w, h);
    };
    glEnable(GL_SCISSOR_TEST);
    setRect(bounds);
    glClearColor(kBackground.x(), kBackground.y(), kBackground.z(), 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glDisable(GL_STENCIL_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LEQUAL);
    for (size_t i = 0; i < views.size(); i++) {
        if (panels[i].w <= 0.0f) continue;
        setRect(panels[i]);
        glClearColor(kPanelColor.x(), kPanelColor.y(), kPanelColor.z(), 1.0f);
        glClearDepth(1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        impl.drawPanel(experiment, runs[views[i].run], impl.geometry[i], visibility, camera, panels[i].w * r);
        glDisable(GL_DEPTH_TEST);
    }
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_PROGRAM_POINT_SIZE);
    glViewport(0, 0, static_cast<GLsizei>(std::lround(viewWidth * r)), static_cast<GLsizei>(std::lround(fbHeight)));

    // Headers above the panels, with one text scale for every panel so the widest line fits.
    nvg::Context &vg = *impl.vg;
    vg.beginFrame(viewWidth, viewHeight, devicePixelRatio);
    vg.fontFaceId(impl.font);
    vg.textAlign(static_cast<int>(nvg::Align::Left | nvg::Align::Top));
    float headerScale = 1.0f;
    vg.fontSize(kTextFontSize);
    for (size_t i = 0; i < panels.size(); i++) {
        for (const auto &line : HeaderLines(experiment, runs[views[i].run], status)) {
            const float width = vg.textBounds(0.0f, 0.0f, line.first.c_str(), nullptr, nullptr);
            if (width > panels[i].w && width > 0.0f) headerScale = std::min(headerScale, panels[i].w / width);
        }
    }
    for (size_t i = 0; i < panels.size(); i++) {
        float x = panels[i].x, y = panels[i].y - kHeaderHeight;
        vg.fontSize(kTitleFontSize * headerScale);
        const std::string title = MethodTitle(runs[views[i].run].method);
        Shadowed(vg, x, y, title, kTextColor);
        y += (kTextLineHeight + kTitleFontSize - kTextFontSize) * headerScale;
        vg.fontSize(kTextFontSize * headerScale);
        for (const auto &[text, color] : HeaderLines(experiment, runs[views[i].run], status)) {
            Shadowed(vg, x, y, text, color);
            y += kTextLineHeight * headerScale;
        }
    }
    vg.endFrame();
}

// -------------------------------------------------------------------------------------------------
// FrameRecorder
// -------------------------------------------------------------------------------------------------

FrameRecorder::~FrameRecorder() { release(); }

void FrameRecorder::release()
{
    if (mFramebuffer) glDeleteFramebuffers(1, &mFramebuffer);
    if (mColor) glDeleteRenderbuffers(1, &mColor);
    if (mDepthStencil) glDeleteRenderbuffers(1, &mDepthStencil);
    mFramebuffer = mColor = mDepthStencil = 0;
}

std::filesystem::path FrameRecorder::UnusedDirectory(const std::filesystem::path &requested)
{
    namespace fs = std::filesystem;
    const auto isFree = [](const fs::path &dir) {
        std::error_code ec;
        if (!fs::exists(dir, ec)) return true;
        return fs::is_directory(dir, ec) && fs::is_empty(dir, ec);
    };
    if (isFree(requested)) return requested;
    // requested/ -> requested_001, requested_002, ...
    fs::path base = requested;
    if (!base.has_filename()) base = base.parent_path();  // trailing separator
    for (int n = 1;; n++) {
        char suffix[16];
        std::snprintf(suffix, sizeof(suffix), "_%03d", n);
        const fs::path candidate = base.parent_path() / (base.filename().string() + suffix);
        if (isFree(candidate)) return candidate;
    }
}

bool FrameRecorder::init(const std::filesystem::path &directory, int width, int height, std::string &error)
{
    release();
    mDirectory = UnusedDirectory(directory);
    mWidth = width;
    mHeight = height;
    mFrame = 0;
    std::error_code ec;
    std::filesystem::create_directories(mDirectory, ec);
    if (ec) {
        error = "could not create " + mDirectory.string() + ": " + ec.message();
        return false;
    }
    glGenFramebuffers(1, &mFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, mFramebuffer);
    glGenRenderbuffers(1, &mColor);
    glBindRenderbuffer(GL_RENDERBUFFER, mColor);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, mColor);
    glGenRenderbuffers(1, &mDepthStencil);
    glBindRenderbuffer(GL_RENDERBUFFER, mDepthStencil);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, mDepthStencil);
    const bool complete = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    if (!complete) {
        release();
        error = "offscreen framebuffer is incomplete";
        return false;
    }
    mPixels.resize(static_cast<size_t>(width) * height * 4);
    return true;
}

bool FrameRecorder::capture(Renderer &renderer, const ExperimentSnapshot &experiment, const Visibility &visibility, const OrbitCamera &camera)
{
    if (!active()) return false;
    GLint previous = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &previous);
    glBindFramebuffer(GL_FRAMEBUFFER, mFramebuffer);
    glViewport(0, 0, mWidth, mHeight);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    const float w = static_cast<float>(mWidth), h = static_cast<float>(mHeight);
    renderer.draw(experiment, visibility, camera, Rect{0.0f, 0.0f, w, h}, w, h, 1.0f, "");
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, mWidth, mHeight, GL_RGBA, GL_UNSIGNED_BYTE, mPixels.data());
    glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(previous));
    // Opaque, top row first.
    const size_t row = static_cast<size_t>(mWidth) * 4;
    for (size_t i = 3; i < mPixels.size(); i += 4) mPixels[i] = 255;
    char name[32];
    std::snprintf(name, sizeof(name), "frame_%06llu.png", static_cast<unsigned long long>(mFrame));
    const std::string file = (mDirectory / name).string();
    stbi_flip_vertically_on_write(1);
    const bool ok = stbi_write_png(file.c_str(), mWidth, mHeight, 4, mPixels.data(), static_cast<int>(row)) != 0;
    if (ok) mFrame++;
    return ok;
}
}  // namespace enright3d
