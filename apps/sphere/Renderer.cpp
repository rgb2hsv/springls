#include "Renderer.hpp"

#include "../common/Palette.hpp"

#include <GL.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace sphere_gui
{
using toybox::SphereSnapshot;
using toybox::Vec3f;
namespace
{
// Line widths in pixels (at a pixel ratio of 1). Springls, particles and the contour
// match the 2D window (single object): the contour 4 px; springls, particles and vertexes are metric, sized
// on the surface relative to the sphere (its radius R about the mesh's centroid), in the 2D window's
// proportions (springl width 0.1, particle radius 0.1, vertex radius 0.05 cells), so they scale with the
// zoom; at least kMinElementPixels across (as the 2D window's minimum).
constexpr float kContourWidth = 4.0f, kReferenceWidth = 1.25f;
// Arrows are heads alone: flat triangles on the surface (in the tangent plane at the sample, orthogonal to the
// surface normal there), kArrowHeadLength times the arrow's length (so they grow with the flow speed),
// kArrowHeadWidth times as wide as long, centered on the arrow and pointing along it, and lifted off the surface
// along the normal by kArrowHeadLift head lengths, so a curved surface does not cut through the flat triangle
// (a head of half length r sinks about r² / 2R into a surface of curvature radius R).
constexpr float kArrowHeadLength = 1.8f, kArrowHeadWidth = 0.675f, kArrowHeadLift = 0.15f;
constexpr float kArrowOpacity = 0.5f;  // blended over what lies under them
constexpr float kSpringlWidthRadii = 0.0006f, kParticleDiameterRadii = 2.0f * kSpringlWidthRadii, kVertexDiameterRadii = kSpringlWidthRadii;
// Normals from the particles, as the 2D window's: half a springl's width wide, a springl's half size (0.5 h at
// the default refinement) long.
constexpr float kNormalWidthRadii = 0.5f * kSpringlWidthRadii, kNormalLengthRadii = 0.0025f;
constexpr float kMinElementPixels = 1.0f;
// The wireframe's and the isolines' widths on the surface, in sphere radii (metric, as the springls).
constexpr float kWireWidthRadii = 0.00025f, kIsoWidthRadii = 0.0004f;
// Lines are pulled toward the eye by this fraction of the length scale, so they sit in front of the surface.
// The contour sits there; the springl layers are pulled a further kLayerPullLengthScales each, so they occlude
// in order: the contour at the back, then springls, vertexes, normals, and particles in front.
constexpr float kPullLengthScales = 0.25f, kLayerPullLengthScales = 0.05f;
// Colors from the toybox palette (common/Palette.hpp), shared by every window (single object: the
// primal springls and particles).
namespace palette = toybox::palette;
const std::array<float, 3> kSpringlColor = palette::kPrimalSpringl.rgb();
const std::array<float, 3> kParticleColor = palette::kPrimalParticle.rgb();
const std::array<float, 3> kVertexColor = palette::kVertex.rgb();
const std::array<float, 3> kNormalColor = palette::kNormal.rgb();
// The surface outside the regions: a dark gray matcap (assets/matcap), so the lines and dots stand out.
const char *const kBackgroundMatcap = "basic_dark";
const std::array<float, 3> kContourColor = palette::kContour.rgb();
const std::array<float, 3> kArrowColor = palette::kArrow.rgb();
const std::array<float, 3> kReferenceColor = palette::kReference.rgb();
// Region tints by object label (label 1 first; cycled); the background (label 0) is untinted. The sphere scenario
// has one object (label 1): the palette's single-object color; further labels take the region colors.
const std::array<palette::Color, palette::kRegions.size()> kRegionTints = {palette::kObject,     palette::kRegions[1], palette::kRegions[2],
                                                                           palette::kRegions[3], palette::kRegions[4], palette::kRegions[5]};

// One vertex per triangle corner (the surface is drawn unindexed): its barycentric coordinate in the face, and
// the face's three corner labels and band distances (the same for the face's three vertexes).
struct SurfaceVertex {
    float position[3], normal[3], bary[3], distances[3];
    float level;         // the face's subdivision level
    float altitudes[3];  // the face's altitudes: corner k's height over the opposite edge (3D)
    int32_t labels[3];
};
struct LineVertex {
    float position[3], color[3];
};
// A capsule end (see kCapsuleVertex): shape = (diameter in sphere radii, kind, unused).
struct CapsuleVertex {
    float position[3], color[3], shape[3];
};

const char *kSurfaceVertex = R"(#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aBary;
layout(location = 3) in vec3 aDistances;
layout(location = 4) in ivec3 aLabels;
layout(location = 5) in float aLevel;
layout(location = 6) in vec3 aAltitudes;
uniform mat4 uView;
uniform mat4 uProjection;
out vec3 vNormal;
out vec3 vViewPosition;
out vec3 vBary;
flat out vec3 vDistances;
flat out ivec3 vLabels;
flat out float vLevel;
flat out vec3 vAltitudes;
void main() {
    vec4 viewPosition = uView * vec4(aPosition, 1.0);
    vNormal = mat3(uView) * aNormal;  // the view is rigid
    vViewPosition = viewPosition.xyz;
    vBary = aBary;
    vDistances = aDistances;
    vLabels = aLabels;
    vLevel = aLevel;
    vAltitudes = aAltitudes;
    gl_Position = uProjection * viewPosition;
}
)";

// The mesh window's matcap lookup (see mesh/ViewRenderer.cpp), times the tint of the region the fragment lies
// in: of the face's corner labels, the one whose signed value (-d at its own corners, +d at the others',
// interpolated linearly) is smallest, ties to the smaller label. The boundary is then a sharp line at the
// band's linear zero crossing, where the contour's crossings are.
// Isolines: d = |the winning label's value| is the interpolated distance to the nearest region boundary (0 on
// it); thin anti-aliased lines (about 1 px, by fwidth) where d is a positive multiple of uIsoSpacing, up to
// below uBandWidth, in green (shaded with the surface). Distances are
// clamped at the band width: no line at the clamp value, none on faces whose corners are all clamped.
const char *kSurfaceFragment = R"(#version 330 core
in vec3 vNormal;
in vec3 vViewPosition;
in vec3 vBary;
flat in vec3 vDistances;
flat in ivec3 vLabels;
flat in float vLevel;
flat in vec3 vAltitudes;
uniform vec3 uIsolineColor;
uniform vec3 uLevelColors[LEVELS];  // the palette's refinement-depth colors, by subdivision level
uniform float uWireWidth;  // the wireframe's line width on the surface (scene units)
uniform float uIsoWidth;   // the isolines' width on the surface (scene units)
uniform sampler2D uMatcap;
uniform sampler2D uBackgroundMatcap;  // outside the regions: a dark matcap, for contrast
uniform float uWireframe;
uniform float uRegions;
uniform float uIsolines;
uniform float uIsoSpacing;
uniform float uBandWidth;
uniform vec3 uTints[TINTS];
out vec4 fragColor;
vec2 matcapUv(vec3 n, vec3 I) {
    float a = 1.0 / (1.0 + I.z);
    float b = -I.x * I.y * a;
    vec3 tangent = vec3(1.0 - I.x * I.x * a, b, -I.x);
    vec3 bitangent = vec3(b, 1.0 - I.y * I.y * a, -I.y);
    vec2 uv = vec2(dot(tangent, n), dot(bitangent, n)) * 0.496 + 0.5;
    uv.y = 1.0 - uv.y;  // image rows are stored top-down
    return uv;
}
// Coverage of a pixel by a line of half width h at distance x from its center (all on the surface, in scene
// units), the pixel's footprint fw wide there (a box filter): metric lines, anti-aliased, fading as they get
// thinner than a pixel.
float lineCoverage(float x, float h, float fw) {
    fw = max(fw, 1e-12);
    return clamp((min(x + 0.5 * fw, h) - max(x - 0.5 * fw, -h)) / fw, 0.0, 1.0);
}
int regionLabel(out float bestValue) {
    int best = vLabels[0];
    bestValue = 1e30;
    for (int k = 0; k < 3; k++) {
        int label = vLabels[k];
        float value = 0.0;
        for (int i = 0; i < 3; i++) value += vBary[i] * (vLabels[i] == label ? -vDistances[i] : vDistances[i]);
        if (value < bestValue || (value == bestValue && label < best)) {
            best = label;
            bestValue = value;
        }
    }
    return best;
}
void main() {
    vec3 n = normalize(vNormal);
    vec3 toEye = normalize(-vViewPosition);
    if (dot(n, toEye) < 0.0) n = -n;  // two-sided, independent of the winding
    float value;
    int label = regionLabel(value);
    vec3 tint = label > 0 ? uTints[(label - 1) % TINTS] : vec3(1.0);
    vec2 uv = matcapUv(n, toEye);
    vec3 color = label > 0 || uRegions < 0.5 ? texture(uMatcap, uv).rgb * mix(vec3(1.0), tint, uRegions) : texture(uBackgroundMatcap, uv).rgb;
    // fwidth before any branch, so the derivative is defined for the whole quad.
    float d = abs(value);
    float w = fwidth(d);
    // A face whose corners are all clamped is flat: its interpolated d is rounding noise around the clamp,
    // which a line at a multiple of the spacing equal to the clamp would turn into speckle.
    bool clamped = min(min(vDistances[0], vDistances[1]), vDistances[2]) >= uBandWidth * (1.0 - 1e-4);
    if (uIsolines > 0.5 && uIsoSpacing > 0.0 && !clamped && w > 1e-6 * uIsoSpacing && d <= uBandWidth + 0.5 * w) {
        float k = floor(d / uIsoSpacing + 0.5);
        float offset = abs(d - k * uIsoSpacing);
        // Lines strictly inside the band (the clamp value itself is not a distance), uIsoWidth wide on the surface.
        float alpha = (k >= 1.0 && k * uIsoSpacing < uBandWidth * (1.0 - 1e-4)) ? lineCoverage(offset, 0.5 * uIsoWidth, w) : 0.0;
        vec3 lineColor = uIsolineColor * (0.35 + 0.65 * length(color) / sqrt(3.0));  // shaded with the surface
        color = mix(color, lineColor, alpha);
    }
    // Wireframe: the triangulation's edges, uWireWidth wide on the surface (half inside each face: the distance
    // to edge k is the barycentric coordinate k times corner k's altitude), colored by the face's subdivision
    // level below the input mesh.
    if (uWireframe > 0.5) {
        vec3 e = vBary * vAltitudes;
        vec3 fe = fwidth(e);
        float wire = max(max(lineCoverage(e.x, 0.5 * uWireWidth, fe.x), lineCoverage(e.y, 0.5 * uWireWidth, fe.y)),
                         lineCoverage(e.z, 0.5 * uWireWidth, fe.z));
        vec3 wireColor = uLevelColors[int(clamp(vLevel, 0.0, float(LEVELS - 1)))];
        color = mix(color, wireColor, 0.8 * wire);
    }
    fragColor = vec4(color, 1.0);
}
)";

const char *kLineVertex = R"(#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aColor;
uniform mat4 uView;
uniform mat4 uProjection;
uniform float uPull;
out vec3 vColor;
void main() {
    vec4 v = uView * vec4(aPosition, 1.0);
    v.xyz -= normalize(v.xyz) * uPull;
    vColor = aColor;
    gl_Position = uProjection * v;
}
)";

// Each segment becomes a screen-space quad uWidth pixels wide; with uHead > 0 it ends in a triangular head
// (an arrow) of that length in pixels (at most half the segment's).
const char *kLineGeometry = R"(#version 330 core
layout(lines) in;
layout(triangle_strip, max_vertices = 7) out;
in vec3 vColor[];
out vec3 gColor;
uniform vec2 uViewport;
uniform float uWidth;       // pixels (the minimum, with uWorldWidth)
uniform float uWorldWidth;  // > 0: the width on the surface, in scene units (pixels by perspective)
uniform float uPixelsPerUnit;  // pixels per scene unit at view depth 1
uniform float uHead;
void emitAt(vec4 base, vec2 offsetPixels) {
    gl_Position = base + vec4(offsetPixels / (0.5 * uViewport) * base.w, 0.0, 0.0);
    EmitVertex();
}
void main() {
    vec4 a = gl_in[0].gl_Position, b = gl_in[1].gl_Position;
    if (a.w <= 0.0 || b.w <= 0.0) return;  // behind the eye
    vec2 sa = a.xy / a.w * 0.5 * uViewport, sb = b.xy / b.w * 0.5 * uViewport;
    vec2 d = sb - sa;
    float len = length(d);
    if (len < 1e-3) return;
    d /= len;
    vec2 side = vec2(-d.y, d.x);
    vec2 na = side * 0.5 * max(uWidth, uWorldWidth * uPixelsPerUnit / a.w);
    vec2 nb = side * 0.5 * max(uWidth, uWorldWidth * uPixelsPerUnit / b.w);
    float head = min(uHead, 0.5 * len);
    vec4 shaftEnd = head > 0.0 ? mix(a, b, 1.0 - 0.6 * head / len) : b;
    gColor = vColor[0];
    emitAt(a, na);
    emitAt(a, -na);
    gColor = vColor[1];
    emitAt(shaftEnd, nb);
    emitAt(shaftEnd, -nb);
    EndPrimitive();
    if (head > 0.0) {
        vec4 base = mix(a, b, 1.0 - head / len);
        vec2 wing = side * 0.45 * head;
        emitAt(base, wing);
        emitAt(base, -wing);
        emitAt(b, vec2(0.0));
        EndPrimitive();
    }
}
)";

// Flat colored triangles (the arrow heads), through kLineVertex, with an opacity.
const char *kFlatFragment = R"(#version 330 core
in vec3 vColor;
uniform float uAlpha;
out vec4 fragColor;
void main() { fragColor = vec4(vColor, uAlpha); }
)";

const char *kLineFragment = R"(#version 330 core
in vec3 gColor;
out vec4 fragColor;
void main() { fragColor = vec4(gColor, 1.0); }
)";

// Springls, their vertexes and particles in one pass, back to front in buffer order (all springls, then all
// vertexes, then all normals, then all particles): each primitive is a capsule, a segment with round ends, of
// diameter aShape.x sphere radii on the surface (at least uMinPixels on screen); a vertex or particle is a
// capsule of zero length, a dot. aShape.y is the kind (0 springl, 1 vertex, 2 normal, 3 particle), shown
// where uShow says, and pulled toward the eye a layer step per kind, so the kinds occlude in that order.
const char *kCapsuleVertex = R"(#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec3 aShape;
uniform mat4 uView;
uniform mat4 uProjection;
uniform float uPull;
uniform float uLayerPull;  // per kind: springls at uPull, vertexes one step closer, particles two
out vec3 vColor;
out vec2 vShape;
void main() {
    vec4 v = uView * vec4(aPosition, 1.0);
    v.xyz -= normalize(v.xyz) * (uPull + aShape.y * uLayerPull);
    vColor = aColor;
    vShape = aShape.xy;
    gl_Position = uProjection * v;
}
)";

const char *kCapsuleGeometry = R"(#version 330 core
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;
in vec3 vColor[];
in vec2 vShape[];
out vec3 gColor;
noperspective out vec2 gPixel;
flat out vec2 gA;
flat out vec2 gB;
flat out float gRadius;
uniform vec2 uViewport;
uniform float uPixelsPerUnit;  // pixels per scene unit at view depth 1
uniform float uSphereRadius;
uniform float uMinPixels;
uniform vec4 uShow;  // per kind: 1 shown
void emitAt(vec4 base, vec2 basePixel, vec2 offsetPixels) {
    gPixel = basePixel + offsetPixels;
    gl_Position = base + vec4(offsetPixels / (0.5 * uViewport) * base.w, 0.0, 0.0);
    EmitVertex();
}
void main() {
    int kind = int(vShape[0].y + 0.5);
    if (uShow[kind] < 0.5) return;
    vec4 a = gl_in[0].gl_Position, b = gl_in[1].gl_Position;
    if (a.w <= 0.0 || b.w <= 0.0) return;  // behind the eye
    vec2 sa = a.xy / a.w * 0.5 * uViewport, sb = b.xy / b.w * 0.5 * uViewport;
    float r = 0.5 * max(uMinPixels, vShape[0].x * uSphereRadius * uPixelsPerUnit / (0.5 * (a.w + b.w)));
    vec2 d = sb - sa;
    float len = length(d);
    d = len > 1e-4 ? d / len : vec2(1.0, 0.0);
    vec2 n = vec2(-d.y, d.x);
    gColor = vColor[0];
    gA = sa;
    gB = sb;
    gRadius = r;
    emitAt(a, sa, (-d + n) * r);
    emitAt(a, sa, (-d - n) * r);
    emitAt(b, sb, (d + n) * r);
    emitAt(b, sb, (d - n) * r);
    EndPrimitive();
}
)";

const char *kCapsuleFragment = R"(#version 330 core
in vec3 gColor;
noperspective in vec2 gPixel;
flat in vec2 gA;
flat in vec2 gB;
flat in float gRadius;
out vec4 fragColor;
void main() {
    vec2 ab = gB - gA;
    float t = dot(ab, ab) > 0.0 ? clamp(dot(gPixel - gA, ab) / dot(ab, ab), 0.0, 1.0) : 0.0;
    if (length(gPixel - (gA + t * ab)) > gRadius) discard;
    fragColor = vec4(gColor, 1.0);
}
)";

GLuint Compile(GLenum type, const char *source, std::string &error)
{
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (ok == GL_FALSE) {
        char log[2048];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        error = std::string("shader compile failed: ") + log;
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

// geometry may be null.
GLuint BuildProgram(const char *vertex, const char *geometry, const char *fragment, std::string &error)
{
    std::vector<GLuint> shaders;
    const auto add = [&](GLenum type, const char *source) {
        const GLuint shader = Compile(type, source, error);
        if (shader != 0) shaders.push_back(shader);
        return shader != 0;
    };
    const bool compiled = add(GL_VERTEX_SHADER, vertex) && (!geometry || add(GL_GEOMETRY_SHADER, geometry)) && add(GL_FRAGMENT_SHADER, fragment);
    if (!compiled) {
        for (GLuint shader : shaders) glDeleteShader(shader);
        return 0;
    }
    const GLuint program = glCreateProgram();
    for (GLuint shader : shaders) glAttachShader(program, shader);
    glLinkProgram(program);
    for (GLuint shader : shaders) glDeleteShader(shader);
    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (ok == GL_FALSE) {
        char log[2048];
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        error = std::string("program link failed: ") + log;
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

void Copy(const Vec3f &v, float *out)
{
    out[0] = v.x();
    out[1] = v.y();
    out[2] = v.z();
}

void Copy(const std::array<float, 3> &c, float *out) { std::copy(c.begin(), c.end(), out); }

// Uploads interleaved vertexes: every 3 floats of VertexT are one attribute, in order.
template <class VertexT>
void Upload(unsigned int &vao, unsigned int &vbo, const std::vector<VertexT> &vertexes)
{
    if (vao == 0) {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
    }
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    constexpr GLuint attributes = sizeof(VertexT) / (3 * sizeof(float));
    for (GLuint i = 0; i < attributes; i++) {
        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i, 3, GL_FLOAT, GL_FALSE, sizeof(VertexT), reinterpret_cast<const void *>(sizeof(float) * 3 * i));
    }
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertexes.size() * sizeof(VertexT)), vertexes.data(), GL_DYNAMIC_DRAW);
}

void UploadLines(unsigned int &vao, unsigned int &vbo, int &count, const std::vector<LineVertex> &vertexes)
{
    Upload(vao, vbo, vertexes);
    count = static_cast<int>(vertexes.size());
    glBindVertexArray(0);
}

void AddCapsule(std::vector<CapsuleVertex> &out, const Vec3f &a, const Vec3f &b, const std::array<float, 3> &color, float diameter, float kind)
{
    CapsuleVertex v;
    Copy(color, v.color);
    v.shape[0] = diameter;
    v.shape[1] = kind;
    v.shape[2] = 0.0f;
    Copy(a, v.position);
    out.push_back(v);
    Copy(b, v.position);
    out.push_back(v);
}

void AddLine(std::vector<LineVertex> &out, const Vec3f &a, const Vec3f &b, const std::array<float, 3> &ca, const std::array<float, 3> &cb)
{
    LineVertex v;
    Copy(a, v.position);
    Copy(ca, v.color);
    out.push_back(v);
    Copy(b, v.position);
    Copy(cb, v.color);
    out.push_back(v);
}

bool IsClosed(const SphereSnapshot &snapshot)
{
    std::unordered_map<uint64_t, int> edges;
    edges.reserve(snapshot.triangles.size() * 3);
    for (const std::array<uint32_t, 3> &t : snapshot.triangles) {
        for (size_t k = 0; k < 3; k++) {
            const uint64_t a = std::min(t[k], t[(k + 1) % 3]), b = std::max(t[k], t[(k + 1) % 3]);
            edges[(a << 32) | b]++;
        }
    }
    return std::all_of(edges.begin(), edges.end(), [](const auto &e) { return e.second == 2; });
}
}  // namespace

Renderer::~Renderer() { release(); }

void Renderer::release()
{
    if (!mMatcaps.empty()) {
        glDeleteTextures(static_cast<GLsizei>(mMatcaps.size()), mMatcaps.data());
        mMatcaps.clear();
    }
    for (Buffer *buffer : {&mSurface, &mSpringls, &mContour, &mReference, &mArrows}) {
        if (buffer->vao != 0) {
            glDeleteVertexArrays(1, &buffer->vao);
            glDeleteBuffers(1, &buffer->vbo);
        }
        *buffer = Buffer();
    }
    if (mFallback != 0) glDeleteTextures(1, &mFallback);
    mFallback = 0;
    if (mSurfaceProgram != 0) glDeleteProgram(mSurfaceProgram);
    if (mLineProgram != 0) glDeleteProgram(mLineProgram);
    if (mCapsuleProgram != 0) glDeleteProgram(mCapsuleProgram);
    if (mFlatProgram != 0) glDeleteProgram(mFlatProgram);
    mSurfaceProgram = mLineProgram = mCapsuleProgram = mFlatProgram = 0;
    mUploaded = false;
}

bool Renderer::init(const std::vector<viewer::Matcap> &matcaps, std::string &error)
{
    release();
    std::string fragment = kSurfaceFragment;
    for (size_t at = fragment.find("TINTS"); at != std::string::npos; at = fragment.find("TINTS", at))
        fragment.replace(at, 5, std::to_string(kRegionTints.size()));
    for (size_t at = fragment.find("LEVELS"); at != std::string::npos; at = fragment.find("LEVELS", at))
        fragment.replace(at, 6, std::to_string(palette::kLevels.size()));
    mSurfaceProgram = BuildProgram(kSurfaceVertex, nullptr, fragment.c_str(), error);
    if (mSurfaceProgram == 0) return false;
    mLineProgram = BuildProgram(kLineVertex, kLineGeometry, kLineFragment, error);
    if (mLineProgram == 0) return false;
    mCapsuleProgram = BuildProgram(kCapsuleVertex, kCapsuleGeometry, kCapsuleFragment, error);
    if (mCapsuleProgram == 0) return false;
    mFlatProgram = BuildProgram(kLineVertex, nullptr, kFlatFragment, error);
    if (mFlatProgram == 0) return false;
    // 1x1 white stands in when there is no matcap, so shading falls back to the plain tint.
    const uint8_t white[4] = {255, 255, 255, 255};
    glGenTextures(1, &mFallback);
    glBindTexture(GL_TEXTURE_2D, mFallback);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, white);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    mMatcaps.assign(matcaps.size(), 0);
    mBackgroundMatcap = -1;
    for (size_t i = 0; i < matcaps.size(); i++) {
        if (matcaps[i].name == kBackgroundMatcap) mBackgroundMatcap = static_cast<int>(i);
    }
    if (!mMatcaps.empty()) glGenTextures(static_cast<GLsizei>(mMatcaps.size()), mMatcaps.data());
    for (size_t i = 0; i < matcaps.size(); i++) {
        glBindTexture(GL_TEXTURE_2D, mMatcaps[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, matcaps[i].width, matcaps[i].height, 0, GL_RGBA, GL_UNSIGNED_BYTE, matcaps[i].pixels.data());
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    return true;
}

void Renderer::setSnapshot(const SphereSnapshot &snap)
{
    // The surface: area-weighted vertex normals; each face's corners carry its labels and band distances.
    const size_t n = snap.vertices.size();
    const bool hasLabels = snap.labels.size() == n, hasDistances = snap.distances.size() == n;
    std::vector<Vec3f> normals(n, Vec3f::Zero());
    for (const std::array<uint32_t, 3> &t : snap.triangles) {
        if (t[0] >= n || t[1] >= n || t[2] >= n) continue;
        const Vec3f face = (snap.vertices[t[1]] - snap.vertices[t[0]]).cross(snap.vertices[t[2]] - snap.vertices[t[0]]);
        for (uint32_t v : t) normals[v] += face;
    }
    std::vector<SurfaceVertex> vertexes;
    vertexes.reserve(snap.triangles.size() * 3);
    for (size_t f = 0; f < snap.triangles.size(); f++) {
        const std::array<uint32_t, 3> &t = snap.triangles[f];
        if (t[0] >= n || t[1] >= n || t[2] >= n) continue;
        SurfaceVertex v{};
        v.level = static_cast<float>(f < snap.faceLevels.size() ? snap.faceLevels[f] : 0);
        {
            const Vec3f &p0 = snap.vertices[t[0]], &p1 = snap.vertices[t[1]], &p2 = snap.vertices[t[2]];
            const float twiceArea = (p1 - p0).cross(p2 - p0).norm();
            const float opposite[3] = {(p2 - p1).norm(), (p0 - p2).norm(), (p1 - p0).norm()};
            for (size_t k = 0; k < 3; k++) v.altitudes[k] = opposite[k] > 0.0f ? twiceArea / opposite[k] : 0.0f;
        }
        for (size_t k = 0; k < 3; k++) {
            v.labels[k] = hasLabels ? snap.labels[t[k]] : 0;
            v.distances[k] = hasDistances ? snap.distances[t[k]] : 0.0f;
        }
        for (size_t k = 0; k < 3; k++) {
            const uint32_t i = t[k];
            Copy(snap.vertices[i], v.position);
            const float norm = normals[i].norm();
            Copy(norm > 0.0f ? Vec3f(normals[i] / norm) : Vec3f(Vec3f::UnitZ()), v.normal);
            for (size_t c = 0; c < 3; c++) v.bary[c] = c == k ? 1.0f : 0.0f;
            vertexes.push_back(v);
        }
    }
    if (mSurface.vao == 0) {
        glGenVertexArrays(1, &mSurface.vao);
        glGenBuffers(1, &mSurface.vbo);
    }
    glBindVertexArray(mSurface.vao);
    glBindBuffer(GL_ARRAY_BUFFER, mSurface.vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertexes.size() * sizeof(SurfaceVertex)), vertexes.data(), GL_DYNAMIC_DRAW);
    const auto floats = [](GLuint location, size_t offset) {
        glEnableVertexAttribArray(location);
        glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(SurfaceVertex), reinterpret_cast<const void *>(offset));
    };
    floats(0, offsetof(SurfaceVertex, position));
    floats(1, offsetof(SurfaceVertex, normal));
    floats(2, offsetof(SurfaceVertex, bary));
    floats(3, offsetof(SurfaceVertex, distances));
    glEnableVertexAttribArray(5);
    glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, sizeof(SurfaceVertex), reinterpret_cast<const void *>(offsetof(SurfaceVertex, level)));
    floats(6, offsetof(SurfaceVertex, altitudes));
    glEnableVertexAttribArray(4);
    glVertexAttribIPointer(4, 3, GL_INT, sizeof(SurfaceVertex), reinterpret_cast<const void *>(offsetof(SurfaceVertex, labels)));
    mSurface.count = static_cast<int>(vertexes.size());
    glBindVertexArray(0);

    // The sphere's radius: the largest distance of a vertex from the vertexes' centroid.
    if (mRadiusVertices != snap.vertices.size()) {
        Vec3f centroid = Vec3f::Zero();
        for (const Vec3f &p : snap.vertices) centroid += p;
        if (!snap.vertices.empty()) centroid /= static_cast<float>(snap.vertices.size());
        float radius = 0.0f;
        for (const Vec3f &p : snap.vertices) radius = std::max(radius, (p - centroid).norm());
        mSceneRadius = radius;
        mRadiusVertices = snap.vertices.size();
    }
    // One buffer, drawn in one pass in this order: all springls (v0 -> particle -> v1, as the 2D window), then
    // all vertexes, then all normals, then all particles.
    std::vector<CapsuleVertex> capsules;
    const bool hasParticles = snap.particles.size() == snap.springls.size();
    for (size_t i = 0; i < snap.springls.size(); i++) {
        const auto &[a, b] = snap.springls[i];
        if (hasParticles) {
            AddCapsule(capsules, a, snap.particles[i], kSpringlColor, kSpringlWidthRadii, 0.0f);
            AddCapsule(capsules, snap.particles[i], b, kSpringlColor, kSpringlWidthRadii, 0.0f);
        } else {
            AddCapsule(capsules, a, b, kSpringlColor, kSpringlWidthRadii, 0.0f);
        }
    }
    for (const auto &[a, b] : snap.springls) {
        AddCapsule(capsules, a, a, kVertexColor, kVertexDiameterRadii, 1.0f);
        AddCapsule(capsules, b, b, kVertexColor, kVertexDiameterRadii, 1.0f);
    }
    if (hasParticles && snap.normals.size() == snap.particles.size()) {
        for (size_t i = 0; i < snap.particles.size(); i++)
            AddCapsule(capsules, snap.particles[i], snap.particles[i] + kNormalLengthRadii * mSceneRadius * snap.normals[i], kNormalColor, kNormalWidthRadii,
                       2.0f);
    }
    if (hasParticles) {
        for (const Vec3f &p : snap.particles) AddCapsule(capsules, p, p, kParticleColor, kParticleDiameterRadii, 3.0f);
    }
    Upload(mSpringls.vao, mSpringls.vbo, capsules);
    mSpringls.count = static_cast<int>(capsules.size());
    glBindVertexArray(0);

    std::vector<LineVertex> lines;
    for (const auto &segment : snap.contour) AddLine(lines, segment.first, segment.second, kContourColor, kContourColor);
    UploadLines(mContour.vao, mContour.vbo, mContour.count, lines);

    if (mClosedVertices != snap.vertices.size() || mClosedTriangles != snap.triangles.size()) {
        mClosed = IsClosed(snap);
        mClosedVertices = snap.vertices.size();
        mClosedTriangles = snap.triangles.size();
    }
    lines.clear();
    for (const std::vector<Vec3f> &polyline : snap.reference) {
        const size_t m = polyline.size();
        const size_t segments = (mClosed && m > 2) ? m : (m > 0 ? m - 1 : 0);
        for (size_t k = 0; k < segments; k++) AddLine(lines, polyline[k], polyline[(k + 1) % m], kReferenceColor, kReferenceColor);
    }
    UploadLines(mReference.vao, mReference.vbo, mReference.count, lines);

    lines.clear();
    float maxSpeed = 0.0f;
    for (const toybox::SphereArrow &a : snap.arrows) maxSpeed = std::max(maxSpeed, a.speed);
    for (const toybox::SphereArrow &a : snap.arrows) {
        // The head in the tangent plane at `from`: the arrow's direction with its normal component removed.
        const Vec3f n = a.normal;
        const Vec3f d = a.to - a.from;
        const Vec3f along = d - n * n.dot(d);
        const float length = d.norm();
        if (!(along.norm() > 1e-6f * length) || !(length > 0.0f)) continue;
        const Vec3f t = along.normalized(), side = n.cross(t);
        const float h = kArrowHeadLength * length;
        const Vec3f center = a.from + 0.5f * d + kArrowHeadLift * h * n;
        const Vec3f tip = center + 0.5f * h * t, back = center - 0.5f * h * t;
        AddLine(lines, back + 0.5f * kArrowHeadWidth * h * side, back - 0.5f * kArrowHeadWidth * h * side, kArrowColor, kArrowColor);
        AddLine(lines, tip, tip, kArrowColor, kArrowColor);
        lines.pop_back();  // three vertexes per triangle
    }
    UploadLines(mArrows.vao, mArrows.vbo, mArrows.count, lines);

    mLengthScale = snap.lengthScale;
    mBandWidth = snap.bandWidth;
    mUploaded = true;
}

void Renderer::draw(const Layers &layers, const Eigen::Matrix4f &view, const Eigen::Matrix4f &projection, int x, int y, int width, int height)
{
    if (mSurfaceProgram == 0 || mLineProgram == 0 || !mUploaded || width <= 0 || height <= 0) return;
    glViewport(x, y, width, height);
    glEnable(GL_SCISSOR_TEST);
    glScissor(x, y, width, height);
    glDepthMask(GL_TRUE);
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);

    glUseProgram(mSurfaceProgram);
    glUniformMatrix4fv(glGetUniformLocation(mSurfaceProgram, "uView"), 1, GL_FALSE, view.data());
    glUniformMatrix4fv(glGetUniformLocation(mSurfaceProgram, "uProjection"), 1, GL_FALSE, projection.data());
    glUniform1f(glGetUniformLocation(mSurfaceProgram, "uRegions"), layers.regions ? 1.0f : 0.0f);
    glUniform1f(glGetUniformLocation(mSurfaceProgram, "uIsolines"), layers.isolines ? 1.0f : 0.0f);
    glUniform1f(glGetUniformLocation(mSurfaceProgram, "uIsoSpacing"), layers.isoSpacing * mLengthScale);
    glUniform1f(glGetUniformLocation(mSurfaceProgram, "uBandWidth"), mBandWidth);
    glUniform1f(glGetUniformLocation(mSurfaceProgram, "uWireframe"), layers.wireframe ? 1.0f : 0.0f);
    glUniform1f(glGetUniformLocation(mSurfaceProgram, "uWireWidth"), kWireWidthRadii * mSceneRadius);
    glUniform3f(glGetUniformLocation(mSurfaceProgram, "uIsolineColor"), palette::kIsoline.r, palette::kIsoline.g, palette::kIsoline.b);
    std::array<float, 3 * palette::kLevels.size()> levelColors;
    for (size_t i = 0; i < palette::kLevels.size(); i++) {
        const std::array<float, 3> c = palette::kLevels[i].rgb();
        std::copy(c.begin(), c.end(), levelColors.begin() + 3 * i);
    }
    glUniform3fv(glGetUniformLocation(mSurfaceProgram, "uLevelColors"), static_cast<GLsizei>(palette::kLevels.size()), levelColors.data());
    glUniform1f(glGetUniformLocation(mSurfaceProgram, "uIsoWidth"), kIsoWidthRadii * mSceneRadius);
    glUniform1i(glGetUniformLocation(mSurfaceProgram, "uMatcap"), 0);
    glActiveTexture(GL_TEXTURE0);
    const bool hasMatcap = layers.matcap >= 0 && static_cast<size_t>(layers.matcap) < mMatcaps.size();
    glBindTexture(GL_TEXTURE_2D, hasMatcap ? mMatcaps[static_cast<size_t>(layers.matcap)] : mFallback);
    glUniform1i(glGetUniformLocation(mSurfaceProgram, "uBackgroundMatcap"), 1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, mBackgroundMatcap >= 0 ? mMatcaps[static_cast<size_t>(mBackgroundMatcap)] : mFallback);
    glActiveTexture(GL_TEXTURE0);
    std::array<float, 3 * kRegionTints.size()> tints;
    for (size_t i = 0; i < kRegionTints.size(); i++) {
        const std::array<float, 3> c = kRegionTints[i].rgb();
        std::copy(c.begin(), c.end(), tints.begin() + 3 * i);
    }
    glUniform3fv(glGetUniformLocation(mSurfaceProgram, "uTints"), static_cast<GLsizei>(kRegionTints.size()), tints.data());
    if (mSurface.count > 0) {
        glBindVertexArray(mSurface.vao);
        glDrawArrays(GL_TRIANGLES, 0, mSurface.count);
    }

    // Line widths scale with the framebuffer: a 1600-pixel-wide view uses them as given.
    const float scale = std::max(1.0f, static_cast<float>(std::max(width, height)) / 1600.0f);
    glUseProgram(mLineProgram);
    glUniformMatrix4fv(glGetUniformLocation(mLineProgram, "uView"), 1, GL_FALSE, view.data());
    glUniformMatrix4fv(glGetUniformLocation(mLineProgram, "uProjection"), 1, GL_FALSE, projection.data());
    glUniform1f(glGetUniformLocation(mLineProgram, "uPull"), kPullLengthScales * mLengthScale);
    glUniform2f(glGetUniformLocation(mLineProgram, "uViewport"), static_cast<float>(width), static_cast<float>(height));
    // Pixels per scene unit at view depth 1 (gl_Position.w is the view depth): the projection's y scale.
    const float pixelsPerUnit = projection(1, 1) * 0.5f * static_cast<float>(height);
    glUniform1f(glGetUniformLocation(mLineProgram, "uPixelsPerUnit"), pixelsPerUnit);
    const GLint worldWidthLocation = glGetUniformLocation(mLineProgram, "uWorldWidth");
    const GLint widthLocation = glGetUniformLocation(mLineProgram, "uWidth");
    const GLint headLocation = glGetUniformLocation(mLineProgram, "uHead");
    const auto drawLines = [&](const Buffer &buffer, bool visible, float lineWidth, float head, float worldWidth = 0.0f) {
        if (!visible || buffer.count == 0) return;
        glUniform1f(widthLocation, lineWidth * scale);
        glUniform1f(worldWidthLocation, worldWidth);
        glUniform1f(headLocation, head * scale);
        glBindVertexArray(buffer.vao);
        glDrawArrays(GL_LINES, 0, buffer.count);
    };
    // The reference; the contour; then springls, vertexes, normals and particles in one pass, on top of the
    // contour (each layer pulled a step closer to the eye, so the depth test keeps that order); the arrows last.
    drawLines(mReference, layers.reference, kReferenceWidth, 0.0f);
    drawLines(mContour, layers.contour, kContourWidth, 0.0f);
    if (mSpringls.count > 0 && (layers.springls || layers.vertices || layers.normals || layers.particles)) {
        glUseProgram(mCapsuleProgram);
        glUniformMatrix4fv(glGetUniformLocation(mCapsuleProgram, "uView"), 1, GL_FALSE, view.data());
        glUniformMatrix4fv(glGetUniformLocation(mCapsuleProgram, "uProjection"), 1, GL_FALSE, projection.data());
        glUniform1f(glGetUniformLocation(mCapsuleProgram, "uPull"), (kPullLengthScales + kLayerPullLengthScales) * mLengthScale);
        glUniform1f(glGetUniformLocation(mCapsuleProgram, "uLayerPull"), kLayerPullLengthScales * mLengthScale);
        glUniform2f(glGetUniformLocation(mCapsuleProgram, "uViewport"), static_cast<float>(width), static_cast<float>(height));
        glUniform1f(glGetUniformLocation(mCapsuleProgram, "uPixelsPerUnit"), pixelsPerUnit);
        glUniform1f(glGetUniformLocation(mCapsuleProgram, "uSphereRadius"), mSceneRadius);
        glUniform1f(glGetUniformLocation(mCapsuleProgram, "uMinPixels"), kMinElementPixels * scale);
        glUniform4f(glGetUniformLocation(mCapsuleProgram, "uShow"), layers.springls ? 1.0f : 0.0f, layers.vertices ? 1.0f : 0.0f, layers.normals ? 1.0f : 0.0f,
                    layers.particles ? 1.0f : 0.0f);
        glBindVertexArray(mSpringls.vao);
        glDrawArrays(GL_LINES, 0, mSpringls.count);
        glUseProgram(mLineProgram);
    }
    // Arrows: flat heads on the surface, pulled toward the eye past every springl layer, so they lie on top of
    // the region, contour and springls; the depth test still hides those on the far side of the surface.
    if (layers.arrows && mArrows.count > 0) {
        glUseProgram(mFlatProgram);
        glUniformMatrix4fv(glGetUniformLocation(mFlatProgram, "uView"), 1, GL_FALSE, view.data());
        glUniformMatrix4fv(glGetUniformLocation(mFlatProgram, "uProjection"), 1, GL_FALSE, projection.data());
        glUniform1f(glGetUniformLocation(mFlatProgram, "uPull"), (kPullLengthScales + 4.0f * kLayerPullLengthScales) * mLengthScale);
        glUniform1f(glGetUniformLocation(mFlatProgram, "uAlpha"), kArrowOpacity);
        // Translucent: blended, tested against the depth buffer but not written to it.
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_FALSE);
        glBindVertexArray(mArrows.vao);
        glDrawArrays(GL_TRIANGLES, 0, mArrows.count);
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
    }

    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_DEPTH_TEST);
}
}  // namespace sphere_gui
