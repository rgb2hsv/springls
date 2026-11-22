#include "ViewRenderer.hpp"

#include <GL.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace viewer
{
using springls::Vec3;
namespace
{
// Layer stacking as polygon offsets (negative = toward the viewer): iso-surface 0, dual springls, then primal.
constexpr float kDualOffset = -2.0f, kPrimalOffset = -4.0f;
// Particles are points, which polygon offset does not affect: they are pulled toward the eye in view space by
// this fraction of the finest cell (well under a springl's size), so they sit in front of their springls.
constexpr float kParticlePullCells = 0.25f;
constexpr float kInputOpacity = 0.4f;
const std::array<float, 3> kInputColor = toybox::palette::kInputMesh.rgb();

// One vertex layout for every surface layer: the springl arm vertexes are scaled about aCenter (the particle;
// the vertex itself for the iso-surface and input mesh) by uScale in the shader, so changing the scale does not
// rebuild the buffers.
struct Vertex {
    float position[3], normal[3], center[3];
};

const char *kSurfaceVertex = R"(#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aCenter;
uniform mat4 uView;
uniform mat4 uProjection;
uniform float uScale;
out vec3 vNormal;
out vec3 vViewPosition;
void main() {
    vec4 viewPosition = uView * vec4(aCenter + (aPosition - aCenter) * uScale, 1.0);
    vNormal = mat3(uView) * aNormal;  // the view is rigid
    vViewPosition = viewPosition.xyz;
    gl_Position = uProjection * viewPosition;
}
)";

// matcapColor is the meshlets lookup: the normal is expressed in an orthonormal basis built around the
// direction to the eye rather than the view axes, which avoids stretching under perspective. The matcap
// texture holds sRGB-encoded values and the framebuffer is not sRGB, so the sample is used as is (meshlets
// samples an sRGB texture and re-encodes explicitly with linearToSrgb into a UNORM swapchain).
const char *kSurfaceFragment = R"(#version 330 core
in vec3 vNormal;
in vec3 vViewPosition;
uniform sampler2D uMatcap;
uniform vec3 uTint;
uniform float uOpacity;
out vec4 fragColor;
vec3 matcapColor(vec3 n, vec3 I) {
    float a = 1.0 / (1.0 + I.z);
    float b = -I.x * I.y * a;
    vec3 tangent = vec3(1.0 - I.x * I.x * a, b, -I.x);
    vec3 bitangent = vec3(b, 1.0 - I.y * I.y * a, -I.y);
    vec2 uv = vec2(dot(tangent, n), dot(bitangent, n)) * 0.496 + 0.5;
    uv.y = 1.0 - uv.y;  // image rows are stored top-down
    return texture(uMatcap, uv).rgb;
}
void main() {
    vec3 n = normalize(vNormal);
    vec3 toEye = normalize(-vViewPosition);
    if (dot(n, toEye) < 0.0) n = -n;  // two-sided, independent of the winding
    fragColor = vec4(matcapColor(n, toEye) * uTint, uOpacity);
}
)";

const char *kPointVertex = R"(#version 330 core
layout(location = 0) in vec3 aPosition;
uniform mat4 uView;
uniform mat4 uProjection;
uniform float uSize;         // diameter in world units
uniform float uPixelsPerUnit; // pixels per world unit at view depth 1
uniform float uPull;
void main() {
    vec4 v = uView * vec4(aPosition, 1.0);
    v.xyz -= normalize(v.xyz) * uPull;
    gl_Position = uProjection * v;
    gl_PointSize = max(uSize * uPixelsPerUnit / max(-v.z, 1e-6), 1.0);
}
)";

const char *kPointFragment = R"(#version 330 core
uniform vec3 uColor;
out vec4 fragColor;
void main() {
    vec2 d = 2.0 * gl_PointCoord - 1.0;
    if (dot(d, d) > 1.0) discard;  // round dots
    fragColor = vec4(uColor, 1.0);
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

GLuint BuildProgram(const char *vertex, const char *fragment, std::string &error)
{
    const GLuint vs = Compile(GL_VERTEX_SHADER, vertex, error);
    if (vs == 0) {
        return 0;
    }
    const GLuint fs = Compile(GL_FRAGMENT_SHADER, fragment, error);
    if (fs == 0) {
        glDeleteShader(vs);
        return 0;
    }
    const GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);
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

void Copy(const Vec3 &v, float *out)
{
    out[0] = v.x();
    out[1] = v.y();
    out[2] = v.z();
}

template <class MeshT>
void UploadMesh(MeshT &mesh, const std::vector<Vertex> &vertexes, const std::vector<uint32_t> &indexes)
{
    if (mesh.vao == 0) {
        glGenVertexArrays(1, &mesh.vao);
        glGenBuffers(1, &mesh.vbo);
        glGenBuffers(1, &mesh.ebo);
    }
    glBindVertexArray(mesh.vao);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo);
    for (GLuint i = 0; i < 3; i++) {
        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<const void *>(sizeof(float) * 3 * i));
    }
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertexes.size() * sizeof(Vertex)), vertexes.data(), GL_STATIC_DRAW);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(indexes.size() * sizeof(uint32_t)), indexes.data(), GL_STATIC_DRAW);
    mesh.indexCount = static_cast<int>(indexes.size());
    glBindVertexArray(0);
}

// Triangles (polygons fanned from their first vertex) with the mesh's vertex normals.
template <class MeshT>
void BuildSurface(const springls::SurfaceMesh3D &surface, MeshT &mesh)
{
    std::vector<Vertex> vertexes(surface.vertexes.size());
    const bool hasNormals = surface.normals.size() == surface.vertexes.size();
    for (size_t i = 0; i < vertexes.size(); i++) {
        Copy(surface.vertexes[i], vertexes[i].position);
        Copy(surface.vertexes[i], vertexes[i].center);
        Copy(hasNormals ? surface.normals[i] : Vec3(0.0f, 0.0f, 1.0f), vertexes[i].normal);
    }
    const size_t K = static_cast<size_t>(surface.elementSize());
    const size_t faces = surface.faceCount();
    std::vector<uint32_t> indexes;
    indexes.reserve(faces * (K - 2) * 3);
    for (size_t f = 0; f < faces; f++) {
        const uint32_t *face = surface.face(f);
        for (size_t k = 1; k + 1 < K; k++) {
            indexes.push_back(face[0]);
            indexes.push_back(face[k]);
            indexes.push_back(face[k + 1]);
        }
    }
    UploadMesh(mesh, vertexes, indexes);
}

template <class PointsT>
void UploadPoints(PointsT &points, const std::vector<float> &positions)
{
    if (points.vao == 0) {
        glGenVertexArrays(1, &points.vao);
        glGenBuffers(1, &points.vbo);
    }
    glBindVertexArray(points.vao);
    glBindBuffer(GL_ARRAY_BUFFER, points.vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(positions.size() * sizeof(float)), positions.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    points.count = static_cast<int>(positions.size() / 3);
    glBindVertexArray(0);
}
}  // namespace

ViewRenderer::~ViewRenderer() { release(); }

void ViewRenderer::release()
{
    if (!mMatcaps.empty()) {
        glDeleteTextures(static_cast<GLsizei>(mMatcaps.size()), mMatcaps.data());
        mMatcaps.clear();
    }
    for (Mesh *mesh : {&mIso, &mInput, &mPrimal, &mDual}) {
        if (mesh->vao != 0) {
            glDeleteVertexArrays(1, &mesh->vao);
            glDeleteBuffers(1, &mesh->vbo);
            glDeleteBuffers(1, &mesh->ebo);
        }
        *mesh = Mesh();
    }
    for (Points *points : {&mPrimalParticles, &mDualParticles}) {
        if (points->vao != 0) {
            glDeleteVertexArrays(1, &points->vao);
            glDeleteBuffers(1, &points->vbo);
        }
        *points = Points();
    }
    if (mFallback != 0) {
        glDeleteTextures(1, &mFallback);
        mFallback = 0;
    }
    if (mSurfaceProgram != 0) {
        glDeleteProgram(mSurfaceProgram);
    }
    if (mPointProgram != 0) {
        glDeleteProgram(mPointProgram);
    }
    mSurfaceProgram = mPointProgram = 0;
    mUploaded = false;
}

bool ViewRenderer::init(const std::vector<Matcap> &matcaps, std::string &error)
{
    release();
    mSurfaceProgram = BuildProgram(kSurfaceVertex, kSurfaceFragment, error);
    if (mSurfaceProgram == 0) {
        return false;
    }
    mPointProgram = BuildProgram(kPointVertex, kPointFragment, error);
    if (mPointProgram == 0) {
        return false;
    }
    // 1x1 white stands in when there is no matcap, so shading falls back to the plain tint.
    const uint8_t white[4] = {255, 255, 255, 255};
    glGenTextures(1, &mFallback);
    glBindTexture(GL_TEXTURE_2D, mFallback);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, white);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    mMatcaps.assign(matcaps.size(), 0);
    if (!mMatcaps.empty()) {
        glGenTextures(static_cast<GLsizei>(mMatcaps.size()), mMatcaps.data());
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
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

unsigned int ViewRenderer::matcapTexture(int index) const
{
    return (index >= 0 && static_cast<size_t>(index) < mMatcaps.size()) ? mMatcaps[static_cast<size_t>(index)] : 0u;
}

void ViewRenderer::upload(const SessionSnapshot &snapshot)
{
    const springls::Manifold3D &m = snapshot.manifold;
    BuildSurface(m.isoSurface, mIso);
    BuildSurface(snapshot.inputMesh, mInput);

    // Springls in one pass: each becomes its particle (vertex 0) followed by its arm vertexes, fanned from
    // the particle, in the arrays of its constellation (0 primal, 1 dual).
    const size_t count = std::min(m.springlCount(), m.springlParticleNormals.size());
    const bool tagged = m.springlConstellations.size() == m.springlCount();
    std::array<std::vector<Vertex>, 2> vertexes;
    std::array<std::vector<uint32_t>, 2> indexes;
    std::array<std::vector<float>, 2> particles;
    const size_t expected = static_cast<size_t>(m.elementSize());
    for (size_t w = 0; w < 2; w++) {
        vertexes[w].reserve(count * (expected + 1));
        indexes[w].reserve(count * 3 * expected);
        particles[w].reserve(count * 3);
    }
    for (size_t n = 0; n < count; n++) {
        const size_t w = (tagged && m.springlConstellations[n] != 0) ? 1 : 0;
        std::vector<Vertex> &vs = vertexes[w];
        std::vector<uint32_t> &is = indexes[w];
        const Vec3 &particle = m.springlParticles[n];
        const uint32_t base = static_cast<uint32_t>(vs.size());
        const uint32_t size = static_cast<uint32_t>(m.springlSize(n));
        const Vec3 *arm = m.springl(n);
        Vertex v;
        Copy(particle, v.position);
        Copy(m.springlParticleNormals[n], v.normal);
        Copy(particle, v.center);
        vs.push_back(v);
        for (uint32_t k = 0; k < size; k++) {
            Copy(arm[k], v.position);
            vs.push_back(v);
        }
        for (uint32_t k = 0; k < size; k++) {
            is.push_back(base);
            is.push_back(base + 1 + k);
            is.push_back(base + 1 + (k + 1) % size);
        }
        particles[w].insert(particles[w].end(), {particle.x(), particle.y(), particle.z()});
    }
    UploadMesh(mPrimal, vertexes[0], indexes[0]);
    UploadMesh(mDual, vertexes[1], indexes[1]);
    UploadPoints(mPrimalParticles, particles[0]);
    UploadPoints(mDualParticles, particles[1]);
    mStamp = snapshot.stamp;
    mUploaded = true;
}

void ViewRenderer::draw(const SessionSnapshot &snapshot, const Layers &layers, const OrbitCamera &camera, int x, int y, int width, int height)
{
    if (mSurfaceProgram == 0 || mPointProgram == 0 || width <= 0 || height <= 0) {
        return;
    }
    if (!mUploaded || snapshot.stamp != mStamp) {
        upload(snapshot);
    }

    glViewport(x, y, width, height);
    glEnable(GL_SCISSOR_TEST);
    glScissor(x, y, width, height);
    glDepthMask(GL_TRUE);
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDisable(GL_CULL_FACE);

    const Eigen::Matrix4f view = camera.view();
    const Eigen::Matrix4f projection = camera.projection(static_cast<float>(width) / static_cast<float>(height));

    glUseProgram(mSurfaceProgram);
    glUniformMatrix4fv(glGetUniformLocation(mSurfaceProgram, "uView"), 1, GL_FALSE, view.data());
    glUniformMatrix4fv(glGetUniformLocation(mSurfaceProgram, "uProjection"), 1, GL_FALSE, projection.data());
    const GLint scaleLocation = glGetUniformLocation(mSurfaceProgram, "uScale");
    const GLint tintLocation = glGetUniformLocation(mSurfaceProgram, "uTint");
    const GLint opacityLocation = glGetUniformLocation(mSurfaceProgram, "uOpacity");
    glUniform1i(glGetUniformLocation(mSurfaceProgram, "uMatcap"), 0);
    glActiveTexture(GL_TEXTURE0);
    const unsigned int matcap = matcapTexture(layers.matcap);
    glBindTexture(GL_TEXTURE_2D, matcap != 0 ? matcap : mFallback);

    const auto drawMesh = [&](const Mesh &mesh, const float *tint, float scale, float opacity, float offset) {
        if (mesh.indexCount == 0) {
            return;
        }
        glUniform1f(scaleLocation, scale);
        glUniform3fv(tintLocation, 1, tint);
        glUniform1f(opacityLocation, opacity);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(offset, offset);
        glBindVertexArray(mesh.vao);
        glDrawElements(GL_TRIANGLES, mesh.indexCount, GL_UNSIGNED_INT, nullptr);
    };
    const float white[3] = {1.0f, 1.0f, 1.0f};

    glDisable(GL_BLEND);
    if (layers.isoSurface) {
        drawMesh(mIso, white, 1.0f, 1.0f, 0.0f);
    }
    if (layers.dual) {
        drawMesh(mDual, layers.dualColor.data(), layers.springlScale, 1.0f, kDualOffset);
    }
    if (layers.primal) {
        drawMesh(mPrimal, layers.primalColor.data(), layers.springlScale, 1.0f, kPrimalOffset);
    }
    glDisable(GL_POLYGON_OFFSET_FILL);

    if (layers.particles) {
        glUseProgram(mPointProgram);
        glEnable(GL_PROGRAM_POINT_SIZE);
        const float finestCell = std::ldexp(1.0f, -snapshot.fit.maxDepth);
        glUniform1f(glGetUniformLocation(mPointProgram, "uPull"), kParticlePullCells * finestCell);
        glUniform1f(glGetUniformLocation(mPointProgram, "uPixelsPerUnit"), 0.5f * static_cast<float>(height) * projection(1, 1));
        glUniformMatrix4fv(glGetUniformLocation(mPointProgram, "uView"), 1, GL_FALSE, view.data());
        glUniformMatrix4fv(glGetUniformLocation(mPointProgram, "uProjection"), 1, GL_FALSE, projection.data());
        glUniform1f(glGetUniformLocation(mPointProgram, "uSize"), layers.particleSize * finestCell);
        const GLint colorLocation = glGetUniformLocation(mPointProgram, "uColor");
        const auto drawPoints = [&](const Points &points, const std::array<float, 3> &color, bool visible) {
            if (!visible || points.count == 0) {
                return;
            }
            glUniform3fv(colorLocation, 1, color.data());
            glBindVertexArray(points.vao);
            glDrawArrays(GL_POINTS, 0, points.count);
        };
        // Particles in the palette's particle colors (as every window).
        drawPoints(mDualParticles, toybox::palette::kDualParticle.rgb(), layers.dual);
        drawPoints(mPrimalParticles, toybox::palette::kPrimalParticle.rgb(), layers.primal);
        glDisable(GL_PROGRAM_POINT_SIZE);
        glUseProgram(mSurfaceProgram);
    }

    // The translucent input mesh last: blended, and it does not write depth.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    if (layers.inputMesh) {
        drawMesh(mInput, kInputColor.data(), 1.0f, kInputOpacity, 0.0f);
    }

    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
    glDisable(GL_POLYGON_OFFSET_FILL);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_DEPTH_TEST);
}
}  // namespace viewer
