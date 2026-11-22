// The intrinsic-springls runner: command line, CSV metrics and per-step PLY frames.
#include "Check.hpp"
#include "IntrinsicTestMeshes.hpp"

#include <intrinsic-springls/Runner.hpp>
#include <intrinsic/IntrinsicTriangulation.hpp>
#include <io/Ply.hpp>
#include <mesh/Icosphere.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

using namespace springls;
namespace fs = std::filesystem;

namespace
{
fs::path ScratchDirectory(const std::string &name)
{
    const fs::path dir = fs::temp_directory_path() / ("springls-" + name);
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

// Writes a test mesh as a PLY file.
bool WriteMesh(const springls_test::TestMesh &mesh, const fs::path &file)
{
    PlyTable table;
    PlyElement vertex{"vertex", mesh.vertices.size(), {}};
    for (const char *axis : {"x", "y", "z"}) vertex.properties.push_back(PlyProperty{axis, PlyType::Float32, false, {}, {0}});
    for (const Vec3 &p : mesh.vertices) {
        for (int k = 0; k < 3; k++) vertex.properties[k].values.push_back(p[k]);
    }
    PlyElement face{"face", mesh.triangles.size(), {PlyProperty{"vertex_indices", PlyType::Int32, true, {}, {0}}}};
    for (const auto &t : mesh.triangles) {
        for (uint32_t v : t) face.properties[0].values.push_back(v);
        face.properties[0].listStart.push_back(static_cast<uint32_t>(face.properties[0].values.size()));
    }
    table.elements = {vertex, face};
    std::ofstream out(file, std::ios::binary);
    std::string error;
    return WritePly(out, table, error);
}

bool ReadTable(const fs::path &file, PlyTable &table)
{
    std::ifstream in(file, std::ios::binary);
    std::string error;
    return in && ReadPly(in, table, error);
}

std::vector<std::string> Lines(const fs::path &file)
{
    std::ifstream in(file);
    std::vector<std::string> lines;
    for (std::string line; std::getline(in, line);) lines.push_back(line);
    return lines;
}
}  // namespace

TEST(runner_parses_arguments)
{
    const char *argv[] = {
        "intrinsic-springls", "--mesh", "m.ply", "--object", "star",    "--speed", "2",     "--loop-distance",  "1.2", "--radius", "0.3",
        "--frames",           "out",    "--csv", "m.csv",    "--steps", "7",       "--set", "fillDistance=0.4", "-j",  "3",        "--verbose"};
    intrinsic::RunnerOptions options;
    std::string error;
    CHECK(intrinsic::ParseArguments(static_cast<int>(std::size(argv)), const_cast<char **>(argv), options, error));
    CHECK(options.session.mesh == "m.ply" && options.session.config.object == LoopObject::Star && options.session.config.speed == 2.0);
    CHECK(options.session.config.loopDistance == 1.2 && options.session.config.radius == 0.3);
    CHECK(options.frames == "out" && options.csv == "m.csv" && options.steps == 7 && options.threads == 3 && options.session.verbose);
    CHECK(options.session.tuning.size() == 1 && options.session.tuning[0].first == "fillDistance" && options.session.tuning[0].second == 0.4);
    // Defaults: the metasphere and the puzzle piece.
    const char *bare[] = {"intrinsic-springls"};
    intrinsic::RunnerOptions defaults;
    CHECK(intrinsic::ParseArguments(1, const_cast<char **>(bare), defaults, error));
    CHECK(defaults.session.mesh.filename() == "metasphere.ply" && defaults.session.config.object == LoopObject::PuzzlePiece && defaults.steps < 0);
    // The automatic loop distance is a fraction of the scenario's (whose full loop on the metasphere takes hours).
    CHECK(defaults.session.config.loopFraction == intrinsic::kDefaultLoopFraction && intrinsic::kDefaultLoopFraction < 1.0);
    const char *pictures[] = {"intrinsic-springls", "--images", "img", "--every", "5", "--image-size", "300"};
    intrinsic::RunnerOptions withImages;
    CHECK(intrinsic::ParseArguments(7, const_cast<char **>(pictures), withImages, error));
    CHECK(withImages.images == "img" && withImages.every == 5 && withImages.imageSize == 300 && defaults.every == 1);
    const char *bad[] = {"intrinsic-springls", "--object", "teapot"};
    CHECK(!intrinsic::ParseArguments(3, const_cast<char **>(bad), defaults, error));
    const char *badSet[] = {"intrinsic-springls", "--set", "noSuchKey=1"};
    CHECK(!intrinsic::ParseArguments(3, const_cast<char **>(badSet), defaults, error));
}

TEST(runner_writes_csv_and_frames)
{
    const fs::path dir = ScratchDirectory("intrinsic-runner");
    const springls_test::TestMesh mesh = springls_test::Icosphere(4);
    CHECK(WriteMesh(mesh, dir / "sphere.ply"));
    intrinsic::RunnerOptions options;
    options.session.mesh = dir / "sphere.ply";
    options.session.config.object = LoopObject::Star;
    options.session.config.radius = 0.4;
    options.frames = dir / "frames";
    options.csv = dir / "metrics.csv";
    options.steps = 2;
    intrinsic::RunnerResult result;
    std::string error;
    CHECK(intrinsic::Run(options, result, error));
    if (!error.empty()) std::printf("  run: %s\n", error.c_str());
    CHECK(result.steps == 2);
    // CSV: header, then one row (the one region) for steps 0..2.
    const auto lines = Lines(options.csv);
    CHECK(!lines.empty() && lines[0] ==
                                "step,time,label,area,relative_area_error,mean_shape_error,max_shape_error,springls,reference_mean_distance,reference_max_"
                                "distance,reference_max_gap");
    CHECK(lines.size() == 1 + 3);
    if (lines.size() > 1) CHECK(lines[1].rfind("0,0,1,", 0) == 0);
    // Frames: the input mesh with region colors, and the springls as 3D polylines.
    PlyTable frame, springls;
    CHECK(ReadTable(options.frames / "frame_000002.ply", frame));
    CHECK(ReadTable(options.frames / "springls_000002.ply", springls));
    const PlyElement *vertex = frame.find("vertex"), *face = frame.find("face");
    CHECK(vertex && face && vertex->count == mesh.vertices.size() && face->count == mesh.triangles.size());
    if (vertex) CHECK(vertex->find("red") && vertex->find("green") && vertex->find("blue"));
    const PlyElement *points = springls.find("vertex"), *edges = springls.find("edge");
    CHECK(points && edges && points->count == 3 * result.final.springls && edges->count == 2 * result.final.springls);
    if (points && points->count > 0) {
        // Springl points lie on the sphere.
        const PlyProperty *x = points->find("x"), *y = points->find("y"), *z = points->find("z");
        double worst = 0.0;
        for (size_t k = 0; k < points->count; k++)
            worst = std::max(worst, std::abs(std::sqrt(x->values[k] * x->values[k] + y->values[k] * y->values[k] + z->values[k] * z->values[k]) - 1.0));
        CHECK(worst < 0.01);
    }
    CHECK(fs::exists(options.frames / "frame_000000.ply"));
    fs::remove_all(dir);
}

TEST(runner_parses_max_level)
{
    std::string error;
    const char *bare[] = {"intrinsic-springls"};
    intrinsic::RunnerOptions defaults;
    CHECK(intrinsic::ParseArguments(1, const_cast<char **>(bare), defaults, error));
    CHECK(defaults.session.maxLevel == 3 && IntrinsicSpringls::kDefaultMaxLevel == 3);
    const char *levels[] = {"intrinsic-springls", "--max-level", "2"};
    intrinsic::RunnerOptions withLevels;
    CHECK(intrinsic::ParseArguments(3, const_cast<char **>(levels), withLevels, error));
    CHECK(withLevels.session.maxLevel == 2);
    const char *negative[] = {"intrinsic-springls", "--max-level", "-1"};
    CHECK(!intrinsic::ParseArguments(3, const_cast<char **>(negative), withLevels, error));
    // The static Loop subdivision of the swept cap is gone: refinement is adaptive.
    const char *subdivide[] = {"intrinsic-springls", "--subdivide", "2"};
    CHECK(!intrinsic::ParseArguments(3, const_cast<char **>(subdivide), withLevels, error));
}

TEST(runner_refines_near_the_object)
{
    const fs::path dir = ScratchDirectory("intrinsic-runner-refine");
    const springls_test::TestMesh mesh = springls_test::Icosphere(4);
    CHECK(WriteMesh(mesh, dir / "sphere.ply"));
    intrinsic::RunnerOptions options;
    options.session.mesh = dir / "sphere.ply";
    options.session.config.object = LoopObject::Star;
    options.session.config.loopDistance = 0.6;
    options.session.config.radius = 0.2;
    options.session.maxLevel = 0;
    options.steps = 1;
    intrinsic::RunnerResult coarse, fine;
    std::string error;
    CHECK(intrinsic::Run(options, coarse, error));
    options.session.maxLevel = 2;
    CHECK(intrinsic::Run(options, fine, error));
    if (!error.empty()) std::printf("  run: %s\n", error.c_str());
    std::printf("  faces %zu -> %zu, length scale %.4f -> %.4f, springls %zu -> %zu\n", coarse.faces, fine.faces, coarse.lengthScale, fine.lengthScale,
                coarse.final.springls, fine.final.springls);
    // Two levels near the object: a quarter of the length scale, far fewer faces than uniform.
    CHECK(fine.faces > coarse.faces && fine.faces < 16 * coarse.faces / 2);
    CHECK(std::abs(fine.lengthScale - 0.25 * coarse.lengthScale) < 1e-12);
    CHECK(fine.final.springls > 2 * coarse.final.springls);
    CHECK(std::abs(fine.loopDistance - coarse.loopDistance) < 1e-12 && std::abs(fine.radius - coarse.radius) < 1e-12);
    fs::remove_all(dir);
}

TEST(runner_renders_images_every_n_steps)
{
    const fs::path dir = ScratchDirectory("intrinsic-runner-images");
    const springls_test::TestMesh mesh = springls_test::Icosphere(4);
    CHECK(WriteMesh(mesh, dir / "sphere.ply"));
    intrinsic::RunnerOptions options;
    options.session.mesh = dir / "sphere.ply";
    options.session.config.object = LoopObject::Star;
    options.session.config.radius = 0.4;
    options.images = dir / "images";
    options.every = 2;
    options.imageSize = 96;
    options.steps = 3;
    intrinsic::RunnerResult result;
    std::string error;
    CHECK(intrinsic::Run(options, result, error));
    // Steps 0 and 2, and the last step.
    for (const char *name : {"image_000000.png", "image_000002.png", "image_000003.png"}) {
        std::ifstream in(options.images / name, std::ios::binary);
        unsigned char header[24] = {};
        in.read(reinterpret_cast<char *>(header), 24);
        CHECK(in && header[1] == 'P' && header[2] == 'N' && header[3] == 'G');
        const unsigned width = (header[16] << 24) | (header[17] << 16) | (header[18] << 8) | header[19];
        CHECK(width == 96);
    }
    CHECK(!fs::exists(options.images / "image_000001.png"));
    fs::remove_all(dir);
}

TEST(icosphere_counts_and_radius)
{
    const LoopMesh mesh = MakeIcosphere(5);
    CHECK(mesh.vertices.size() == 10242 && mesh.triangles.size() == 20480);
    double worst = 0.0;
    for (const Vec3 &p : mesh.vertices) worst = std::max(worst, std::abs(static_cast<double>(p.norm()) - 1.0));
    CHECK(worst < 1e-6);
    IntrinsicTriangulation tri;
    std::string error;
    CHECK(tri.build(mesh.vertices, mesh.triangles, error));
    const LoopMesh shifted = MakeIcosphere(2, 2.0, Vec3d(1, 2, 3));
    CHECK(shifted.vertices.size() == 162);
    for (const Vec3 &p : shifted.vertices) CHECK(std::abs((p.cast<double>() - Vec3d(1, 2, 3)).norm() - 2.0) < 1e-5);
}

TEST(sphere_session_default_is_icosphere)
{
    intrinsic::SessionOptions options;
    options.config.object = LoopObject::Star;
    intrinsic::Session session;
    std::string error;
    CHECK(session.load(options, error));
    if (!error.empty()) std::printf("  load: %s\n", error.c_str());
    CHECK(session.inputTriangulation().vertexCount() == 10242);
    CHECK(session.triangulation().vertexCount() > 10242);  // refined near the object
    CHECK(!session.complete() && !session.objectLabels().empty() && session.lengthScale() > 0.0);
    const double t0 = session.simulation().time();
    CHECK(session.step());
    CHECK(session.simulation().time() > t0);
    CHECK(session.metrics().springls > 0);
}

TEST(sphere_session_rejects_open_mesh)
{
    const fs::path dir = ScratchDirectory("sphere-session-open");
    springls_test::TestMesh mesh;
    mesh.vertices = {Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)};
    mesh.triangles = {{0, 1, 2}};
    CHECK(WriteMesh(mesh, dir / "triangle.ply"));
    intrinsic::SessionOptions options;
    options.mesh = dir / "triangle.ply";
    intrinsic::Session session;
    std::string error;
    CHECK(!session.load(options, error));
    CHECK(error.find("boundary") != std::string::npos);
    fs::remove_all(dir);
}
