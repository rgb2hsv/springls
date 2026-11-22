#include <intrinsic-springls/Runner.hpp>

#include <intrinsic-springls/Render.hpp>

#include <io/Ply.hpp>
#include <mesh/Icosphere.hpp>
#include <scenarios/EnrightTest2D.hpp>

#include <argparse/argparse.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

namespace springls::intrinsic
{
const char *const kCsvHeader =
    "step,time,label,area,relative_area_error,mean_shape_error,max_shape_error,springls,reference_mean_distance,reference_max_distance,reference_max_gap";

std::string DefaultMesh() { return std::string(SPRINGLS_ASSET_DIR) + "/model/metasphere.ply"; }

bool ParseArguments(int argc, char **argv, RunnerOptions &options, std::string &error, bool *help)
{
    argparse::ArgumentParser p("intrinsic-springls", "1.0", argparse::default_arguments::none);
    p.add_description("Springls on a triangle-mesh surface: an object rotates once about the z axis and back, the mesh adaptively refined near it.");
    p.add_argument("-h", "--help").flag().help("shows this help");
    p.add_argument("--mesh").default_value(DefaultMesh()).help("closed triangle mesh (PLY)");
    p.add_argument("--object").default_value(std::string("puzzle")).help("puzzle or star");
    p.add_argument("--speed").scan<'g', double>().default_value(1.0).help("speed of the object's center (mesh units per unit time)");
    p.add_argument("--loop-distance")
        .scan<'g', double>()
        .default_value(0.0)
        .help("geodesic distance of the object's center from the loop center (0: an eighth of half the distance to the vertex nearest -z)");
    p.add_argument("--radius").scan<'g', double>().default_value(0.0).help("object size r (0: loop distance / 6)");
    p.add_argument("--frames").default_value(std::string()).help("directory for per-step PLY frames");
    p.add_argument("--csv").default_value(std::string()).help("per-step metrics file");
    p.add_argument("--steps").scan<'i', int>().default_value(-1).help("number of steps (default: the whole loop)");
    p.add_argument("--set")
        .default_value(std::vector<std::string>{})
        .append()
        .help("tuning override key=value (SpringLevelSet2D keys, lengths in h_t, the refined edge length: mean input edge / 2^max-level)");
    p.add_argument("--images").default_value(std::string()).help("directory for PNG views of the object");
    p.add_argument("--every").scan<'i', int>().default_value(1).help("frames and images every this many steps (and at the last)");
    p.add_argument("--image-size").scan<'i', int>().default_value(512).help("image width and height in pixels");
    p.add_argument("--max-level").scan<'i', int>().default_value(IntrinsicSpringls::kDefaultMaxLevel).help("adaptive refinement levels near the springls");
    p.add_argument("-j", "--threads").scan<'i', int>().default_value(0).help("worker threads (0: OpenMP default)");
    p.add_argument("--verbose").flag().help("progress on stderr");
    try {
        p.parse_args(argc, argv);
    } catch (const std::exception &e) {
        error = std::string("error: ") + e.what();
        return false;
    }
    if (p.get<bool>("--help")) {
        if (help) *help = true;
        std::ostringstream text;
        text << p;
        error = text.str();
        return false;
    }
    options.session.mesh = p.get<std::string>("--mesh");  // default: the metasphere
    const std::string object = p.get<std::string>("--object");
    if (object == "puzzle") {
        options.session.config.object = LoopObject::PuzzlePiece;
    } else if (object == "star") {
        options.session.config.object = LoopObject::Star;
    } else {
        error = "error: --object expects puzzle or star, got '" + object + "'";
        return false;
    }
    options.session.config.speed = p.get<double>("--speed");
    options.session.config.loopDistance = p.get<double>("--loop-distance");
    options.session.config.loopFraction = kDefaultLoopFraction;
    options.session.config.radius = p.get<double>("--radius");
    options.frames = p.get<std::string>("--frames");
    options.csv = p.get<std::string>("--csv");
    options.steps = p.get<int>("--steps");
    options.threads = p.get<int>("--threads");
    options.session.verbose = p.get<bool>("--verbose");
    options.images = p.get<std::string>("--images");
    options.every = p.get<int>("--every");
    options.imageSize = p.get<int>("--image-size");
    if (options.every < 1 || options.imageSize < 16) {
        error = "error: --every expects >= 1 and --image-size >= 16";
        return false;
    }
    options.session.maxLevel = p.get<int>("--max-level");
    if (options.session.maxLevel < 0) {
        error = "error: --max-level expects a level count >= 0";
        return false;
    }
    options.session.tuning.clear();
    const std::vector<std::string> keys = TuningKeys();
    for (const std::string &entry : p.get<std::vector<std::string>>("--set")) {
        const size_t eq = entry.find('=');
        double value = 0.0;
        try {
            if (eq == std::string::npos) throw std::invalid_argument(entry);
            value = std::stod(entry.substr(eq + 1));
        } catch (const std::exception &) {
            error = "error: --set expects key=value, got '" + entry + "'";
            return false;
        }
        const std::string key = entry.substr(0, eq);
        if (std::find(keys.begin(), keys.end(), key) == keys.end()) {
            error = "error: unknown tuning key '" + key + "'";
            return false;
        }
        options.session.tuning.emplace_back(key, value);
    }
    return true;
}

void WriteCsvRows(std::ostream &out, int step, const LoopScenario::Metrics &m, const std::vector<int> &labels)
{
    for (size_t k = 0; k < labels.size() && k < m.areas.size(); k++) {
        out << step << ',' << m.time << ',' << labels[k] << ',' << m.areas[k] << ',' << m.relativeAreaErrors[k] << ',' << m.meanShapeError << ','
            << m.maxShapeError << ',' << m.springls << ',' << m.referenceMeanDistance << ',' << m.referenceMaxDistance << ',' << m.referenceMaxGap << '\n';
    }
}

namespace
{
std::filesystem::path Numbered(const std::filesystem::path &directory, const char *prefix, int step)
{
    char name[64];
    std::snprintf(name, sizeof(name), "%s_%06d.ply", prefix, step);
    return directory / name;
}

bool Save(const std::filesystem::path &file, const PlyTable &table, std::string &error)
{
    std::ofstream out(file, std::ios::binary);
    if (!out) {
        error = "cannot write " + file.string();
        return false;
    }
    return WritePly(out, table, error);
}

PlyProperty Scalar(const char *name, PlyType type) { return PlyProperty{name, type, false, {}, {0}}; }
}  // namespace

bool WriteFrame(const std::filesystem::path &directory, int step, const IntrinsicSpringls &sim, std::string &error)
{
    const IntrinsicTriangulation &tri = sim.triangulation();
    const IntrinsicTriangulation &input = tri.input();
    // The input mesh, vertices colored inside / outside.
    PlyTable mesh;
    PlyElement vertex{"vertex",
                      input.vertexCount(),
                      {Scalar("x", PlyType::Float32), Scalar("y", PlyType::Float32), Scalar("z", PlyType::Float32), Scalar("red", PlyType::UInt8),
                       Scalar("green", PlyType::UInt8), Scalar("blue", PlyType::UInt8)}};
    for (uint32_t v = 0; v < input.vertexCount(); v++) {
        const Vec3 &p = tri.inputVertices()[v];
        double rgb[3];
        LabelColor(sim.phi()[v] < 0.0 ? IntrinsicSpringls::kRegionLabel : 0, rgb);
        for (int c = 0; c < 3; c++) {
            vertex.properties[c].values.push_back(p[c]);
            vertex.properties[3 + c].values.push_back(rgb[c]);
        }
    }
    PlyElement face{"face", input.faceCount(), {PlyProperty{"vertex_indices", PlyType::Int32, true, {}, {0}}}};
    for (uint32_t f = 0; f < input.faceCount(); f++) {
        for (int c = 0; c < 3; c++) face.properties[0].values.push_back(input.tail(3 * f + c));
        face.properties[0].listStart.push_back(static_cast<uint32_t>(face.properties[0].values.size()));
    }
    mesh.elements = {std::move(vertex), std::move(face)};
    // Springls: v0 - particle - v1 per springl, on the input mesh.
    PlyTable springls;
    const size_t N = sim.springls().size();
    PlyElement points{
        "vertex", 3 * N, {Scalar("x", PlyType::Float32), Scalar("y", PlyType::Float32), Scalar("z", PlyType::Float32), Scalar("label", PlyType::Int32)}};
    PlyElement edges{"edge", 2 * N, {Scalar("vertex1", PlyType::Int32), Scalar("vertex2", PlyType::Int32)}};
    for (size_t s = 0; s < N; s++) {
        const IntrinsicSpringl &sp = sim.springls()[s];
        for (const SurfacePoint *p : {&sp.v0, &sp.particle, &sp.v1}) {
            const Vec3d x = tri.toExtrinsic(*p).position;
            for (int c = 0; c < 3; c++) points.properties[c].values.push_back(x[c]);
            points.properties[3].values.push_back(IntrinsicSpringls::kRegionLabel);
        }
        for (int e = 0; e < 2; e++) {
            edges.properties[0].values.push_back(static_cast<double>(3 * s + e));
            edges.properties[1].values.push_back(static_cast<double>(3 * s + e + 1));
        }
    }
    springls.elements = {std::move(points), std::move(edges)};
    return Save(Numbered(directory, "frame", step), mesh, error) && Save(Numbered(directory, "springls", step), springls, error);
}

bool Session::load(const SessionOptions &options, std::string &error, std::ostream *log)
{
    mSim.reset();
    mLabels.clear();
    mComplete = false;
    LoopMesh mesh;
    std::string name = "icosphere";
    if (options.mesh.empty()) {
        mesh = MakeIcosphere(5);
    } else {
        name = options.mesh.string();
        PolygonMesh input;
        if (!LoadPly(options.mesh, input, error)) {
            error = name + ": " + error;
            return false;
        }
        mesh = LoopMesh{input.positions, Triangulate(input)};
    }
    IntrinsicTriangulation &tri = mTri;
    LoopScenario &scenario = mScenario;
    if (!tri.build(mesh.vertices, mesh.triangles, error)) {
        error = name + ": " + error;
        return false;
    }
    const size_t flips = tri.flipToDelaunay();
    if (log)
        *log << "mesh " << name << ": " << tri.vertexCount() << " vertices, " << tri.faceCount() << " faces, " << flips << " Delaunay flips, mean edge "
             << tri.meanEdgeLength() << '\n';
    if (!scenario.setup(tri, options.config, error)) return false;
    if (log) {
        *log << "loop: center vertex " << scenario.center() << ", distance " << scenario.loopDistance() << ", object radius " << scenario.radius()
             << ", angular speed " << scenario.angularSpeed() << ", period " << scenario.period() << '\n';
    }
    mSim = std::make_unique<IntrinsicSpringls>(tri);
    IntrinsicSpringls &sim = *mSim;
    sim.verbose = options.verbose;
    SpringLevelSet2D probe;  // carries the 2D tuning keys onto the intrinsic tuning
    probe.tuning = sim.tuning;
    for (const auto &[key, value] : options.tuning) {
        if (!SetTuning(probe, key, value)) {
            error = "unknown tuning key '" + key + "'";
            return false;
        }
    }
    sim.tuning = probe.tuning;
    sim.setMaxLevel(options.maxLevel);
    sim.setInitialRegion(scenario.initialPhi());
    // The simulation's points live on its refined triangulation: the analytic field there.
    sim.setVelocity([this](const SurfacePoint &p, double t) { return mScenario.velocity(mSim->triangulation(), p, t); }, scenario.schedule());
    // The reference's points are relocated through every refinement.
    mReference = IntrinsicReference();
    sim.setTrackedPoints([this] { return mReference.points(); });
    if (!sim.init(error)) return false;
    mLabels = {IntrinsicSpringls::kRegionLabel};
    mReference.initialize(sim.triangulation(), sim.contour());
    return true;
}

LoopScenario::Metrics Session::metrics() const
{
    LoopScenario::Metrics m = mScenario.measure(*mSim);
    const double h = lengthScale();
    if (h <= 0.0) return m;
    const double cap = 20.0 * h;
    std::vector<SurfacePoint> points;
    for (const IntrinsicCurve &curve : mSim->contour()) points.insert(points.end(), curve.points.begin(), curve.points.end());
    double sum = 0.0;
    const IntrinsicTriangulation &tri = mSim->triangulation();
    for (double d : mReference.distancesAt(tri, points, cap)) {
        d /= h;
        sum += d;
        m.referenceMaxDistance = std::max(m.referenceMaxDistance, d);
    }
    const size_t n = points.size();
    m.referenceMeanDistance = (n > 0) ? sum / static_cast<double>(n) : 0.0;
    m.referenceMaxGap = mReference.maxGap(tri) / h;
    return m;
}

bool Session::step()
{
    if (mComplete || !mSim) return false;
    const double before = mSim->time();
    if (!mSim->step()) mComplete = true;
    const LoopScenario &scenario = mScenario;
    const IntrinsicSpringls::Schedule schedule = scenario.schedule();
    const IntrinsicTriangulation &tri = mSim->triangulation();
    mReference.advance(tri, [&](const SurfacePoint &p, double t) { return scenario.velocity(tri, p, t) * static_cast<double>(schedule.weight(t)); }, before,
                       mSim->time(), 0.25 * lengthScale(), {0.5 * scenario.period()});
    return !mComplete;
}

bool Run(const RunnerOptions &options, RunnerResult &result, std::string &error, std::ostream *log)
{
    SetThreads(options.threads);
    Session session;
    if (!session.load(options.session, error, log)) return false;
    const IntrinsicTriangulation &tri = session.triangulation();
    const LoopScenario &scenario = session.scenario();
    const IntrinsicSpringls &sim = session.simulation();
    const std::vector<int> &labels = session.objectLabels();
    result.vertices = tri.vertexCount();
    result.faces = tri.faceCount();
    result.loopDistance = scenario.loopDistance();
    result.radius = scenario.radius();
    result.lengthScale = session.lengthScale();
    std::ofstream csv;
    if (!options.csv.empty()) {
        if (options.csv.has_parent_path()) std::filesystem::create_directories(options.csv.parent_path());
        csv.open(options.csv);
        if (!csv) {
            error = "cannot write " + options.csv.string();
            return false;
        }
        csv << kCsvHeader << '\n';
    }
    if (!options.frames.empty()) std::filesystem::create_directories(options.frames);
    if (!options.images.empty()) std::filesystem::create_directories(options.images);
    // Views follow the object: centered on its springls, from outside the surface.
    Vec3d meshCenter = Vec3d::Zero();
    for (const Vec3 &p : tri.inputVertices()) meshCenter += p.cast<double>();
    meshCenter /= static_cast<double>(tri.inputVertices().size());
    const auto image = [&](int step) {
        RenderView view;
        view.size = options.imageSize;
        view.halfWidth = 2.5 * scenario.radius();
        Vec3d c = session.inputTriangulation().toExtrinsic(scenario.objectCenter()).position;
        if (!sim.springls().empty()) {
            c = Vec3d::Zero();
            for (const IntrinsicSpringl &s : sim.springls()) c += sim.triangulation().toExtrinsic(s.particle).position;
            c /= static_cast<double>(sim.springls().size());
        }
        view.center = c;
        view.normal = (c - meshCenter).normalized();
        view.up = (std::abs(view.normal.z()) < 0.9) ? Vec3d(0, 0, 1) : Vec3d(1, 0, 0);
        char name[64];
        std::snprintf(name, sizeof(name), "image_%06d.png", step);
        return RenderPng(options.images / name, sim, view, error);
    };
    const auto record = [&](int step, bool last) {
        const LoopScenario::Metrics m = session.metrics();
        if (csv.is_open()) WriteCsvRows(csv, step, m, labels);
        result.final = m;
        if (step % options.every != 0 && !last) return true;
        if (!options.frames.empty() && !WriteFrame(options.frames, step, sim, error)) return false;
        if (!options.images.empty() && !image(step)) return false;
        return true;
    };
    if (!record(0, false)) return false;
    const auto start = std::chrono::steady_clock::now();
    result.steps = 0;
    bool more = true;
    while (more && (options.steps < 0 || result.steps < options.steps)) {
        more = session.step();
        result.steps++;
        const bool last = !more || (options.steps >= 0 && result.steps >= options.steps);
        if (!record(result.steps, last)) return false;
    }
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (log) {
        *log << "done: " << result.steps << " steps, " << result.seconds << " s, t = " << result.final.time << " of " << scenario.period() << '\n';
        for (size_t k = 0; k < labels.size(); k++) *log << "  label " << labels[k] << ": relative area error " << result.final.relativeAreaErrors[k] << '\n';
        *log << "  shape error mean " << result.final.meanShapeError << " h, max " << result.final.maxShapeError << " h; springls " << result.final.springls
             << '\n';
    }
    return true;
}
}  // namespace springls::intrinsic
