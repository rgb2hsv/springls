#include "SphereSimulation.hpp"

#include <core/Common.hpp>
#include <intrinsic/IntrinsicContour.hpp>
#include <intrinsic/IntrinsicTrace.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <unordered_map>

using namespace springls;

namespace toybox
{
namespace
{
Vec3f Position(const IntrinsicTriangulation &tri, const SurfacePoint &p) { return tri.toExtrinsic(p).position.cast<float>(); }

// Face centroids over the whole mesh, in face order, greedily kept when at least `spacing` input edge lengths H
// (in 3D) from every kept one (a hash grid of cells that size finds the kept ones near a candidate).
std::vector<SurfacePoint> Centroids(const intrinsic::Session &session, double spacing)
{
    const IntrinsicTriangulation &tri = session.inputTriangulation();  // the scenario's fields live on it
    const double h = session.inputTriangulation().meanEdgeLength();    // H: the display unit, independent of max level
    const double minDistance = spacing * h, minDistance2 = minDistance * minDistance;
    std::vector<SurfacePoint> samples;
    std::vector<Vec3d> positions;
    std::unordered_map<uint64_t, std::vector<uint32_t>> grid;  // cell -> kept samples
    const auto cellOf = [&](const Vec3d &x) { return (x / minDistance).array().floor().cast<int64_t>().eval(); };
    const auto key = [](int64_t i, int64_t j, int64_t k) {
        return (static_cast<uint64_t>(i & 0x1FFFFF) << 42) | (static_cast<uint64_t>(j & 0x1FFFFF) << 21) | static_cast<uint64_t>(k & 0x1FFFFF);
    };
    for (uint32_t f = 0; f < tri.faceCount(); f++) {
        SurfacePoint p;
        p.face = f;
        p.bary = Vec3d(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
        const Vec3d x = tri.toExtrinsic(p).position;
        const auto c = cellOf(x);
        bool apart = true;
        for (int64_t di = -1; di <= 1 && apart; di++) {
            for (int64_t dj = -1; dj <= 1 && apart; dj++) {
                for (int64_t dk = -1; dk <= 1 && apart; dk++) {
                    const auto it = grid.find(key(c[0] + di, c[1] + dj, c[2] + dk));
                    if (it == grid.end()) continue;
                    for (uint32_t k : it->second) {
                        if ((x - positions[k]).squaredNorm() < minDistance2) {
                            apart = false;
                            break;
                        }
                    }
                }
            }
        }
        if (!apart) continue;
        grid[key(c[0], c[1], c[2])].push_back(static_cast<uint32_t>(positions.size()));
        samples.push_back(p);
        positions.push_back(x);
    }
    return samples;
}

std::vector<SphereArrow> ArrowsAt(const intrinsic::Session &session, const std::vector<SurfacePoint> &samples, double arrowLength)
{
    const IntrinsicTriangulation &tri = session.inputTriangulation();  // the scenario's fields live on it
    const LoopScenario &scenario = session.scenario();
    const double t = session.simulation().time();
    const double weight = scenario.schedule().weight(t);
    const double h = session.inputTriangulation().meanEdgeLength();  // H
    std::vector<Vec2d> velocities;
    double maxSpeed = 0.0;
    for (const SurfacePoint &p : samples) {
        velocities.push_back(scenario.velocity(p, t));
        maxSpeed = std::max(maxSpeed, velocities.back().norm());
    }
    // Vertex normals of the input mesh: area-weighted face normals (the cross product's length is twice the area).
    std::vector<Vec3d> vertexNormals(tri.vertexCount(), Vec3d::Zero());
    for (uint32_t f = 0; f < tri.faceCount(); f++) {
        const uint32_t v0 = tri.tail(3 * f), v1 = tri.tail(3 * f + 1), v2 = tri.tail(3 * f + 2);
        const Vec3d p0 = tri.vertexPosition(v0), p1 = tri.vertexPosition(v1), p2 = tri.vertexPosition(v2);
        const Vec3d n = (p1 - p0).cross(p2 - p0);
        vertexNormals[v0] += n;
        vertexNormals[v1] += n;
        vertexNormals[v2] += n;
    }
    std::vector<SphereArrow> arrows;
    arrows.reserve(samples.size());
    for (size_t i = 0; i < samples.size(); i++) {
        SphereArrow a;
        a.from = Position(tri, samples[i]);
        Vec3d normal = Vec3d::Zero();
        for (int c = 0; c < 3; c++) normal += samples[i].bary[c] * vertexNormals[tri.tail(3 * samples[i].face + c)].normalized();
        if (normal.squaredNorm() > 0.0) a.normal = normal.normalized().cast<float>();
        a.speed = static_cast<float>(velocities[i].norm());
        const Vec2d displacement = maxSpeed > 0.0 ? Vec2d(velocities[i] * (arrowLength * h * weight / maxSpeed)) : Vec2d(Vec2d::Zero());
        a.to = Position(tri, Trace(tri, samples[i], displacement).end);
        arrows.push_back(a);
    }
    return arrows;
}
}  // namespace

bool ObjectFrame(const intrinsic::Session &session, Vec3f &center, Vec3f &normal)
{
    const IntrinsicTriangulation &tri = session.triangulation();
    const IntrinsicTriangulation &input = tri.input();
    const IntrinsicSpringls &sim = session.simulation();
    struct Part {
        double area;
        Vec3d position, normal;
    };
    std::vector<Part> parts;
    double area = 0.0;
    Vec3d sum = Vec3d::Zero(), normalSum = Vec3d::Zero();
    {
        for (const RegionPiece &piece : RegionPieces(tri, [&](uint32_t v) { return sim.phi()[v]; })) {
            const IntrinsicTriangulation::ExtrinsicPoint x = tri.toExtrinsic(piece.centroid);
            const Vec3 &a = tri.inputVertices()[input.tail(3 * x.face)], &b = tri.inputVertices()[input.tail(3 * x.face + 1)],
                       &c = tri.inputVertices()[input.tail(3 * x.face + 2)];
            const Vec3d n = (b - a).cross(c - a).cast<double>();
            const Vec3d unit = n.norm() > 0.0 ? Vec3d(n.normalized()) : Vec3d(Vec3d::Zero());
            parts.push_back({piece.area, x.position, unit});
            area += piece.area;
            sum += piece.area * x.position;
            normalSum += piece.area * unit;
        }
    }
    if (!(area > 0.0)) return false;
    const Vec3d centroid = sum / area;
    const Part *nearest = &parts.front();
    for (const Part &p : parts)
        if ((p.position - centroid).squaredNorm() < (nearest->position - centroid).squaredNorm()) nearest = &p;
    // The normals cancel when the regions wrap the surface: no side to look from, nor a level to lift to.
    if (normalSum.norm() < 0.5 * area) {
        center = nearest->position.cast<float>();
        normal = nearest->normal.norm() > 0.0 ? Vec3f(nearest->normal.cast<float>()) : Vec3f(Vec3f::UnitZ());
        return true;
    }
    const Vec3d n = normalSum.normalized();
    center = (centroid + n * n.dot(nearest->position - centroid)).cast<float>();
    normal = n.cast<float>();
    return true;
}

std::vector<SphereArrow> SampleArrows(const intrinsic::Session &session, double spacing, double arrowLength)
{
    return ArrowsAt(session, Centroids(session, spacing), arrowLength);
}

SphereSimulation::SphereSimulation(const Entry &entry, const SphereOptions &options) : mEntry(entry), mOptions(options) {}

bool SphereSimulation::init(std::string &error)
{
    if (mOptions.threads > 0) SetThreads(mOptions.threads);
    const auto start = std::chrono::steady_clock::now();
    if (!mSession.load(mOptions.session, error)) return false;
    mMetrics = mSession.metrics();
    mSteps = 0;
    mSampleSpacing = 0;
    mSamples.clear();
    mSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    mStamp = NextSnapshotStamp();
    return true;
}

bool SphereSimulation::step()
{
    if (complete()) return false;
    const auto start = std::chrono::steady_clock::now();
    const bool more = mSession.step();
    mSteps++;
    mMetrics = mSession.metrics();
    mSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    mStamp = NextSnapshotStamp();
    return more;
}

double SphereSimulation::time() const { return mSession.simulation().time(); }
uint64_t SphereSimulation::iteration() const { return mSession.simulation().iteration(); }
size_t SphereSimulation::springlCount() const { return mSession.simulation().springls().size(); }

void SphereSimulation::csvHeader(std::ostream &out) const { out << intrinsic::kCsvHeader << '\n'; }

void SphereSimulation::csvRow(std::ostream &out, double) const
{
    intrinsic::WriteCsvRows(out, static_cast<int>(iteration()), mMetrics, mSession.objectLabels());
}

SphereSnapshot SphereSimulation::snapshot(double arrowSpacing, double arrowLength)
{
    SphereSnapshot snap;
    snap.stamp = mStamp;
    snap.steps = mSteps;
    snap.complete = mSession.complete();
    snap.metrics = mMetrics;
    const LoopScenario &scenario = mSession.scenario();
    snap.period = scenario.period();
    snap.objectRadius = static_cast<float>(scenario.radius());
    snap.lengthScale = static_cast<float>(mSession.lengthScale());
    snap.bandWidth = static_cast<float>(IntrinsicSpringls::kBandWidth * mSession.lengthScale());
    const IntrinsicSpringls &sim = mSession.simulation();
    const IntrinsicTriangulation &tri = mSession.triangulation();
    const IntrinsicTriangulation &input = tri.input();
    snap.time = sim.time();

    // The simulation's refined triangulation: input and inserted vertices, every face; phi is per vertex of it.
    snap.contractedTotal = sim.contractedTotal();
    snap.filledTotal = sim.filledTotal();
    snap.vertices.reserve(tri.vertexCount());
    for (uint32_t v = 0; v < tri.vertexCount(); v++) snap.vertices.push_back(tri.vertexPosition(v).cast<float>());
    snap.triangles.reserve(tri.faceCount());
    snap.faceLevels.reserve(tri.faceCount());
    for (uint32_t f = 0; f < tri.faceCount(); f++) {
        snap.triangles.push_back({tri.tail(3 * f), tri.tail(3 * f + 1), tri.tail(3 * f + 2)});
        snap.faceLevels.push_back(tri.faceLevel(f));
    }
    snap.labels.resize(snap.vertices.size());
    snap.distances.resize(snap.vertices.size());
    for (size_t v = 0; v < snap.vertices.size(); v++) {
        snap.labels[v] = sim.phi()[v] < 0.0 ? IntrinsicSpringls::kRegionLabel : 0;
        snap.distances[v] = static_cast<float>(std::abs(sim.phi()[v]));
    }

    for (const IntrinsicSpringl &s : sim.springls()) {
        snap.springls.emplace_back(Position(tri, s.v0), Position(tri, s.v1));
        snap.particles.push_back(Position(tri, s.particle));
        const IntrinsicTriangulation::ExtrinsicPoint x = tri.toExtrinsic(s.particle);
        const IntrinsicTriangulation &input = tri.input();
        const Vec3 &a = tri.inputVertices()[input.tail(3 * x.face)], &b = tri.inputVertices()[input.tail(3 * x.face + 1)],
                   &c = tri.inputVertices()[input.tail(3 * x.face + 2)];
        const Vec3f up = (b - a).cross(c - a).cast<float>();
        const Vec3f outward = (snap.springls.back().second - snap.springls.back().first).cross(up);
        snap.normals.push_back(outward.norm() > 0.0f ? Vec3f(outward.normalized()) : Vec3f(Vec3f::Zero()));
        snap.springlLabels.push_back(IntrinsicSpringls::kRegionLabel);
    }
    if (!ObjectFrame(mSession, snap.objectCenter, snap.objectNormal)) {
        // No region area: the scenario's object center, and the input face's normal there.
        const IntrinsicTriangulation::ExtrinsicPoint x = mSession.inputTriangulation().toExtrinsic(scenario.objectCenter());
        const Vec3 &a = tri.inputVertices()[input.tail(3 * x.face)], &b = tri.inputVertices()[input.tail(3 * x.face + 1)],
                   &c = tri.inputVertices()[input.tail(3 * x.face + 2)];
        const Vec3 n = (b - a).cross(c - a);
        snap.objectCenter = x.position.cast<float>();
        snap.objectNormal = n.norm() > 0.0f ? Vec3f(n.normalized()) : Vec3f(Vec3f::UnitZ());
    }

    for (const IntrinsicCurve &curve : sim.contour()) {
        const size_t n = curve.points.size();
        const size_t segments = curve.closed ? n : (n > 0 ? n - 1 : 0);
        for (size_t k = 0; k < segments; k++) {
            const uint32_t f = curve.faces[k];
            snap.contour.emplace_back(Position(tri, ContourPointInFace(tri, curve, k, f)), Position(tri, ContourPointInFace(tri, curve, (k + 1) % n, f)));
            snap.contourLabels.push_back(curve.label);
        }
    }

    for (const std::vector<SurfacePoint> &curve : mSession.reference().curves()) {
        std::vector<Vec3f> polyline;
        polyline.reserve(curve.size());
        for (const SurfacePoint &p : curve) polyline.push_back(Position(tri, p));
        snap.reference.push_back(std::move(polyline));
    }

    if (mSampleSpacing != arrowSpacing || mSamples.empty()) {
        mSamples = Centroids(mSession, arrowSpacing);
        mSampleSpacing = arrowSpacing;
    }
    snap.arrows = ArrowsAt(mSession, mSamples, arrowLength);
    return snap;
}

void SphereSimulation::printInit(std::ostream &out) const
{
    const IntrinsicTriangulation &tri = mSession.triangulation();
    const LoopScenario &scenario = mSession.scenario();
    out << "mesh " << (mOptions.session.mesh.empty() ? std::string("icosphere") : mOptions.session.mesh.string()) << ": " << tri.vertexCount() << " vertices, "
        << tri.faceCount() << " faces, loop distance " << scenario.loopDistance() << ", radius " << scenario.radius() << ", period " << scenario.period()
        << ", length scale " << mSession.lengthScale() << ", " << springlCount() << " springls" << std::endl;
    if (scenario.flow() == springls::LoopFlow::Hodge) {
        out << "flow: Hodge (divergence-free part of the projected rotation; removed exact part " << scenario.exactFraction() << ", remainder "
            << scenario.remainderFraction() << " of its norm)" << std::endl;
    } else {
        out << "flow: rotation" << std::endl;
    }
}

void SphereSimulation::printStep(std::ostream &out) const
{
    const auto flags = out.flags();
    const auto precision = out.precision();
    out << "step " << mSteps << ": t " << std::fixed << std::setprecision(4) << time() << " / " << mSession.scenario().period() << ", springls "
        << springlCount() << ", area error";
    out << std::scientific << std::setprecision(2);
    for (size_t k = 0; k < mMetrics.relativeAreaErrors.size(); k++)
        out << (k ? ", " : " ") << mSession.objectLabels()[k] << ": " << mMetrics.relativeAreaErrors[k];
    out << ", shape error mean " << mMetrics.meanShapeError << " max " << mMetrics.maxShapeError << " h, ref " << mMetrics.referenceMeanDistance << "/"
        << mMetrics.referenceMaxDistance << " h, gap " << mMetrics.referenceMaxGap << " h" << std::endl;
    out.flags(flags);
    out.precision(precision);
}

void SphereSimulation::summary(std::ostream &out, double seconds) const
{
    const auto flags = out.flags();
    const auto precision = out.precision();
    out << "Sphere: " << mSteps << " steps, t " << std::fixed << std::setprecision(4) << time() << " / " << mSession.scenario().period()
        << (complete() ? " (complete)" : "") << ", " << std::setprecision(2) << seconds << " s, final shape error mean " << std::scientific
        << mMetrics.meanShapeError << " max " << mMetrics.maxShapeError << std::endl;
    out.flags(flags);
    out.precision(precision);
}
}  // namespace toybox
