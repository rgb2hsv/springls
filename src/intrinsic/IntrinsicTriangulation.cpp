#include <intrinsic/IntrinsicTriangulation.hpp>

#include <io/Ply.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <numeric>
#include <utility>

namespace springls
{
namespace
{
constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kPiD = 3.14159265358979323846264338327950;

double Cross(const Vec2d &a, const Vec2d &b) { return a.x() * b.y() - a.y() * b.x(); }
Vec2d Perp(const Vec2d &a) { return Vec2d(-a.y(), a.x()); }  // +90° rotation
double WrapAngle(double a)
{
    a = std::fmod(a, kTwoPi);
    if (a < 0.0) a += kTwoPi;
    return (a >= kTwoPi) ? 0.0 : a;
}
// Interior angle opposite side `opposite` in a triangle with sides a, b around it (law of cosines).
double AngleFromLengths(double a, double b, double opposite)
{
    const double c = (a * a + b * b - opposite * opposite) / (2.0 * a * b);
    return std::acos(std::clamp(c, -1.0, 1.0));
}
// Heron's formula in the ordered form that is stable for slivers.
double AreaFromLengths(double a, double b, double c)
{
    if (a < b) std::swap(a, b);
    if (a < c) std::swap(a, c);
    if (b < c) std::swap(b, c);
    const double q = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c));
    return 0.25 * std::sqrt(std::max(0.0, q));
}
// Third corner of a triangle laid out on base [(0,0), (base,0)] with sides `fromOrigin` and `fromEnd`,
// above (side = +1) or below (side = -1) the base.
Vec2d PlaceCorner(double base, double fromOrigin, double fromEnd, double side)
{
    const double x = (fromOrigin * fromOrigin + base * base - fromEnd * fromEnd) / (2.0 * base);
    return Vec2d(x, side * std::sqrt(std::max(0.0, fromOrigin * fromOrigin - x * x)));
}
void CleanBary(Vec3d &b)
{
    for (int c = 0; c < 3; c++) b[c] = std::max(0.0, b[c]);
    const double s = b.sum();
    b = (s > 0.0) ? Vec3d(b / s) : Vec3d(1, 0, 0);
}
// Barycentric coordinates of x in the planar triangle (p0, p1, p2).
Vec3d PlanarBary(const Vec2d &p0, const Vec2d &p1, const Vec2d &p2, const Vec2d &x)
{
    const double A = Cross(p1 - p0, p2 - p0);
    return Vec3d(Cross(p1 - x, p2 - x) / A, Cross(p2 - x, p0 - x) / A, Cross(p0 - x, p1 - x) / A);
}
// 3D position of a point on the input mesh.
Vec3d InputPosition(const IntrinsicTriangulation &in, const SurfacePoint &p)
{
    Vec3d x = Vec3d::Zero();
    for (int c = 0; c < 3; c++) x += p.bary[c] * in.inputVertices()[in.tail(3 * p.face + c)].cast<double>();
    return x;
}
}  // namespace

// The surface points an operation relocates, bucketed by face so each operation touches only the
// points in the faces it changes.
struct IntrinsicTriangulation::PointSet {
    const std::vector<SurfacePoint *> &points;
    std::vector<std::vector<uint32_t>> byFace;
    PointSet(const std::vector<SurfacePoint *> &p, size_t faces) : points(p), byFace(faces)
    {
        for (uint32_t n = 0; n < p.size(); n++) {
            if (p[n] && p[n]->face < faces) byFace[p[n]->face].push_back(n);
        }
    }
    std::vector<uint32_t> take(uint32_t f)
    {
        std::vector<uint32_t> out;
        if (f < byFace.size()) out.swap(byFace[f]);
        return out;
    }
    const Vec3d &bary(uint32_t n) const { return points[n]->bary; }
    void place(uint32_t n, uint32_t f, const Vec3d &b)
    {
        if (f >= byFace.size()) byFace.resize(f + 1);
        points[n]->face = f;
        points[n]->bary = b;
        byFace[f].push_back(n);
    }
};

bool IntrinsicTriangulation::build(const Points3 &vertices, const std::vector<std::array<uint32_t, 3>> &triangles, std::string &error)
{
    *this = IntrinsicTriangulation();
    const size_t V = vertices.size(), F = triangles.size();
    if (F == 0) {
        error = "mesh has no triangles";
        return false;
    }
    mVertices = vertices;
    mTail.resize(3 * F);
    for (size_t f = 0; f < F; f++) {
        const auto &t = triangles[f];
        if (t[0] >= V || t[1] >= V || t[2] >= V) {
            error = "triangle " + std::to_string(f) + " has a vertex index out of range";
            return false;
        }
        if (t[0] == t[1] || t[1] == t[2] || t[2] == t[0]) {
            error = "triangle " + std::to_string(f) + " repeats a vertex";
            return false;
        }
        for (int c = 0; c < 3; c++) mTail[3 * f + c] = t[c];
    }
    // Undirected edge use counts, then twins by matching each directed halfedge with its reverse.
    std::map<std::pair<uint32_t, uint32_t>, int> uses;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> directed;
    for (uint32_t he = 0; he < 3 * F; he++) {
        const uint32_t a = mTail[he], b = mTail[next(he)];
        uses[{std::min(a, b), std::max(a, b)}]++;
    }
    size_t boundary = 0;
    for (const auto &[key, count] : uses) {
        if (count > 2) {
            error = "non-manifold edge " + std::to_string(key.first) + "-" + std::to_string(key.second) + " (" + std::to_string(count) + " triangles)";
            return false;
        }
        if (count == 1) boundary++;
    }
    if (boundary > 0) {
        error = "mesh has " + std::to_string(boundary) + " boundary edges (a closed surface is required)";
        return false;
    }
    for (uint32_t he = 0; he < 3 * F; he++) {
        const auto key = std::make_pair(mTail[he], mTail[next(he)]);
        if (!directed.emplace(key, he).second) {
            error = "inconsistently oriented triangles at edge " + std::to_string(key.first) + "-" + std::to_string(key.second);
            return false;
        }
    }
    mTwin.resize(3 * F);
    mEdge.assign(3 * F, UINT32_MAX);
    for (uint32_t he = 0; he < 3 * F; he++) {
        mTwin[he] = directed.at({mTail[next(he)], mTail[he]});
        if (mEdge[he] == UINT32_MAX) {
            const uint32_t e = static_cast<uint32_t>(mLength.size());
            mEdge[he] = mEdge[mTwin[he]] = e;
            mEdgeHalfedge.push_back(he);
            mLength.push_back((vertices[mTail[he]].cast<double>() - vertices[mTail[next(he)]].cast<double>()).norm());
        }
    }
    mVertexHalfedge.assign(V, UINT32_MAX);
    std::vector<uint32_t> outgoing(V, 0);
    for (uint32_t he = 0; he < 3 * F; he++) {
        if (mVertexHalfedge[mTail[he]] == UINT32_MAX) mVertexHalfedge[mTail[he]] = he;
        outgoing[mTail[he]]++;
    }
    for (uint32_t v = 0; v < V; v++) {
        if (mVertexHalfedge[v] == UINT32_MAX) {
            error = "vertex " + std::to_string(v) + " is not used by any triangle";
            return false;
        }
        if (degree(v) != outgoing[v]) {
            error = "non-manifold vertex " + std::to_string(v) + " (more than one fan of triangles)";
            return false;
        }
    }
    for (uint32_t e = 0; e < mLength.size(); e++) {
        if (!(mLength[e] > 0.0)) {
            error = "edge " + std::to_string(e) + " has zero length";
            return false;
        }
    }
    // Angle sums, then signposts: running sums of normalized corner angles from each vertex's
    // reference halfedge, counter-clockwise.
    mAngleSum.assign(V, 0.0);
    for (uint32_t he = 0; he < 3 * F; he++) mAngleSum[mTail[he]] += cornerAngle(he);
    mSignpost.assign(3 * F, 0.0);
    for (uint32_t v = 0; v < V; v++) {
        const uint32_t start = mVertexHalfedge[v];
        double angle = 0.0;
        uint32_t he = start;
        do {
            mSignpost[he] = WrapAngle(angle);
            angle += cornerAngle(he) * kTwoPi / mAngleSum[v];
            he = nextOutgoing(he);
        } while (he != start);
    }
    mFaceLevel.assign(F, 0);
    mGreenPartner.assign(F, kNoFace);
    mGreenMid.assign(F, 0);
    mInput = std::make_shared<const IntrinsicTriangulation>(*this);
    return true;
}

void IntrinsicTriangulation::resetToInput()
{
    if (!mInput) return;
    const IntrinsicTriangulation &in = *mInput;
    mTail = in.mTail;
    mTwin = in.mTwin;
    mEdge = in.mEdge;
    mSignpost = in.mSignpost;
    mLength = in.mLength;
    mEdgeHalfedge = in.mEdgeHalfedge;
    mVertexHalfedge = in.mVertexHalfedge;
    mAngleSum = in.mAngleSum;
    mFaceLevel.assign(in.faceCount(), 0);
    mGreenPartner.assign(in.faceCount(), kNoFace);
    mGreenMid.assign(in.faceCount(), 0);
    mInserted.clear();
}

bool IntrinsicTriangulation::load(const std::filesystem::path &ply, std::string &error)
{
    PolygonMesh mesh;
    if (!LoadPly(ply, mesh, error)) return false;
    return build(mesh.positions, Triangulate(mesh), error);
}

size_t IntrinsicTriangulation::degree(uint32_t v) const
{
    const uint32_t start = mVertexHalfedge[v];
    size_t n = 0;
    uint32_t he = start;
    do {
        n++;
        he = nextOutgoing(he);
    } while (he != start && n <= mTail.size());
    return n;
}

double IntrinsicTriangulation::meanEdgeLength() const
{
    double sum = 0.0;
    for (double l : mLength) sum += l;
    return mLength.empty() ? 0.0 : sum / static_cast<double>(mLength.size());
}

double IntrinsicTriangulation::cornerAngle(uint32_t he) const
{
    return AngleFromLengths(halfedgeLength(he), halfedgeLength(prev(he)), halfedgeLength(next(he)));
}

double IntrinsicTriangulation::faceArea(uint32_t f) const
{
    return AreaFromLengths(halfedgeLength(3 * f), halfedgeLength(3 * f + 1), halfedgeLength(3 * f + 2));
}

std::array<Vec2d, 3> IntrinsicTriangulation::layout(uint32_t f) const
{
    const double l01 = halfedgeLength(3 * f), l12 = halfedgeLength(3 * f + 1), l20 = halfedgeLength(3 * f + 2);
    return {Vec2d(0, 0), Vec2d(l01, 0), PlaceCorner(l01, l20, l12, 1.0)};
}

double IntrinsicTriangulation::cotanWeight(uint32_t e) const
{
    // cot of the angle opposite he in face(he): (b² + c² − a²) / (4 A).
    const auto cotOpposite = [&](uint32_t he) {
        const double a = halfedgeLength(he), b = halfedgeLength(next(he)), c = halfedgeLength(prev(he));
        const double area = AreaFromLengths(a, b, c);
        return (b * b + c * c - a * a) / (4.0 * std::max(area, 1e-300));
    };
    const uint32_t he = mEdgeHalfedge[e];
    return 0.5 * (cotOpposite(he) + cotOpposite(mTwin[he]));
}

Vec2d IntrinsicTriangulation::toLayout(const SurfacePoint &p) const
{
    const auto P = layout(p.face);
    return p.bary[0] * P[0] + p.bary[1] * P[1] + p.bary[2] * P[2];
}

SurfacePoint IntrinsicTriangulation::cornerPoint(uint32_t f, int corner)
{
    SurfacePoint p;
    p.face = f;
    p.bary = Vec3d::Zero();
    p.bary[corner] = 1.0;
    return p;
}

SurfacePoint IntrinsicTriangulation::halfedgePoint(uint32_t he, double t)
{
    SurfacePoint p;
    p.face = he / 3;
    p.bary = Vec3d::Zero();
    p.bary[he % 3] = 1.0 - t;
    p.bary[(he % 3 + 1) % 3] = t;
    return p;
}

bool IntrinsicTriangulation::flipEdge(uint32_t e, const std::vector<SurfacePoint *> &points)
{
    if (points.empty()) return flipEdgeImpl(e, nullptr);
    PointSet set(points, faceCount());
    return flipEdgeImpl(e, &set);
}

bool IntrinsicTriangulation::flipEdgeImpl(uint32_t e, PointSet *points)
{
    const uint32_t he = mEdgeHalfedge[e], tw = mTwin[he];
    const uint32_t A = face(he), B = face(tw);
    if (A == B) return false;
    const uint32_t a1 = next(he), a2 = next(a1);  // j→k, k→i
    const uint32_t b1 = next(tw), b2 = next(b1);  // i→l, l→j
    const uint32_t i = mTail[he], j = mTail[tw], k = mTail[a2], l = mTail[b2];
    // Endpoints keep at least two edges (degree >= 3 before the flip).
    if (degree(i) < 3 || degree(j) < 3) return false;
    const double lij = mLength[e], ljk = halfedgeLength(a1), lki = halfedgeLength(a2), lil = halfedgeLength(b1), llj = halfedgeLength(b2);
    const Vec2d pk = PlaceCorner(lij, lki, ljk, 1.0), pl = PlaceCorner(lij, lil, llj, -1.0);
    // Strictly convex quadrilateral: segment kl crosses the open segment ij.
    constexpr double kConvexTolerance = 1e-9;
    if (pk.y() <= kConvexTolerance * lij || pl.y() >= -kConvexTolerance * lij) return false;
    const double x0 = pl.x() + (pk.x() - pl.x()) * (-pl.y()) / (pk.y() - pl.y());
    if (x0 <= kConvexTolerance * lij || x0 >= (1.0 - kConvexTolerance) * lij) return false;
    const double lkl = (pk - pl).norm();
    dissolveGreen(A);
    dissolveGreen(B);
    // Signpost update: a known direction at k (k→i) plus the new corner angle at k, normalized.
    const double thetaK = AngleFromLengths(lki, lkl, lil), thetaL = AngleFromLengths(llj, lkl, ljk);
    const double phiKL = WrapAngle(mSignpost[a2] + thetaK * kTwoPi / mAngleSum[k]);
    const double phiLK = WrapAngle(mSignpost[b2] + thetaL * kTwoPi / mAngleSum[l]);
    std::vector<uint32_t> inA, inB;
    if (points) {
        inA = points->take(A);
        inB = points->take(B);
    }
    const int cA = static_cast<int>(he % 3), cB = static_cast<int>(tw % 3);
    // New faces A = (k, i, l) and B = (l, j, k); the four outer halfedges move to new slots.
    const uint32_t oldSlots[4] = {a2, b1, b2, a1};
    const uint32_t newSlots[4] = {3 * A + 0, 3 * A + 1, 3 * B + 0, 3 * B + 1};
    struct Data {
        uint32_t tail, twin, edge;
        double signpost;
    } moved[4];
    for (int n = 0; n < 4; n++) moved[n] = {mTail[oldSlots[n]], mTwin[oldSlots[n]], mEdge[oldSlots[n]], mSignpost[oldSlots[n]]};
    const auto remap = [&](uint32_t h) {
        for (int n = 0; n < 4; n++) {
            if (oldSlots[n] == h) return newSlots[n];
        }
        return h;
    };
    for (int n = 0; n < 4; n++) {
        const uint32_t s = newSlots[n];
        mTail[s] = moved[n].tail;
        mTwin[s] = remap(moved[n].twin);
        mEdge[s] = moved[n].edge;
        mSignpost[s] = moved[n].signpost;
    }
    for (int n = 0; n < 4; n++) {
        const uint32_t s = newSlots[n];
        mTwin[mTwin[s]] = s;
        mEdgeHalfedge[mEdge[s]] = s;
    }
    const uint32_t lk = 3 * A + 2, kl = 3 * B + 2;
    mTail[lk] = l;
    mTail[kl] = k;
    mTwin[lk] = kl;
    mTwin[kl] = lk;
    mEdge[lk] = mEdge[kl] = e;
    mSignpost[lk] = phiLK;
    mSignpost[kl] = phiKL;
    mEdgeHalfedge[e] = kl;
    mLength[e] = lkl;
    mVertexHalfedge[k] = 3 * A + 0;
    mVertexHalfedge[i] = 3 * A + 1;
    mVertexHalfedge[l] = 3 * B + 0;
    mVertexHalfedge[j] = 3 * B + 1;
    mFaceLevel[A] = mFaceLevel[B] = std::max(mFaceLevel[A], mFaceLevel[B]);
    if (!points) return true;
    // Relocation in the hinge layout (i at the origin, j on +x, k above, l below). Points on an outer
    // edge (or at a corner) keep their coordinates along it exactly; the rest are mapped in the plane.
    const Vec2d pi(0, 0), pj(lij, 0);
    const auto placePlanar = [&](uint32_t n, const Vec2d &x) {
        Vec3d wA = PlanarBary(pk, pi, pl, x), wB = PlanarBary(pl, pj, pk, x);
        const bool inNewA = wA.minCoeff() >= wB.minCoeff();
        Vec3d &w = inNewA ? wA : wB;
        CleanBary(w);
        points->place(n, inNewA ? A : B, w);
    };
    for (uint32_t n : inA) {
        const Vec3d &b = points->bary(n);
        const double bi = b[cA], bj = b[(cA + 1) % 3], bk = b[(cA + 2) % 3];
        if (bj == 0.0) {
            points->place(n, A, Vec3d(bk, bi, 0.0));  // on k-i
        } else if (bi == 0.0) {
            points->place(n, B, Vec3d(0.0, bj, bk));  // on j-k
        } else {
            placePlanar(n, bj * pj + bk * pk);
        }
    }
    for (uint32_t n : inB) {
        const Vec3d &b = points->bary(n);
        const double bj = b[cB], bi = b[(cB + 1) % 3], bl = b[(cB + 2) % 3];
        if (bj == 0.0) {
            points->place(n, A, Vec3d(0.0, bi, bl));  // on i-l
        } else if (bi == 0.0) {
            points->place(n, B, Vec3d(bl, bj, 0.0));  // on l-j
        } else {
            placePlanar(n, bj * pj + bl * pl);
        }
    }
    return true;
}

uint32_t IntrinsicTriangulation::splitEdge(uint32_t e, const std::vector<SurfacePoint *> &points)
{
    if (points.empty()) return splitEdgeImpl(e, nullptr);
    PointSet set(points, faceCount());
    return splitEdgeImpl(e, &set);
}

uint32_t IntrinsicTriangulation::splitEdgeImpl(uint32_t e, PointSet *points)
{
    const uint32_t he = mEdgeHalfedge[e], tw = mTwin[he];
    const uint32_t A = face(he), B = face(tw);
    if (A == B) return UINT32_MAX;
    dissolveGreen(A);
    dissolveGreen(B);
    const uint32_t a1 = next(he), a2 = next(a1);  // j→k, k→i
    const uint32_t b1 = next(tw), b2 = next(b1);  // i→l, l→j
    const uint32_t i = mTail[he], j = mTail[tw], k = mTail[a2], l = mTail[b2];
    const double lij = mLength[e], ljk = halfedgeLength(a1), lki = halfedgeLength(a2), lil = halfedgeLength(b1), llj = halfedgeLength(b2);
    const double half = 0.5 * lij;
    // Hinge layout: i at the origin, j on +x, k above, l below; the midpoint m on the base.
    const Vec2d pk = PlaceCorner(lij, lki, ljk, 1.0), pl = PlaceCorner(lij, lil, llj, -1.0), pm(half, 0.0);
    const double lkm = (pk - pm).norm(), llm = (pl - pm).norm();
    // The midpoint on the input mesh, traced from an input-vertex endpoint when there is one. The
    // trace's end direction is that of m→head(s): signpost 0 for m→j, π for m→i.
    const size_t nIn = mVertices.size();
    const bool fromJ = i >= nIn && j < nIn;
    const uint32_t s = fromJ ? tw : he;
    const auto P = layout(face(s));
    const int cs = static_cast<int>(s % 3);
    const Trace r = inputTrace(s, 0.5 * (P[(cs + 1) % 3] - P[cs]));
    Inserted inserted;
    inserted.point = {r.end.face, r.end.bary, InputPosition(input(), r.end)};
    inserted.frame = std::atan2(r.direction.y(), r.direction.x()) - (fromJ ? kPiD : 0.0);
    // Signposts. The endpoints keep theirs (same direction, shorter edge); the opposite corners get the
    // signpost update from their known halfedge k→i / l→j; m's frame starts at m→j.
    const double sIJ = mSignpost[he], sJI = mSignpost[tw];
    const double sKM = WrapAngle(mSignpost[a2] + AngleFromLengths(lki, lkm, half) * kTwoPi / mAngleSum[k]);
    const double sLM = WrapAngle(mSignpost[b2] + AngleFromLengths(llj, llm, half) * kTwoPi / mAngleSum[l]);
    const double sMK = AngleFromLengths(half, lkm, ljk);
    const double sML = WrapAngle(kPiD + AngleFromLengths(half, llm, lil));
    std::vector<uint32_t> inA, inB;
    if (points) {
        inA = points->take(A);
        inB = points->take(B);
    }
    const int cA = static_cast<int>(he % 3), cB = static_cast<int>(tw % 3);
    // New faces: A = (k, i, m), C = (m, j, k), B = (l, j, m), D = (m, i, l). The four outer halfedges
    // move to their new slots.
    const uint32_t C = static_cast<uint32_t>(faceCount()), D = C + 1;
    const uint32_t oldSlots[4] = {a2, a1, b2, b1};
    const uint32_t newSlots[4] = {3 * A + 0, 3 * C + 1, 3 * B + 0, 3 * D + 1};
    struct Data {
        uint32_t tail, twin, edge;
        double signpost;
    } moved[4];
    for (int n = 0; n < 4; n++) moved[n] = {mTail[oldSlots[n]], mTwin[oldSlots[n]], mEdge[oldSlots[n]], mSignpost[oldSlots[n]]};
    const auto remap = [&](uint32_t h) {
        for (int n = 0; n < 4; n++) {
            if (oldSlots[n] == h) return newSlots[n];
        }
        return h;
    };
    const size_t H = 3 * (static_cast<size_t>(D) + 1);
    mTail.resize(H);
    mTwin.resize(H);
    mEdge.resize(H);
    mSignpost.resize(H);
    const uint32_t m = static_cast<uint32_t>(vertexCount());
    const uint32_t eJ = static_cast<uint32_t>(edgeCount()), eK = eJ + 1, eL = eJ + 2;
    const auto set = [&](uint32_t h, uint32_t tail, uint32_t twin, uint32_t edge, double signpost) {
        mTail[h] = tail;
        mTwin[h] = twin;
        mEdge[h] = edge;
        mSignpost[h] = signpost;
    };
    for (int n = 0; n < 4; n++) set(newSlots[n], moved[n].tail, remap(moved[n].twin), moved[n].edge, moved[n].signpost);
    set(3 * A + 1, i, 3 * D + 0, e, sIJ);
    set(3 * A + 2, m, 3 * C + 2, eK, sMK);
    set(3 * C + 0, m, 3 * B + 1, eJ, 0.0);
    set(3 * C + 2, k, 3 * A + 2, eK, sKM);
    set(3 * B + 1, j, 3 * C + 0, eJ, sJI);
    set(3 * B + 2, m, 3 * D + 2, eL, sML);
    set(3 * D + 0, m, 3 * A + 1, e, kPiD);
    set(3 * D + 2, l, 3 * B + 2, eL, sLM);
    for (int n = 0; n < 4; n++) {
        const uint32_t h = newSlots[n];
        mTwin[mTwin[h]] = h;
        mEdgeHalfedge[mEdge[h]] = h;
    }
    mLength[e] = half;
    mEdgeHalfedge[e] = 3 * A + 1;
    mLength.insert(mLength.end(), {half, lkm, llm});
    mEdgeHalfedge.insert(mEdgeHalfedge.end(), {3 * C + 0, 3 * A + 2, 3 * B + 2});
    mVertexHalfedge.push_back(3 * C + 0);
    mVertexHalfedge[i] = 3 * A + 1;
    mVertexHalfedge[j] = 3 * B + 1;
    mVertexHalfedge[k] = 3 * A + 0;
    mVertexHalfedge[l] = 3 * B + 0;
    mAngleSum.push_back(kTwoPi);
    mInserted.push_back(inserted);
    mFaceLevel.push_back(mFaceLevel[A]);
    mFaceLevel.push_back(mFaceLevel[B]);
    mGreenPartner.insert(mGreenPartner.end(), {kNoFace, kNoFace});
    mGreenMid.insert(mGreenMid.end(), {0u, 0u});
    if (!points) return m;
    // Exact map: m = (i + j) / 2, so i = 2m − j (or j = 2m − i) in the barycentric combination.
    for (uint32_t n : inA) {
        const Vec3d &b = points->bary(n);
        const double bi = b[cA], bj = b[(cA + 1) % 3], bk = b[(cA + 2) % 3];
        if (bi >= bj) {
            points->place(n, A, Vec3d(bk, bi - bj, 2.0 * bj));
        } else {
            points->place(n, C, Vec3d(2.0 * bi, bj - bi, bk));
        }
    }
    for (uint32_t n : inB) {
        const Vec3d &b = points->bary(n);
        const double bj = b[cB], bi = b[(cB + 1) % 3], bl = b[(cB + 2) % 3];
        if (bj >= bi) {
            points->place(n, B, Vec3d(bl, bj - bi, 2.0 * bi));
        } else {
            points->place(n, D, Vec3d(2.0 * bj, bi - bj, bl));
        }
    }
    return m;
}

void IntrinsicTriangulation::dissolveGreen(uint32_t f)
{
    const uint32_t g = mGreenPartner[f];
    if (g == kNoFace) return;
    mGreenPartner[f] = mGreenPartner[g] = kNoFace;
}

uint32_t IntrinsicTriangulation::greenHalfedge(uint32_t f, bool inner) const
{
    // The halfedge of green half f shared with its partner (inner), or the one opposite the midpoint (outer).
    for (uint32_t he = 3 * f; he < 3 * f + 3; he++) {
        if (inner && face(mTwin[he]) == mGreenPartner[f]) return he;
        if (!inner && mTail[he] != mGreenMid[f] && head(he) != mGreenMid[f]) return he;
    }
    return UINT32_MAX;
}

size_t IntrinsicTriangulation::refine(const std::vector<uint8_t> &faceMarks, const std::vector<SurfacePoint *> &points)
{
    const uint32_t F0 = static_cast<uint32_t>(faceCount()), E0 = static_cast<uint32_t>(edgeCount());
    // Red-green closure. A face to refine is split 1:4 (red); a face with two or three split edges too; one
    // with a single split edge is bisected (green), and the two halves are remembered as a green pair. A
    // green pair is never split or bisected again: whenever one of its halves is marked or any of its edges
    // must split, the bisection is undone into its parent's Loop pattern (upgrade): the parent's other two
    // edges split at their midpoints and the bisecting edge flipped, which is the parent's 1:4 split.
    std::vector<uint8_t> red(F0, 0), upgrade(F0, 0), edgeMark(E0, 0);
    std::vector<uint32_t> edgeWork;
    const auto markEdge = [&](uint32_t e) {
        if (edgeMark[e]) return;
        edgeMark[e] = 1;
        edgeWork.push_back(e);
    };
    const auto makeRed = [&](uint32_t f) {
        if (red[f]) return;
        red[f] = 1;
        for (int c = 0; c < 3; c++) markEdge(mEdge[3 * f + c]);
    };
    const auto requestUpgrade = [&](uint32_t f) {
        const uint32_t g = mGreenPartner[f];
        if (upgrade[f]) return;
        upgrade[f] = upgrade[g] = 1;
        markEdge(mEdge[greenHalfedge(f, false)]);
        markEdge(mEdge[greenHalfedge(g, false)]);
    };
    for (uint32_t f = 0; f < F0 && f < faceMarks.size(); f++) {
        if (!faceMarks[f]) continue;
        if (mGreenPartner[f] != kNoFace) {
            requestUpgrade(f);
        } else {
            makeRed(f);
        }
    }
    while (!edgeWork.empty()) {
        const uint32_t e = edgeWork.back();
        edgeWork.pop_back();
        for (uint32_t g : {face(mEdgeHalfedge[e]), face(mTwin[mEdgeHalfedge[e]])}) {
            if (red[g] || upgrade[g]) continue;
            if (mGreenPartner[g] != kNoFace) {
                requestUpgrade(g);
                continue;
            }
            int marked = 0;
            for (int d = 0; d < 3; d++) marked += edgeMark[mEdge[3 * g + d]];
            if (marked >= 2) makeRed(g);
        }
    }
    PointSet set(points, F0);
    PointSet *ps = points.empty() ? nullptr : &set;
    // root: the face before this refinement that each face comes from. In a red face, the first split joins
    // a midpoint to the opposite corner and the later ones join midpoints; flipping that first edge afterwards
    // (inside the flat parent face, a parallelogram) gives Loop's pattern.
    std::vector<uint32_t> root(F0), firstEdge(F0, UINT32_MAX);
    std::iota(root.begin(), root.end(), 0u);
    const std::vector<int> parentLevel(mFaceLevel);
    std::vector<uint8_t> edgeDone(E0, 0), processed(F0, 0);
    std::vector<std::array<uint32_t, 3>> greens;  // bisected faces: half, other half, midpoint
    // Splits edge e (once). Each side's face is part of a red face (the first split there is remembered), of
    // a green pair still to be upgraded (its pattern), or else bisected: a new green pair.
    const auto split = [&](uint32_t e) {
        if (edgeDone[e]) return;
        edgeDone[e] = 1;
        const uint32_t A = face(mEdgeHalfedge[e]), B = face(mTwin[mEdgeHalfedge[e]]);
        const uint32_t edgesBefore = static_cast<uint32_t>(edgeCount()), facesBefore = static_cast<uint32_t>(faceCount());
        const uint32_t m = splitEdgeImpl(e, ps);
        if (m == UINT32_MAX) return;
        root.push_back(root[A]);
        root.push_back(root[B]);
        const uint32_t sides[2][3] = {{A, facesBefore, edgesBefore + 1}, {B, facesBefore + 1, edgesBefore + 2}};
        for (const auto &side : sides) {
            const uint32_t r = root[side[0]];
            if (red[r]) {
                if (firstEdge[r] == UINT32_MAX) firstEdge[r] = side[2];
            } else if (!(upgrade[r] && !processed[r])) {
                greens.push_back({side[0], side[1], m});
            }
        }
    };
    // Upgrades first, coarsest first (a finer pair's outer edge may be a coarser pair's half edge, which must
    // split only once that coarser pair is a 1:4 split): the parent's outer edges, then the bisecting edge
    // flipped (k-n-m-o is a parallelogram inside the flat parent face).
    struct Upgrade {
        uint32_t faces[2], inner, outer[2];
        int level;
    };
    std::vector<Upgrade> upgrades;
    for (uint32_t f = 0; f < F0; f++) {
        const uint32_t g = mGreenPartner[f];
        if (!upgrade[f] || g < f) continue;
        upgrades.push_back({{f, g}, mEdge[greenHalfedge(f, true)], {mEdge[greenHalfedge(f, false)], mEdge[greenHalfedge(g, false)]}, mFaceLevel[f]});
    }
    std::stable_sort(upgrades.begin(), upgrades.end(), [](const Upgrade &a, const Upgrade &b) { return a.level < b.level; });
    for (const Upgrade &u : upgrades) dissolveGreen(u.faces[0]);
    for (const Upgrade &u : upgrades) {
        for (uint32_t e : u.outer) split(e);
        [[maybe_unused]] const bool flipped = flipEdgeImpl(u.inner, ps);
        assert(flipped);
        processed[u.faces[0]] = processed[u.faces[1]] = 1;
    }
    const size_t bisectionsUndone = upgrades.size();
    for (uint32_t e = 0; e < E0; e++) {
        if (edgeMark[e]) split(e);
    }
    size_t count = bisectionsUndone;
    for (uint32_t f = 0; f < F0; f++) {
        if (!red[f]) continue;
        count++;
        if (firstEdge[f] == UINT32_MAX) continue;
        // The quad is a parallelogram inside the flat parent face and both ends have degree >= 3.
        [[maybe_unused]] const bool flipped = flipEdgeImpl(firstEdge[f], ps);
        assert(flipped);
    }
    for (uint32_t g = 0; g < faceCount(); g++) {
        if (red[root[g]] || upgrade[root[g]]) mFaceLevel[g] = parentLevel[root[g]] + 1;
    }
    for (const auto &[half, other, m] : greens) {
        mGreenPartner[half] = other;
        mGreenPartner[other] = half;
        mGreenMid[half] = mGreenMid[other] = m;
    }
    flipToDelaunayImpl(ps);
    return count;
}

size_t IntrinsicTriangulation::flipToDelaunay(const std::vector<SurfacePoint *> &points)
{
    if (points.empty()) return flipToDelaunayImpl(nullptr);
    PointSet set(points, faceCount());
    return flipToDelaunayImpl(&set);
}

size_t IntrinsicTriangulation::flipToDelaunayImpl(PointSet *points)
{
    constexpr double kDelaunayTolerance = 1e-13;
    const size_t E = mLength.size();
    std::deque<uint32_t> queue;
    std::vector<uint8_t> queued(E, 1);
    for (uint32_t e = 0; e < E; e++) queue.push_back(e);
    size_t flips = 0;
    const size_t maxFlips = 1000 * std::max<size_t>(E, 1);
    while (!queue.empty() && flips < maxFlips) {
        const uint32_t e = queue.front();
        queue.pop_front();
        queued[e] = 0;
        if (cotanWeight(e) >= -kDelaunayTolerance) continue;
        // Green halves keep their bisection (refine undoes it when they refine): their edges are not flipped.
        const uint32_t h0 = mEdgeHalfedge[e];
        if (mGreenPartner[face(h0)] != kNoFace || mGreenPartner[face(mTwin[h0])] != kNoFace) continue;
        if (!flipEdgeImpl(e, points)) continue;
        flips++;
        const uint32_t he = mEdgeHalfedge[e];
        for (uint32_t h : {next(he), prev(he), next(mTwin[he]), prev(mTwin[he])}) {
            const uint32_t q = mEdge[h];
            if (!queued[q]) {
                queued[q] = 1;
                queue.push_back(q);
            }
        }
    }
    return flips;
}

double IntrinsicTriangulation::polarAngle(uint32_t he, const Vec2d &w) const
{
    const auto P = layout(face(he));
    const int c = static_cast<int>(he % 3);
    const Vec2d e = P[(c + 1) % 3] - P[c];
    const double alpha = std::atan2(Cross(e, w), e.dot(w));
    return WrapAngle(mSignpost[he] + alpha * kTwoPi / mAngleSum[mTail[he]]);
}

void IntrinsicTriangulation::vertexDirection(uint32_t v, double angle, SurfacePoint &point, Vec2d &direction, uint32_t *along) const
{
    angle = WrapAngle(angle);
    const uint32_t start = mVertexHalfedge[v];
    uint32_t he = start, best = start;
    double bestExcess = std::numeric_limits<double>::infinity(), bestOffset = 0.0;
    do {
        const double width = cornerAngle(he) * kTwoPi / mAngleSum[v];
        const double offset = WrapAngle(angle - mSignpost[he]);
        const double excess = offset - width;  // <= 0 inside this corner's wedge
        if (excess <= 0.0) {
            best = he;
            bestOffset = offset;
            break;
        }
        // Rounding near the wedge end: the closest wedge wins.
        if (excess < bestExcess || (kTwoPi - offset) < bestExcess) {
            if (kTwoPi - offset < excess) {
                bestExcess = kTwoPi - offset;
                bestOffset = 0.0;
            } else {
                bestExcess = excess;
                bestOffset = width;
            }
            best = he;
        }
        he = nextOutgoing(he);
    } while (he != start);
    const double theta = cornerAngle(best);
    const double width = theta * kTwoPi / mAngleSum[v];
    // A direction along one of the wedge's edges (up to rounding in the normalized angles): the trace
    // walks that edge exactly instead of a direction nudged into the wedge.
    constexpr double kAlignTolerance = 1e-12;
    if (along) {
        *along = UINT32_MAX;
        if (bestOffset <= kAlignTolerance) {
            *along = best;
        } else if (width - bestOffset <= kAlignTolerance) {
            *along = nextOutgoing(best);
        }
        if (*along != UINT32_MAX) {
            const auto P = layout(face(*along));
            const int c = static_cast<int>(*along % 3);
            direction = (P[(c + 1) % 3] - P[c]).normalized();
            point = cornerPoint(face(*along), c);
            return;
        }
    }
    // Strictly inside the wedge, so the trace leaves through the opposite edge: the nudge must stay
    // well above rounding of the barycentric rates, even for skinny corners.
    constexpr double kWedgeMargin = 1e-9;
    const double alpha = std::clamp(bestOffset * mAngleSum[v] / kTwoPi, kWedgeMargin * theta, (1.0 - kWedgeMargin) * theta);
    const auto P = layout(face(best));
    const int c = static_cast<int>(best % 3);
    const Vec2d e = (P[(c + 1) % 3] - P[c]).normalized();
    direction = std::cos(alpha) * e + std::sin(alpha) * Perp(e);
    point = cornerPoint(face(best), c);
}

IntrinsicTriangulation::Trace IntrinsicTriangulation::trace(const SurfacePoint &start, const Vec2d &displacement, std::vector<EdgeCrossing> *crossings) const
{
    const double length = displacement.norm();
    if (!(length > 0.0)) {
        Trace result;
        result.end = start;
        return result;
    }
    const Vec2d d = displacement / length;
    return traceLoop(start, d, length, d, crossings);
}

IntrinsicTriangulation::Trace IntrinsicTriangulation::traceFromVertex(uint32_t v, double angle, double distance, std::vector<EdgeCrossing> *crossings) const
{
    SurfacePoint point;
    Vec2d d;
    uint32_t along = UINT32_MAX;
    vertexDirection(v, angle, point, d, &along);
    if (!(distance > 0.0)) {
        Trace result;
        result.end = point;
        result.startDirection = result.direction = d;
        return result;
    }
    return traceLoop(point, d, distance, d, crossings, along);
}

IntrinsicTriangulation::Trace IntrinsicTriangulation::traceLoop(SurfacePoint point, Vec2d d, double remaining, const Vec2d &startDirection,
                                                                std::vector<EdgeCrossing> *crossings, uint32_t along, double alongStart, bool moved) const
{
    constexpr double kVertexTolerance = 1e-10;
    constexpr double kNegativeRate = 1e-14;
    const size_t maxSteps = 4 * mTail.size() + 1000;
    CleanBary(point.bary);
    for (size_t step = 0; step < maxSteps && remaining > 0.0; step++) {
        if (along != UINT32_MAX) {
            // Exactly along halfedge `along` from parameter alongStart: end on it, or reach its head and
            // continue through the vertex with the incoming direction.
            const double L = halfedgeLength(along), left = L * (1.0 - alongStart);
            const auto P = layout(face(along));
            const int c = static_cast<int>(along % 3);
            d = (P[(c + 1) % 3] - P[c]).normalized();
            if (remaining < left) {
                point = halfedgePoint(along, alongStart + remaining / L);
                remaining = 0.0;
                break;
            }
            remaining -= left;
            moved = true;
            const uint32_t he = next(along);  // out of the head, in face(along)
            along = UINT32_MAX;
            alongStart = 0.0;
            point = cornerPoint(face(he), static_cast<int>(he % 3));
            if (!(remaining > 0.0)) break;
            vertexDirection(mTail[he], polarAngle(he, -d) + kPiD, point, d);
            continue;
        }
        const uint32_t f = point.face;
        const auto P = layout(f);
        const double A2 = Cross(P[1] - P[0], P[2] - P[0]);
        Vec3d delta;
        for (int c = 0; c < 3; c++) delta[c] = Cross(P[(c + 2) % 3] - P[(c + 1) % 3], d) / A2;
        double tExit = std::numeric_limits<double>::infinity();
        int cExit = -1;
        for (int c = 0; c < 3; c++) {
            if (delta[c] < -kNegativeRate) {
                const double t = std::max(0.0, point.bary[c]) / (-delta[c]);
                if (t < tExit) {
                    tExit = t;
                    cExit = c;
                }
            }
        }
        if (cExit < 0 || tExit >= remaining) {
            point.bary += remaining * delta;
            CleanBary(point.bary);
            remaining = 0.0;
            break;
        }
        point.bary += tExit * delta;
        point.bary[cExit] = 0.0;
        remaining -= tExit;
        if (tExit > 0.0) moved = true;
        const int c1 = (cExit + 1) % 3, c2 = (cExit + 2) % 3;
        if (point.bary[c1] < kVertexTolerance || point.bary[c2] < kVertexTolerance) {
            // Through a vertex: continue with equal (normalized) angles on both sides of the path. A
            // trace that starts at the vertex keeps its own direction instead.
            const int m = (point.bary[c1] < kVertexTolerance) ? c2 : c1;
            const uint32_t he = 3 * f + m;
            const double angle = moved ? polarAngle(he, -d) + kPiD : polarAngle(he, d);
            vertexDirection(mTail[he], angle, point, d);
            continue;
        }
        const uint32_t he = 3 * f + c1;  // the edge opposite cExit, from corner c1 to c2
        const double s = point.bary[c2] / (point.bary[c1] + point.bary[c2]);
        if (crossings) crossings->push_back({he, s});
        const uint32_t tw = mTwin[he];
        const uint32_t g = face(tw);
        const int ct = static_cast<int>(tw % 3);
        const auto Q = layout(g);
        const Vec2d uf = (P[c2] - P[c1]).normalized();
        const Vec2d ug = (Q[ct] - Q[(ct + 1) % 3]).normalized();
        d = d.dot(uf) * ug + d.dot(Perp(uf)) * Perp(ug);
        d.normalize();
        point.face = g;
        point.bary = Vec3d::Zero();
        point.bary[ct] = s;
        point.bary[(ct + 1) % 3] = 1.0 - s;
    }
    Trace result;
    result.end = point;
    result.startDirection = startDirection;
    result.direction = d;
    return result;
}

IntrinsicTriangulation::Trace IntrinsicTriangulation::inputTrace(uint32_t he, const Vec2d &w) const
{
    const IntrinsicTriangulation &in = input();
    const uint32_t v = mTail[he];
    const double angle = polarAngle(he, w);
    if (v < mVertices.size()) return in.traceFromVertex(v, angle, w.norm());
    // An inserted vertex: from its point on the input mesh, its frame turned into that face's layout.
    const Inserted &s = mInserted[v - mVertices.size()];
    const double a = s.frame + angle;
    SurfacePoint start;
    start.face = s.point.face;
    start.bary = s.point.bary;
    const Vec2d d(std::cos(a), std::sin(a));
    // On an input edge (a zero barycentric) and heading along it: walk the edge exactly.
    for (int c0 = 0; c0 < 3; c0++) {
        if (start.bary[c0] != 0.0) continue;
        const uint32_t g = 3 * start.face + (c0 + 1) % 3;  // the input edge, in the stored face
        const auto P = in.layout(start.face);
        const int cg = static_cast<int>(g % 3);
        const Vec2d u = (P[(cg + 1) % 3] - P[cg]).normalized();
        const double turn = std::atan2(Cross(u, d), u.dot(d));
        const double t = start.bary[(cg + 1) % 3];  // parameter of the point along g
        constexpr double kAlignTolerance = 1e-12;
        if (std::abs(turn) <= kAlignTolerance) return in.traceLoop(start, u, w.norm(), u, nullptr, g, t, false);
        if (kPiD - std::abs(turn) <= kAlignTolerance) return in.traceLoop(start, -u, w.norm(), -u, nullptr, in.twin(g), 1.0 - t, false);
    }
    return in.trace(start, w.norm() * d);
}

IntrinsicTriangulation::ExtrinsicPoint IntrinsicTriangulation::vertexExtrinsic(uint32_t v) const
{
    if (v >= mVertices.size()) return mInserted[v - mVertices.size()].point;
    const SurfacePoint p = input().vertexPoint(v);
    return {p.face, p.bary, mVertices[v].cast<double>()};
}

Vec3d IntrinsicTriangulation::vertexPosition(uint32_t v) const
{
    return (v < mVertices.size()) ? Vec3d(mVertices[v].cast<double>()) : mInserted[v - mVertices.size()].point.position;
}

IntrinsicTriangulation::ExtrinsicPoint IntrinsicTriangulation::toExtrinsic(uint32_t intrinsicFace, const Vec3d &bary) const
{
    if (!mInput) {
        SurfacePoint p;
        p.face = intrinsicFace;
        p.bary = bary;
        return {intrinsicFace, bary, InputPosition(*this, p)};
    }
    // Trace on the input mesh from the corner nearest the point, along the point's direction in the
    // shared vertex frame (an inserted corner traces from its own point on the input mesh).
    const auto P = layout(intrinsicFace);
    const Vec2d x = bary[0] * P[0] + bary[1] * P[1] + bary[2] * P[2];
    int m = 0;
    for (int c = 1; c < 3; c++) {
        if (bary[c] > bary[m]) m = c;
    }
    const uint32_t he = 3 * intrinsicFace + m;
    const Vec2d w = x - P[m];
    if (w.norm() <= 1e-14 * halfedgeLength(he)) return vertexExtrinsic(mTail[he]);
    const Trace r = inputTrace(he, w);
    return {r.end.face, r.end.bary, InputPosition(input(), r.end)};
}

IntrinsicTriangulation::ExtrinsicFrame IntrinsicTriangulation::extrinsicFrame(const SurfacePoint &p) const
{
    ExtrinsicFrame frame;
    if (!mInput) {
        frame.point = toExtrinsic(p);
        return frame;
    }
    const auto P = layout(p.face);
    const Vec2d x = p.bary[0] * P[0] + p.bary[1] * P[1] + p.bary[2] * P[2];
    int m = 0;
    for (int c = 1; c < 3; c++) {
        if (p.bary[c] > p.bary[m]) m = c;
    }
    const uint32_t he = 3 * p.face + m;
    const Vec2d w = x - P[m];
    // Into p.face, from p: the corner's bisector at a vertex, the inward normal of the edge p lies on, or
    // none inside the face.
    Vec2d inward = Vec2d::Zero();
    if (w.norm() > 1e-14 * halfedgeLength(he)) {
        // As toExtrinsic: the geodesic from the corner, its end direction on the input mesh.
        const Trace r = inputTrace(he, w);
        frame.point = {r.end.face, r.end.bary, InputPosition(input(), r.end)};
        frame.rotation = std::atan2(w.y(), w.x()) - std::atan2(r.direction.y(), r.direction.x());
        for (int c = 0; c < 3; c++) {
            if (p.bary[c] != 0.0) continue;
            const Vec2d edge = P[(c + 2) % 3] - P[(c + 1) % 3];  // the edge opposite corner c, counter-clockwise
            inward = Vec2d(-edge.y(), edge.x());
        }
    } else {
        // At the corner's vertex: the geodesic along the corner's bisector.
        const Vec2d d = (P[(m + 1) % 3] - P[m]).normalized() + (P[(m + 2) % 3] - P[m]).normalized();
        inward = d;
        const double angle = polarAngle(he, d);
        const uint32_t v = mTail[he];
        double inputAngle = 0.0;
        if (v >= mVertices.size()) {
            const Inserted &s = mInserted[v - mVertices.size()];
            frame.point = s.point;
            inputAngle = s.frame + angle;  // inserted vertices are flat: polar angles are true angles
        } else {
            SurfacePoint start;
            Vec2d direction;
            input().vertexDirection(v, angle, start, direction);
            frame.point = {start.face, start.bary, mVertices[v].cast<double>()};
            inputAngle = std::atan2(direction.y(), direction.x());
        }
        frame.rotation = std::atan2(d.y(), d.x()) - inputAngle;
    }
    // On an input edge (one zero barycentric): the input face on p.face's side of it.
    const Vec3d &b = frame.point.bary;
    const int zeros = (b[0] == 0.0) + (b[1] == 0.0) + (b[2] == 0.0);
    if (zeros != 1 || inward.squaredNorm() == 0.0) return frame;
    const IntrinsicTriangulation &in = input();
    const int c0 = (b[0] == 0.0) ? 0 : (b[1] == 0.0) ? 1 : 2;
    const uint32_t g = 3 * frame.point.face + (c0 + 1) % 3;  // the input edge, in point.face
    const auto L = in.layout(frame.point.face);
    const Vec2d ug = L[(c0 + 2) % 3] - L[(c0 + 1) % 3];
    const double c = std::cos(-frame.rotation), s = std::sin(-frame.rotation);
    const Vec2d inwardInput(c * inward.x() - s * inward.y(), s * inward.x() + c * inward.y());
    if (Cross(ug, inwardInput) >= 0.0) return frame;
    const uint32_t tg = in.twin(g);
    const auto T = in.layout(in.face(tg));
    const int ct = static_cast<int>(tg % 3);
    const Vec2d ut = T[(ct + 1) % 3] - T[ct];  // g's direction reversed, in the twin's layout
    const double t = b[(c0 + 2) % 3];          // p's parameter along g
    const SurfacePoint q = halfedgePoint(tg, 1.0 - t);
    frame.point.face = q.face;
    frame.point.bary = q.bary;
    frame.rotation += std::atan2(ug.y(), ug.x()) - std::atan2(-ut.y(), -ut.x());
    return frame;
}

Vec2d IntrinsicTriangulation::extrinsicToLayout(const ExtrinsicFrame &frame, const Vec3d &v) const
{
    const IntrinsicTriangulation &in = input();
    const uint32_t F = frame.point.face;
    Vec3d X[3];
    for (int c = 0; c < 3; c++) X[c] = mVertices[in.tail(3 * F + c)].cast<double>();
    const Vec3d e1 = X[1] - X[0], e2 = X[2] - X[0];
    // v's component in the face's plane, as e1, e2 coordinates (the normal part drops out).
    Eigen::Matrix2d G;
    G << e1.dot(e1), e1.dot(e2), e1.dot(e2), e2.dot(e2);
    const Eigen::Vector2d a = G.inverse() * Eigen::Vector2d(e1.dot(v), e2.dot(v));
    const auto L = in.layout(F);
    const Vec2d u = a[0] * (L[1] - L[0]) + a[1] * (L[2] - L[0]);
    const double c = std::cos(frame.rotation), s = std::sin(frame.rotation);
    return Vec2d(c * u.x() - s * u.y(), s * u.x() + c * u.y());
}
}  // namespace springls
