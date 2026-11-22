#include <spatial/SpringlHashGrid2D.hpp>

namespace springls
{
namespace
{
// Extra cells around the springls' bounding region when the table is (re)sized, so springls can
// drift a while before the table has to grow again.
constexpr int64_t kTableMargin = 8;
}  // namespace

void SpringlHashGrid2D::ensureTable(int64_t x0, int64_t y0, int64_t x1, int64_t y1)
{
    if (mTableW > 0 && inTable(x0, y0) && inTable(x1, y1)) return;
    if (mTableW > 0) {  // keep everything already covered
        x0 = std::min(x0, mTableX0);
        y0 = std::min(y0, mTableY0);
        x1 = std::max(x1, mTableX0 + mTableW - 1);
        y1 = std::max(y1, mTableY0 + mTableH - 1);
    }
    mTableX0 = x0 - kTableMargin;
    mTableY0 = y0 - kTableMargin;
    mTableW = x1 - x0 + 1 + 2 * kTableMargin;
    mTableH = y1 - y0 + 1 + 2 * kTableMargin;
    mTable.assign(static_cast<size_t>(mTableW * mTableH), -1);
    for (size_t b = 0; b < mBuckets.size(); b++) {
        if (mBuckets[b].empty()) continue;
        const auto &cell = mBucketCell[b];
        mTable[static_cast<size_t>((cell[1] - mTableY0) * mTableW + (cell[0] - mTableX0))] = static_cast<int>(b);
    }
}

void SpringlHashGrid2D::insert(uint32_t springl, int64_t cx, int64_t cy)
{
    ensureTable(cx, cy, cx, cy);
    int &entry = mTable[static_cast<size_t>((cy - mTableY0) * mTableW + (cx - mTableX0))];
    if (entry < 0) {
        if (!mFreeBuckets.empty()) {
            entry = mFreeBuckets.back();
            mFreeBuckets.pop_back();
        } else {
            entry = static_cast<int>(mBuckets.size());
            mBuckets.emplace_back();
            mBucketCell.emplace_back();
        }
        mBucketCell[entry] = {cx, cy};
    }
    std::vector<uint32_t> &items = mBuckets[entry];
    // Keep buckets sorted by index so iteration order (and tie breaking) is deterministic.
    const auto pos = std::lower_bound(items.begin(), items.end(), springl);
    const size_t slot = static_cast<size_t>(pos - items.begin());
    items.insert(pos, springl);
    for (size_t k = slot; k < items.size(); k++) mItemSlot[items[k]] = static_cast<uint32_t>(k);
    mItemCx[springl] = static_cast<int32_t>(cx);
    mItemCy[springl] = static_cast<int32_t>(cy);
}

void SpringlHashGrid2D::remove(uint32_t springl)
{
    const int64_t cx = mItemCx[springl], cy = mItemCy[springl];
    int &entry = mTable[static_cast<size_t>((cy - mTableY0) * mTableW + (cx - mTableX0))];
    std::vector<uint32_t> &items = mBuckets[entry];
    const size_t slot = mItemSlot[springl];
    items.erase(items.begin() + static_cast<std::ptrdiff_t>(slot));
    for (size_t k = slot; k < items.size(); k++) mItemSlot[items[k]] = static_cast<uint32_t>(k);
    if (items.empty()) {
        mFreeBuckets.push_back(entry);
        entry = -1;
    }
}

void SpringlHashGrid2D::rebuild(const Points2 &particles)
{
    const size_t N = particles.size();
    for (std::vector<uint32_t> &bucket : mBuckets) bucket.clear();
    mBuckets.resize(0);
    mBucketCell.clear();
    mFreeBuckets.clear();
    mItemCx.assign(N, 0);
    mItemCy.assign(N, 0);
    mItemSlot.assign(N, 0);
    mTableW = mTableH = 0;
    if (N > 0) {
        int64_t x0 = cellCoord(particles[0].x()), x1 = x0, y0 = cellCoord(particles[0].y()), y1 = y0;
        for (size_t s = 1; s < N; s++) {
            const int64_t cx = cellCoord(particles[s].x()), cy = cellCoord(particles[s].y());
            x0 = std::min(x0, cx);
            x1 = std::max(x1, cx);
            y0 = std::min(y0, cy);
            y1 = std::max(y1, cy);
        }
        ensureTable(x0, y0, x1, y1);
    } else {
        mTable.clear();
    }
    // Springls are inserted in index order, so each bucket comes out sorted.
    for (size_t s = 0; s < N; s++) {
        insert(static_cast<uint32_t>(s), cellCoord(particles[s].x()), cellCoord(particles[s].y()));
    }
    mValid = true;
}

void SpringlHashGrid2D::update(const Points2 &particles, const Points2 &vertexes)
{
    const size_t N = particles.size();
    // Reach: the longest arm, so a query box widened by it finds every segment it could touch.
    float reach = 0.0f;
    for (size_t s = 0; s < N; s++) {
        reach = std::max({reach, distance(particles[s], vertexes[2 * s]), distance(particles[s], vertexes[2 * s + 1])});
    }
    mReach = reach;
    if (!mValid || N < mItemCx.size()) {
        rebuild(particles);
        return;
    }
    // Existing springls: move only those whose particle changed cell.
    const size_t existing = mItemCx.size();
    for (size_t s = 0; s < existing; s++) {
        const int64_t cx = cellCoord(particles[s].x()), cy = cellCoord(particles[s].y());
        if (cx == mItemCx[s] && cy == mItemCy[s]) continue;
        remove(static_cast<uint32_t>(s));
        insert(static_cast<uint32_t>(s), cx, cy);
    }
    // Appended springls.
    mItemCx.resize(N, 0);
    mItemCy.resize(N, 0);
    mItemSlot.resize(N, 0);
    for (size_t s = existing; s < N; s++) {
        insert(static_cast<uint32_t>(s), cellCoord(particles[s].x()), cellCoord(particles[s].y()));
    }
}

size_t SpringlHashGrid2D::memoryBytes() const
{
    size_t bytes = sizeof(*this);
    for (const auto &bucket : mBuckets) bytes += sizeof(bucket) + bucket.capacity() * sizeof(uint32_t);
    bytes += mTable.capacity() * sizeof(int) + mBucketCell.capacity() * sizeof(mBucketCell[0]);
    bytes += (mItemCx.capacity() + mItemCy.capacity()) * sizeof(int32_t) + mItemSlot.capacity() * sizeof(uint32_t) + mFreeBuckets.capacity() * sizeof(int);
    return bytes;
}
}  // namespace springls
