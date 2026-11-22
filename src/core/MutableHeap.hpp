#pragma once
// Priority queues whose items can be found and re-prioritized by key (decrease-key), for front
// propagation such as fast marching: one item per key, so a value that improves moves its item in
// place instead of pushing a duplicate.
//
// Our copy of hoc::MutableHeap from header-only-club (BSD 3-Clause, Copyright (c) 2022, Blake Lucas),
// at commit 535ba83 ("Add DenseMutableHeap and speed up MutableHeap"), adapted to this library: namespace
// springls, plain assert() in place of hoc's Expects, no hoc headers.
//
// BasicMutableHeap is a binary min-heap (with the default std::less) in slots 1..size of one array, slot 0
// a sentinel. Its `Index` maps a key to the key's heap slot:
// - MutableHeap<Key, T>: a hash map, for any hashable key.
// - DenseMutableHeap<T>: a flat array indexed by a uint32_t key (vertex, face or voxel ids). Lookups and
//   the index writes on every sift step are array accesses; memory is O(largest key). clear() resets only
//   the live entries, so one heap reused across many runs costs O(items) per run, not O(key range).
//
// Ties in value are broken by the smaller key, so the order is deterministic (the same as a
// std::priority_queue of (value, key) pairs with std::greater).
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace springls
{
namespace detail
{
// Key -> heap slot lookup backed by a hash map. Works for any hashable key.
template <class Key>
struct HashHeapIndex {
    static constexpr uint32_t npos = std::numeric_limits<uint32_t>::max();

    [[nodiscard]] uint32_t find(const Key &key) const
    {
        const auto it = map.find(key);
        return it == map.end() ? npos : it->second;
    }
    void set(const Key &key, uint32_t slot) { map[key] = slot; }
    void erase(const Key &key) { map.erase(key); }
    void reserve(size_t n) { map.reserve(n); }
    void reserveKeys(size_t n) { map.reserve(n); }
    template <class Items>
    void clear(const Items &)
    {
        map.clear();
    }

    std::unordered_map<Key, uint32_t> map;
};

// Key -> heap slot lookup backed by a flat array, for dense uint32_t keys.
struct DenseHeapIndex {
    static constexpr uint32_t npos = std::numeric_limits<uint32_t>::max();

    [[nodiscard]] uint32_t find(uint32_t key) const { return key < slots.size() ? slots[key] : npos; }
    void set(uint32_t key, uint32_t slot)
    {
        if (key >= slots.size()) slots.resize(std::max<size_t>(size_t(key) + 1, slots.size() * 2), npos);
        slots[key] = slot;
    }
    void erase(uint32_t key)
    {
        if (key < slots.size()) slots[key] = npos;
    }
    void reserve(size_t) {}
    // Pre-sizes for keys in [0, n) so pushes never grow the table.
    void reserveKeys(size_t n)
    {
        if (n > slots.size()) slots.resize(n, npos);
    }
    // Resets only the live slots, so clearing is O(heap size), not O(key range).
    template <class Items>
    void clear(const Items &items)
    {
        for (const auto &item : items) slots[item.key] = npos;
    }

    std::vector<uint32_t> slots;
};
}  // namespace detail

// Priority queue with O(log n) update by key. Slots 1..size are heap nodes; slot 0 is a sentinel.
// `Index` maps a key to its heap slot; use the MutableHeap / DenseMutableHeap aliases below. Key and T
// must be default-constructible (scratch and growth slots).
template <class Key, class T, class Compare, class Index>
class BasicMutableHeap
{
    static_assert(std::is_default_constructible_v<Key> && std::is_default_constructible_v<T>, "MutableHeap requires default-constructible Key and T");
    static constexpr uint32_t npos = Index::npos;

public:
    struct Item {
        Key key{};
        T value{};
        Item() = default;
        Item(const Key &keyIn, const T &valueIn) : key(keyIn), value(valueIn) {}
        // The same order as a default (std::less) heap: smaller value first, ties by smaller key.
        bool operator<(const Item &item) const
        {
            if (value < item.value) return true;
            if (item.value < value) return false;
            return key < item.key;
        }
    };

    explicit BasicMutableHeap(const Compare &compare = Compare()) : mCompare(compare), mSize(0)
    {
        mStorage.resize(1);  // slot 0: sentinel / scratch
    }

    [[nodiscard]] bool empty() const noexcept { return mSize == 0; }

    // Reserves heap slots and key -> slot capacity (fewer reallocations / rehashes in bulk use).
    void reserve(size_t sz)
    {
        if (sz == 0) return;
        // + 1 for one-based indexing (slot 0 holds no heap data).
        if (mStorage.size() < sz + 1) mStorage.resize(sz + 1);
        mIndex.reserve(sz);
    }

    // Reserves room for keys in [0, n). For a dense heap this pre-sizes the lookup table.
    void reserveKeys(size_t n) { mIndex.reserveKeys(n); }

    [[nodiscard]] std::optional<Item> top() const
    {
        if (empty()) return std::nullopt;
        return mStorage[1];
    }

    // Heap items in storage order (not sorted); the sentinel slot is excluded.
    [[nodiscard]] std::span<const Item> items() const noexcept { return {mStorage.data() + 1, mSize}; }

    [[nodiscard]] size_t size() const noexcept { return mSize; }

    [[nodiscard]] bool contains(const Key &key) const { return mIndex.find(key) != npos; }

    // Changes the value of a key in the heap. False if the key is not in the heap.
    bool exchange(const Key &key, const T &value)
    {
        const uint32_t index = mIndex.find(key);
        if (index == npos) return false;
        Item &v = mStorage[index];
        // The key is unchanged, so the value alone decides the direction.
        const bool up = mCompare(value, v.value);
        v.value = value;
        if (up) {
            siftUp(index);
        } else {
            siftDown(index);
        }
        return true;
    }

    // Adds a key that is not in the heap (asserted).
    bool push(const Key &key, const T &value)
    {
        assert(mIndex.find(key) == npos && "MutableHeap: duplicate keys are not allowed");
        assert(mSize < size_t(npos) - 1 && "MutableHeap: full");
        size_t hole = ++mSize;
        if (hole + 1 > mStorage.size()) mStorage.resize(std::max(size_t(2), mStorage.size()) << 1);
        const Item item{key, value};
        mStorage[0] = item;  // the sentinel stops the loop at the root
        size_t nextHole = hole >> 1;
        for (; compare(item, mStorage[nextHole]); hole = nextHole, nextHole >>= 1) {
            mStorage[hole] = std::move(mStorage[nextHole]);
            mIndex.set(mStorage[hole].key, static_cast<uint32_t>(hole));
        }
        mStorage[hole] = item;
        mIndex.set(key, static_cast<uint32_t>(hole));
        return true;
    }

    // Removes the top item (nothing if empty).
    void pop()
    {
        if (mSize == 0) return;
        mIndex.erase(mStorage[1].key);
        const size_t last = mSize--;
        if (last > 1) {
            mStorage[1] = std::move(mStorage[last]);
            siftDown(1);
        }
    }

    // Removes the first item, in storage order, whose key satisfies eraseFunc. False if none does.
    bool erase_first(const std::function<bool(const Key &key)> &eraseFunc)
    {
        for (size_t n = 1; n <= mSize; ++n) {
            if (eraseFunc(mStorage[n].key)) {
                const Key key = mStorage[n].key;
                return erase(key);
            }
        }
        return false;
    }

    // Removes a key. False if it is not in the heap.
    bool erase(const Key &key)
    {
        const uint32_t idx = mIndex.find(key);
        if (idx == npos) return false;
        mIndex.erase(key);
        const size_t last = mSize--;
        if (idx != last) {
            const bool up = compare(mStorage[last], mStorage[idx]);
            mStorage[idx] = std::move(mStorage[last]);
            if (up) {
                siftUp(idx);
            } else {
                siftDown(idx);
            }
        }
        return true;
    }

    // Empties the heap but keeps its allocated storage.
    void clear()
    {
        mIndex.clear(items());
        mSize = 0;
    }

    // True if the heap order holds and every item's index entry points at its slot.
    bool isHeap() const
    {
        for (size_t n = 1; n <= mSize; ++n) {
            const Item &it = mStorage[n];
            if (mIndex.find(it.key) != n) return false;
            if (n > 1 && compare(it, mStorage[n >> 1])) return false;
        }
        return true;
    }

    std::string toString() const
    {
        std::stringstream ss;
        ss << "[";
        for (size_t n = 1; n <= mSize; ++n) {
            const Item &it = mStorage[n];
            ss << std::to_string(it.key) << ":" << std::to_string(it.value) << ((n < mSize) ? ", " : "");
        }
        ss << "]";
        return ss.str();
    }

private:
    [[nodiscard]] bool compare(const Item &a, const Item &b) const
    {
        if (mCompare(a.value, b.value)) return true;
        if (mCompare(b.value, a.value)) return false;
        return a.key < b.key;
    }

    // Sifts the item in slot i toward the leaves. Always writes its final index entry.
    void siftDown(size_t i)
    {
        Item tmp = std::move(mStorage[i]);
        for (;;) {
            size_t j = i << 1;
            if (j > mSize) break;
            if (j < mSize && compare(mStorage[j + 1], mStorage[j])) ++j;
            if (!compare(mStorage[j], tmp)) break;
            mStorage[i] = std::move(mStorage[j]);
            mIndex.set(mStorage[i].key, static_cast<uint32_t>(i));
            i = j;
        }
        mStorage[i] = std::move(tmp);
        mIndex.set(mStorage[i].key, static_cast<uint32_t>(i));
    }

    // Sifts the item in slot k toward the root. Always writes its final index entry.
    void siftUp(size_t k)
    {
        Item v = std::move(mStorage[k]);
        size_t parent = k >> 1;
        while (parent > 0 && compare(v, mStorage[parent])) {
            mStorage[k] = std::move(mStorage[parent]);
            mIndex.set(mStorage[k].key, static_cast<uint32_t>(k));
            k = parent;
            parent = k >> 1;
        }
        mStorage[k] = std::move(v);
        mIndex.set(mStorage[k].key, static_cast<uint32_t>(k));
    }

    Compare mCompare{};
    size_t mSize{0};
    std::vector<Item> mStorage;
    Index mIndex;
};

// Mutable heap for any hashable key.
template <class Key, class T, class Compare = std::less<T>>
using MutableHeap = BasicMutableHeap<Key, T, Compare, detail::HashHeapIndex<Key>>;

// Mutable heap for dense uint32_t keys (vertex ids, linear pixel / voxel indices): lookups are a flat
// array access instead of a hash; memory is O(largest key).
template <class T, class Compare = std::less<T>>
using DenseMutableHeap = BasicMutableHeap<uint32_t, T, Compare, detail::DenseHeapIndex>;
}  // namespace springls
