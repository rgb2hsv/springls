// MutableHeap / DenseMutableHeap: order, decrease-key, erase and reuse against a reference.
#include "Check.hpp"

#include <core/MutableHeap.hpp>

#include <cstdint>
#include <map>
#include <random>
#include <utility>
#include <vector>

using namespace springls;

namespace
{
// Drives a heap and a reference (a map key -> value; the order is the smallest (value, key)) with the same
// random pushes, value changes, erases and pops, and checks they agree at every pop.
template <class Heap>
void CheckAgainstReference(Heap &heap, uint32_t seed, uint32_t keys, int operations)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> keyDist(0, keys - 1);
    std::uniform_int_distribution<int> valueDist(0, 50);  // few distinct values: many ties, broken by key
    std::uniform_int_distribution<int> opDist(0, 9);
    std::map<uint32_t, double> reference;
    const auto referenceTop = [&]() {
        auto best = reference.begin();
        for (auto it = reference.begin(); it != reference.end(); ++it) {
            if (it->second < best->second || (it->second == best->second && it->first < best->first)) best = it;
        }
        return best;
    };
    for (int n = 0; n < operations; n++) {
        const int op = opDist(rng);
        const uint32_t key = keyDist(rng);
        const double value = valueDist(rng);
        if (op < 5) {
            // Push, or change the value of a key already in (both directions).
            if (reference.count(key)) {
                CHECK(heap.exchange(key, value));
            } else {
                CHECK(heap.push(key, value));
            }
            reference[key] = value;
        } else if (op < 7) {
            CHECK(heap.erase(key) == (reference.erase(key) == 1));
        } else if (!reference.empty()) {
            const auto top = heap.top();
            const auto expected = referenceTop();
            CHECK(top.has_value());
            CHECK(top->key == expected->first);
            CHECK(top->value == expected->second);
            heap.pop();
            reference.erase(expected);
        }
        CHECK(heap.size() == reference.size());
        CHECK(heap.isHeap());
    }
    // Drain in order.
    while (!reference.empty()) {
        const auto expected = referenceTop();
        CHECK(heap.top()->key == expected->first);
        heap.pop();
        reference.erase(expected);
    }
    CHECK(heap.empty());
    CHECK(!heap.top().has_value());
}
}  // namespace

TEST(mutable_heap_matches_reference)
{
    MutableHeap<uint32_t, double> heap;
    CheckAgainstReference(heap, 1, 200, 4000);
}

TEST(dense_mutable_heap_matches_reference)
{
    DenseMutableHeap<double> heap;
    heap.reserveKeys(64);  // smaller than the key range: the table also grows on demand
    CheckAgainstReference(heap, 2, 300, 4000);
}

TEST(dense_mutable_heap_clear_and_reuse)
{
    // Reused across runs, as fast marching would: clear() forgets every key, keeps storage.
    DenseMutableHeap<double> heap;
    for (uint32_t run = 0; run < 3; run++) {
        for (uint32_t k = 0; k < 20; k++) heap.push(k * 7, double((k * 13) % 5));
        CHECK(heap.size() == 20);
        CHECK(heap.contains(7));
        heap.clear();
        CHECK(heap.empty());
        CHECK(!heap.contains(7));
        CHECK(heap.isHeap());
        CheckAgainstReference(heap, 10 + run, 150, 1500);
    }
}

TEST(mutable_heap_ties_by_smaller_key)
{
    // The same pop order as a std::priority_queue of (value, key) with std::greater.
    DenseMutableHeap<double> heap;
    heap.push(5, 1.0);
    heap.push(2, 1.0);
    heap.push(9, 0.5);
    heap.push(3, 1.0);
    std::vector<uint32_t> order;
    while (!heap.empty()) {
        order.push_back(heap.top()->key);
        heap.pop();
    }
    CHECK((order == std::vector<uint32_t>{9, 2, 3, 5}));
}

TEST(mutable_heap_decrease_key_moves_to_top)
{
    DenseMutableHeap<double> heap;
    for (uint32_t k = 0; k < 10; k++) heap.push(k, 10.0 + k);
    CHECK(heap.exchange(8, 1.0));
    CHECK(heap.top()->key == 8);
    CHECK(heap.exchange(8, 100.0));  // and back down
    CHECK(heap.top()->key == 0);
    CHECK(!heap.exchange(42, 0.0));  // not in the heap
    CHECK(!heap.erase(42));
    CHECK(heap.isHeap());
}
