/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
#include <algorithm>

extern "C" {
#include <ucs/datastruct/interval_tree.h>
}

class test_interval_tree : public ucs::test {
protected:
    using interval_vector_t = std::vector<std::pair<uint64_t, uint64_t>>;

    void init()
    {
        ucs::test::init();
        ucs_interval_tree_init(&m_tree, &m_ops);
    }

    void cleanup()
    {
        ucs_interval_tree_cleanup(&m_tree);
        ucs::test::cleanup();
    }

    bool is_fully_covered(const std::pair<uint64_t, uint64_t> &interval)
    {
        return ucs_interval_tree_is_single_range(&m_tree, interval.first,
                                                 interval.second);
    }

    bool is_any_covered(const interval_vector_t &intervals)
    {
        return std::any_of(
                intervals.begin(), intervals.end(),
                [this](const std::pair<uint64_t, uint64_t> &interval) {
                    return ucs_interval_tree_is_single_range(&m_tree,
                                                             interval.first,
                                                             interval.second);
                });
    }

    void insert_intervals(const interval_vector_t &intervals)
    {
        for (const auto &interval : intervals) {
            ucs_status_t status = ucs_interval_tree_insert(&m_tree,
                                                           interval.first,
                                                           interval.second);
            ASSERT_UCS_OK(status, << "\nFailed to insert interval ["
                                  << interval.first << ", " << interval.second
                                  << "]");
        }
    }

private:
    ucs_interval_tree_t     m_tree = {NULL, {NULL}};
    ucs_interval_tree_ops_t m_ops  = {(ucs_interval_tree_alloc_node_func_t)malloc,
                                      (ucs_interval_tree_free_node_func_t)free,
                                      NULL};
};

UCS_TEST_F(test_interval_tree, single_interval) {
    insert_intervals({{10, 20}});
    EXPECT_TRUE(is_fully_covered({10, 20}));
    EXPECT_FALSE(is_any_covered({{0, 9}, {21, 30}, {5, 15}, {15, 25}}));
}

UCS_TEST_F(test_interval_tree, multiple_non_overlapping) {
    insert_intervals({{10, 20}, {30, 40}, {50, 60}});
    EXPECT_FALSE(is_any_covered({{30, 40}, {50, 60}}));
    insert_intervals({{20, 30}, {40, 50}, {60, 70}});
    EXPECT_TRUE(is_fully_covered({10, 70}));
}

UCS_TEST_F(test_interval_tree, adjacent_intervals) {
    insert_intervals({{0, 10}, {10, 20}, {20, 30}});
    EXPECT_TRUE(is_fully_covered({0, 30}));
    EXPECT_FALSE(is_any_covered({{15, 40}, {40, 50}}));
}

UCS_TEST_F(test_interval_tree, overlapping_intervals) {
    insert_intervals({{10, 20}, {15, 25}, {23, 30}});
    EXPECT_TRUE(is_fully_covered({10, 30}));
    EXPECT_FALSE(is_any_covered({{3, 15}, {3, 4}}));
}

UCS_TEST_F(test_interval_tree, contained_interval) {
    insert_intervals({{10, 30}, {15, 20}});
    EXPECT_FALSE(is_any_covered({{15, 20}}));
    EXPECT_TRUE(is_fully_covered({10, 30}));
}

UCS_TEST_F(test_interval_tree, continuous_range) {
    interval_vector_t intervals;
    for (int i = 0; i < 10; i++) {
        intervals.push_back({i * 10, (i + 1) * 10});
    }

    insert_intervals(intervals);
    EXPECT_TRUE(is_fully_covered({0, 100}));
}

UCS_TEST_F(test_interval_tree, out_of_order_insertion) {
    insert_intervals({{30, 40}, {10, 20}, {50, 60}, {0, 5}});
    EXPECT_FALSE(is_any_covered({{0, 5}, {10, 20}, {30, 40}, {50, 60}}));

    insert_intervals({{30, 50}, {20, 40}, {10, 30}, {0, 20}});
    EXPECT_TRUE(is_fully_covered({0, 60}));
}

UCS_TEST_F(test_interval_tree, zero_length_insert) {
    insert_intervals({{10, 10}});
    EXPECT_TRUE(is_fully_covered({10, 10}));

    insert_intervals({{10, 20}, {15, 15}});
    EXPECT_TRUE(is_fully_covered({10, 20}));
}

UCS_TEST_F(test_interval_tree, large_intervals) {
    insert_intervals({{1000000, 2000000}, {2000001, 3000000}});
    EXPECT_FALSE(is_any_covered({{1000000, 2000000}, {2000001, 3000000}, 
                                 {1000000, 3000000}}));
    insert_intervals({{1000000, 3000000}});
    EXPECT_TRUE(is_fully_covered({1000000, 3000000}));
}

UCS_TEST_F(test_interval_tree, merge_multiple_overlaps) {
    insert_intervals({{10, 20}, {15, 25}, {22, 30}, {12, 28}});
    EXPECT_TRUE(is_fully_covered({10, 30}));

    insert_intervals({{4, 7}, {0, 5}});
    EXPECT_FALSE(is_any_covered({{0, 7}, {10, 30}}));

    insert_intervals({{40, 50}, {35, 45}});
    EXPECT_FALSE(is_any_covered({{35, 50}, {0, 7}, {10, 30}}));

    insert_intervals({{0, 50}});
    EXPECT_TRUE(is_fully_covered({0, 50}));
}


