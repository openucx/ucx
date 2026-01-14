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
        ASSERT_UCS_OK(ucs_interval_tree_init(&m_tree, &m_ops, 0, 0));
    }

    void cleanup()
    {
        ucs_interval_tree_cleanup(&m_tree);
        ucs::test::cleanup();
    }

    bool are_all_covered(const interval_vector_t &intervals)
    {
        return std::all_of(
                intervals.begin(), intervals.end(),
                [this](const std::pair<uint64_t, uint64_t> &interval) {
                    return ucs_interval_tree_range_covered(&m_tree,
                                                           interval.first,
                                                           interval.second);
                });
    }

    bool is_any_covered(const interval_vector_t &intervals)
    {
        return std::any_of(
                intervals.begin(), intervals.end(),
                [this](const std::pair<uint64_t, uint64_t> &interval) {
                    return ucs_interval_tree_range_covered(&m_tree,
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

    ucs_interval_tree_t     m_tree;
    ucs_interval_tree_ops_t m_ops = {(ucs_interval_tree_alloc_node_func_t)malloc,
                                     (ucs_interval_tree_free_node_func_t)free,
                                     NULL};
};

UCS_TEST_F(test_interval_tree, single_interval) {
    insert_intervals({{10, 20}});
    EXPECT_TRUE(are_all_covered({{10, 20}, {12, 18}, {10, 15}}));
    EXPECT_FALSE(is_any_covered({{0, 9}, {21, 30}, {5, 15}, {15, 25}}));
}

UCS_TEST_F(test_interval_tree, multiple_non_overlapping) {
    insert_intervals({{10, 20}, {30, 40}, {50, 60}});
    EXPECT_TRUE(are_all_covered({{10, 20}, {30, 40}, {50, 60}}));
    EXPECT_FALSE(is_any_covered({{10, 40}, {10, 60}, {20, 30}}));
}

UCS_TEST_F(test_interval_tree, adjacent_intervals) {
    insert_intervals({{0, 10}, {10, 20}, {20, 30}});
    EXPECT_TRUE(are_all_covered({{0, 10}, {10, 20}, {20, 30}, {0, 20}, {0, 30}, 
                                {10, 30}, {20, 30}, {0, 30}}));
    EXPECT_FALSE(is_any_covered({{15, 40}, {40, 50}}));
}

UCS_TEST_F(test_interval_tree, overlapping_intervals) {
    insert_intervals({{10, 20}, {15, 25}});
    EXPECT_TRUE(are_all_covered({{10, 25}, {10, 20}, {15, 25}, {12, 23}}));
    EXPECT_FALSE(is_any_covered({{3, 15}, {3, 4}}));
}

UCS_TEST_F(test_interval_tree, contained_interval) {
    insert_intervals({{10, 30}, {15, 20}, {50, 60}, {40, 70}});
    EXPECT_TRUE(are_all_covered({{10, 30}, {40, 70}}));
}

UCS_TEST_F(test_interval_tree, continuous_range) {
    interval_vector_t intervals;
    for (int i = 0; i < 10; i++) {
        intervals.push_back({i * 10, (i + 1) * 10});
    }

    insert_intervals(intervals);
    EXPECT_TRUE(are_all_covered({{0, 100}, {0, 50}, {50, 100}, {25, 75}}));
}

UCS_TEST_F(test_interval_tree, out_of_order_insertion) {
    insert_intervals({{30, 40}, {10, 20}, {50, 60}, {0, 5}});
    EXPECT_TRUE(are_all_covered({{0, 5}, {10, 20}, {30, 40}, {50, 60}}));
    EXPECT_FALSE(is_any_covered({{0, 15}, {15, 35}, {35, 55}, {55, 65}}));

    insert_intervals({{30, 50}, {20, 40}, {10, 30}, {0, 20}});
    EXPECT_TRUE(are_all_covered({{0, 50}}));
}

UCS_TEST_F(test_interval_tree, invalid_range) {
    ucs_status_t status;

    status = ucs_interval_tree_insert(&m_tree, 20, 10);
    EXPECT_EQ(status, UCS_ERR_INVALID_PARAM);
    EXPECT_FALSE(is_any_covered({{20, 10}}));

    ucs_interval_tree_t tree;
    status = ucs_interval_tree_init(&tree, &m_ops, 100, 50);
    EXPECT_EQ(status, UCS_ERR_INVALID_PARAM);
}

UCS_TEST_F(test_interval_tree, zero_length_insert) {
    insert_intervals({{10, 10}});
    EXPECT_TRUE(are_all_covered({{10, 10}}));

    insert_intervals({{10, 20}, {15, 15}});
    EXPECT_TRUE(are_all_covered({{15, 15}}));
}

UCS_TEST_F(test_interval_tree, large_intervals) {
    insert_intervals({{1000000, 2000000}, {2000001, 3000000}});
    EXPECT_TRUE(are_all_covered({{1000000, 2000000}, {2000001, 3000000}}));
    EXPECT_FALSE(is_any_covered({{1000000, 3000000}}));
}

UCS_TEST_F(test_interval_tree, merge_multiple_overlaps) {
    insert_intervals({{10, 20}, {15, 25}, {22, 30}, {12, 28}});
    EXPECT_TRUE(are_all_covered({{10, 30}}));

    insert_intervals({{4, 7}, {0, 5}});
    EXPECT_TRUE(are_all_covered({{0, 7}, {10, 30}}));

    insert_intervals({{40, 50}, {35, 45}});
    EXPECT_TRUE(are_all_covered({{35, 50}, {0, 7}, {10, 30}}));
    EXPECT_FALSE(is_any_covered({{0, 50}}));

    insert_intervals({{0, 50}});
    EXPECT_TRUE(are_all_covered({{0, 50}}));
}

UCS_TEST_F(test_interval_tree, init_with_root) {
    ucs_interval_tree_cleanup(&m_tree);
    ASSERT_UCS_OK(ucs_interval_tree_init(&m_tree, &m_ops, 100, 200));

    EXPECT_TRUE(are_all_covered({{100, 200}, {120, 180}, {100, 150}}));
    EXPECT_FALSE(is_any_covered({{50, 150}, {150, 250}, {50, 250}}));
}
