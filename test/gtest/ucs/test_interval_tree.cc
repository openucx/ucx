/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>

#include <algorithm>
#include <cmath>

extern "C" {
#include <ucs/datastruct/interval_tree.h>
#include <ucs/datastruct/mpool.h>
}

class test_interval_tree : public ucs::test {
protected:
    using interval_t        = std::pair<uint64_t, uint64_t>;
    using interval_vector_t = std::vector<interval_t>;

    void init()
    {
        ucs::test::init();

        /* Initialize memory pool for interval tree nodes */
        ucs_mpool_params_t mp_params;
        ucs_mpool_params_reset(&mp_params);
        mp_params.elem_size       = sizeof(ucs_interval_node_t);
        mp_params.elems_per_chunk = 256;
        mp_params.ops             = &m_mpool_ops;
        mp_params.name            = "test_interval_tree_nodes";

        ucs_status_t status = ucs_mpool_init(&mp_params, &m_mpool);
        ASSERT_UCS_OK(status);

        ucs_interval_tree_init(&m_tree, &m_mpool);
    }

    void cleanup()
    {
        ucs_interval_tree_cleanup(&m_tree);
        ucs_mpool_cleanup(&m_mpool, 0);
        ucs::test::cleanup();
    }

    bool is_fully_covered(const interval_t &interval) const
    {
        return ucs_interval_tree_is_equal_range(&m_tree, interval.first,
                                                 interval.second);
    }

    bool is_any_covered(const interval_vector_t &intervals) const
    {
        return std::any_of(intervals.begin(), intervals.end(),
                           [this](const interval_t &interval) {
                               return is_fully_covered(interval);
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

    /* Red-Black tree test helpers (use public node layout) */
    static size_t tree_height(const ucs_interval_node_t *node)
    {
        if (node == NULL) {
            return 0;
        }
        return 1 + std::max(tree_height(node->left), tree_height(node->right));
    }

    size_t tree_height() const
    {
        return tree_height(m_tree.root);
    }

    void collect_inorder(const ucs_interval_node_t *node,
                         interval_vector_t &out) const
    {
        if (node == NULL) {
            return;
        }
        collect_inorder(node->left, out);
        out.push_back({node->start, node->end});
        collect_inorder(node->right, out);
    }

    interval_vector_t collect_inorder() const
    {
        interval_vector_t out;
        collect_inorder(m_tree.root, out);
        return out;
    }

    /* Returns black height if valid, or (size_t)-1 if RB invariant violated */
    static size_t check_rb_invariant(const ucs_interval_node_t *node,
                                    int parent_red,
                                    size_t *out_black_height)
    {
        if (node == NULL) {
            *out_black_height = 1;
            return 1;
        }
        if (parent_red && node->color == UCS_INTERVAL_NODE_RED) {
            return (size_t)-1; /* double red */
        }
        size_t left_bh, right_bh;
        if (check_rb_invariant(node->left, node->color == UCS_INTERVAL_NODE_RED,
                               &left_bh) == (size_t)-1) {
            return (size_t)-1;
        }
        if (check_rb_invariant(node->right,
                               node->color == UCS_INTERVAL_NODE_RED,
                               &right_bh) == (size_t)-1) {
            return (size_t)-1;
        }
        if (left_bh != right_bh) {
            return (size_t)-1; /* unequal black heights */
        }
        *out_black_height = left_bh + (node->color == UCS_INTERVAL_NODE_BLACK ? 1 : 0);
        return *out_black_height;
    }

    bool check_rb_invariant() const
    {
        if (m_tree.root == NULL) {
            return true;
        }
        if (m_tree.root->color != UCS_INTERVAL_NODE_BLACK) {
            return false; /* root must be black */
        }
        size_t bh;
        return check_rb_invariant(m_tree.root, 0, &bh) != (size_t)-1;
    }

    ucs_interval_tree_t m_tree;
    ucs_mpool_t         m_mpool;

private:
    static ucs_mpool_ops_t m_mpool_ops;
};

ucs_mpool_ops_t test_interval_tree::m_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = NULL,
    .obj_cleanup   = NULL,
    .obj_str       = NULL
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

UCS_TEST_F(test_interval_tree, length_one_insert) {
    insert_intervals({{10, 10}});
    EXPECT_TRUE(is_fully_covered({10, 10}));

    insert_intervals({{10, 20}, {15, 15}});
    EXPECT_TRUE(is_fully_covered({10, 20}));
}

UCS_TEST_F(test_interval_tree, length_zero_insert) {
    insert_intervals({{10, 9}});
    EXPECT_TRUE(is_fully_covered({10, 9}));

    insert_intervals({{10, 20}, {20, 19}});
    EXPECT_TRUE(is_fully_covered({10, 20}));
}

UCS_TEST_F(test_interval_tree, large_intervals) {
    insert_intervals({{1000000, 2000000}, {2000002, 3000000}});
    EXPECT_FALSE(is_any_covered(
            {{1000000, 2000000}, {2000002, 3000000}, {1000000, 3000000}}));
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

UCS_TEST_F(test_interval_tree, adjacent_discrete_integers) {
    insert_intervals({{1, 3}});
    EXPECT_TRUE(is_fully_covered({1, 3}));

    insert_intervals({{4, 6}});
    EXPECT_TRUE(is_fully_covered({1, 6}));
    EXPECT_FALSE(is_any_covered({{1, 3}, {4, 6}}));

    insert_intervals({{10, 12}});
    EXPECT_FALSE(is_fully_covered({10, 12}));

    insert_intervals({{14, 16}});
    EXPECT_FALSE(is_fully_covered({10, 16}));
}

/* Red-Black tree balance: height after N ascending inserts must be O(log N) */
UCS_TEST_F(test_interval_tree, height_ascending_insert) {
    const size_t n = 1000;
    for (size_t i = 0; i < n; i++) {
        uint64_t start = i * 100;
        uint64_t end   = start + 50;
        ucs_status_t status = ucs_interval_tree_insert(&m_tree, start, end);
        ASSERT_UCS_OK(status);
    }
    size_t h = tree_height();
    size_t max_h = 2 * (size_t)std::ceil(std::log2((double)(n + 1)));
    EXPECT_LE(h, max_h) << "height " << h << " > 2*ceil(log2(" << n << "+1)) = " << max_h;
}

/* Inorder traversal must be sorted by interval start */
UCS_TEST_F(test_interval_tree, inorder_sorted) {
    insert_intervals({{30, 40}, {10, 20}, {50, 60}, {0, 5}, {25, 28}});
    interval_vector_t in = collect_inorder();
    for (size_t i = 1; i < in.size(); i++) {
        EXPECT_LE(in[i - 1].first, in[i].first)
            << "inorder not sorted at index " << i;
    }
}

/* Red-Black invariants: root black, no double red, same black height on all paths */
UCS_TEST_F(test_interval_tree, red_black_invariant) {
    insert_intervals({{10, 20}, {30, 40}, {50, 60}, {0, 5}, {25, 28},
                      {70, 80}, {45, 55}, {100, 110}});
    EXPECT_TRUE(check_rb_invariant()) << "RB invariant violated";
}

/* Large ascending insert (degenerate without balancing), then check height and invariant */
UCS_TEST_F(test_interval_tree, degenerate_ascending_then_invariant) {
    const size_t n = 500;
    for (size_t i = 0; i < n; i++) {
        uint64_t start = i * 2;
        uint64_t end   = start + 1;
        ucs_status_t status = ucs_interval_tree_insert(&m_tree, start, end);
        ASSERT_UCS_OK(status);
    }
    EXPECT_TRUE(check_rb_invariant()) << "RB invariant violated after ascending inserts";
    size_t h = tree_height();
    size_t max_h = 2 * (size_t)std::ceil(std::log2((double)(n + 1)));
    EXPECT_LE(h, max_h) << "height " << h << " > 2*ceil(log2(" << n << "+1)) = " << max_h;
}
