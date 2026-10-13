/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>

#include "rbtree_check.h"

#include <algorithm>
#include <vector>

extern "C" {
#include <ucs/datastruct/interval_map.h>
}

class test_interval_map : public ucs::test {
protected:
    /* Nodes are intrusive, so the test owns them and can check that a pointer
     * handed out at insert time stays valid across unrelated removals. */
    struct entry {
        ucs_interval_map_node_t node;
        unsigned                id;
        bool                    live;
    };

    void init()
    {
        ucs::test::init();
        ucs_interval_map_init(&m_map);
        m_entries.resize(NUM_ENTRIES);
        for (unsigned i = 0; i < NUM_ENTRIES; ++i) {
            m_entries[i].id   = i;
            m_entries[i].live = false;
        }
    }

    void insert(unsigned idx, uint64_t start, uint64_t end)
    {
        ucs_interval_map_insert(&m_map, &m_entries[idx].node, start, end);
        m_entries[idx].live = true;
    }

    void remove(unsigned idx)
    {
        ucs_interval_map_remove(&m_map, &m_entries[idx].node);
        m_entries[idx].live = false;
    }

    entry *find(uint64_t start, uint64_t end)
    {
        ucs_interval_map_node_t *node =
                ucs_interval_map_find_containing(&m_map, start, end);
        return (node == NULL) ? NULL : ucs_container_of(node, entry, node);
    }

    /* Brute-force reference: is any live interval a superset of the range? */
    bool ref_contains(uint64_t start, uint64_t end)
    {
        for (const entry &e : m_entries) {
            if (e.live && (e.node.start <= start) && (e.node.end >= end)) {
                return true;
            }
        }
        return false;
    }

    /* Recompute a subtree's maximum end independently of the augmentation */
    static uint64_t subtree_max(const ucs_rbtree_node_t *rb_node)
    {
        const ucs_interval_map_node_t *node =
                ucs_derived_of(rb_node, ucs_interval_map_node_t);
        uint64_t max = node->end;

        if (rb_node->left != NULL) {
            max = std::max(max, subtree_max(rb_node->left));
        }
        if (rb_node->right != NULL) {
            max = std::max(max, subtree_max(rb_node->right));
        }
        return max;
    }

    /* max_end is a per-node property, so it goes through the visitor. The
     * red-black invariants themselves belong to the core and are checked
     * there. */
    static void check_max_end(const ucs_rbtree_node_t *rb_node)
    {
        EXPECT_EQ(subtree_max(rb_node),
                  ucs_derived_of(rb_node, ucs_interval_map_node_t)->max_end);
    }

    void collect_starts(const ucs_rbtree_node_t *rb_node,
                        std::vector<uint64_t> &out) const
    {
        if (rb_node == NULL) {
            return;
        }

        collect_starts(rb_node->left, out);
        out.push_back(ucs_derived_of(rb_node, ucs_interval_map_node_t)->start);
        collect_starts(rb_node->right, out);
    }

    void validate(size_t expected)
    {
        std::vector<uint64_t> starts;

        EXPECT_TRUE(rbtree_check::validate(&m_map.rb, expected, check_max_end));
        EXPECT_EQ(expected, ucs_interval_map_count(&m_map));

        /* Ordering is the property that an in-order walk is sorted by start.
         * Comparing a node only against its immediate children is weaker and
         * would not establish it. */
        collect_starts(m_map.rb.root, starts);
        EXPECT_TRUE(std::is_sorted(starts.begin(), starts.end()));
    }

    static void collect_cb(ucs_interval_map_node_t *node, void *arg)
    {
        std::vector<unsigned> *ids = static_cast<std::vector<unsigned>*>(arg);
        ids->push_back(ucs_container_of(node, entry, node)->id);
    }

    std::vector<unsigned> overlapping(uint64_t start, uint64_t end)
    {
        std::vector<unsigned> ids;
        ucs_interval_map_foreach_overlapping(&m_map, start, end, collect_cb,
                                             &ids);
        std::sort(ids.begin(), ids.end());
        return ids;
    }

    static const unsigned NUM_ENTRIES = 256;

    ucs_interval_map_t m_map;
    std::vector<entry> m_entries;
};


/* The lookup contract: a request is served by any interval containing it. */
UCS_TEST_F(test_interval_map, containment) {
    insert(0, 1000, 7000);

    EXPECT_EQ(&m_entries[0], find(1000, 7000)); /* exact                */
    EXPECT_EQ(&m_entries[0], find(1000, 4000)); /* shares the start     */
    EXPECT_EQ(&m_entries[0], find(2500, 4000)); /* strictly inside      */
    EXPECT_EQ(&m_entries[0], find(6999, 7000)); /* last byte            */
    EXPECT_EQ(NULL, find(1000, 9000));          /* extends past the end */
    EXPECT_EQ(NULL, find(500, 4000));           /* starts before        */
    EXPECT_EQ(NULL, find(8000, 9000));          /* disjoint             */
    validate(1);
}

UCS_TEST_F(test_interval_map, empty_map) {
    EXPECT_EQ(0u, ucs_interval_map_count(&m_map));
    EXPECT_EQ(NULL, find(0, 1));
    validate(0);
}

/* Overlapping intervals must stay distinct rather than coalescing. */
UCS_TEST_F(test_interval_map, overlapping_coexist) {
    insert(0, 1000, 4000);
    insert(1, 3000, 6000);
    validate(2);

    EXPECT_EQ(&m_entries[0], find(1000, 4000));
    EXPECT_EQ(&m_entries[1], find(3000, 6000));
    /* Spans both but is contained by neither */
    EXPECT_EQ(NULL, find(1000, 6000));

    /* The shared region is covered by both; either answer is correct */
    entry *e = find(3000, 4000);
    EXPECT_TRUE((e == &m_entries[0]) || (e == &m_entries[1]));
}

/* Removing one interval must not disturb an overlapping one. */
UCS_TEST_F(test_interval_map, remove_keeps_overlapping) {
    insert(0, 1000, 4000);
    insert(1, 3000, 6000);

    remove(0);
    validate(1);
    EXPECT_EQ(&m_entries[1], find(3000, 6000));
    EXPECT_EQ(NULL, find(1000, 2000));

    /* The surviving node was never relocated */
    EXPECT_EQ(3000u, m_entries[1].node.start);
    EXPECT_EQ(6000u, m_entries[1].node.end);
}

/* A nested interval is served by its enclosing one. */
UCS_TEST_F(test_interval_map, nested) {
    insert(0, 1000, 8000);
    insert(1, 2000, 3000);
    validate(2);

    EXPECT_TRUE(find(2000, 3000) != NULL);
    EXPECT_EQ(&m_entries[0], find(1000, 8000));

    remove(1);
    EXPECT_EQ(&m_entries[0], find(2000, 3000));
}

UCS_TEST_F(test_interval_map, duplicates) {
    insert(0, 100, 200);
    insert(1, 100, 200);
    insert(2, 100, 200);
    validate(3);

    remove(1);
    validate(2);
    EXPECT_TRUE(find(100, 200) != NULL);

    remove(0);
    remove(2);
    validate(0);
    EXPECT_EQ(NULL, find(100, 200));
}

UCS_TEST_F(test_interval_map, foreach_overlapping) {
    insert(0, 100, 200);
    insert(1, 150, 250);
    insert(2, 300, 400);
    insert(3, 50, 500);

    /* Touching but not overlapping: [200,300) meets 0 at its exclusive end */
    EXPECT_EQ(std::vector<unsigned>({1, 3}), overlapping(200, 300));
    EXPECT_EQ(std::vector<unsigned>({0, 1, 3}), overlapping(100, 160));
    EXPECT_EQ(std::vector<unsigned>({3}), overlapping(50, 60));
    EXPECT_TRUE(overlapping(1000, 2000).empty());
    validate(4);
}

/* Randomized insert/remove, cross-checked against a brute-force reference and
 * with every red-black and augmentation invariant re-verified as it goes. */
UCS_TEST_F(test_interval_map, random_stress) {
    static const unsigned NUM_ITERS = 20000;
    size_t live                     = 0;

    for (unsigned i = 0; i < NUM_ITERS; ++i) {
        const unsigned idx = ucs::rand() % NUM_ENTRIES;

        if (!m_entries[idx].live) {
            const uint64_t start = ucs::rand() % 900;
            insert(idx, start, start + 1 + (ucs::rand() % 200));
            ++live;
        } else {
            remove(idx);
            --live;
        }

        if ((i % 101) != 0) {
            continue;
        }

        validate(live);

        const uint64_t start = ucs::rand() % 1000;
        const uint64_t end   = start + 1 + (ucs::rand() % 150);
        ucs_interval_map_node_t *node =
                ucs_interval_map_find_containing(&m_map, start, end);
        EXPECT_EQ(ref_contains(start, end), node != NULL);
        if (node != NULL) {
            EXPECT_LE(node->start, start);
            EXPECT_GE(node->end, end);
        }

        std::vector<unsigned> expected;
        for (const entry &e : m_entries) {
            if (e.live && (e.node.start < end) && (start < e.node.end)) {
                expected.push_back(e.id);
            }
        }
        EXPECT_EQ(expected, overlapping(start, end));

        ASSERT_FALSE(HasFailure());
    }
}
