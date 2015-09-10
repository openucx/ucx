/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include <ucs/gtest/test.h>
#include <ucs/gtest/test_helpers.h>

extern "C" {
#include <ucs/sys/sys.h>
#include <ucs/datastruct/arbiter.h>
}
#include <set>

class test_arbiter : public ucs::test {
protected:

    static ucs_arbiter_cb_result_t count_elems(ucs_arbiter_t *arbitrer,
                                               ucs_arbiter_elem_t *elem,
                                               void *arg)
    {
        int *counter = (int*)arg;
        --(*counter);
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    }

    static ucs_arbiter_cb_result_t resched_groups(ucs_arbiter_t *arbitrer,
                                                  ucs_arbiter_elem_t *elem,
                                                  void *arg)
    {
        int *counter = (int*)arg;
        if (*counter == 0) {
            return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
        } else {
            --(*counter);
            return UCS_ARBITER_CB_RESULT_RESCHED_GROUP;
        }
    }

    struct arb_elem {
        unsigned           group_idx;
        unsigned           elem_idx;
        bool               last;
        ucs_arbiter_elem_t elem;
    };

    void skip_empty_groups()
    {
        while (m_empty_groups.find(m_expected_group_idx) != m_empty_groups.end()) {
            advance_expected_group();
        }
    }

    void advance_expected_group()
    {
        ++m_expected_group_idx;
        if (m_expected_group_idx >= m_num_groups) {
            m_expected_group_idx = 0;
        }
    }

    static void release_element(arb_elem *e)
    {
        memset(e, 0xCC, sizeof(*e)); /* Invalidate memory to catch use-after-free bugs */
        delete e;
    }

    ucs_arbiter_cb_result_t dispatch(ucs_arbiter_t *arbiter,
                                     ucs_arbiter_elem_t *elem)
    {
        arb_elem *e = ucs_container_of(elem, arb_elem, elem);

        skip_empty_groups();

        EXPECT_EQ(m_expected_group_idx,               e->group_idx);
        EXPECT_EQ(m_expected_elem_idx[e->group_idx],  e->elem_idx);

        advance_expected_group();

        /* Sometimes we just move to the next group */
        if ((rand() % 5) == 0) {
            return UCS_ARBITER_CB_RESULT_NEXT_GROUP;
        }

        /* Sometimes we want to detach the whole group */
        if ((rand() % 10) == 0) {
            m_empty_groups.insert(e->group_idx);
            m_detached_groups.insert(e->group_idx);
            return UCS_ARBITER_CB_RESULT_DESCHED_GROUP;
        }

        ++m_expected_elem_idx[e->group_idx];

        if (e->last) {
            m_empty_groups.insert(e->group_idx);
        }
        release_element(e);

        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    }

    static ucs_arbiter_cb_result_t dispatch_cb(ucs_arbiter_t *arbiter,
                                               ucs_arbiter_elem_t *elem,
                                               void *arg)
    {
        test_arbiter *self = (test_arbiter *)arg;
        return self->dispatch(arbiter, elem);
    }

    static ucs_arbiter_cb_result_t purge_cb(ucs_arbiter_t *arbiter,
                                            ucs_arbiter_elem_t *elem,
                                            void *arg)
    {
        arb_elem *e = ucs_container_of(elem, arb_elem, elem);
        release_element(e);
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    }

protected:
    std::set<unsigned>    m_empty_groups;
    std::set<unsigned>    m_detached_groups;
    std::vector<unsigned> m_expected_elem_idx;
    unsigned              m_expected_group_idx;
    unsigned              m_num_groups;
};


UCS_TEST_F(test_arbiter, add_purge) {

    ucs_arbiter_t arbiter;

    ucs_arbiter_group_t group1;
    ucs_arbiter_group_t group2;

    ucs_arbiter_init(&arbiter);
    ucs_arbiter_group_init(&group1);
    ucs_arbiter_group_init(&group2);


    ucs_arbiter_elem_t elem1;
    ucs_arbiter_elem_t elem2_1;
    ucs_arbiter_elem_t elem2_2;

    /* test internal function */
    ucs_arbiter_group_push_elem_always(&group1, &elem1);
    ucs_arbiter_group_push_elem_always(&group2, &elem2_1);
    ucs_arbiter_group_push_elem_always(&group2, &elem2_2);

    ucs_arbiter_group_schedule(&arbiter, &group1);
    ucs_arbiter_group_schedule(&arbiter, &group2);

    int count = 3;
    ucs_arbiter_dispatch_nonempty(&arbiter, 3, count_elems, &count);

    EXPECT_EQ(0, count);

    ucs_arbiter_group_cleanup(&group2);
    ucs_arbiter_group_cleanup(&group1);
    ucs_arbiter_cleanup(&arbiter);
}

UCS_TEST_F(test_arbiter, multiple_dispatch) {
    m_num_groups = 20;

    ucs_arbiter_t arbiter;
    ucs_arbiter_init(&arbiter);

    std::vector<ucs_arbiter_group_t> groups(m_num_groups);
    for (unsigned i = 0; i < m_num_groups; ++i) {
        ucs_arbiter_group_init(&groups[i]);

        unsigned num_elems = rand() % 9;

        for (unsigned j = 0; j < num_elems; ++j) {
            arb_elem *e = new arb_elem;
            e->group_idx = i;
            e->elem_idx  = j;
            e->last      = (j == num_elems - 1);
            ucs_arbiter_elem_init(&e->elem);
            ucs_arbiter_group_push_elem(&groups[i], &e->elem);
            /* coverity[leaked_storage] */
        }

        if (num_elems == 0) {
            m_empty_groups.insert(i);
        }

        ucs_arbiter_group_schedule(&arbiter, &groups[i]);
    }

    m_expected_group_idx = 0;
    m_expected_elem_idx.resize(m_num_groups, 0);
    std::fill(m_expected_elem_idx.begin(), m_expected_elem_idx.end(), 0);

    ucs_arbiter_dispatch(&arbiter, 1, dispatch_cb, this);

    ASSERT_TRUE(arbiter.current == NULL);

    /* Release detached groups */
    for (unsigned i = 0; i < m_num_groups; ++i) {
        if (m_detached_groups.find(i) != m_detached_groups.end()) {
            ucs_arbiter_group_purge(&arbiter, &groups[i], purge_cb, NULL);
        }
        ucs_arbiter_group_cleanup(&groups[i]);
    }

    ucs_arbiter_cleanup(&arbiter);
}

UCS_TEST_F(test_arbiter, resched) {

    ucs_arbiter_t arbiter;

    ucs_arbiter_group_t group1;
    ucs_arbiter_group_t group2;

    ucs_arbiter_init(&arbiter);
    ucs_arbiter_group_init(&group1);
    ucs_arbiter_group_init(&group2);


    ucs_arbiter_elem_t elem1;
    ucs_arbiter_elem_t elem2_1;

    ucs_arbiter_elem_init(&elem1);
    ucs_arbiter_elem_init(&elem2_1);
    ucs_arbiter_group_push_elem(&group1, &elem1);
    ucs_arbiter_group_push_elem(&group2, &elem2_1);

    ucs_arbiter_group_schedule(&arbiter, &group1);
    ucs_arbiter_group_schedule(&arbiter, &group2);

    int count = 2;
    ucs_arbiter_dispatch_nonempty(&arbiter, 1, resched_groups, &count);

    EXPECT_EQ(0, count);

    count = 1;
    ucs_arbiter_dispatch_nonempty(&arbiter, 1, resched_groups, &count);
    EXPECT_EQ(0, count);

    /* one group with one elem should be there */
    count = 1;
    ucs_arbiter_dispatch_nonempty(&arbiter, 3, count_elems, &count);
    EXPECT_EQ(0, count);
    ASSERT_TRUE(arbiter.current == NULL);

    ucs_arbiter_group_cleanup(&group2);
    ucs_arbiter_group_cleanup(&group1);
    ucs_arbiter_cleanup(&arbiter);
}

