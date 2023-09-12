/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2023. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>

extern "C" {
#include "ucs/datastruct/usage_tracker.h"
}

class test_usage_tracker : public ucs::test {
protected:
    using entries_vec_t = std::vector<uint64_t>;

    virtual void init()
    {
        ucs::test::init();
        ASSERT_UCS_OK(ucs_usage_tracker_create(&m_params, &m_usage_tracker));
    }

    virtual void cleanup()
    {
        ucs_usage_tracker_destroy(m_usage_tracker);
        ucs::test::cleanup();
    }

    void touch_all(const entries_vec_t &input)
    {
        for (auto &entry : input) {
            ucs_usage_tracker_touch_key(m_usage_tracker, (void*)entry);
        }
    }

    static void update_cb(void *entry, void *arg)
    {
        auto results = reinterpret_cast<entries_vec_t*>(arg);
        results->push_back((uint64_t)entry);
    }

    void verify_rank(const entries_vec_t &expected, const entries_vec_t &actual,
                     const std::string &operation)
    {
        ASSERT_EQ(expected.size(), actual.size()) << operation;

        for (int i = 0; i < actual.size(); ++i) {
            ASSERT_TRUE(std::find(expected.begin(), expected.end(),
                                  actual[i]) != expected.end())
                    << "index " << i << ", elem: " << actual[i] << operation;
        }
    }

    void
    verify(const entries_vec_t &exp_promoted, const entries_vec_t &exp_demoted)
    {
        verify_rank(exp_promoted, m_promoted, "promotion");
        verify_rank(exp_demoted, m_demoted, "demotion");

        for (int i = 0; i < m_promoted.size(); ++i) {
            bool promoted = (std::find(m_demoted.begin(), m_demoted.end(),
                                       m_promoted[i]) == m_demoted.end());
            ASSERT_EQ(promoted,
                      ucs_usage_tracker_is_promoted(m_usage_tracker,
                                                    (void*)m_promoted[i]))
                    << "index=" << i << ", entry=" << m_promoted[i];
        }

        for (int i = 0; i < m_demoted.size(); ++i) {
            ASSERT_FALSE(ucs_usage_tracker_is_promoted(m_usage_tracker,
                                                       (void*)m_demoted[i]))
                    << "index=" << i << ", entry=" << m_demoted[i];
        }
    }

    ucs_usage_tracker_params_t m_params = {30, 10, 0.1, update_cb, &m_promoted,
                                           update_cb, &m_demoted, {0.2, 0.8}};

    entries_vec_t              m_promoted;
    entries_vec_t              m_demoted;
    ucs_usage_tracker_h        m_usage_tracker;
};

/* Tests promotion of entries */
UCS_TEST_F(test_usage_tracker, promote) {
    entries_vec_t elements, promoted, demoted;

    /* Entries are initialized */
    for (int i = 0; i < m_params.promote_capacity; ++i) {
        elements.push_back(i);
    }

    /* Entries are added and progressed */
    for (int i = 0; i < 10; ++i) {
        ucs_usage_tracker_progress(m_usage_tracker);
        touch_all(elements);
    }

    /* Entries are promoted */
    promoted = {elements.begin(), elements.begin() + m_params.promote_thresh};
    verify(promoted, demoted);
}

/* Tests stability of promoted entries (do not demote if score is less
 * than a threshold). */
UCS_TEST_F(test_usage_tracker, stability) {
    ucs_usage_tracker_destroy(m_usage_tracker);
    m_params.promote_capacity = 10;
    ASSERT_UCS_OK(ucs_usage_tracker_create(&m_params, &m_usage_tracker));

    entries_vec_t elements1, elements2, promoted, demoted;
    for (int i = 0; i < m_params.promote_capacity; ++i) {
        elements1.push_back(i);
        /* Init with different values */
        elements2.push_back(i + m_params.promote_capacity);
    }

    /* Initial entries inserted into usage tracker. */
    touch_all(elements1);
    ucs_usage_tracker_progress(m_usage_tracker);

    /* New entries are inserted, but removed immediately because the score
     * is not high enough (diff is too small). */
    touch_all(elements2);
    ucs_usage_tracker_progress(m_usage_tracker);

    promoted = {elements1.begin(), elements1.begin() + m_params.promote_thresh};
    verify(promoted, demoted);
}

/* Tests demotion of entries */
UCS_TEST_F(test_usage_tracker, demote) {
    ucs_usage_tracker_destroy(m_usage_tracker);
    m_params.promote_capacity = 10;
    ASSERT_UCS_OK(ucs_usage_tracker_create(&m_params, &m_usage_tracker));

    entries_vec_t elements1, elements2, demoted, promoted;
    for (int i = 0; i < m_params.promote_capacity; ++i) {
        elements1.push_back(i);
        /* second vector values need to be different from first so start
         * from 'promote_capacity' */
        elements2.push_back(i + m_params.promote_capacity);
    }

    /* Add initial entries (progress only few times to prevent them
     * from getting a score which is too high). */
    for (int i = 0; i < 5; ++i) {
        ucs_usage_tracker_progress(m_usage_tracker);
        touch_all(elements1);
    }

    /* Add more entries and progress enough times so that the old entries will
     * be removed. */
    for (int i = 0; i < 10; ++i) {
        ucs_usage_tracker_progress(m_usage_tracker);
        touch_all(elements2);
    }

    /* Expect old entries to be demoted. */
    demoted = {elements1.begin(), elements1.begin() + m_params.promote_thresh};
    promoted = {demoted.begin(), demoted.end()};
    promoted.insert(promoted.end(), elements2.begin(), elements2.end());
    verify(promoted, demoted);
}
