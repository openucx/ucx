/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2017. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

extern "C" {
#include <uct/api/uct.h>
}
#include <common/test.h>
#include "uct_test.h"


class test_uct_progress : public uct_test {
public:
    virtual void init()
    {
        uct_test::init();
        m_entities.push_back(create_entity(0));
    }

protected:
    uct_worker_h worker(unsigned index = 0)
    {
        return ent(index).worker();
    }

    uct_iface_h iface(unsigned index = 0)
    {
        return ent(index).iface();
    }

    static unsigned count_progress(void *arg)
    {
        test_uct_progress *self = reinterpret_cast<test_uct_progress*>(arg);
        ++self->m_count;
        return 1;
    }

    unsigned m_count{0};
};


UCS_TEST_P(test_uct_progress, random_enable_disable)
{
    for (int i = 0; i < 100; ++i) {
        unsigned flags = 0;
        if (ucs::rand() % 2) {
            flags |= UCT_PROGRESS_SEND;
        }
        if (ucs::rand() % 2) {
            flags |= UCT_PROGRESS_RECV;
        }
        if (ucs::rand() % 2) {
            uct_iface_progress_enable(iface(), flags);
        } else {
            uct_iface_progress_disable(iface(), flags);
        }
        progress();
    }
}

UCS_TEST_P(test_uct_progress, oneshot_progress)
{
    int prog_id = UCS_CALLBACKQ_ID_NULL;
    uct_worker_progress_register_safe(worker(), count_progress, this,
                                      UCS_CALLBACKQ_FLAG_ONESHOT, &prog_id);
    EXPECT_NE(UCS_CALLBACKQ_ID_NULL, prog_id);

    EXPECT_EQ(0, m_count);
    unsigned count = progress();
    EXPECT_GE(count, 1);

    /* The callback should be removed by now */
    count = progress();
    EXPECT_EQ(0, count);

    EXPECT_EQ(1, m_count);
}

UCS_TEST_P(test_uct_progress, oneshot_progress_remove)
{
    int prog_id = UCS_CALLBACKQ_ID_NULL;
    uct_worker_progress_register_safe(worker(), count_progress, this,
                                      UCS_CALLBACKQ_FLAG_ONESHOT, &prog_id);
    EXPECT_NE(UCS_CALLBACKQ_ID_NULL, prog_id);

    uct_worker_progress_unregister_safe(worker(), &prog_id);
    EXPECT_EQ(UCS_CALLBACKQ_ID_NULL, prog_id);

    unsigned count = progress();
    EXPECT_EQ(0, count);
    EXPECT_EQ(0, m_count);
}

UCT_INSTANTIATE_TEST_CASE(test_uct_progress);
