/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <uct/uct_test.h>

extern "C" {
#include <ucs/time/time.h>
#include <ucs/datastruct/queue.h>
#include <ucs/arch/atomic.h>
#include <ucs/arch/bitops.h>
}


// FIXME: Add SRD transport to UCT_TEST_IB_TLS when possible
class test_srd : public uct_test {
public:
    virtual void init();

protected:
    entity *m_e1, *m_e2;
};

void test_srd::init()
{
    uct_test::init();

    m_e1 = uct_test::create_entity(0, NULL);
    m_e2 = uct_test::create_entity(0, NULL);
    m_entities.push_back(m_e1);
    m_entities.push_back(m_e2);
}

UCS_TEST_P(test_srd, ep_creations)
{
    for (int i = 0; i < 4; ++i) {
        m_e1->create_ep(i, 0);
    }
}

UCS_TEST_P(test_srd, ep_connect_to_iface)
{
    m_e1->connect_to_iface(0, *m_e2);
}

UCS_TEST_P(test_srd, ep_connect_to_ep)
{
    m_e1->connect_to_ep(0, *m_e2, 0, 0);
}

UCT_INSTANTIATE_SRD_TEST_CASE(test_srd)
