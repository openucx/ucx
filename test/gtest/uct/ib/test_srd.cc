/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <uct/uct_test.h>


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

    m_e1->connect_to_iface(0, *m_e2);
}

UCS_TEST_P(test_srd, am_short_outstanding)
{
    ucs_status_t status;
    uint64_t header = 0x1234567843210987;
    char payload[]  = "the payload";

    status = uct_ep_am_short(m_e1->ep(0), 31, header++, payload, sizeof(payload));
    ASSERT_UCS_OK(status);
}

UCS_TEST_P(test_srd, am_short)
{
    ucs_status_t status;
    uint64_t header = 0x1234567843210987;
    char payload[]  = "the payload";
    int count       = 10;

    /* TODO: Add RX checks when available */
    status = uct_ep_am_short(m_e1->ep(0), 31, header++, payload, sizeof(payload));
    ASSERT_UCS_OK(status);
    status = uct_ep_am_short(m_e1->ep(0), 14, header, payload, sizeof(payload));
    ASSERT_UCS_OK(status);

    while (count-- > 0) {
        short_progress_loop();
    }
}

UCS_TEST_P(test_srd, am_short_failure)
{
    scoped_log_handler slh(wrap_errors_logger);
    uint64_t header = 0x1234567843210987;
    char payload[4096];
    ucs_status_t status;

    status = uct_ep_am_short(m_e1->ep(0), 32, header, payload, 32);
    ASSERT_UCS_STATUS_EQ(UCS_ERR_INVALID_PARAM, status);
    status = uct_ep_am_short(m_e1->ep(0), 14, header, payload, sizeof(payload));
    ASSERT_UCS_STATUS_EQ(UCS_ERR_INVALID_PARAM, status);
}

UCT_INSTANTIATE_SRD_TEST_CASE(test_srd)
