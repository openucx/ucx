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
    entity *m_e1, *m_e2, *m_e3;
};

void test_srd::init()
{
    uct_test::init();

    m_e1 = uct_test::create_entity(0, NULL);
    m_e2 = uct_test::create_entity(0, NULL);
    m_e3 = uct_test::create_entity(0, NULL);
    m_entities.push_back(m_e1);
    m_entities.push_back(m_e2);
    m_entities.push_back(m_e3);

    m_e1->connect_to_iface(0, *m_e2);
    m_e3->connect_to_iface(0, *m_e2);
}

UCS_TEST_P(test_srd, am_short_outstanding)
{
    int count       = 40;
    uint64_t header = 0x1234567843210987;
    char payload[]  = "the payload";

    while (count-- > 0) {
        ASSERT_UCS_OK(uct_ep_am_short(m_e1->ep(0), 31, header++, payload,
                                      sizeof(payload)));
    }
}

UCS_TEST_P(test_srd, am_short)
{
    int count       = 20;
    uint64_t header = 0x1234567843210987;
    char payload[]  = "the payload";
    struct stats {
        int      count;
        int      bytes;
        uint64_t hdr;
    } stats = {}, expect = {};

    auto counter_func = [](void *arg, void *data, size_t length,
                           unsigned flags) {
        struct stats *stats = reinterpret_cast<struct stats*>(arg);

        stats->count += strncmp("the payload",
                                reinterpret_cast<char*>(data) +
                                        sizeof(uint64_t),
                                length - sizeof(uint64_t)) == 0;
        stats->bytes += length;
        stats->hdr   += *reinterpret_cast<uint64_t*>(data);
        return UCS_OK;
    };

    ASSERT_UCS_OK(uct_iface_set_am_handler(m_e2->iface(), 31, counter_func,
                                           &stats, 0));
    ASSERT_UCS_OK(uct_iface_set_am_handler(m_e2->iface(), 14, counter_func,
                                           &stats, 0));
    for (auto i = 0; i < count; i++) {
        expect.count += 3;
        expect.bytes += (sizeof(header) + sizeof(payload)) * 3 - 6;

        expect.hdr += header;
        ASSERT_UCS_OK(uct_ep_am_short(m_e1->ep(0), 31, header++, payload,
                                      sizeof(payload)));
        expect.hdr += header;
        ASSERT_UCS_OK(uct_ep_am_short(m_e1->ep(0), 31, header++, payload,
                                      sizeof(payload)));
        expect.hdr += header;
        ASSERT_UCS_OK(uct_ep_am_short(m_e3->ep(0), 14, header--, payload,
                                      sizeof(payload) - 6));
    }

    wait_for_value(&stats.count, expect.count, true);
    EXPECT_EQ(expect.bytes, stats.bytes);
    EXPECT_EQ(expect.count, stats.count);
    EXPECT_EQ(expect.hdr,   stats.hdr);
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

UCS_TEST_P(test_srd, am_short_no_resource)
{
    int count = 400;
    int i;
    ucs_status_t status;

    for (i = 0; i < count; i++) {
        status = uct_ep_am_short(m_e1->ep(0), 30, 0x0, "test", 4);
        if (status != UCS_OK) {
            break;
        }
    }

    ASSERT_LT(100, i);
    ASSERT_GT(count, i);
    ASSERT_UCS_STATUS_EQ(UCS_ERR_NO_RESOURCE, status);
}

UCT_INSTANTIATE_SRD_TEST_CASE(test_srd)
