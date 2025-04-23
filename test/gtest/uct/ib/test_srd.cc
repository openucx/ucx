/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <uct/uct_test.h>
#include <uct/ib/efa/srd/srd_ep.h>

// FIXME: Add SRD transport to UCT_TEST_IB_TLS when possible
class test_srd : public uct_test {
public:
    virtual void init();

protected:
    struct completion : uct_completion_t {
        completion() :
            uct_completion_t({.func = counter_cb,
                              .count = 1,
                              .status = UCS_OK}),
            m_count(0)
        {
        }

        static void counter_cb(uct_completion_t *comp)
        {
            reinterpret_cast<struct completion*>(comp)->m_count++;
        }

        int m_count;
    };

    void test_am(uct_ep_am_short_func_t sender)
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
            ASSERT_UCS_OK(sender(m_e1->ep(0), 31, header++, payload,
                                 sizeof(payload)));
            expect.hdr += header;
            ASSERT_UCS_OK(sender(m_e1->ep(0), 31, header++, payload,
                                 sizeof(payload)));
            expect.hdr += header;
            ASSERT_UCS_OK(sender(m_e3->ep(0), 14, header--, payload,
                                 sizeof(payload) - 6));
        }

        wait_for_value(&stats.count, expect.count, true);
        EXPECT_EQ(expect.bytes, stats.bytes);
        EXPECT_EQ(expect.count, stats.count);
        EXPECT_EQ(expect.hdr,   stats.hdr);
    }

    void progress_ctl()
    {
        while (!(((uct_srd_ep_t*)m_e1->ep(0))->flags &
               UCT_SRD_EP_FLAG_AH_ADDED)) {
            short_progress_loop();
        }
    }

    void pending_purge_check(int expected_count)
    {
        int count = 0;
        uct_ep_pending_purge(
                m_e1->ep(0),
                [](uct_pending_req_t*, void *arg) {
                    (*reinterpret_cast<int*>(arg))++;
                },
                &count);
        ASSERT_EQ(expected_count, count);
    }

    ucs_status_t test_get_bcopy()
    {
        mapped_buffer srcbuf(4096, 0ul, *m_e2);
        mapped_buffer dstbuf(4096, 0ul, *m_e1);

        m_comp.count   = 1;
        m_comp.m_count = 0;

        srcbuf.pattern_fill(m_seed);
        ucs_status_t status = uct_ep_get_bcopy(m_e1->ep(0),
                                               (uct_unpack_callback_t)memcpy,
                                               dstbuf.ptr(), dstbuf.length(),
                                               srcbuf.addr(), srcbuf.rkey(),
                                               &m_comp);
        if ((status != UCS_INPROGRESS) &&
            (status != UCS_OK)) {
            return status;
        }

        ASSERT_UCS_STATUS_EQ(UCS_INPROGRESS, status);
        wait_for_value(&m_comp.m_count, 1, true);
        dstbuf.pattern_check(m_seed);
        return UCS_OK;
    }

    ucs_status_t test_get_zcopy()
    {
        mapped_buffer srcbuf(4096, 0ul, *m_e2);
        mapped_buffer dstbuf(4096, 0ul, *m_e1);

        m_comp.count   = 1;
        m_comp.m_count = 0;

        UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, dstbuf.ptr(), dstbuf.length(),
                                dstbuf.memh(), 1);

        srcbuf.pattern_fill(m_seed);
        ucs_status_t status = uct_ep_get_zcopy(m_e1->ep(0), iov, iovcnt,
                                               srcbuf.addr(), srcbuf.rkey(),
                                               &m_comp);
        if ((status != UCS_INPROGRESS) &&
            (status != UCS_OK)) {
            return status;
        }

        ASSERT_UCS_STATUS_EQ(UCS_INPROGRESS, status);
        wait_for_value(&m_comp.m_count, 1, true);
        dstbuf.pattern_check(m_seed);
        return UCS_OK;
    }

    template<typename F> void test_fence(F &&send_cb, bool wait_progress = true)
    {
        int count = 0;
        int c     = 1;
        ASSERT_UCS_OK(uct_iface_set_am_handler(
                m_e2->iface(), 31,
                [](void *arg, void *data, size_t length, unsigned flags) {
                    (*reinterpret_cast<int*>(arg))++;
                    return UCS_OK;
                },
                &count, 0));

        progress_ctl();

        ASSERT_UCS_OK(uct_ep_am_short(m_e1->ep(0), 31, 0x123, &c, 1));
        ASSERT_UCS_OK(uct_ep_fence(m_e1->ep(0), 0));
        ASSERT_UCS_STATUS_EQ(UCS_ERR_NO_RESOURCE, send_cb());

        if (wait_progress) {
            wait_for_value(&count, 1, true);
            EXPECT_EQ(1, count);
        }

        ASSERT_UCS_OK(send_cb());
        ASSERT_UCS_OK(send_cb());
    }

    void test_flush_comp(int flags, ucs_status_t expect_status)
    {
        int c = 1;
        completion comp[2];

        auto noop = [](void *arg, void *data, size_t length, unsigned flags) {
            return UCS_OK;
        };

        ASSERT_UCS_OK(
                uct_iface_set_am_handler(m_e2->iface(), 31, noop, NULL, 0));
        ASSERT_UCS_OK(uct_ep_am_short(m_e1->ep(0), 31, 0x123, &c, 1));

        ASSERT_UCS_STATUS_EQ(UCS_INPROGRESS,
                             uct_ep_flush(m_e1->ep(0), flags, &comp[0]));
        ASSERT_UCS_STATUS_EQ(UCS_INPROGRESS,
                             uct_ep_flush(m_e1->ep(0), flags, &comp[1]));
        wait_for_value(&comp[1].m_count, 1, true);

        ASSERT_EQ(comp[0].m_count, 1);
        ASSERT_EQ(comp[1].m_count, 1);
        ASSERT_EQ(expect_status, comp[0].status);
        ASSERT_EQ(expect_status, comp[1].status);
    }

    completion                m_comp;
    static constexpr uint64_t m_seed = 0x54321;

    static constexpr int m_req_count = 3;
    uct_pending_req_t m_req[m_req_count];
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
    test_am(uct_ep_am_short);
}

UCS_TEST_P(test_srd, am_bcopy)
{
    test_am([](uct_ep_h tl_ep, uint8_t id, uint64_t hdr, const void *buffer,
               unsigned length) {
        struct iovec iov[] = {{.iov_base = &hdr,
                               .iov_len = sizeof(hdr)},
                              {.iov_base = const_cast<void*>(buffer),
                               .iov_len = length}};

        auto pack = [](void *dest, void *arg) {
            const struct iovec *iov = reinterpret_cast<struct iovec*>(arg);
            size_t total            = 0;
            for (int count = 2; count-- > 0; total += iov->iov_len, iov++) {
                memcpy(dest, iov->iov_base, iov->iov_len);
                dest = reinterpret_cast<uint8_t*>(dest) + iov->iov_len;
            }
            return total;
        };

        return uct_ep_am_bcopy(tl_ep, id, pack, iov, 0) > -1 ?
                       UCS_OK :
                       UCS_ERR_NO_RESOURCE;
    });
}

UCS_TEST_P(test_srd, am_short_iov)
{
    test_am([](uct_ep_h tl_ep, uint8_t id, uint64_t hdr, const void *buffer,
               unsigned length) {
        uint8_t srcbuf[length + sizeof(hdr)];

        memcpy(srcbuf, &hdr, sizeof(hdr));
        memcpy(srcbuf + sizeof(hdr), buffer, length);
        UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, srcbuf, sizeof(srcbuf), NULL, 1);
        return uct_ep_am_short_iov(tl_ep, id, iov, iovcnt);
    });
}

UCS_TEST_P(test_srd, am_zcopy)
{
    mapped_buffer srcbuf(4000, 0ul, *m_e1);
    mapped_buffer dstbuf(4000, 0ul, *m_e1);
    uint8_t id   = 13;
    uint64_t hdr = 0x1234567843126543llu;

    auto check_func = [](void *arg, void *data, size_t length,
                         unsigned flags) -> ucs_status_t {
        if ((4000 + 8 == length) &&
            (0x1234567843126543llu == *reinterpret_cast<uint64_t*>(data))) {
            memcpy(arg, reinterpret_cast<uint8_t*>(data) + 8, length - 8);
        }

        return UCS_OK;
    };

    ASSERT_UCS_OK(uct_iface_set_am_handler(m_e2->iface(), id, check_func,
                                           dstbuf.ptr(), 0));

    srcbuf.pattern_fill(m_seed);
    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, srcbuf.ptr(), srcbuf.length(),
                            srcbuf.memh(), 1);
    ucs_status_t status = uct_ep_am_zcopy(m_e1->ep(0), id, &hdr, sizeof(hdr),
                                          iov, iovcnt, 0, &m_comp);
    ASSERT_UCS_STATUS_EQ(UCS_INPROGRESS, status);

    wait_for_value(&m_comp.m_count, 1, true);
    dstbuf.pattern_check(m_seed);
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

UCS_TEST_P(test_srd, am_short_iov_failure)
{
    scoped_log_handler slh(wrap_errors_logger);
    char payload[64];
    ucs_status_t status;
    uct_iov_t iov;

    status = uct_ep_am_short_iov(m_e1->ep(0), 32, &iov, 1);
    ASSERT_UCS_STATUS_EQ(UCS_ERR_INVALID_PARAM, status);
    status     = uct_ep_am_short_iov(m_e1->ep(0), 14, &iov, 2);
    iov.length = sizeof(payload);
    iov.buffer = payload;
    iov.count  = 1;
    status     = uct_ep_am_short_iov(m_e1->ep(0), 14, &iov, 1);
    ASSERT_UCS_STATUS_EQ(UCS_ERR_INVALID_PARAM, status);
}

UCS_TEST_P(test_srd, am_bcopy_failure)
{
    scoped_log_handler slh(wrap_errors_logger);
    ssize_t size;
    size_t length = 12;

    auto pack = [](void *dest, void *arg) {
        auto length = *reinterpret_cast<size_t*>(arg);
        memset(dest, 0, length);
        return length;
    };

    size = uct_ep_am_bcopy(m_e1->ep(0), 32, pack, &length, 0);
    ASSERT_UCS_STATUS_EQ(UCS_ERR_INVALID_PARAM,
                         static_cast<ucs_status_t>(size));
}

UCS_TEST_P(test_srd, am_zcopy_failure)
{
    scoped_log_handler slh(wrap_errors_logger);
    ucs_status_t status;
    uct_iov_t iov[3];

    for (auto i = 0; i < 3; ++i) {
        iov[i].length = 64;
        iov[i].count  = 1;
    }

    status = uct_ep_am_zcopy(m_e1->ep(0), 32, NULL, 0, iov, 1, 0, NULL);
    ASSERT_UCS_STATUS_EQ(UCS_ERR_INVALID_PARAM, status);
    status = uct_ep_am_zcopy(m_e1->ep(0), 14, NULL, 0, iov, 2, 0, NULL);
    ASSERT_UCS_STATUS_EQ(UCS_ERR_INVALID_PARAM, status);
    status = uct_ep_am_zcopy(m_e1->ep(0), 14, NULL, UINT32_MAX, iov, 1, 0,
                             NULL);
    ASSERT_UCS_STATUS_EQ(UCS_ERR_INVALID_PARAM, status);
    iov[0].length = UINT32_MAX;
    status        = uct_ep_am_zcopy(m_e1->ep(0), 14, NULL, 0, iov, 1, 0, NULL);
    ASSERT_UCS_STATUS_EQ(UCS_ERR_INVALID_PARAM, status);
}

UCS_TEST_P(test_srd, get_bcopy_failure)
{
    scoped_log_handler slh(wrap_errors_logger);
    ucs_status_t status;

    size_t length = 4097;
    status = uct_ep_get_bcopy(m_e1->ep(0), NULL, NULL, length, 0, 0, NULL);
    ASSERT_UCS_STATUS_EQ(UCS_ERR_INVALID_PARAM, status);
}

UCS_TEST_P(test_srd, get_zcopy_failure)
{
    scoped_log_handler slh(wrap_errors_logger);
    uct_iov_t iov[3];
    ucs_status_t status;

    for (auto i = 0; i < 3; ++i) {
        iov[i].length = (i == 0 ? 0 : 64);
        iov[i].count  = 1;
    }

    progress_ctl();

    status = uct_ep_get_zcopy(m_e1->ep(0), iov, 2, 0, 0, NULL);
    ASSERT_UCS_STATUS_EQ(UCS_ERR_INVALID_PARAM, status);
    iov[0].length = UCS_GBYTE * 2;
    status        = uct_ep_get_zcopy(m_e1->ep(0), iov, 1, 0, 0, NULL);
    ASSERT_UCS_STATUS_EQ(UCS_ERR_INVALID_PARAM, status);
}

UCS_TEST_P(test_srd, get_zcopy_destroy_outstanding)
{
    mapped_buffer srcbuf(4096, 0ul, *m_e2);
    mapped_buffer dstbuf(4096, 0ul, *m_e1);

    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, dstbuf.ptr(), dstbuf.length(),
                            dstbuf.memh(), 1);

    progress_ctl();

    ucs_status_t status = uct_ep_get_zcopy(m_e1->ep(0), iov, iovcnt,
                                           srcbuf.addr(), srcbuf.rkey(),
                                           &m_comp);
    ASSERT_UCS_STATUS_EQ(UCS_INPROGRESS, status);
    ASSERT_EQ(0, m_comp.m_count);
}

UCS_TEST_P(test_srd, get_zcopy)
{
    progress_ctl();

    ASSERT_UCS_OK(test_get_zcopy());
}

UCS_TEST_P(test_srd, get_bcopy)
{
    progress_ctl();

    ASSERT_UCS_OK(test_get_bcopy());
}

UCS_TEST_P(test_srd, put_zcopy)
{
    mapped_buffer srcbuf(4096, 0ul, *m_e1);
    mapped_buffer dstbuf(4096, 0ul, *m_e2);

    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, srcbuf.ptr(), srcbuf.length(),
                            srcbuf.memh(), 1);

    progress_ctl();
    srcbuf.pattern_fill(m_seed);
    ucs_status_t status = uct_ep_put_zcopy(m_e1->ep(0), iov, iovcnt,
                                           dstbuf.addr(), dstbuf.rkey(),
                                           &m_comp);
    ASSERT_UCS_STATUS_EQ(UCS_INPROGRESS, status);
    wait_for_value(&m_comp.m_count, 1, true);
    dstbuf.pattern_check(m_seed);
}

UCS_TEST_P(test_srd, put_bcopy)
{
    ssize_t size = 4096;
    mapped_buffer srcbuf(size, 0ul, *m_e1);
    mapped_buffer dstbuf(size, 0ul, *m_e2);

    auto pack_put_bcopy = [](void *dest, void *arg) {
        size_t length = 4096;
        memcpy(dest, arg, length);
        return length;
    };

    progress_ctl();
    srcbuf.pattern_fill(m_seed);
    ssize_t length = uct_ep_put_bcopy(m_e1->ep(0),
                                      pack_put_bcopy,
                                      srcbuf.ptr(),
                                      dstbuf.addr(), dstbuf.rkey());
    ASSERT_EQ(size, length);
    short_progress_loop();
    dstbuf.pattern_check(m_seed);
}

UCS_TEST_P(test_srd, get_bcopy_no_resource)
{
    int i, count = 400;
    mapped_buffer srcbuf(4096, 0ul, *m_e2);
    mapped_buffer dstbuf(4096, 0ul, *m_e1);
    ucs_status_t status;

    progress_ctl();

    for (i = 0; i < count; i++) {
        status = uct_ep_get_bcopy(m_e1->ep(0), (uct_unpack_callback_t)memcpy,
                                  dstbuf.ptr(), dstbuf.length(), srcbuf.addr(),
                                  srcbuf.rkey(), NULL);
        if (status != UCS_INPROGRESS) {
            break;
        }
    }

    ASSERT_LT(100, i);
    ASSERT_GT(count, i);
    ASSERT_UCS_STATUS_EQ(UCS_ERR_NO_RESOURCE, status);
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

UCS_TEST_P(test_srd, am_short_no_resource_with_pending)
{
    uint8_t c = 23;

    m_req[0].func = [](uct_pending_req_t*) { return UCS_OK; };

    /* Cannot add pending as no RMA operation was posted */
    ASSERT_UCS_STATUS_EQ(UCS_ERR_BUSY,
                         uct_ep_pending_add(m_e1->ep(0), &m_req[0], 0));
    ASSERT_UCS_STATUS_EQ(UCS_OK, uct_ep_am_short(m_e1->ep(0), 31, 0x1, &c, 1));
}

UCS_TEST_P(test_srd, pending_busy_if_ready_to_send)
{
    progress_ctl();
    ASSERT_UCS_STATUS_EQ(UCS_ERR_BUSY,
                         uct_ep_pending_add(m_e1->ep(0), &m_req[0], 0));
}

UCS_TEST_P(test_srd, am_no_res_after_rma_no_resource)
{
    int c = 1;
    ASSERT_UCS_STATUS_EQ(UCS_ERR_NO_RESOURCE, test_get_bcopy());
    ASSERT_UCS_STATUS_EQ(UCS_ERR_NO_RESOURCE,
                        uct_ep_am_short(m_e1->ep(0), 31, 0x1, &c, 1));
}

UCS_TEST_P(test_srd, pending_dispatch)
{
    uint8_t c = 23;
    auto i    = 0;
    static int req_count;
    ucs_status_t status;

    auto noop_func = [](void *arg, void *data, size_t length, unsigned flags) {
        return UCS_OK;
    };

    ASSERT_UCS_OK(
            uct_iface_set_am_handler(m_e2->iface(), 31, noop_func, NULL, 0));

    for (; i < 400; i++) {
        status = uct_ep_am_short(m_e1->ep(0), 31, 0x1, &c, 1);
        if (status == UCS_ERR_NO_RESOURCE) {
            break;
        }
    }

    ASSERT_LT(0, i);
    ASSERT_LT(i, 400);

    req_count = 0;
    for (auto i = 0; i < 3; i++) {
        m_req[i].func = [](uct_pending_req_t*) {
            req_count++;
            if (req_count == 1) {
                return UCS_ERR_NO_RESOURCE;
            } else if (req_count == 2) {
                return UCS_INPROGRESS;
            } else {
                return UCS_OK;
            }
        };

        ASSERT_UCS_OK(uct_ep_pending_add(m_e1->ep(0), &m_req[i], 0));
    }

    wait_for_value(&req_count, 5, true);
    pending_purge_check(0);
}

UCS_TEST_P(test_srd, get_bcopy_ah_pending)
{
    mapped_buffer srcbuf(4096, 0ul, *m_e2);
    mapped_buffer dstbuf(4096, 0ul, *m_e1);
    ucs_status_t status;

    m_req[0].func = [](uct_pending_req_t*) { return UCS_OK; };
    ASSERT_UCS_STATUS_EQ(UCS_ERR_BUSY,
                         uct_ep_pending_add(m_e1->ep(0), &m_req[0], 0));
    status = uct_ep_get_bcopy(m_e1->ep(0), (uct_unpack_callback_t)memcpy,
                              dstbuf.ptr(), dstbuf.length(), srcbuf.addr(),
                              srcbuf.rkey(), NULL);
    ASSERT_UCS_STATUS_EQ(UCS_ERR_NO_RESOURCE, status);

    /* AM cannot be posted after EP has seen RMA */
    char c = 1;
    status = uct_ep_am_short(m_e1->ep(0), 31, 0x1, &c, 1);
    ASSERT_UCS_STATUS_EQ(UCS_ERR_NO_RESOURCE, status);

    ASSERT_UCS_OK(uct_ep_pending_add(m_e1->ep(0), &m_req[0], 0));
}

UCS_TEST_P(test_srd, destroy_ep_before_ctl_resp)
{
    m_e1->connect_to_iface(1, *m_e2);
    m_e1->destroy_ep(1);

    progress_ctl();
}

UCS_TEST_P(test_srd, fence_no_resource_am_short)
{
    test_fence([&]() {
        int c = 1;
        return uct_ep_am_short(m_e1->ep(0), 31, 0x123, &c, 1);
    });
}

UCS_TEST_P(test_srd, fence_no_resource_get_bcopy)
{
    test_fence([&]() { return test_get_bcopy(); });
}

UCS_TEST_P(test_srd, fence_no_resource_get_zcopy)
{
    test_fence([&]() { return test_get_zcopy(); });
}

UCS_TEST_P(test_srd, fence_no_resource_pending)
{
    int step = 0;
    ucs_status_t status;

    m_req[0].func = [](uct_pending_req_t*) { return UCS_OK; };

    test_fence([&]() {
        step++;
        if (step == 1) {
            /* Can add to pending since fence is pending */
            status = uct_ep_pending_add(m_e1->ep(0), &m_req[0], 0);
            ASSERT_UCS_OK(status);
            return UCS_ERR_NO_RESOURCE;
        }

        /* Could send, should not be able to add to pending */
        ASSERT_UCS_STATUS_EQ(UCS_ERR_BUSY,
                             uct_ep_pending_add(m_e1->ep(0), &m_req[1], 0));
        return UCS_OK;
    });

    /* Pending must have already been executed */
    pending_purge_check(0);
}

UCS_TEST_P(test_srd, flush_no_outstanding)
{
    ASSERT_UCS_OK(uct_ep_flush(m_e1->ep(0), 0, NULL));
    ASSERT_UCS_OK(uct_ep_flush(m_e1->ep(0), UCT_FLUSH_FLAG_CANCEL, NULL));
}

UCS_TEST_P(test_srd, flush_pending_canceled_no_outstanding)
{
    mapped_buffer srcbuf(4096, 0ul, *m_e2);
    mapped_buffer dstbuf(4096, 0ul, *m_e1);
    ucs_status_t status;

    m_req[0].func = [](uct_pending_req_t*) { return UCS_OK; };
    status = uct_ep_get_bcopy(m_e1->ep(0), (uct_unpack_callback_t)memcpy,
                              dstbuf.ptr(), dstbuf.length(), srcbuf.addr(),
                              srcbuf.rkey(), NULL);
    ASSERT_UCS_STATUS_EQ(UCS_ERR_NO_RESOURCE, status);

    /* Pending added as RMA was seen and no CTL message received */
    ASSERT_UCS_OK(uct_ep_pending_add(m_e1->ep(0), &m_req[0], 0));
    ASSERT_UCS_OK(uct_ep_flush(m_e1->ep(0), UCT_FLUSH_FLAG_CANCEL, NULL));

    pending_purge_check(0);
}

UCS_TEST_P(test_srd, flush_outstanding)
{
    int c = 1;

    ASSERT_UCS_OK(uct_ep_am_short(m_e1->ep(0), 31, 0x123, &c, 1));
    ASSERT_UCS_STATUS_EQ(UCS_INPROGRESS, uct_ep_flush(m_e1->ep(0), 0, NULL));
}

UCS_TEST_P(test_srd, flush_pending_canceled_outstanding)
{
    int c         = 1;
    m_req[0].func = [](uct_pending_req_t*) { return UCS_OK; };
    ucs_status_t status;

    for (auto count = 10000; count > 0; --count) {
        status = uct_ep_am_short(m_e1->ep(0), 31, 0x123, &c, 1);
        if (status != UCS_OK) {
            break;
        }
    }

    ASSERT_UCS_STATUS_EQ(UCS_ERR_NO_RESOURCE, status);
    ASSERT_UCS_OK(uct_ep_pending_add(m_e1->ep(0), &m_req[0], 0));
    ASSERT_UCS_STATUS_EQ(UCS_INPROGRESS,
                         uct_ep_flush(m_e1->ep(0), UCT_FLUSH_FLAG_CANCEL,
                                      NULL));

    pending_purge_check(0);
}

UCS_TEST_P(test_srd, flush_comp)
{
    test_flush_comp(0, UCS_OK);
}

UCS_TEST_P(test_srd, flush_comp_cancel)
{
    test_flush_comp(UCT_FLUSH_FLAG_CANCEL, UCS_ERR_CANCELED);
}

UCS_TEST_P(test_srd, iface_flush_comp_unsupported)
{
    ASSERT_UCS_STATUS_EQ(UCS_ERR_UNSUPPORTED,
                         uct_iface_flush(m_e1->iface(), 0, &m_comp));
}

UCS_TEST_P(test_srd, iface_flush_no_outstanding)
{
    m_e1->connect_to_iface(1, *m_e2);
    m_e1->connect_to_iface(2, *m_e2);
    m_e1->connect_to_iface(3, *m_e3);

    ASSERT_UCS_OK(uct_iface_flush(m_e1->iface(), 0, NULL));
}

UCS_TEST_P(test_srd, iface_flush_inprogress)
{
    m_e1->connect_to_iface(1, *m_e2);
    m_e1->connect_to_iface(2, *m_e2);
    m_e1->connect_to_iface(3, *m_e3);

    int c = 1, count = 0;
    auto counter_func = [](void *arg, void *data, size_t length,
                           unsigned flags) {
        (*reinterpret_cast<int*>(arg))++;
        return UCS_OK;
    };

    ASSERT_UCS_OK(uct_iface_set_am_handler(m_e2->iface(), 31, counter_func,
                                           &count, 0));

    ASSERT_UCS_OK(uct_ep_am_short(m_e1->ep(1), 31, 0x123, &c, 1));
    ASSERT_UCS_STATUS_EQ(UCS_INPROGRESS,
                         uct_iface_flush(m_e1->iface(), 0, NULL));
    ASSERT_UCS_STATUS_EQ(UCS_INPROGRESS,
                         uct_iface_flush(m_e1->iface(), 0, NULL));
    wait_for_value(&count, 1, true);
    ASSERT_EQ(1, count);
    ASSERT_UCS_OK(uct_iface_flush(m_e1->iface(), 0, NULL));
}

UCS_TEST_P(test_srd, iface_fence_none)
{
    ASSERT_UCS_OK(uct_iface_fence(m_e1->iface(), 0));
}

UCS_TEST_P(test_srd, iface_fence_outstanding)
{
    m_e1->connect_to_iface(1, *m_e2);
    m_e1->connect_to_iface(2, *m_e2);
    m_e1->connect_to_iface(3, *m_e3);

    progress_ctl();

    int c = 1, count = 0;
    auto counter_func = [](void *arg, void *data, size_t length,
                           unsigned flags) {
        (*reinterpret_cast<int*>(arg))++;
        return UCS_OK;
    };

    ASSERT_UCS_OK(uct_iface_set_am_handler(m_e2->iface(), 31, counter_func,
                                           &count, 0));
    ASSERT_UCS_OK(uct_ep_am_short(m_e1->ep(1), 31, 0x123, &c, 1));
    ASSERT_UCS_OK(uct_ep_am_short(m_e1->ep(2), 31, 0x123, &c, 1));

    m_req[0].func = [](uct_pending_req_t*) { return UCS_OK; };
    /* Cannot be added we are ready to send */
    ASSERT_UCS_STATUS_EQ(UCS_ERR_BUSY,
                         uct_ep_pending_add(m_e1->ep(0), &m_req[0], 0));
    ASSERT_UCS_OK(uct_iface_fence(m_e1->iface(), 0));
    ASSERT_UCS_OK(uct_ep_pending_add(m_e1->ep(0), &m_req[0], 0));

    ASSERT_UCS_STATUS_EQ(UCS_ERR_NO_RESOURCE,
                         uct_ep_am_short(m_e1->ep(0), 31, 0x123, &c, 1));
    ASSERT_UCS_STATUS_EQ(UCS_ERR_NO_RESOURCE,
                         uct_ep_am_short(m_e1->ep(1), 31, 0x123, &c, 1));
    ASSERT_UCS_STATUS_EQ(UCS_ERR_NO_RESOURCE,
                         uct_ep_am_short(m_e1->ep(2), 31, 0x123, &c, 1));
    ASSERT_UCS_STATUS_EQ(UCS_ERR_NO_RESOURCE,
                         uct_ep_am_short(m_e1->ep(3), 31, 0x123, &c, 1));

    wait_for_value(&count, 2, true);
    ASSERT_EQ(2, count);

    /* Iface fence on empty outstanding queues */
    ASSERT_UCS_OK(uct_iface_fence(m_e1->iface(), 0));

    /* Pending cannot be added as we are ready to transmit */
    ASSERT_UCS_STATUS_EQ(UCS_ERR_BUSY,
                         uct_ep_pending_add(m_e1->ep(0), &m_req[1], 0));

    ASSERT_UCS_OK(uct_ep_am_short(m_e1->ep(1), 31, 0x123, &c, 1));
    wait_for_value(&count, 3, true);
    ASSERT_EQ(3, count);

    /* Pending was executed too */
    pending_purge_check(0);
}

UCT_INSTANTIATE_SRD_TEST_CASE(test_srd)
