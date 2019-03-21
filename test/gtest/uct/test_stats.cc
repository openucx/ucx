
/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
extern "C" {
#include <uct/api/uct.h>
#include <uct/base/uct_iface.h>

#include <ucs/time/time.h>
}
#include <common/test.h>
#include "uct_test.h"
#include "uct_p2p_test.h"

#ifdef ENABLE_STATS

class test_uct_stats : public uct_p2p_test {
public:
    test_uct_stats() : uct_p2p_test(0), lbuf(NULL), rbuf(NULL)  {
        m_comp.func  = NULL;
        m_comp.count = 0;
    }

    virtual void init() {
        stats_activate();
        uct_p2p_test::init();
    }

    void init_bufs(size_t min, size_t max)
    {
        size_t size = ucs_max(min, ucs_min(64ul, max));
        lbuf = new mapped_buffer(size, 0, sender(), 0, sender().md_attr().cap.mem_type);
        rbuf = new mapped_buffer(size, 0, receiver(), 0, sender().md_attr().cap.mem_type);
    }

    virtual void cleanup() {
        delete lbuf;
        delete rbuf;
        uct_p2p_test::cleanup();
        stats_restore();
    }

    uct_base_ep_t *uct_ep(const entity &e)
    {
            return ucs_derived_of(e.ep(0), uct_base_ep_t);
    }

    uct_base_iface_t *uct_iface(const entity &e)
    {
            return ucs_derived_of(e.iface(), uct_base_iface_t);
    }

    static ucs_status_t am_handler(void *arg, void *data, size_t length,
                                   unsigned flags) {
        return UCS_OK;
    }

    static void purge_cb(uct_pending_req_t *r, void *arg)
    {
    }

    void check_tx_counters(int op, int type, size_t len) {
        uint64_t v;

        v = UCS_STATS_GET_COUNTER(uct_ep(sender())->stats, op);
        EXPECT_EQ(1UL, v);
        v = UCS_STATS_GET_COUNTER(uct_ep(sender())->stats, type);
        EXPECT_EQ(len, v);
    }

    void check_am_rx_counters(size_t len) {
        uint64_t v;

        ucs_time_t deadline = ucs::get_deadline();
        do {
            short_progress_loop();
            v = UCS_STATS_GET_COUNTER(uct_iface(receiver())->stats, UCT_IFACE_STAT_RX_AM);
        } while ((ucs_get_time() < deadline) && !v);

        EXPECT_EQ(1UL, v);
        v = UCS_STATS_GET_COUNTER(uct_iface(receiver())->stats, UCT_IFACE_STAT_RX_AM_BYTES);
        EXPECT_EQ(len, v);
    }

    void check_atomic_counters() {
        uint64_t v;

        v = UCS_STATS_GET_COUNTER(uct_ep(sender())->stats, UCT_EP_STAT_ATOMIC);
        EXPECT_EQ(1UL, v);
        /* give atomic chance to complete */
        short_progress_loop();
    }

    int fill_tx_q(int n) {
        int count_wait;
        int i, max;
        size_t len;

        max = (n == 0) ? 1024 : n;

        for (count_wait = i = 0; i < max; i++) {
            len = uct_ep_am_bcopy(sender_ep(), 0, mapped_buffer::pack, lbuf, 0);
            if (len != lbuf->length()) {
                if (n == 0) {
                    return 1;
                }
                count_wait++;
            }
        }
        return count_wait;
    }

    void init_completion() {
        m_comp.count = 2;
        m_comp.func  = NULL;
    }

    void wait_for_completion(ucs_status_t status) {

        EXPECT_TRUE(UCS_INPROGRESS == status || UCS_OK == status);
        if (status == UCS_OK) {
            --m_comp.count;
        }

        ucs_time_t deadline = ucs::get_deadline();
        do {
            short_progress_loop();
        } while ((ucs_get_time() < deadline) && (m_comp.count > 1));
        EXPECT_EQ(1, m_comp.count);
    }

protected:
    mapped_buffer *lbuf, *rbuf;
    uct_completion_t m_comp;
};


/* test basic stat counters:
 * am, put, get, amo, flush and fence
 */
UCS_TEST_P(test_uct_stats, am_short)
{
    uint64_t hdr=0xdeadbeef, send_data=0xfeedf00d;
    ucs_status_t status;

    check_caps(UCT_IFACE_FLAG_AM_SHORT);
    init_bufs(0, sender().iface_attr().cap.am.max_short);

    status = uct_iface_set_am_handler(receiver().iface(), 0, am_handler,
                                      0, UCT_CB_FLAG_ASYNC);
    EXPECT_UCS_OK(status);

    UCT_TEST_CALL_AND_TRY_AGAIN(uct_ep_am_short(sender_ep(), 0, hdr, &send_data,
                                                sizeof(send_data)), status);
    EXPECT_UCS_OK(status);

    check_tx_counters(UCT_EP_STAT_AM, UCT_EP_STAT_BYTES_SHORT,
                      sizeof(hdr) + sizeof(send_data));
    check_am_rx_counters(sizeof(hdr) + sizeof(send_data));
}

UCS_TEST_P(test_uct_stats, am_bcopy)
{
    ssize_t v;
    ucs_status_t status;

    check_caps(UCT_IFACE_FLAG_AM_BCOPY);
    init_bufs(0, sender().iface_attr().cap.am.max_bcopy);

    status = uct_iface_set_am_handler(receiver().iface(), 0, am_handler, 0, UCT_CB_FLAG_ASYNC);
    EXPECT_UCS_OK(status);

    UCT_TEST_CALL_AND_TRY_AGAIN(uct_ep_am_bcopy(sender_ep(), 0, mapped_buffer::pack,
                                                lbuf, 0), v);
    EXPECT_EQ((ssize_t)lbuf->length(), v);

    check_tx_counters(UCT_EP_STAT_AM, UCT_EP_STAT_BYTES_BCOPY, lbuf->length());
    check_am_rx_counters(lbuf->length());
}

UCS_TEST_P(test_uct_stats, am_zcopy)
{
    ucs_status_t status;

    check_caps(UCT_IFACE_FLAG_AM_ZCOPY);
    init_bufs(0, sender().iface_attr().cap.am.max_zcopy);

    status = uct_iface_set_am_handler(receiver().iface(), 0, am_handler, 0, UCT_CB_FLAG_ASYNC);
    EXPECT_UCS_OK(status);

    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, lbuf->ptr(), lbuf->length(), lbuf->memh(),
                            sender().iface_attr().cap.am.max_iov);

    UCT_TEST_CALL_AND_TRY_AGAIN(uct_ep_am_zcopy(sender_ep(), 0, 0, 0,
                                                iov, iovcnt, 0, NULL), status);
    EXPECT_TRUE(UCS_INPROGRESS == status || UCS_OK == status);

    check_tx_counters(UCT_EP_STAT_AM, UCT_EP_STAT_BYTES_ZCOPY, lbuf->length());
    check_am_rx_counters(lbuf->length());
}


UCS_TEST_P(test_uct_stats, put_short)
{
    uint64_t send_data=0xfeedf00d;
    ucs_status_t status;

    check_caps(UCT_IFACE_FLAG_PUT_SHORT);
    init_bufs(0, sender().iface_attr().cap.put.max_short);

    UCT_TEST_CALL_AND_TRY_AGAIN(uct_ep_put_short(sender_ep(), &send_data, sizeof(send_data),
                                                 rbuf->addr(), rbuf->rkey()), status);
    EXPECT_UCS_OK(status);

    check_tx_counters(UCT_EP_STAT_PUT, UCT_EP_STAT_BYTES_SHORT,
                      sizeof(send_data));
}

UCS_TEST_P(test_uct_stats, put_bcopy)
{
    ssize_t v;

    check_caps(UCT_IFACE_FLAG_PUT_BCOPY);
    init_bufs(0, sender().iface_attr().cap.put.max_bcopy);

    UCT_TEST_CALL_AND_TRY_AGAIN(uct_ep_put_bcopy(sender_ep(), mapped_buffer::pack, lbuf,
                                                 rbuf->addr(), rbuf->rkey()), v);
    EXPECT_EQ((ssize_t)lbuf->length(), v);

    check_tx_counters(UCT_EP_STAT_PUT, UCT_EP_STAT_BYTES_BCOPY,
                      lbuf->length());
}

UCS_TEST_P(test_uct_stats, put_zcopy)
{
    ucs_status_t status;

    check_caps(UCT_IFACE_FLAG_PUT_ZCOPY);
    init_bufs(0, sender().iface_attr().cap.put.max_zcopy);

    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, lbuf->ptr(), lbuf->length(), lbuf->memh(),
                            sender().iface_attr().cap.put.max_iov);

    UCT_TEST_CALL_AND_TRY_AGAIN(
        uct_ep_put_zcopy(sender_ep(), iov, iovcnt, rbuf->addr(),
                         rbuf->rkey(), 0), status);
    EXPECT_TRUE(UCS_INPROGRESS == status || UCS_OK == status);

    check_tx_counters(UCT_EP_STAT_PUT, UCT_EP_STAT_BYTES_ZCOPY,
                      lbuf->length());
}


UCS_TEST_P(test_uct_stats, get_bcopy)
{
    ucs_status_t status;

    check_caps(UCT_IFACE_FLAG_GET_BCOPY);
    init_bufs(0, sender().iface_attr().cap.get.max_bcopy);

    init_completion();
    UCT_TEST_CALL_AND_TRY_AGAIN(
        uct_ep_get_bcopy(sender_ep(), (uct_unpack_callback_t)memcpy,
                         lbuf->ptr(), lbuf->length(),
                         rbuf->addr(), rbuf->rkey(), &m_comp), status);
    wait_for_completion(status);

    short_progress_loop();
    check_tx_counters(UCT_EP_STAT_GET, UCT_EP_STAT_BYTES_BCOPY,
                      lbuf->length());
}

UCS_TEST_P(test_uct_stats, get_zcopy)
{
    ucs_status_t status;

    check_caps(UCT_IFACE_FLAG_GET_ZCOPY);
    init_bufs(sender().iface_attr().cap.get.min_zcopy,
              sender().iface_attr().cap.get.max_zcopy);

    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, lbuf->ptr(), lbuf->length(), lbuf->memh(),
                            sender().iface_attr().cap.get.max_iov);

    init_completion();
    UCT_TEST_CALL_AND_TRY_AGAIN(
        uct_ep_get_zcopy(sender_ep(), iov, iovcnt, rbuf->addr(),
                         rbuf->rkey(), &m_comp), status);
    wait_for_completion(status);

    short_progress_loop();
    check_tx_counters(UCT_EP_STAT_GET, UCT_EP_STAT_BYTES_ZCOPY,
                      lbuf->length());
}

#define TEST_STATS_ATOMIC_POST(_op, _val)                                      \
UCS_TEST_P(test_uct_stats, atomic_post_ ## _op ## _val)                        \
{                                                                              \
    ucs_status_t status;                                                       \
    check_atomics(UCS_BIT(UCT_ATOMIC_OP_ ## _op), OP ## _val);                 \
    init_bufs(sizeof(uint##_val##_t), sizeof(uint##_val##_t));                 \
    status = uct_ep_atomic ##_val##_post(sender_ep(), (UCT_ATOMIC_OP_ ## _op), \
                                         1, rbuf->addr(), rbuf->rkey());       \
    EXPECT_UCS_OK(status);                                                     \
    check_atomic_counters();                                                   \
}

TEST_STATS_ATOMIC_POST(ADD, 32)
TEST_STATS_ATOMIC_POST(ADD, 64)
TEST_STATS_ATOMIC_POST(AND, 32)
TEST_STATS_ATOMIC_POST(AND, 64)
TEST_STATS_ATOMIC_POST(OR,  32)
TEST_STATS_ATOMIC_POST(OR,  64)
TEST_STATS_ATOMIC_POST(XOR, 32)
TEST_STATS_ATOMIC_POST(XOR, 64)


#define TEST_STATS_ATOMIC_FETCH(_op, _val)                                              \
UCS_TEST_P(test_uct_stats, atomic_fetch_## _op ## _val)                                 \
{                                                                                       \
    ucs_status_t status;                                                                \
    uint##_val##_t result;                                                              \
                                                                                        \
    check_atomics(UCS_BIT(UCT_ATOMIC_OP_ ## _op), FOP ## _val);                         \
    init_bufs(sizeof(result), sizeof(result));                                          \
                                                                                        \
    init_completion();                                                                  \
    status = uct_ep_atomic##_val##_fetch(sender_ep(), (UCT_ATOMIC_OP_ ## _op), 1,       \
                                         &result, rbuf->addr(), rbuf->rkey(), &m_comp); \
    wait_for_completion(status);                                                        \
                                                                                        \
    check_atomic_counters();                                                            \
}

TEST_STATS_ATOMIC_FETCH(ADD,  32)
TEST_STATS_ATOMIC_FETCH(ADD,  64)
TEST_STATS_ATOMIC_FETCH(AND,  32)
TEST_STATS_ATOMIC_FETCH(AND,  64)
TEST_STATS_ATOMIC_FETCH(OR,   32)
TEST_STATS_ATOMIC_FETCH(OR,   64)
TEST_STATS_ATOMIC_FETCH(XOR,  32)
TEST_STATS_ATOMIC_FETCH(XOR,  64)
TEST_STATS_ATOMIC_FETCH(SWAP, 32)
TEST_STATS_ATOMIC_FETCH(SWAP, 64)

#define TEST_STATS_ATOMIC_CSWAP(val) \
UCS_TEST_P(test_uct_stats, atomic_cswap##val) \
{ \
    ucs_status_t status; \
    uint##val##_t result; \
\
    check_atomics(UCS_BIT(UCT_ATOMIC_OP_CSWAP), FOP ## val); \
    init_bufs(sizeof(result), sizeof(result)); \
\
    init_completion(); \
    UCT_TEST_CALL_AND_TRY_AGAIN( \
        uct_ep_atomic_cswap##val (sender_ep(), 1, 2, rbuf->addr(), \
                                  rbuf->rkey(), &result, &m_comp), \
        status); \
    wait_for_completion(status); \
\
    check_atomic_counters(); \
}

TEST_STATS_ATOMIC_CSWAP(32)
TEST_STATS_ATOMIC_CSWAP(64)

UCS_TEST_P(test_uct_stats, flush)
{
    ucs_status_t status;
    uint64_t v;

    if (sender_ep()) {
        status = uct_ep_flush(sender_ep(), 0, NULL);
        EXPECT_UCS_OK(status);
        v = UCS_STATS_GET_COUNTER(uct_ep(sender())->stats, UCT_EP_STAT_FLUSH);
        EXPECT_EQ(1UL, v);
        v = UCS_STATS_GET_COUNTER(uct_ep(sender())->stats, UCT_EP_STAT_FLUSH_WAIT);
        EXPECT_EQ(0UL, v);
    }

    status = uct_iface_flush(sender().iface(), 0, NULL);
    EXPECT_UCS_OK(status);
    v = UCS_STATS_GET_COUNTER(uct_iface(sender())->stats, UCT_IFACE_STAT_FLUSH);
    EXPECT_EQ(1UL, v);
    v = UCS_STATS_GET_COUNTER(uct_iface(sender())->stats, UCT_IFACE_STAT_FLUSH_WAIT);
    EXPECT_EQ(0UL, v);
}

UCS_TEST_P(test_uct_stats, fence)
{
    ucs_status_t status;
    uint64_t v;

    if (sender_ep()) {
        status = uct_ep_fence(sender_ep(), 0);
        EXPECT_UCS_OK(status);
        v = UCS_STATS_GET_COUNTER(uct_ep(sender())->stats, UCT_EP_STAT_FENCE);
        EXPECT_EQ(1UL, v);
    }

    status = uct_iface_fence(sender().iface(), 0);
    EXPECT_UCS_OK(status);
    v = UCS_STATS_GET_COUNTER(uct_iface(sender())->stats, UCT_IFACE_STAT_FENCE);
    EXPECT_EQ(1UL, v);
}

/* flush test only check stats on tls with am_bcopy
 * TODO: full test matrix
 */
UCS_TEST_P(test_uct_stats, flush_wait_iface)
{
    uint64_t v;
    uint64_t count_wait;
    ucs_status_t status;

    check_caps(UCT_IFACE_FLAG_AM_BCOPY);
    init_bufs(0, sender().iface_attr().cap.am.max_bcopy);

    status = uct_iface_set_am_handler(receiver().iface(), 0, am_handler, 0, UCT_CB_FLAG_ASYNC);
    EXPECT_UCS_OK(status);

    fill_tx_q(0);
    count_wait = 0;
    do {
        status = uct_iface_flush(sender().iface(), 0, NULL);
        if (status == UCS_INPROGRESS) {
            count_wait++;
        }
        progress();
    } while (status != UCS_OK);

    v = UCS_STATS_GET_COUNTER(uct_iface(sender())->stats, UCT_IFACE_STAT_FLUSH);
    EXPECT_EQ(1UL, v);
    v = UCS_STATS_GET_COUNTER(uct_iface(sender())->stats, UCT_IFACE_STAT_FLUSH_WAIT);
    EXPECT_EQ(count_wait, v);
}

UCS_TEST_P(test_uct_stats, flush_wait_ep)
{
    uint64_t v;
    uint64_t count_wait;
    ucs_status_t status;

    check_caps(UCT_IFACE_FLAG_AM_BCOPY);
    init_bufs(0, sender().iface_attr().cap.am.max_bcopy);

    status = uct_iface_set_am_handler(receiver().iface(), 0, am_handler, 0, UCT_CB_FLAG_ASYNC);
    EXPECT_UCS_OK(status);

    fill_tx_q(0);
    count_wait = 0;
    do {
        status = uct_ep_flush(sender_ep(), 0, NULL);
        if (status == UCS_INPROGRESS) {
            count_wait++;
        }
        progress();
    } while (status != UCS_OK);

    v = UCS_STATS_GET_COUNTER(uct_ep(sender())->stats, UCT_EP_STAT_FLUSH);
    EXPECT_EQ(1UL, v);
    v = UCS_STATS_GET_COUNTER(uct_ep(sender())->stats, UCT_EP_STAT_FLUSH_WAIT);
    EXPECT_EQ(count_wait, v);
}

/* fence test only check stats on tls with am_bcopy
 * TODO: full test matrix
 */
UCS_TEST_P(test_uct_stats, fence_iface)
{
    uint64_t v;
    ucs_status_t status;

    check_caps(UCT_IFACE_FLAG_AM_BCOPY);
    init_bufs(0, sender().iface_attr().cap.am.max_bcopy);

    status = uct_iface_set_am_handler(receiver().iface(), 0, am_handler, 0, UCT_CB_FLAG_ASYNC);
    EXPECT_UCS_OK(status);

    fill_tx_q(0);

    status = uct_iface_fence(sender().iface(), 0);
    EXPECT_UCS_OK(status);

    fill_tx_q(0);

    v = UCS_STATS_GET_COUNTER(uct_iface(sender())->stats, UCT_IFACE_STAT_FENCE);
    EXPECT_EQ(1UL, v);
}

UCS_TEST_P(test_uct_stats, fence_ep)
{
    uint64_t v;
    ucs_status_t status;

    check_caps(UCT_IFACE_FLAG_AM_BCOPY);
    init_bufs(0, sender().iface_attr().cap.am.max_bcopy);

    status = uct_iface_set_am_handler(receiver().iface(), 0, am_handler, 0, UCT_CB_FLAG_ASYNC);
    EXPECT_UCS_OK(status);

    fill_tx_q(0);

    status = uct_ep_fence(sender_ep(), 0);
    EXPECT_UCS_OK(status);

    fill_tx_q(0);

    v = UCS_STATS_GET_COUNTER(uct_ep(sender())->stats, UCT_EP_STAT_FENCE);
    EXPECT_EQ(1UL, v);
}

UCS_TEST_P(test_uct_stats, tx_no_res)
{
    uint64_t v, count;
    ucs_status_t status;

    check_caps(UCT_IFACE_FLAG_AM_BCOPY);
    init_bufs(0, sender().iface_attr().cap.am.max_bcopy);

    status = uct_iface_set_am_handler(receiver().iface(), 0, am_handler, 0, UCT_CB_FLAG_ASYNC);
    EXPECT_UCS_OK(status);
    count = fill_tx_q(1024);
    v = UCS_STATS_GET_COUNTER(uct_ep(sender())->stats, UCT_EP_STAT_NO_RES);
    EXPECT_EQ(count, v);
    v = UCS_STATS_GET_COUNTER(uct_ep(sender())->stats, UCT_EP_STAT_AM);
    EXPECT_EQ(1024-count, v);
}

UCS_TEST_P(test_uct_stats, pending_add)
{
    const int num_reqs = 5;
    uct_pending_req_t p_reqs[num_reqs];
    uint64_t v;
    ssize_t len;

    check_caps(UCT_IFACE_FLAG_AM_BCOPY | UCT_IFACE_FLAG_PENDING);
    init_bufs(0, sender().iface_attr().cap.am.max_bcopy);

    EXPECT_UCS_OK(uct_iface_set_am_handler(receiver().iface(), 0, am_handler, 0,
                                           UCT_CB_FLAG_ASYNC));

    // Check that counter is not increased if pending_add returns NOT_OK
    EXPECT_EQ(uct_ep_pending_add(sender().ep(0), &p_reqs[0], 0),
              UCS_ERR_BUSY);
    v = UCS_STATS_GET_COUNTER(uct_ep(sender())->stats, UCT_EP_STAT_PENDING);
    EXPECT_EQ(0ul, v);

    // Check that counter gets increased on every successfull pending_add returns NOT_OK
    fill_tx_q(0);

    UCT_TEST_CALL_AND_TRY_AGAIN(
        uct_ep_am_bcopy(sender_ep(), 0, mapped_buffer::pack,
                        lbuf, 0), len);
    if (len == (ssize_t)lbuf->length()) {
        UCS_TEST_SKIP_R("Can't add to pending");
    }

    for (int i = 0; i < num_reqs; ++i) {
        p_reqs[i].func = NULL;
        EXPECT_UCS_OK(uct_ep_pending_add(sender().ep(0), &p_reqs[i], 0));
    }
    uct_ep_pending_purge(sender().ep(0), purge_cb, NULL);

    v = UCS_STATS_GET_COUNTER(uct_ep(sender())->stats, UCT_EP_STAT_PENDING);
    EXPECT_EQ(static_cast<uint64_t>(num_reqs), v);
}

UCT_INSTANTIATE_TEST_CASE(test_uct_stats);
#endif
