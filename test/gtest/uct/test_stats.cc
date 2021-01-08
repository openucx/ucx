
/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "uct_test.h"
#include "uct_p2p_test.h"
#include <common/test.h>
extern "C" {
#include <uct/api/uct.h>
#include <uct/base/uct_iface.h>

#include <ucs/time/time.h>
}

#ifdef ENABLE_STATS

#define EXPECT_STAT(_side, _uct_obj, _stat, _exp_val) \
    do { \
        uint64_t v = UCS_STATS_GET_COUNTER(_uct_obj(_side())->stats, _stat); \
        EXPECT_EQ(get_cntr_init(UCS_PP_MAKE_STRING(_side), \
                                UCS_PP_MAKE_STRING(_stat)) + (_exp_val), v); \
    } while (0)


class test_uct_stats : public uct_p2p_test {
public:
    test_uct_stats() : uct_p2p_test(0), lbuf(NULL), rbuf(NULL) {
        m_comp.func   = NULL;
        m_comp.count  = 0;
        m_comp.status = UCS_OK;
    }

    virtual void init() {
        stats_activate();
        uct_p2p_test::init();

        // Sender EP
        collect_cntr_init("sender", uct_ep(sender())->stats,
                          UCS_PP_MAKE_STRING(UCT_EP_STAT_FLUSH),
                          UCT_EP_STAT_FLUSH);
        collect_cntr_init("sender", uct_ep(sender())->stats,
                          UCS_PP_MAKE_STRING(UCT_EP_STAT_FLUSH_WAIT),
                          UCT_EP_STAT_FLUSH_WAIT);
        collect_cntr_init("sender", uct_ep(sender())->stats,
                          UCS_PP_MAKE_STRING(UCT_EP_STAT_FENCE),
                          UCT_EP_STAT_FENCE);
        collect_cntr_init("sender", uct_ep(sender())->stats,
                          UCS_PP_MAKE_STRING(UCT_EP_STAT_AM),
                          UCT_EP_STAT_AM);
        collect_cntr_init("sender", uct_ep(sender())->stats,
                          UCS_PP_MAKE_STRING(UCT_EP_STAT_NO_RES),
                          UCT_EP_STAT_NO_RES);
        collect_cntr_init("sender", uct_ep(sender())->stats,
                          UCS_PP_MAKE_STRING(UCT_EP_STAT_PENDING),
                          UCT_EP_STAT_PENDING);
        collect_cntr_init("sender", uct_ep(sender())->stats,
                          UCS_PP_MAKE_STRING(UCT_EP_STAT_ATOMIC),
                          UCT_EP_STAT_ATOMIC);

        // Sender IFACE
        collect_cntr_init("sender", uct_iface(sender())->stats,
                          UCS_PP_MAKE_STRING(UCT_IFACE_STAT_FLUSH),
                          UCT_IFACE_STAT_FLUSH);
        collect_cntr_init("sender", uct_iface(sender())->stats,
                          UCS_PP_MAKE_STRING(UCT_IFACE_STAT_FLUSH_WAIT),
                          UCT_IFACE_STAT_FLUSH_WAIT);
        collect_cntr_init("sender", uct_iface(sender())->stats,
                          UCS_PP_MAKE_STRING(UCT_IFACE_STAT_FENCE),
                          UCT_IFACE_STAT_FENCE);

        // Receiver IFACE
        collect_cntr_init("receiver", uct_iface(receiver())->stats,
                          UCS_PP_MAKE_STRING(UCT_IFACE_STAT_RX_AM),
                          UCT_IFACE_STAT_RX_AM);
        collect_cntr_init("receiver", uct_iface(receiver())->stats,
                          UCS_PP_MAKE_STRING(UCT_IFACE_STAT_RX_AM_BYTES),
                          UCT_IFACE_STAT_RX_AM_BYTES);
    }

    void collect_cntr_init(std::string side, ucs_stats_node_t *stats_node,
                           std::string stat_name, unsigned stat) {
        cntr_init[side][stat_name] = UCS_STATS_GET_COUNTER(stats_node, stat);
    }

    size_t get_cntr_init(std::string side, std::string stat_name) {
        return cntr_init[side][stat_name];
    }

    void init_bufs(size_t min, size_t max)
    {
        size_t size = ucs_max(min, ucs_min(64ul, max));
        uint8_t mem_type_index;

        ucs_assert(sender().md_attr().cap.access_mem_types != 0);
        mem_type_index = ucs_ffs64(sender().md_attr().cap.access_mem_types);

        lbuf = new mapped_buffer(size, 0, sender(), 0, (ucs_memory_type_t)mem_type_index);
        rbuf = new mapped_buffer(size, 0, receiver(), 0, (ucs_memory_type_t)mem_type_index);
    }

    virtual void cleanup() {
        flush();
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

    void check_am_rx_counters(size_t len) {
        uint64_t iface_rx_am_init = get_cntr_init("receiver",
                                                  UCS_PP_MAKE_STRING(UCT_IFACE_STAT_RX_AM));
        uint64_t v;

        ucs_time_t deadline = ucs::get_deadline();
        do {
            short_progress_loop();
            v = UCS_STATS_GET_COUNTER(uct_iface(receiver())->stats, UCT_IFACE_STAT_RX_AM);
        } while ((ucs_get_time() < deadline) && (v == iface_rx_am_init));

        EXPECT_STAT(receiver, uct_iface, UCT_IFACE_STAT_RX_AM, 1UL);
        EXPECT_STAT(receiver, uct_iface, UCT_IFACE_STAT_RX_AM_BYTES, len);
    }

    void check_atomic_counters() {
        EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_ATOMIC, 1UL);
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
        m_comp.count  = 2;
        m_comp.status = UCS_OK;
        m_comp.func   = NULL;
    }

    void wait_for_completion(ucs_status_t status) {

        EXPECT_FALSE(UCS_STATUS_IS_ERR(status));
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
    std::map< std::string, std::map< std::string, uint64_t > > cntr_init;
};


/* test basic stat counters:
 * am, put, get, amo, flush and fence
 */
UCS_TEST_SKIP_COND_P(test_uct_stats, am_short,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT))
{
    uint64_t hdr=0xdeadbeef, send_data=0xfeedf00d;
    ucs_status_t status;

    init_bufs(0, sender().iface_attr().cap.am.max_short);

    status = uct_iface_set_am_handler(receiver().iface(), 0, am_handler,
                                      0, UCT_CB_FLAG_ASYNC);
    EXPECT_UCS_OK(status);

    UCT_TEST_CALL_AND_TRY_AGAIN(uct_ep_am_short(sender_ep(), 0, hdr, &send_data,
                                                sizeof(send_data)), status);
    EXPECT_UCS_OK(status);

    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_AM, 1UL);
    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_BYTES_SHORT,
                sizeof(hdr) + sizeof(send_data));
    check_am_rx_counters(sizeof(hdr) + sizeof(send_data));
}

UCS_TEST_SKIP_COND_P(test_uct_stats, am_short_iov,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT))
{
    ucs_status_t status;

    init_bufs(0, sender().iface_attr().cap.am.max_short);

    status = uct_iface_set_am_handler(receiver().iface(), 0, am_handler, 0,
                                      UCT_CB_FLAG_ASYNC);
    EXPECT_UCS_OK(status);

    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, lbuf->ptr(), lbuf->length(), lbuf->memh(),
                            sender().iface_attr().cap.am.max_iov);

    UCT_TEST_CALL_AND_TRY_AGAIN(uct_ep_am_short_iov(sender_ep(), 0, iov, iovcnt),
                                status);
    EXPECT_UCS_OK(status);

    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_AM, 1UL);
    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_BYTES_SHORT, lbuf->length());
    check_am_rx_counters(lbuf->length());
}

UCS_TEST_SKIP_COND_P(test_uct_stats, am_bcopy,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY))
{
    ssize_t v;
    ucs_status_t status;

    init_bufs(0, sender().iface_attr().cap.am.max_bcopy);

    status = uct_iface_set_am_handler(receiver().iface(), 0, am_handler, 0, UCT_CB_FLAG_ASYNC);
    EXPECT_UCS_OK(status);

    UCT_TEST_CALL_AND_TRY_AGAIN(uct_ep_am_bcopy(sender_ep(), 0, mapped_buffer::pack,
                                                lbuf, 0), v);
    EXPECT_EQ((ssize_t)lbuf->length(), v);

    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_AM, 1UL);
    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_BYTES_BCOPY,
                lbuf->length());
    check_am_rx_counters(lbuf->length());
}

UCS_TEST_SKIP_COND_P(test_uct_stats, am_zcopy,
                     !check_caps(UCT_IFACE_FLAG_AM_ZCOPY))
{
    ucs_status_t status;

    init_bufs(0, sender().iface_attr().cap.am.max_zcopy);

    status = uct_iface_set_am_handler(receiver().iface(), 0, am_handler, 0, UCT_CB_FLAG_ASYNC);
    EXPECT_UCS_OK(status);

    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, lbuf->ptr(), lbuf->length(), lbuf->memh(),
                            sender().iface_attr().cap.am.max_iov);

    UCT_TEST_CALL_AND_TRY_AGAIN(uct_ep_am_zcopy(sender_ep(), 0, 0, 0,
                                                iov, iovcnt, 0, NULL), status);
    EXPECT_TRUE(UCS_INPROGRESS == status || UCS_OK == status);

    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_AM, 1UL);
    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_BYTES_ZCOPY,
                lbuf->length());
    check_am_rx_counters(lbuf->length());
}


UCS_TEST_SKIP_COND_P(test_uct_stats, put_short,
                     !check_caps(UCT_IFACE_FLAG_PUT_SHORT))
{
    uint64_t send_data=0xfeedf00d;
    ucs_status_t status;

    init_bufs(0, sender().iface_attr().cap.put.max_short);

    UCT_TEST_CALL_AND_TRY_AGAIN(uct_ep_put_short(sender_ep(), &send_data, sizeof(send_data),
                                                 rbuf->addr(), rbuf->rkey()), status);
    EXPECT_UCS_OK(status);

    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_PUT, 1UL);
    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_BYTES_SHORT,
                sizeof(send_data));
}

UCS_TEST_SKIP_COND_P(test_uct_stats, put_bcopy,
                     !check_caps(UCT_IFACE_FLAG_PUT_BCOPY))
{
    ssize_t v;

    init_bufs(0, sender().iface_attr().cap.put.max_bcopy);

    UCT_TEST_CALL_AND_TRY_AGAIN(uct_ep_put_bcopy(sender_ep(), mapped_buffer::pack, lbuf,
                                                 rbuf->addr(), rbuf->rkey()), v);
    EXPECT_EQ((ssize_t)lbuf->length(), v);

    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_PUT, 1UL);
    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_BYTES_BCOPY,
                lbuf->length());
}

UCS_TEST_SKIP_COND_P(test_uct_stats, put_zcopy,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY))
{
    ucs_status_t status;

    init_bufs(0, sender().iface_attr().cap.put.max_zcopy);

    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, lbuf->ptr(), lbuf->length(), lbuf->memh(),
                            sender().iface_attr().cap.put.max_iov);

    UCT_TEST_CALL_AND_TRY_AGAIN(
        uct_ep_put_zcopy(sender_ep(), iov, iovcnt, rbuf->addr(),
                         rbuf->rkey(), 0), status);
    EXPECT_FALSE(UCS_STATUS_IS_ERR(status));

    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_PUT, 1UL);
    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_BYTES_ZCOPY,
                lbuf->length());
}


UCS_TEST_SKIP_COND_P(test_uct_stats, get_bcopy,
                     !check_caps(UCT_IFACE_FLAG_GET_BCOPY))
{
    ucs_status_t status;

    init_bufs(0, sender().iface_attr().cap.get.max_bcopy);

    init_completion();
    UCT_TEST_CALL_AND_TRY_AGAIN(
        uct_ep_get_bcopy(sender_ep(), (uct_unpack_callback_t)memcpy,
                         lbuf->ptr(), lbuf->length(),
                         rbuf->addr(), rbuf->rkey(), &m_comp), status);
    wait_for_completion(status);

    short_progress_loop();
    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_GET, 1UL);
    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_BYTES_BCOPY,
                lbuf->length());
}

UCS_TEST_SKIP_COND_P(test_uct_stats, get_zcopy,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY))
{
    ucs_status_t status;

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
    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_GET, 1UL);
    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_BYTES_ZCOPY,
                lbuf->length());
}

#define TEST_STATS_ATOMIC_POST(_op, _val) \
UCS_TEST_SKIP_COND_P(test_uct_stats, atomic_post_ ## _op ## _val, \
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_ ## _op), OP ## _val)) \
{ \
    ucs_status_t status; \
    init_bufs(sizeof(uint##_val##_t), sizeof(uint##_val##_t)); \
    status = uct_ep_atomic ##_val##_post(sender_ep(), (UCT_ATOMIC_OP_ ## _op), \
                                         1, rbuf->addr(), rbuf->rkey()); \
    EXPECT_UCS_OK(status); \
    check_atomic_counters(); \
}

TEST_STATS_ATOMIC_POST(ADD, 32)
TEST_STATS_ATOMIC_POST(ADD, 64)
TEST_STATS_ATOMIC_POST(AND, 32)
TEST_STATS_ATOMIC_POST(AND, 64)
TEST_STATS_ATOMIC_POST(OR,  32)
TEST_STATS_ATOMIC_POST(OR,  64)
TEST_STATS_ATOMIC_POST(XOR, 32)
TEST_STATS_ATOMIC_POST(XOR, 64)


#define TEST_STATS_ATOMIC_FETCH(_op, _val) \
UCS_TEST_SKIP_COND_P(test_uct_stats, atomic_fetch_## _op ## _val, \
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_ ## _op), FOP ## _val)) \
{ \
    ucs_status_t status; \
    uint##_val##_t result; \
    \
    init_bufs(sizeof(result), sizeof(result)); \
    \
    init_completion(); \
    status = uct_ep_atomic##_val##_fetch(sender_ep(), (UCT_ATOMIC_OP_ ## _op), 1, \
                                         &result, rbuf->addr(), rbuf->rkey(), &m_comp); \
    wait_for_completion(status); \
    \
    check_atomic_counters(); \
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
UCS_TEST_SKIP_COND_P(test_uct_stats, atomic_cswap##val, \
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_CSWAP), FOP ## val)) \
{ \
    ucs_status_t status; \
    uint##val##_t result; \
    \
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

    if (sender_ep()) {
        status = uct_ep_flush(sender_ep(), 0, NULL);
        EXPECT_UCS_OK(status);
        EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_FLUSH, 1Ul);
        EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_FLUSH_WAIT, 0UL);
    }

    status = uct_iface_flush(sender().iface(), 0, NULL);
    EXPECT_UCS_OK(status);
    EXPECT_STAT(sender, uct_iface, UCT_IFACE_STAT_FLUSH, 1UL);
    EXPECT_STAT(sender, uct_iface, UCT_IFACE_STAT_FLUSH_WAIT, 0UL);
}

UCS_TEST_P(test_uct_stats, fence)
{
    ucs_status_t status;

    if (sender_ep()) {
        status = uct_ep_fence(sender_ep(), 0);
        EXPECT_UCS_OK(status);
        EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_FENCE, 1UL);
    }

    status = uct_iface_fence(sender().iface(), 0);
    EXPECT_UCS_OK(status);
    EXPECT_STAT(sender, uct_iface, UCT_IFACE_STAT_FENCE, 1UL);
}

/* flush test only check stats on tls with am_bcopy
 * TODO: full test matrix
 */
UCS_TEST_SKIP_COND_P(test_uct_stats, flush_wait_iface,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY))
{
    uint64_t count_wait;
    ucs_status_t status;

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

    EXPECT_STAT(sender, uct_iface, UCT_IFACE_STAT_FLUSH, 1UL);
    EXPECT_STAT(sender, uct_iface, UCT_IFACE_STAT_FLUSH_WAIT, count_wait);
}

UCS_TEST_SKIP_COND_P(test_uct_stats, flush_wait_ep,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY))
{
    uint64_t count_wait;
    ucs_status_t status;

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

    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_FLUSH, 1UL);
    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_FLUSH_WAIT, count_wait);
}

/* fence test only check stats on tls with am_bcopy
 * TODO: full test matrix
 */
UCS_TEST_SKIP_COND_P(test_uct_stats, fence_iface,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY))
{
    ucs_status_t status;

    init_bufs(0, sender().iface_attr().cap.am.max_bcopy);

    status = uct_iface_set_am_handler(receiver().iface(), 0, am_handler, 0, UCT_CB_FLAG_ASYNC);
    EXPECT_UCS_OK(status);

    fill_tx_q(0);

    status = uct_iface_fence(sender().iface(), 0);
    EXPECT_UCS_OK(status);

    fill_tx_q(0);

    EXPECT_STAT(sender, uct_iface, UCT_IFACE_STAT_FENCE, 1UL);
}

UCS_TEST_SKIP_COND_P(test_uct_stats, fence_ep,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY))
{
    ucs_status_t status;

    init_bufs(0, sender().iface_attr().cap.am.max_bcopy);

    status = uct_iface_set_am_handler(receiver().iface(), 0, am_handler, 0, UCT_CB_FLAG_ASYNC);
    EXPECT_UCS_OK(status);

    fill_tx_q(0);

    status = uct_ep_fence(sender_ep(), 0);
    EXPECT_UCS_OK(status);

    fill_tx_q(0);

    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_FENCE, 1UL);
}

UCS_TEST_SKIP_COND_P(test_uct_stats, tx_no_res,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY))
{
    uint64_t count;
    ucs_status_t status;

    init_bufs(0, sender().iface_attr().cap.am.max_bcopy);

    status = uct_iface_set_am_handler(receiver().iface(), 0, am_handler, 0, UCT_CB_FLAG_ASYNC);
    EXPECT_UCS_OK(status);
    count = fill_tx_q(1024);

    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_NO_RES, count);
    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_AM, 1024 - count);
}

UCS_TEST_SKIP_COND_P(test_uct_stats, pending_add,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY |
                                 UCT_IFACE_FLAG_PENDING))
{
    const size_t num_reqs = 5;
    uct_pending_req_t p_reqs[num_reqs];
    ssize_t len;

    init_bufs(0, sender().iface_attr().cap.am.max_bcopy);

    EXPECT_UCS_OK(uct_iface_set_am_handler(receiver().iface(), 0, am_handler, 0,
                                           UCT_CB_FLAG_ASYNC));

    // Check that counter is not increased if pending_add returns NOT_OK
    EXPECT_EQ(UCS_ERR_BUSY, uct_ep_pending_add(sender().ep(0), &p_reqs[0], 0));
    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_PENDING, 0UL);

    // Check that counter gets increased on every successfull pending_add returns NOT_OK
    fill_tx_q(0);

    UCT_TEST_CALL_AND_TRY_AGAIN(
        uct_ep_am_bcopy(sender_ep(), 0, mapped_buffer::pack,
                        lbuf, 0), len);
    if (len == (ssize_t)lbuf->length()) {
        UCS_TEST_SKIP_R("Can't add to pending");
    }

    for (size_t i = 0; i < num_reqs; ++i) {
        p_reqs[i].func = NULL;
        EXPECT_UCS_OK(uct_ep_pending_add(sender().ep(0), &p_reqs[i], 0));
    }
    uct_ep_pending_purge(sender().ep(0), purge_cb, NULL);

    EXPECT_STAT(sender, uct_ep, UCT_EP_STAT_PENDING, num_reqs);
}

UCT_INSTANTIATE_TEST_CASE(test_uct_stats);
#endif
