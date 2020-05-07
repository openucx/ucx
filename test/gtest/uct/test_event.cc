/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.All rights reserved.
* See file LICENSE for terms.
*/

extern "C" {
#include <poll.h>
#include <uct/api/uct.h>
#include <ucs/time/time.h>
}
#include <common/test.h>
#include "uct_test.h"
#include "uct_p2p_test.h"

class test_uct_event : public uct_test {
public:
    void init() {
        uct_test::init();

        m_e1 = uct_test::create_entity(0);
        m_entities.push_back(m_e1);

        check_skip_test();

        m_e2 = uct_test::create_entity(0);
        m_entities.push_back(m_e2);

        m_e1->connect(0, *m_e2, 0);
        m_e2->connect(0, *m_e1, 0);

        m_am_count = 0;
    }

    typedef struct {
        unsigned length;
        /* data follows */
    } recv_desc_t;

    static ucs_status_t am_handler(void *arg, void *data, size_t length,
                                   unsigned flags) {
        recv_desc_t *my_desc  = (recv_desc_t *) arg;
        uint64_t *test_ib_hdr = (uint64_t *) data;
        uint64_t *actual_data = (uint64_t *) test_ib_hdr + 1;
        unsigned data_length  = length - sizeof(test_ib_hdr);

        my_desc->length = data_length;
        if (*test_ib_hdr == 0xbeef) {
            memcpy(my_desc + 1, actual_data , data_length);
        }

        ++m_am_count;
        return UCS_OK;
    }

    void cleanup() {
        uct_test::cleanup();
    }

    void test_recv_am(unsigned arm_flags, unsigned send_flags);

    static size_t pack_u64(void *dest, void *arg)
    {
        *reinterpret_cast<uint64_t*>(dest) = *reinterpret_cast<uint64_t*>(arg);
        return sizeof(uint64_t);
    }

    void arm(entity *e, unsigned arm_flags) {
        ucs_status_t status;
        for (int i = 0; i < 10; ++i) {
            /* have several retries for arming, in case a transport has spurious
             * events */
            status = uct_iface_event_arm(e->iface(), arm_flags);
            if (status == UCS_OK) {
                break;
            }
        }
        ASSERT_EQ(UCS_OK, status);
    }

protected:
    entity *m_e1, *m_e2;
    static int m_am_count;
};

int test_uct_event::m_am_count = 0;

void test_uct_event::test_recv_am(unsigned arm_flags, unsigned send_flags)
{
    uint64_t send_data = 0xdeadbeef;
    int am_send_count = 0;
    ssize_t res;
    recv_desc_t *recv_buffer;
    struct pollfd wakeup_fd;
    ucs_status_t status;

    recv_buffer = (recv_desc_t *)malloc(sizeof(*recv_buffer) +
                                        sizeof(send_data));
    recv_buffer->length = 0; /* Initialize length to 0 */

    /* give a chance to finish connection for some transports (ib/ud, tcp) */
    flush();

    /* set a callback for the uct to invoke for receiving the data */
    uct_iface_set_am_handler(m_e2->iface(), 0, am_handler, recv_buffer, 0);

    /* create receiver wakeup */
    status = uct_iface_event_fd_get(m_e2->iface(), &wakeup_fd.fd);
    ASSERT_EQ(UCS_OK, status);

    wakeup_fd.events = POLLIN;
    EXPECT_EQ(0, poll(&wakeup_fd, 1, 0));

    arm(m_e2, arm_flags);

    EXPECT_EQ(0, poll(&wakeup_fd, 1, 0));

    /* send the data */
    res = uct_ep_am_bcopy(m_e1->ep(0), 0, pack_u64, &send_data, send_flags);
    ASSERT_EQ((ssize_t)sizeof(send_data), res);
    ++am_send_count;

    /* make sure the file descriptor IS signaled ONCE */
    ASSERT_EQ(1, poll(&wakeup_fd, 1, 1000*ucs::test_time_multiplier()));

    for (;;) {
        if ((progress() == 0) && (m_am_count == am_send_count)) {
            status = uct_iface_event_arm(m_e2->iface(), arm_flags);
            if (status != UCS_ERR_BUSY) {
                break;
            }
        }
    }
    ASSERT_EQ(UCS_OK, status);

    arm(m_e2, arm_flags);

    /* send the data again */
    res = uct_ep_am_bcopy(m_e1->ep(0), 0, pack_u64, &send_data, send_flags);
    ASSERT_EQ((ssize_t)sizeof(send_data), res);
    ++am_send_count;

    /* make sure the file descriptor IS signaled */
    ASSERT_EQ(1, poll(&wakeup_fd, 1, 1000*ucs::test_time_multiplier()));

    while (m_am_count < am_send_count) {
        progress();
    }

    m_e1->flush();

    free(recv_buffer);
}

UCS_TEST_SKIP_COND_P(test_uct_event, am,
                     !check_caps(UCT_IFACE_FLAG_CB_SYNC |
                                 UCT_IFACE_FLAG_AM_BCOPY) ||
                     !check_event_caps(UCT_IFACE_FLAG_EVENT_RECV))
{
    test_recv_am(UCT_EVENT_RECV, 0);
}

UCS_TEST_SKIP_COND_P(test_uct_event, sig_am,
                     !check_caps(UCT_IFACE_FLAG_CB_SYNC |
                                 UCT_IFACE_FLAG_AM_BCOPY) ||
                     !check_event_caps(UCT_IFACE_FLAG_EVENT_RECV_SIG))
{
    test_recv_am(UCT_EVENT_RECV_SIG, UCT_SEND_FLAG_SIGNALED);
}

UCT_INSTANTIATE_NO_SELF_TEST_CASE(test_uct_event);


class uct_p2p_test_event : public uct_p2p_test {
public:
    uct_p2p_test_event(): uct_p2p_test(0) {}

    static ucs_log_level_t orig_log_level;
    static unsigned flushed_qp_num;

    ucs_status_t send(uct_ep_h ep, const mapped_buffer &sendbuf,
                      const mapped_buffer &recvbuf) {
        return uct_ep_put_short(ep, sendbuf.ptr(), sendbuf.length(),
                                recvbuf.addr(), recvbuf.rkey());
    }

    static ucs_log_func_rc_t
    last_wqe_check_log(const char *file, unsigned line, const char *function,
                       ucs_log_level_t level,
                       const ucs_log_component_config_t *comp_conf,
                       const char *message, va_list ap)
    {
        std::string msg = format_message(message, ap);

        sscanf(msg.c_str(), "IB Async event on %*s SRQ-attached QP 0x%x was flushed", &flushed_qp_num);

        return (level <= orig_log_level) ? UCS_LOG_FUNC_RC_CONTINUE
            : UCS_LOG_FUNC_RC_STOP;
    }

    uint32_t check_flush_qp(entity &e) {
        unsigned char *addr = (unsigned char *)alloca(e.iface_attr().ep_addr_len);

        uct_ep_get_address(e.ep(0), (uct_ep_addr_t *)addr);
        uint32_t qp_num = addr[0] |
            ((uint32_t)addr[1] << 8) |
            ((uint32_t)addr[2] << 16);

        e.destroy_eps();
        uint64_t timeout = 100000000;
        while (flushed_qp_num != qp_num && timeout > 0) {
            timeout--;
            ucs_memory_bus_load_fence();
        }

        return timeout > 0;
    }
};

UCS_TEST_P(uct_p2p_test_event, last_wqe, "ASYNC_EVENTS=y")
{
    const p2p_resource *r = dynamic_cast<const p2p_resource*>(GetParam());
    ucs_assert_always(r != NULL);

    mapped_buffer sendbuf(0, 0, sender());
    mapped_buffer recvbuf(0, 0, receiver());

    ucs_log_push_handler(last_wqe_check_log);
    orig_log_level = ucs_global_opts.log_component.log_level;
    ucs_global_opts.log_component.log_level = UCS_LOG_LEVEL_DEBUG;
    if (!ucs_log_is_enabled(UCS_LOG_LEVEL_DEBUG)) {
        UCS_TEST_SKIP;
    }

    UCS_TEST_SCOPE_EXIT() {
        ucs_global_opts.log_component.log_level = orig_log_level;
        ucs_log_pop_handler();
    } UCS_TEST_SCOPE_EXIT_END

    blocking_send(static_cast<send_func_t>(&uct_p2p_test_event::send),
                  sender_ep(), sendbuf, recvbuf, true);

    if (r->loopback) {
        ASSERT_TRUE(check_flush_qp(sender()));
    } else {
        ASSERT_TRUE(check_flush_qp(sender()));
        ASSERT_TRUE(check_flush_qp(receiver()));
    }
}

ucs_log_level_t uct_p2p_test_event::orig_log_level;
unsigned uct_p2p_test_event::flushed_qp_num;

_UCT_INSTANTIATE_TEST_CASE(uct_p2p_test_event, rc_mlx5);
