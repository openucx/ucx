/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) The University of Tennessee and The University
*               of Tennessee Research Foundation. 2016. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "libperf_int.h"

extern "C" {
#include <ucs/debug/log.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/sys/math.h>
#include <ucs/sys/sys.h>
}


template <ucx_perf_cmd_t CMD, ucx_perf_test_type_t TYPE, uct_perf_data_layout_t DATA, bool ONESIDED>
class uct_perf_test_runner {
public:

    typedef uint8_t psn_t;

    uct_perf_test_runner(ucx_perf_context_t &perf) :
        m_perf(perf),
        m_max_outstanding(m_perf.params.max_outstanding),
        m_send_b_count(0)

    {
        ucs_assert_always(m_max_outstanding > 0);

        m_completion.count = 1;
        m_completion.func  = NULL;

        ucs_status_t status;
        uct_iface_attr_t attr;
        status = uct_iface_query(m_perf.uct.iface, &attr);
        ucs_assert_always(status == UCS_OK);
        if (attr.cap.flags & (UCT_IFACE_FLAG_AM_SHORT|UCT_IFACE_FLAG_AM_BCOPY|UCT_IFACE_FLAG_AM_ZCOPY)) {
            status = uct_iface_set_am_handler(m_perf.uct.iface, UCT_PERF_TEST_AM_ID,
                                              am_hander, m_perf.recv_buffer, UCT_CB_FLAG_SYNC);
            ucs_assert_always(status == UCS_OK);
        }
    }

    ~uct_perf_test_runner() {
        uct_iface_set_am_handler(m_perf.uct.iface, UCT_PERF_TEST_AM_ID, NULL, NULL, UCT_CB_FLAG_SYNC);
    }

    /**
     * Make uct_iov_t iov[msg_size_cnt] array with pointer elements to
     * original buffer
     */
    static void uct_perf_get_buffer_iov(uct_iov_t *iov, void *buffer,
                                        unsigned header_size, uct_mem_h memh,
                                        const ucx_perf_context_t *perf)
    {
        const size_t iovcnt    = perf->params.msg_size_cnt;
        size_t iov_length_it, iov_it;

        ucs_assert(UCT_PERF_DATA_LAYOUT_ZCOPY == DATA);
        ucs_assert(NULL != perf->params.msg_size_list);
        ucs_assert(iovcnt > 0);
        ucs_assert(perf->params.msg_size_list[0] >= header_size);

        iov_length_it = 0;
        for (iov_it = 0; iov_it < iovcnt; ++iov_it) {
            iov[iov_it].buffer = (char *)buffer + iov_length_it + header_size;
            iov[iov_it].length = perf->params.msg_size_list[iov_it] - header_size;
            iov[iov_it].memh   = memh;
            iov[iov_it].stride = 0;
            iov[iov_it].count  = 1;

            if (perf->params.iov_stride) {
                iov_length_it += perf->params.iov_stride - header_size;
            } else {
                iov_length_it += iov[iov_it].length;
            }

            header_size        = 0; /* should be zero for next iterations */
        }

        ucs_debug("IOV buffer filled by %lu slices with total length %lu",
                  iovcnt, iov_length_it);
    }

    void uct_perf_test_prepare_iov_buffer() {
        if (UCT_PERF_DATA_LAYOUT_ZCOPY == DATA) {
            size_t start_iov_buffer_size = 0;
            if (UCX_PERF_CMD_AM == CMD) {
                start_iov_buffer_size = m_perf.params.am_hdr_size;
            }
            uct_perf_get_buffer_iov(m_perf.uct.iov, m_perf.send_buffer,
                                    start_iov_buffer_size, m_perf.uct.send_mem.memh,
                                    &m_perf);
        }
    }

    /**
     * Get the length between beginning of the IOV first buffer and the latest byte
     * in the latest IOV buffer.
     */
    size_t uct_perf_get_buffer_extent(const ucx_perf_params_t *params)
    {
        size_t length;

        if ((UCT_PERF_DATA_LAYOUT_ZCOPY == DATA) && params->iov_stride) {
            length = ((params->msg_size_cnt - 1) * params->iov_stride) +
                     params->msg_size_list[params->msg_size_cnt - 1];
        } else {
            length = ucx_perf_get_message_size(params);
        }

        return length;
    }

    void UCS_F_ALWAYS_INLINE progress_responder() {
        if (!ONESIDED) {
            uct_worker_progress(m_perf.uct.worker);
        }
    }

    void UCS_F_ALWAYS_INLINE progress_requestor() {
        uct_worker_progress(m_perf.uct.worker);
    }

    void UCS_F_ALWAYS_INLINE wait_for_window(bool send_window)
    {
        while (send_window && (outstanding() >= m_max_outstanding)) {
            progress_requestor();
        }
    }

    static ucs_status_t am_hander(void *arg, void *data, size_t length,
                                  unsigned flags)
    {
        ucs_assert(UCS_CIRCULAR_COMPARE8(*(psn_t*)arg, <=, *(psn_t*)data));
        *(psn_t*)arg = *(psn_t*)data;
        return UCS_OK;
    }

    static size_t pack_cb(void *dest, void *arg)
    {
        uct_perf_test_runner *self = (uct_perf_test_runner *)arg;
        size_t length = ucx_perf_get_message_size(&self->m_perf.params);

        memcpy(dest, self->m_perf.send_buffer, length);
        return length;
    }

    ucs_status_t UCS_F_ALWAYS_INLINE
    send(uct_ep_h ep, psn_t sn, psn_t prev_sn, void *buffer, unsigned length,
         uint64_t remote_addr, uct_rkey_t rkey, uct_completion_t *comp)
    {
        uint64_t am_short_hdr;
        size_t header_size;
        ssize_t packed_len;

        /* coverity[switch_selector_expr_is_constant] */
        switch (CMD) {
        case UCX_PERF_CMD_AM:
            /* coverity[switch_selector_expr_is_constant] */
            switch (DATA) {
            case UCT_PERF_DATA_LAYOUT_SHORT:
                am_short_hdr = sn;
                return uct_ep_am_short(ep, UCT_PERF_TEST_AM_ID, am_short_hdr,
                                       (char*)buffer + sizeof(am_short_hdr),
                                       length - sizeof(am_short_hdr));
            case UCT_PERF_DATA_LAYOUT_BCOPY:
                *(psn_t*)buffer = sn;
                packed_len = uct_ep_am_bcopy(ep, UCT_PERF_TEST_AM_ID, pack_cb,
                                             (void*)this, 0);
                return (packed_len >= 0) ? UCS_OK : (ucs_status_t)packed_len;
            case UCT_PERF_DATA_LAYOUT_ZCOPY:
                *(psn_t*)buffer = sn;
                header_size = m_perf.params.am_hdr_size;
                return uct_ep_am_zcopy(ep, UCT_PERF_TEST_AM_ID, buffer, header_size,
                                       m_perf.uct.iov, m_perf.params.msg_size_cnt,
                                       0, comp);
            default:
                return UCS_ERR_INVALID_PARAM;
            }
        case UCX_PERF_CMD_PUT:
            if (TYPE == UCX_PERF_TEST_TYPE_PINGPONG) {
                /* Put the control word at the latest byte of the IOV message */
                *((psn_t*)buffer + uct_perf_get_buffer_extent(&m_perf.params) - 1) = sn;
            }
            /* coverity[switch_selector_expr_is_constant] */
            switch (DATA) {
            case UCT_PERF_DATA_LAYOUT_SHORT:
                return uct_ep_put_short(ep, buffer, length, remote_addr, rkey);
            case UCT_PERF_DATA_LAYOUT_BCOPY:
                packed_len = uct_ep_put_bcopy(ep, pack_cb, (void*)this, remote_addr, rkey);
                return (packed_len >= 0) ? UCS_OK : (ucs_status_t)packed_len;
            case UCT_PERF_DATA_LAYOUT_ZCOPY:
                return uct_ep_put_zcopy(ep, m_perf.uct.iov, m_perf.params.msg_size_cnt,
                                        remote_addr, rkey, comp);
            default:
                return UCS_ERR_INVALID_PARAM;
            }
        case UCX_PERF_CMD_GET:
            /* coverity[switch_selector_expr_is_constant] */
            switch (DATA) {
            case UCT_PERF_DATA_LAYOUT_BCOPY:
                return uct_ep_get_bcopy(ep, (uct_unpack_callback_t)memcpy,
                                        buffer, length, remote_addr, rkey, comp);
            case UCT_PERF_DATA_LAYOUT_ZCOPY:
                return uct_ep_get_zcopy(ep, m_perf.uct.iov, m_perf.params.msg_size_cnt,
                                        remote_addr, rkey, comp);
            default:
                return UCS_ERR_INVALID_PARAM;
            }
        case UCX_PERF_CMD_ADD:
            if (length == sizeof(uint32_t)) {
                return uct_ep_atomic32_post(ep, UCT_ATOMIC_OP_ADD, sn - prev_sn, remote_addr, rkey);
            } else if (length == sizeof(uint64_t)) {
                return uct_ep_atomic64_post(ep, UCT_ATOMIC_OP_ADD, sn - prev_sn, remote_addr, rkey);
            } else {
                return UCS_ERR_INVALID_PARAM;
            }
        case UCX_PERF_CMD_FADD:
            if (length == sizeof(uint32_t)) {
                return uct_ep_atomic32_fetch(ep, UCT_ATOMIC_OP_ADD, sn - prev_sn,
                                             (uint32_t*)buffer, remote_addr, rkey, comp);
            } else if (length == sizeof(uint64_t)) {
                return uct_ep_atomic64_fetch(ep, UCT_ATOMIC_OP_ADD, sn - prev_sn,
                                             (uint64_t*)buffer, remote_addr, rkey, comp);
            } else {
                return UCS_ERR_INVALID_PARAM;
            }
        case UCX_PERF_CMD_SWAP:
            if (length == sizeof(uint32_t)) {
                return uct_ep_atomic32_fetch(ep, UCT_ATOMIC_OP_SWAP, sn,
                                             (uint32_t*)buffer, remote_addr, rkey, comp);
            } else if (length == sizeof(uint64_t)) {
                return uct_ep_atomic64_fetch(ep, UCT_ATOMIC_OP_SWAP, sn,
                                             (uint64_t*)buffer, remote_addr, rkey, comp);
            } else {
                return UCS_ERR_INVALID_PARAM;
            }
        case UCX_PERF_CMD_CSWAP:
            if (length == sizeof(uint32_t)) {
                return uct_ep_atomic_cswap32(ep, prev_sn, sn, remote_addr, rkey,
                                             (uint32_t*)buffer, comp);
            } else if (length == sizeof(uint64_t)) {
                return uct_ep_atomic_cswap64(ep, prev_sn, sn, remote_addr, rkey,
                                             (uint64_t*)buffer, comp);
            } else {
                return UCS_ERR_INVALID_PARAM;
            }
        default:
            return UCS_ERR_INVALID_PARAM;
        }
    }

    void UCS_F_ALWAYS_INLINE
    send_b(uct_ep_h ep, psn_t sn, psn_t prev_sn, void *buffer, unsigned length,
           uint64_t remote_addr, uct_rkey_t rkey, uct_completion_t *comp)
    {
        ucs_status_t status;
        for (;;) {
            status = send(ep, sn, prev_sn, buffer, length, remote_addr, rkey, comp);
            if (ucs_likely(status == UCS_OK)) {
                if ((m_send_b_count++ % N_SEND_B_PER_PROGRESS) == 0) {
                    progress_requestor();
                }
                return;
            } else if (status == UCS_INPROGRESS) {
                ++m_completion.count;
                progress_requestor();
                ucs_assert((comp == NULL) || (outstanding() <= m_max_outstanding));
                return;
            } else if (status == UCS_ERR_NO_RESOURCE) {
                progress_requestor();
                continue;
            } else {
                ucs_error("Failed to send: %s", ucs_status_string(status));
                return;
            }
        };
    }

    ucs_status_t run_pingpong()
    {
        psn_t send_sn, *recv_sn;
        unsigned my_index;
        uct_ep_h ep;
        uint64_t remote_addr;
        uct_rkey_t rkey;
        void *buffer;
        size_t length;

        length = ucx_perf_get_message_size(&m_perf.params);
        ucs_assert(length >= sizeof(psn_t));

        /* coverity[switch_selector_expr_is_constant] */
        switch (CMD) {
        case UCX_PERF_CMD_AM:
        case UCX_PERF_CMD_ADD:
            recv_sn = (psn_t*)m_perf.recv_buffer;
            break;
        case UCX_PERF_CMD_PUT:
            /* since polling on data, must be end of the buffer */
            recv_sn = (psn_t*)m_perf.recv_buffer + length - 1;
            break;
        default:
            ucs_error("Cannot run this test in ping-pong mode");
            return UCS_ERR_INVALID_PARAM;
        }

        uct_perf_test_prepare_iov_buffer();

        *recv_sn  = -1;
        uct_perf_barrier(&m_perf);

        my_index = rte_call(&m_perf, group_index);

        ucx_perf_test_start_clock(&m_perf);

        buffer      = m_perf.send_buffer;
        remote_addr = m_perf.uct.peers[1 - my_index].remote_addr + m_perf.offset;
        rkey        = m_perf.uct.peers[1 - my_index].rkey.rkey;
        ep          = m_perf.uct.peers[1 - my_index].ep;

        send_sn = 0;
        if (my_index == 0) {
            UCX_PERF_TEST_FOREACH(&m_perf) {
                send_b(ep, send_sn, send_sn - 1, buffer, length, remote_addr,
                       rkey, NULL);
                ucx_perf_update(&m_perf, 1, length);
                while (*recv_sn != send_sn) {
                    progress_responder();
                }
                ++send_sn;
            }
        } else if (my_index == 1) {
            UCX_PERF_TEST_FOREACH(&m_perf) {
                while (*recv_sn != send_sn) {
                    progress_responder();
                }
                send_b(ep, send_sn, send_sn - 1, buffer, length, remote_addr,
                       rkey, NULL);
                ucx_perf_update(&m_perf, 1, length);
                ++send_sn;
            }
        }

        uct_perf_iface_flush_b(&m_perf);
        return UCS_OK;
    }

    ucs_status_t run_stream_req_uni(bool flow_control, bool send_window,
                                    bool direction_to_responder)
    {
        unsigned long remote_addr;
        volatile psn_t *recv_sn;
        psn_t sn, send_sn;
        uct_rkey_t rkey;
        void *buffer;
        unsigned fc_window;
        unsigned my_index;
        unsigned length;
        uct_ep_h ep;

        length = ucx_perf_get_message_size(&m_perf.params);
        ucs_assert(length >= sizeof(psn_t));
        ucs_assert(m_perf.params.uct.fc_window <= ((psn_t)-1) / 2);

        memset(m_perf.send_buffer, 0, length);
        memset(m_perf.recv_buffer, 0, length);

        uct_perf_test_prepare_iov_buffer();

        recv_sn  = direction_to_responder ? (psn_t*)m_perf.recv_buffer :
                                            (psn_t*)m_perf.send_buffer;
        my_index = rte_call(&m_perf, group_index);

        uct_perf_barrier(&m_perf);

        ucx_perf_test_start_clock(&m_perf);

        ep          = m_perf.uct.peers[1 - my_index].ep;
        buffer      = m_perf.send_buffer;
        remote_addr = m_perf.uct.peers[1 - my_index].remote_addr + m_perf.offset;
        rkey        = m_perf.uct.peers[1 - my_index].rkey.rkey;
        fc_window   = m_perf.params.uct.fc_window;

        if (my_index == 1) {
            /* send_sn is the next SN to send */
            if (flow_control) {
                send_sn     = 1;
            } else{
                send_sn     = 0; /* Remote buffer will remain 0 throughout the test */
            }
            *(psn_t*)buffer = send_sn;

            UCX_PERF_TEST_FOREACH(&m_perf) {
                if (flow_control) {
                    /* Wait until getting ACK from responder */
                    ucs_assertv(UCS_CIRCULAR_COMPARE8(send_sn - 1, >=, *recv_sn),
                                "recv_sn=%d iters=%ld", *recv_sn, m_perf.current.iters);
                    while (UCS_CIRCULAR_COMPARE8(send_sn, >, (psn_t)(*recv_sn + fc_window))) {
                        progress_responder();
                    }
                }

                /* Wait until we have enough sends completed, then take
                 * the next completion handle in the window. */
                wait_for_window(send_window);

                if (flow_control) {
                    send_b(ep, send_sn, send_sn - 1, buffer, length, remote_addr,
                           rkey, &m_completion);
                    ++send_sn;
                } else {
                    send_b(ep, send_sn, send_sn, buffer, length, remote_addr,
                           rkey, &m_completion);
                }

                ucx_perf_update(&m_perf, 1, length);
            }

            if (!flow_control) {
                /* Send "sentinel" value */
                if (direction_to_responder) {
                    wait_for_window(send_window);
                    *(psn_t*)buffer = 2;
                    send_b(ep, 2, send_sn, buffer, length, remote_addr, rkey,
                           &m_completion);
                } else {
                    *(psn_t*)m_perf.recv_buffer = 2;
                }
            } else {
                /* Wait for last ACK, to make sure no more messages will arrive. */
                ucs_assert(direction_to_responder);
                while (UCS_CIRCULAR_COMPARE8((psn_t)(send_sn - 1), >, *recv_sn)) {
                    progress_responder();
                }
            }
        } else if (my_index == 0) {
            if (flow_control) {
                /* Since we're doing flow control, we can count exactly how
                 * many packets were received.
                 */
                send_sn = (psn_t)-1; /* Last SN we have sent (as acknowledgment) */
                ucs_assert(direction_to_responder);
                UCX_PERF_TEST_FOREACH(&m_perf) {
                    sn = *recv_sn;
                    progress_responder();
                    if (UCS_CIRCULAR_COMPARE8(sn, >, (psn_t)(send_sn + (fc_window / 2)))) {
                        /* Send ACK every half-window */
                        wait_for_window(send_window);
                        send_b(ep, sn, send_sn, buffer, length, remote_addr,
                               rkey, &m_completion);
                        send_sn = sn;
                    }

                    /* Calculate number of iterations */
                    m_perf.current.iters +=
                                    (psn_t)(sn - (psn_t)m_perf.current.iters);
                }

                /* Send ACK for last packet */
                if (UCS_CIRCULAR_COMPARE8(*recv_sn, >, send_sn)) {
                    wait_for_window(send_window);
                    send_b(ep, *recv_sn, send_sn, buffer, length, remote_addr,
                           rkey, &m_completion);
                }
            } else {
                /* Wait for "sentinel" value */
                ucs_time_t poll_time = ucs_get_time();
                while (*recv_sn != 2) {
                    progress_responder();
                    if (!direction_to_responder) {
                        if (ucs_get_time() > poll_time + ucs_time_from_msec(1.0)) {
                            wait_for_window(send_window);
                            send_b(ep, 0, 0, buffer, length, remote_addr, rkey,
                                   &m_completion);
                            poll_time = ucs_get_time();
                        }
                    }
                }
            }
        }

        uct_perf_iface_flush_b(&m_perf);
        ucs_assert(outstanding() == 0);
        if (my_index == 1) {
            ucx_perf_update(&m_perf, 0, 0);
        }

        return UCS_OK;
    }

    ucs_status_t run()
    {
        bool zcopy = (DATA == UCT_PERF_DATA_LAYOUT_ZCOPY);

        /* coverity[switch_selector_expr_is_constant] */
        switch (TYPE) {
        case UCX_PERF_TEST_TYPE_PINGPONG:
            return run_pingpong();
        case UCX_PERF_TEST_TYPE_STREAM_UNI:
            /* coverity[switch_selector_expr_is_constant] */
            switch (CMD) {
            case UCX_PERF_CMD_PUT:
                return run_stream_req_uni(false, /* No need for flow control for RMA */
                                          zcopy, /* ZCOPY can return INPROGRESS */
                                          true /* data goes to responder */);
            case UCX_PERF_CMD_ADD:
                return run_stream_req_uni(false, /* No need for flow control for RMA */
                                          false, /* This atomic does not wait for reply */
                                          true /* Data goes to responder */);
            case UCX_PERF_CMD_AM:
                return run_stream_req_uni(true, /* Need flow control for active messages,
                                                   because they are handled in SW */
                                          zcopy, /* ZCOPY can return INPROGRESS */
                                          true /* data goes to responder */);
            case UCX_PERF_CMD_GET:
                return run_stream_req_uni(false, /* No flow control for RMA/AMO */
                                          true, /* Waiting for replies */
                                          false /* For GET, data is delivered to requester */ );
            case UCX_PERF_CMD_FADD:
            case UCX_PERF_CMD_SWAP:
            case UCX_PERF_CMD_CSWAP:
                return run_stream_req_uni(false, /* No flow control for RMA/AMO */
                                          true, /* Waiting for replies */
                                          true /* For atomics, data goes both ways, but
                                                     the request is easier to predict */ );
            default:
                return UCS_ERR_INVALID_PARAM;
            }
        case UCX_PERF_TEST_TYPE_STREAM_BI:
        default:
            return UCS_ERR_INVALID_PARAM;
        }
    }

private:
    inline unsigned outstanding() {
        return m_completion.count - 1;
    }

    ucx_perf_context_t &m_perf;
    const unsigned     m_max_outstanding;
    uct_completion_t   m_completion;
    int                m_send_b_count;
    const static int   N_SEND_B_PER_PROGRESS = 16;
};


#define TEST_CASE(_perf, _cmd, _type, _data, _onesided) \
    if (((_perf)->params.command == (_cmd)) && \
        ((_perf)->params.test_type == (_type)) && \
        ((_perf)->params.uct.data_layout == (_data)) && \
        (!!((_perf)->params.flags & UCX_PERF_TEST_FLAG_ONE_SIDED) == !!(_onesided))) \
    { \
        uct_perf_test_runner<_cmd, _type, _data, _onesided> r(*_perf); \
        return r.run(); \
    }
#define TEST_CASE_ALL_OSD(_perf, _case, _data) \
   TEST_CASE(_perf, UCS_PP_TUPLE_0 _case, UCS_PP_TUPLE_1 _case, _data, true) \
   TEST_CASE(_perf, UCS_PP_TUPLE_0 _case, UCS_PP_TUPLE_1 _case, _data, false)
#define TEST_CASE_ALL_DATA(_perf, _case) \
   TEST_CASE_ALL_OSD(_perf, _case, UCT_PERF_DATA_LAYOUT_SHORT) \
   TEST_CASE_ALL_OSD(_perf, _case, UCT_PERF_DATA_LAYOUT_BCOPY) \
   TEST_CASE_ALL_OSD(_perf, _case, UCT_PERF_DATA_LAYOUT_ZCOPY)

ucs_status_t uct_perf_test_dispatch(ucx_perf_context_t *perf)
{
    UCS_PP_FOREACH(TEST_CASE_ALL_DATA, perf,
        (UCX_PERF_CMD_AM,  UCX_PERF_TEST_TYPE_PINGPONG),
        (UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_PINGPONG),
        (UCX_PERF_CMD_ADD, UCX_PERF_TEST_TYPE_PINGPONG),
        (UCX_PERF_CMD_AM,  UCX_PERF_TEST_TYPE_STREAM_UNI),
        (UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_STREAM_UNI),
        (UCX_PERF_CMD_GET, UCX_PERF_TEST_TYPE_STREAM_UNI),
        (UCX_PERF_CMD_ADD, UCX_PERF_TEST_TYPE_STREAM_UNI),
        (UCX_PERF_CMD_FADD, UCX_PERF_TEST_TYPE_STREAM_UNI),
        (UCX_PERF_CMD_SWAP, UCX_PERF_TEST_TYPE_STREAM_UNI),
        (UCX_PERF_CMD_CSWAP, UCX_PERF_TEST_TYPE_STREAM_UNI)
        );

    ucs_error("Invalid test case");
    return UCS_ERR_INVALID_PARAM;
}
