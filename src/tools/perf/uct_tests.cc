/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "libperf_int.h"

extern "C" {
#include <ucs/debug/log.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/sys/math.h>
}


template <ucx_perf_cmd_t CMD, ucx_perf_test_type_t TYPE, uct_perf_data_layout_t DATA, bool ONESIDED>
class uct_perf_test_runner {
public:

    typedef uint8_t psn_t;

    uct_perf_test_runner(ucx_perf_context_t &perf) :
        m_perf(perf),
        m_outstanding(0),
        m_max_outstanding(m_perf.params.max_outstanding)

    {
        ucs_assert_always(m_max_outstanding > 0);

        uct_iface_attr_t attr;
        ucs_status_t status = uct_iface_query(m_perf.uct.iface, &attr);
        ucs_assert_always(status == UCS_OK);

        m_completion_size = sizeof (comp_t) + attr.completion_priv_len;
        void *completions_buffer = malloc(m_completion_size * m_max_outstanding);
        ucs_assert_always(completions_buffer != NULL);

        m_completions = (comp_t**)calloc(m_max_outstanding, sizeof(comp_t*));
        ucs_assert_always(m_completions != NULL);
        for (unsigned i = 0; i < m_max_outstanding; ++i) {
            m_completions[i] = (comp_t*)((char*)completions_buffer + i * m_completion_size);
            m_completions[i]->self     = this;
            m_completions[i]->uct.func = completion_func();
        }

        status = uct_iface_set_am_handler(m_perf.uct.iface, UCT_PERF_TEST_AM_ID, am_hander,
                                          m_perf.recv_buffer);
        ucs_assert_always(status == UCS_OK);
    }

    ~uct_perf_test_runner() {
        uct_iface_set_am_handler(m_perf.uct.iface, UCT_PERF_TEST_AM_ID, NULL, NULL);
        free(m_completions[0]);
        free(m_completions);
    }

    void UCS_F_ALWAYS_INLINE progress_responder() {
        if (!ONESIDED) {
            uct_worker_progress(m_perf.uct.worker);
        }
    }

    void UCS_F_ALWAYS_INLINE progress_requestor() {
        uct_worker_progress(m_perf.uct.worker);
    }

    static ucs_status_t am_hander(void *arg, void *data, size_t length, void *desc)
    {
        ucs_assert(UCS_CIRCULAR_COMPARE8(*(psn_t*)arg, <=, *(psn_t*)data));
        *(psn_t*)arg = *(psn_t*)data;
        return UCS_OK;
    }

    static void zcopy_completion_cb(uct_completion_t *comp, void *data)
    {
        uct_perf_test_runner *self = ucs_container_of(comp, comp_t, uct)->self;
        --self->m_outstanding;
    }

    static void fetch_completion_cb(uct_completion_t *comp, void *data)
    {
        uct_perf_test_runner *self = ucs_container_of(comp, comp_t, uct)->self;
        *(psn_t*)self->m_perf.send_buffer = *(psn_t*)data;
        --self->m_outstanding;
    }

    static uct_completion_callback_t completion_func() {
        switch (CMD) {
        case UCX_PERF_CMD_AM:
            switch (DATA) {
            case UCT_PERF_DATA_LAYOUT_ZCOPY:
                return zcopy_completion_cb;
            default:
                return NULL;
            }
        case UCX_PERF_CMD_PUT:
            switch (DATA) {
            case UCT_PERF_DATA_LAYOUT_ZCOPY:
                return zcopy_completion_cb;
            default:
                return NULL;
            }
        case UCX_PERF_CMD_GET:
            switch (DATA) {
            case UCT_PERF_DATA_LAYOUT_BCOPY:
                return fetch_completion_cb;
            case UCT_PERF_DATA_LAYOUT_ZCOPY:
                return zcopy_completion_cb;
            default:
                return NULL;
            }
        case UCX_PERF_CMD_ADD:
            return NULL;
        case UCX_PERF_CMD_FADD:
        case UCX_PERF_CMD_SWAP:
        case UCX_PERF_CMD_CSWAP:
            return fetch_completion_cb;
        default:
            return NULL;
        }
    }

    ucs_status_t UCS_F_ALWAYS_INLINE
    send(uct_ep_h ep, psn_t sn, psn_t prev_sn, void *buffer, unsigned length,
         uint64_t remote_addr, uct_rkey_t rkey, uct_completion_t *comp)
    {
        uint64_t am_short_hdr;
        size_t header_size;

        switch (CMD) {
        case UCX_PERF_CMD_AM:
            switch (DATA) {
            case UCT_PERF_DATA_LAYOUT_SHORT:
                am_short_hdr = sn;
                return uct_ep_am_short(ep, UCT_PERF_TEST_AM_ID, am_short_hdr,
                                       (char*)buffer + sizeof(am_short_hdr),
                                       length - sizeof(am_short_hdr));
            case UCT_PERF_DATA_LAYOUT_BCOPY:
                *(psn_t*)buffer = sn;
                return uct_ep_am_bcopy(ep, UCT_PERF_TEST_AM_ID,
                                       (uct_pack_callback_t)memcpy, buffer, length);
            case UCT_PERF_DATA_LAYOUT_ZCOPY:
                *(psn_t*)buffer = sn;
                header_size = m_perf.params.am_hdr_size;
                return uct_ep_am_zcopy(ep, UCT_PERF_TEST_AM_ID,
                                       buffer, header_size,
                                       (char*)buffer + header_size, length - header_size,
                                       m_perf.uct.send_memh, comp);
            default:
                return UCS_ERR_INVALID_PARAM;
            }
        case UCX_PERF_CMD_PUT:
            if (TYPE == UCX_PERF_TEST_TYPE_PINGPONG) {
                *((psn_t*)buffer + length - 1) = sn;
            }
            switch (DATA) {
            case UCT_PERF_DATA_LAYOUT_SHORT:
                return uct_ep_put_short(ep, buffer, length, remote_addr, rkey);
            case UCT_PERF_DATA_LAYOUT_BCOPY:
                return uct_ep_put_bcopy(ep, (uct_pack_callback_t)memcpy, buffer,
                                        length, remote_addr, rkey);
            case UCT_PERF_DATA_LAYOUT_ZCOPY:
                return uct_ep_put_zcopy(ep, buffer, length, m_perf.uct.send_memh,
                                        remote_addr, rkey, comp);
            default:
                return UCS_ERR_INVALID_PARAM;
            }
        case UCX_PERF_CMD_GET:
            switch (DATA) {
            case UCT_PERF_DATA_LAYOUT_BCOPY:
                return uct_ep_get_bcopy(ep, length, remote_addr, rkey, comp);
            case UCT_PERF_DATA_LAYOUT_ZCOPY:
                return uct_ep_get_zcopy(ep, buffer, length, m_perf.uct.send_memh,
                                        remote_addr, rkey, comp);
            default:
                return UCS_ERR_INVALID_PARAM;
            }
        case UCX_PERF_CMD_ADD:
            if (length == sizeof(uint32_t)) {
                return uct_ep_atomic_add32(ep, sn - prev_sn, remote_addr, rkey);
            } else if (length == sizeof(uint64_t)) {
                return uct_ep_atomic_add64(ep, sn - prev_sn, remote_addr, rkey);
            } else {
                return UCS_ERR_INVALID_PARAM;
            }
        case UCX_PERF_CMD_FADD:
            if (length == sizeof(uint32_t)) {
                return uct_ep_atomic_fadd32(ep, sn - prev_sn, remote_addr, rkey,
                                            comp);
            } else if (length == sizeof(uint64_t)) {
                return uct_ep_atomic_fadd64(ep, sn - prev_sn, remote_addr, rkey,
                                            comp);
            } else {
                return UCS_ERR_INVALID_PARAM;
            }
        case UCX_PERF_CMD_SWAP:
            if (length == sizeof(uint32_t)) {
                return uct_ep_atomic_swap32(ep, sn, remote_addr, rkey, comp);
            } else if (length == sizeof(uint64_t)) {
                return uct_ep_atomic_swap64(ep, sn, remote_addr, rkey, comp);
            } else {
                return UCS_ERR_INVALID_PARAM;
            }
        case UCX_PERF_CMD_CSWAP:
            if (length == sizeof(uint32_t)) {
                return uct_ep_atomic_cswap32(ep, sn, prev_sn, remote_addr, rkey,
                                             comp);
            } else if (length == sizeof(uint64_t)) {
                return uct_ep_atomic_cswap64(ep, sn, prev_sn, remote_addr, rkey,
                                             comp);
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
            progress_requestor();
            if (ucs_likely(status == UCS_OK)) {
                return;
            } else if (status == UCS_INPROGRESS) {
                ++m_outstanding;
                ucs_assert((comp == NULL) || (m_outstanding <= m_max_outstanding));
                return;
            } else if (status == UCS_ERR_NO_RESOURCE) {
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

        ucs_assert(m_perf.params.message_size >= sizeof(psn_t));

        switch (CMD) {
        case UCX_PERF_CMD_AM:
        case UCX_PERF_CMD_ADD:
            recv_sn = (psn_t*)m_perf.recv_buffer;
            break;
        case UCX_PERF_CMD_PUT:
            /* since polling on data, must be end of the buffer */
            recv_sn = (psn_t*)m_perf.recv_buffer +
                            m_perf.params.message_size - 1;
            break;
        default:
            ucs_error("Cannot run this test in ping-pong mode");
            return UCS_ERR_INVALID_PARAM;
        }

        *recv_sn  = -1;
        rte_call(&m_perf, barrier);

        my_index = rte_call(&m_perf, group_index);

        ucx_perf_test_start_clock(&m_perf);

        buffer      = m_perf.send_buffer;
        length      = m_perf.params.message_size;
        remote_addr = m_perf.uct.peers[1 - my_index].remote_addr;
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
        unsigned completion_index, fc_window;
        unsigned my_index;
        unsigned length;
        uct_ep_h ep;

        ucs_assert(m_perf.params.message_size >= sizeof(psn_t));
        ucs_assert(m_perf.params.uct.fc_window <= ((psn_t)-1) / 2);

        recv_sn  = direction_to_responder ? (psn_t*)m_perf.recv_buffer :
                                            (psn_t*)m_perf.send_buffer;
        *recv_sn = 0;
        send_sn  = 1;
        my_index = rte_call(&m_perf, group_index);

        rte_call(&m_perf, barrier);

        ucx_perf_test_start_clock(&m_perf);

        ep          = m_perf.uct.peers[1 - my_index].ep;
        buffer      = m_perf.send_buffer;
        length      = m_perf.params.message_size;
        remote_addr = m_perf.uct.peers[1 - my_index].remote_addr;
        rkey        = m_perf.uct.peers[1 - my_index].rkey.rkey;
        fc_window   = m_perf.params.uct.fc_window;

        completion_index  = 0;
        *(psn_t*)buffer = send_sn;

        if (my_index == 0) {
            UCX_PERF_TEST_FOREACH(&m_perf) {
                if (flow_control) {
                    /* Wait until getting ACK from responder */
                    ucs_assertv(UCS_CIRCULAR_COMPARE8(send_sn - 1, >=, *recv_sn),
                                "recv_sn=%d iters=%ld", *recv_sn, m_perf.current.iters);
                    while (UCS_CIRCULAR_COMPARE8(send_sn, >, (psn_t)(*recv_sn + fc_window))) {
                        progress_responder();
                    }
                }
                if (send_window) {
                    /* Wait until we have enough sends completed, then take
                     * the next completion handle in the window. */
                    while (m_outstanding >= m_max_outstanding) {
                        progress_requestor();
                    }
                }

                send_b(ep, send_sn, send_sn - 1, buffer, length, remote_addr,
                       rkey, &completion(completion_index)->uct);
                ++completion_index;

                ucx_perf_update(&m_perf, 1, length);
                if (flow_control) {
                    ++send_sn;
                }
            }

            if (!flow_control) {
                /* Send "sentinel" value */
                if (direction_to_responder) {
                    while (m_outstanding >= m_max_outstanding) {
                        progress_requestor();
                    }
                    *(psn_t*)buffer = 2;
                    send_b(ep, 2, 1, buffer, length, remote_addr, rkey,
                           &completion(completion_index)->uct);
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
        } else if (my_index == 1) {
            if (flow_control) {
                /* Since we're doing flow control, we can count exactly how
                 * many packets were received.
                 */
                ucs_assert(direction_to_responder);
                UCX_PERF_TEST_FOREACH(&m_perf) {
                    sn = *recv_sn;
                    progress_responder();
                    if (UCS_CIRCULAR_COMPARE8(sn, >, (psn_t)(send_sn + (fc_window / 2)))) {
                        /* Send ACK every half-window */
                        send_b(ep, sn, send_sn, buffer, length, remote_addr,
                               rkey, &completion(completion_index)->uct);
                        ++completion_index;
                        send_sn = sn;
                    }

                    /* Calculate number of iterations */
                    m_perf.current.iters +=
                                    (psn_t)(sn - (psn_t)m_perf.current.iters);
                }

                /* Send ACK for last packet */
                if (UCS_CIRCULAR_COMPARE8(*recv_sn, >, send_sn)) {
                    send_b(ep, *recv_sn, send_sn, buffer, length, remote_addr,
                           rkey, &completion(completion_index)->uct);
                }
            } else {
                /* Wait for "sentinel" value */
                ucs_time_t poll_time = ucs_get_time();
                while (*recv_sn != 2) {
                    progress_responder();
                    if (!direction_to_responder) {
                        if (ucs_get_time() > poll_time + ucs_time_from_msec(1.0)) {
                            send_b(ep, 0, 0, buffer, length, remote_addr, rkey,
                                   &completion(completion_index)->uct);
                            poll_time = ucs_get_time();
                        }
                    }
                }
            }
        }

        uct_perf_iface_flush_b(&m_perf);
        ucs_assert(m_outstanding == 0);
        if (my_index == 0) {
            ucx_perf_update(&m_perf, 0, 0);
        }

        return UCS_OK;
    }

    ucs_status_t run()
    {
        bool zcopy = (DATA == UCT_PERF_DATA_LAYOUT_ZCOPY);

        switch (TYPE) {
        case UCX_PERF_TEST_TYPE_PINGPONG:
            return run_pingpong();
        case UCX_PERF_TEST_TYPE_STREAM_UNI:
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
            case UCX_PERF_CMD_FADD:
            case UCX_PERF_CMD_SWAP:
            case UCX_PERF_CMD_CSWAP:
                return run_stream_req_uni(false, /* No flow control for RMA/AMO */
                                          true, /* Waiting for replies */
                                          false /* For GET, data is delivered to requestor.
                                                   For atomics, data goes both ways, but
                                                     the reply is easier to predict */ );
            default:
                return UCS_ERR_INVALID_PARAM;
            }
        case UCX_PERF_TEST_TYPE_STREAM_BI:
        default:
            return UCS_ERR_INVALID_PARAM;
        }
    }

private:
    typedef struct {
        uct_perf_test_runner *self;
        uct_completion_t     uct;
    } comp_t;

    comp_t *completion(unsigned index) {
        return m_completions[index % m_max_outstanding];
    }

    ucx_perf_context_t &m_perf;
    unsigned           m_outstanding;
    const unsigned     m_max_outstanding;
    size_t             m_completion_size;
    comp_t             **m_completions;
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
