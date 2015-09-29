/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "libperf_int.h"

extern "C" {
#include <ucs/debug/log.h>
#include <ucs/sys/math.h>
}
#include <ucs/sys/preprocessor.h>


template <ucx_perf_cmd_t CMD, ucx_perf_test_type_t TYPE, bool ONESIDED>
class ucp_perf_test_runner {
public:
    static const ucp_tag_t TAG = 0x1337a880u;

    typedef uint8_t psn_t;

    ucp_perf_test_runner(ucx_perf_context_t &perf) :
        m_perf(perf),
        m_outstanding(0),
        m_max_outstanding(m_perf.params.max_outstanding)

    {
        ucs_assert_always(m_max_outstanding > 0);
    }

    void UCS_F_ALWAYS_INLINE progress_responder() {
        if (!ONESIDED) {
            ucp_worker_progress(m_perf.ucp.worker);
        }
    }

    void UCS_F_ALWAYS_INLINE progress_requestor() {
        ucp_worker_progress(m_perf.ucp.worker);
    }

    ucs_status_t UCS_F_ALWAYS_INLINE
    send(ucp_ep_h ep, void *buffer, unsigned length, uint8_t sn,
         uint64_t remote_addr, ucp_rkey_h rkey)
    {
        switch (CMD) {
        case UCX_PERF_CMD_TAG:
            return ucp_tag_send(ep, buffer, length, ucp_dt_make_contig(1),
                                TAG);
        case UCX_PERF_CMD_PUT:
            *((uint8_t*)buffer + length - 1) = sn;
            return ucp_put(ep, buffer, length, remote_addr, rkey);
        case UCX_PERF_CMD_GET:
            return ucp_get(ep, buffer, length, remote_addr, rkey);
        case UCX_PERF_CMD_ADD:
            if (length == sizeof(uint32_t)) {
                return ucp_atomic_add32(ep, 1, remote_addr, rkey);
            } else if (length == sizeof(uint64_t)) {
                return ucp_atomic_add64(ep, 1, remote_addr, rkey);
            } else {
                return UCS_ERR_INVALID_PARAM;
            }
        case UCX_PERF_CMD_FADD:
            if (length == sizeof(uint32_t)) {
                return ucp_atomic_fadd32(ep, 0, remote_addr, rkey, (uint32_t*)buffer);
            } else if (length == sizeof(uint64_t)) {
                return ucp_atomic_fadd64(ep, 0, remote_addr, rkey, (uint64_t*)buffer);
            } else {
                return UCS_ERR_INVALID_PARAM;
            }
        case UCX_PERF_CMD_SWAP:
            if (length == sizeof(uint32_t)) {
                return ucp_atomic_swap32(ep, 0, remote_addr, rkey, (uint32_t*)buffer);
            } else if (length == sizeof(uint64_t)) {
                return ucp_atomic_swap64(ep, 0, remote_addr, rkey, (uint64_t*)buffer);
            } else {
                return UCS_ERR_INVALID_PARAM;
            }
        case UCX_PERF_CMD_CSWAP:
            if (length == sizeof(uint32_t)) {
                return ucp_atomic_cswap32(ep, 0, 0, remote_addr, rkey, (uint32_t*)buffer);
            } else if (length == sizeof(uint64_t)) {
                return ucp_atomic_cswap64(ep, 0, 0, remote_addr, rkey, (uint64_t*)buffer);
            } else {
                return UCS_ERR_INVALID_PARAM;
            }
        default:
            return UCS_ERR_INVALID_PARAM;
        }
    }

    ucs_status_t UCS_F_ALWAYS_INLINE
    recv(ucp_worker_h worker, void *buffer, unsigned length, uint8_t sn)
    {
        ucp_tag_recv_info_t comp;
        volatile uint8_t *ptr;

        switch (CMD) {
        case UCX_PERF_CMD_TAG:
            return ucp_tag_recv(worker, buffer, length, ucp_dt_make_contig(1),
                                TAG, 0, &comp);
        case UCX_PERF_CMD_PUT:
            switch (TYPE) {
            case UCX_PERF_TEST_TYPE_PINGPONG:
                ptr = (volatile uint8_t*)buffer + length - 1;
                while (*ptr != sn) {
                    progress_responder();
                }
                return UCS_OK;
            case UCX_PERF_TEST_TYPE_STREAM_UNI:
                return UCS_OK;
            default:
                return UCS_ERR_INVALID_PARAM;
            }
        case UCX_PERF_CMD_GET:
        case UCX_PERF_CMD_ADD:
        case UCX_PERF_CMD_FADD:
        case UCX_PERF_CMD_SWAP:
        case UCX_PERF_CMD_CSWAP:
            switch (TYPE) {
            case UCX_PERF_TEST_TYPE_STREAM_UNI:
                progress_responder();
                return UCS_OK;
            default:
                return UCS_ERR_INVALID_PARAM;
            }
        default:
            return UCS_ERR_INVALID_PARAM;
        }
    }

    ucs_status_t run_pingpong()
    {
        unsigned my_index;
        ucp_worker_h worker;
        ucp_ep_h ep;
        void *send_buffer, *recv_buffer;
        uint64_t remote_addr;
        uint8_t sn;
        ucp_rkey_h rkey;
        size_t length;

        ucs_assert(m_perf.params.message_size >= sizeof(psn_t));

        *((volatile uint8_t*)m_perf.recv_buffer + m_perf.params.message_size - 1) = -1;

        rte_call(&m_perf, barrier);

        my_index = rte_call(&m_perf, group_index);

        ucx_perf_test_start_clock(&m_perf);

        send_buffer = m_perf.send_buffer;
        recv_buffer = m_perf.recv_buffer;
        length      = m_perf.params.message_size;
        worker      = m_perf.ucp.worker;
        ep          = m_perf.ucp.peers[1 - my_index].ep;
        remote_addr = m_perf.ucp.peers[1 - my_index].remote_addr;
        rkey        = m_perf.ucp.peers[1 - my_index].rkey;
        sn          = 0;

        if (my_index == 0) {
            UCX_PERF_TEST_FOREACH(&m_perf) {
                send(ep, send_buffer, length, sn, remote_addr, rkey);
                recv(worker, recv_buffer, length, sn);
                ucx_perf_update(&m_perf, 1, length);
                ++sn;
            }
        } else if (my_index == 1) {
            UCX_PERF_TEST_FOREACH(&m_perf) {
                recv(worker, recv_buffer, length, sn);
                send(ep, send_buffer, length, sn, remote_addr, rkey);
                ucx_perf_update(&m_perf, 1, length);
                ++sn;
            }
        }

        ucp_worker_flush(m_perf.ucp.worker);
        rte_call(&m_perf, barrier);
        return UCS_OK;
    }

    ucs_status_t run_stream_uni()
    {
        unsigned my_index;
        ucp_worker_h worker;
        ucp_ep_h ep;
        void *send_buffer, *recv_buffer;
        uint64_t remote_addr;
        ucp_rkey_h rkey;
        size_t length;
        uint8_t sn;

        ucs_assert(m_perf.params.message_size >= sizeof(psn_t));

        rte_call(&m_perf, barrier);

        my_index = rte_call(&m_perf, group_index);

        ucx_perf_test_start_clock(&m_perf);

        send_buffer = m_perf.send_buffer;
        recv_buffer = m_perf.recv_buffer;
        length      = m_perf.params.message_size;
        worker      = m_perf.ucp.worker;
        ep          = m_perf.ucp.peers[1 - my_index].ep;
        remote_addr = m_perf.ucp.peers[1 - my_index].remote_addr;
        rkey        = m_perf.ucp.peers[1 - my_index].rkey;
        sn          = 0;

        if (my_index == 0) {
            UCX_PERF_TEST_FOREACH(&m_perf) {
                recv(worker, recv_buffer, length, sn);
                ucx_perf_update(&m_perf, 1, length);
                ++sn;
            }
        } else if (my_index == 1) {
            UCX_PERF_TEST_FOREACH(&m_perf) {
                send(ep, send_buffer, length, sn, remote_addr, rkey);
                ucx_perf_update(&m_perf, 1, length);
                ++sn;
            }
        }

        ucp_worker_flush(m_perf.ucp.worker);
        rte_call(&m_perf, barrier);
        return UCS_OK;
    }

    ucs_status_t run()
    {
        switch (TYPE) {
        case UCX_PERF_TEST_TYPE_PINGPONG:
            return run_pingpong();
        case UCX_PERF_TEST_TYPE_STREAM_UNI:
            return run_stream_uni();
        case UCX_PERF_TEST_TYPE_STREAM_BI:
        default:
            return UCS_ERR_INVALID_PARAM;
        }
    }

private:
    ucx_perf_context_t &m_perf;
    unsigned           m_outstanding;
    const unsigned     m_max_outstanding;
};


#define TEST_CASE(_perf, _cmd, _type, _onesided) \
    if (((_perf)->params.command == (_cmd)) && \
        ((_perf)->params.test_type == (_type)) && \
        (!!((_perf)->params.flags & UCX_PERF_TEST_FLAG_ONE_SIDED) == !!(_onesided))) \
    { \
        ucp_perf_test_runner<_cmd, _type, _onesided> r(*_perf); \
        return r.run(); \
    }
#define TEST_CASE_ALL_OSD(_perf, _case) \
   TEST_CASE(_perf, UCS_PP_TUPLE_0 _case, UCS_PP_TUPLE_1 _case, true) \
   TEST_CASE(_perf, UCS_PP_TUPLE_0 _case, UCS_PP_TUPLE_1 _case, false)


ucs_status_t ucp_perf_test_dispatch(ucx_perf_context_t *perf)
{
    UCS_PP_FOREACH(TEST_CASE_ALL_OSD, perf,
        (UCX_PERF_CMD_TAG,   UCX_PERF_TEST_TYPE_PINGPONG),
        (UCX_PERF_CMD_TAG,   UCX_PERF_TEST_TYPE_STREAM_UNI),
        (UCX_PERF_CMD_PUT,   UCX_PERF_TEST_TYPE_PINGPONG),
        (UCX_PERF_CMD_PUT,   UCX_PERF_TEST_TYPE_STREAM_UNI),
        (UCX_PERF_CMD_GET,   UCX_PERF_TEST_TYPE_STREAM_UNI),
        (UCX_PERF_CMD_ADD,   UCX_PERF_TEST_TYPE_STREAM_UNI),
        (UCX_PERF_CMD_FADD,  UCX_PERF_TEST_TYPE_STREAM_UNI),
        (UCX_PERF_CMD_SWAP,  UCX_PERF_TEST_TYPE_STREAM_UNI),
        (UCX_PERF_CMD_CSWAP, UCX_PERF_TEST_TYPE_STREAM_UNI)
        );

    ucs_error("Invalid test case");
    return UCS_ERR_INVALID_PARAM;
}
