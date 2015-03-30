/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
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
    send(ucp_ep_h ep, void *buffer, unsigned length)
    {
        switch (CMD) {
        case UCX_PERF_CMD_TAG:
            return ucp_tag_send(ep, buffer, length, TAG);
        default:
            return UCS_ERR_INVALID_PARAM;
        }
    }

    ucs_status_t UCS_F_ALWAYS_INLINE
    recv(ucp_worker_h worker, void *buffer, unsigned length)
    {
        ucp_tag_recv_completion_t comp;
        switch (CMD) {
        case UCX_PERF_CMD_TAG:
            return ucp_tag_recv(worker, buffer, length, TAG, 0, &comp);
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
        size_t length;

        ucs_assert(m_perf.params.message_size >= sizeof(psn_t));

        rte_call(&m_perf, barrier);

        my_index = rte_call(&m_perf, group_index);

        ucx_perf_test_start_clock(&m_perf);

        send_buffer = m_perf.send_buffer;
        recv_buffer = m_perf.recv_buffer;
        length      = m_perf.params.message_size;
        worker      = m_perf.ucp.worker;
        ep          = m_perf.ucp.peers[1 - my_index].ep;

        if (my_index == 0) {
            UCX_PERF_TEST_FOREACH(&m_perf) {
                send(ep, send_buffer, length);
                recv(worker, recv_buffer, length);
                ucx_perf_update(&m_perf, 1, length);
            }
        } else if (my_index == 1) {
            UCX_PERF_TEST_FOREACH(&m_perf) {
                recv(worker, recv_buffer, length);
                send(ep, send_buffer, length);
                ucx_perf_update(&m_perf, 1, length);
            }
        }
        return UCS_OK;
    }

    ucs_status_t run_stream_uni()
    {
        unsigned my_index;
        ucp_worker_h worker;
        ucp_ep_h ep;
        void *send_buffer, *recv_buffer;
        size_t length;

        ucs_assert(m_perf.params.message_size >= sizeof(psn_t));

        rte_call(&m_perf, barrier);

        my_index = rte_call(&m_perf, group_index);

        ucx_perf_test_start_clock(&m_perf);

        send_buffer = m_perf.send_buffer;
        recv_buffer = m_perf.recv_buffer;
        length      = m_perf.params.message_size;
        worker      = m_perf.ucp.worker;
        ep          = m_perf.ucp.peers[1 - my_index].ep;

        if (my_index == 0) {
            UCX_PERF_TEST_FOREACH(&m_perf) {
                recv(worker, recv_buffer, length);
                ucx_perf_update(&m_perf, 1, length);
            }
        } else if (my_index == 1) {
            UCX_PERF_TEST_FOREACH(&m_perf) {
                send(ep, send_buffer, length);
                ucx_perf_update(&m_perf, 1, length);
            }
        }
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
        (UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_PINGPONG),
        (UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_STREAM_UNI)
        );

    ucs_error("Invalid test case");
    return UCS_ERR_INVALID_PARAM;
}
