/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) The University of Tennessee and The University
*               of Tennessee Research Foundation. 2016. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "libperf_int.h"

#include <ucs/sys/preprocessor.h>
#include <limits>


#define UCP_PERF_LAST_ITER_SN    1


template <ucx_perf_cmd_t CMD, ucx_perf_test_type_t TYPE, unsigned FLAGS>
class ucp_perf_test_runner {
public:
    static const unsigned AM_ID     = 1;
    static const ucp_tag_t TAG      = 0x1337a880u;
    static const ucp_tag_t TAG_MASK = (FLAGS & UCX_PERF_TEST_FLAG_TAG_WILDCARD) ?
                                      0 : (ucp_tag_t)-1;

    typedef uint8_t psn_t;

    ucp_perf_test_runner(ucx_perf_context_t &perf)
        : m_perf(perf),
          m_recvs_outstanding(0),
          m_sends_outstanding(0),
          m_max_outstanding(m_perf.params.max_outstanding),
          m_am_rx_buffer(NULL),
          m_am_rx_length(0ul)

    {
        memset(&m_am_rx_params, 0, sizeof(m_am_rx_params));
        memset(&m_send_params, 0, sizeof(m_send_params));

        ucs_assert_always(m_max_outstanding > 0);

        set_am_handler(am_data_handler, this, UCP_AM_FLAG_WHOLE_MSG);

        if (CMD == UCX_PERF_CMD_ADD) {
            m_atomic_op = UCP_ATOMIC_OP_ADD;
        } else if (CMD == UCX_PERF_CMD_FADD) {
            m_atomic_op = UCP_ATOMIC_OP_ADD;
        } else if (CMD == UCX_PERF_CMD_SWAP) {
            m_atomic_op = UCP_ATOMIC_OP_SWAP;
        } else if (CMD == UCX_PERF_CMD_CSWAP) {
            m_atomic_op = UCP_ATOMIC_OP_CSWAP;
        } else {
            m_atomic_op = UCP_ATOMIC_OP_LAST;
        }
    }

    ~ucp_perf_test_runner()
    {
        set_am_handler(NULL, this, 0);
    }

    void set_am_handler(ucp_am_recv_callback_t cb, void *arg, unsigned flags)
    {
        if (CMD == UCX_PERF_CMD_AM) {
            ucp_am_handler_param_t param;
            param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                               UCP_AM_HANDLER_PARAM_FIELD_CB |
                               UCP_AM_HANDLER_PARAM_FIELD_ARG;
            param.id         = AM_ID;
            param.cb         = cb;
            param.arg        = arg;

            if (flags != 0) {
                param.field_mask |= UCP_AM_HANDLER_PARAM_FIELD_FLAGS;
                param.flags       = flags;
            }

            ucs_status_t status = ucp_worker_set_am_recv_handler(
                                      m_perf.ucp.worker, &param);
            ucs_assert_always(status == UCS_OK);
        }
    }

    void create_iov_buffer(ucp_dt_iov_t *iov, void *buffer)
    {
        size_t iov_length_it, iov_it;
        const size_t iovcnt = m_perf.params.msg_size_cnt;

        ucs_assert(NULL != m_perf.params.msg_size_list);
        ucs_assert(iovcnt > 0);

        iov_length_it = 0;
        for (iov_it = 0; iov_it < iovcnt; ++iov_it) {
            iov[iov_it].buffer = (char *)buffer + iov_length_it;
            iov[iov_it].length = m_perf.params.msg_size_list[iov_it];

            if (m_perf.params.iov_stride) {
                iov_length_it += m_perf.params.iov_stride;
            } else {
                iov_length_it += iov[iov_it].length;
            }
        }
    }

    ucp_datatype_t ucp_perf_test_get_datatype(ucp_perf_datatype_t datatype, ucp_dt_iov_t *iov,
                                              size_t *length, void **buffer_p)
    {
        ucp_datatype_t type = ucp_dt_make_contig(1);
        if ((CMD == UCX_PERF_CMD_ADD) || (CMD == UCX_PERF_CMD_FADD) ||
            (CMD == UCX_PERF_CMD_SWAP) || (CMD == UCX_PERF_CMD_CSWAP)) {
            ucs_assert(m_atomic_op != UCP_ATOMIC_OP_LAST);
            type      = ucp_dt_make_contig(*length);
        } else if (UCP_PERF_DATATYPE_IOV == datatype) {
            *buffer_p = iov;
            *length   = m_perf.params.msg_size_cnt;
            type      = ucp_dt_make_iov();
        }
        return type;
    }
    /**
     * Make ucp_dt_iov_t iov[msg_size_cnt] array with pointer elements to
     * original buffer
     */
    void ucp_perf_test_prepare_iov_buffers()
    {
        if (UCP_PERF_DATATYPE_IOV == m_perf.params.ucp.send_datatype) {
            create_iov_buffer(m_perf.ucp.send_iov, m_perf.send_buffer);
        }
        if (UCP_PERF_DATATYPE_IOV == m_perf.params.ucp.recv_datatype) {
            create_iov_buffer(m_perf.ucp.recv_iov, m_perf.recv_buffer);
        }
    }

    void ucp_perf_init_common_params(size_t *total_length, size_t *send_length,
                                     ucp_datatype_t *send_dt,
                                     void **send_buffer, size_t *recv_length,
                                     ucp_datatype_t *recv_dt,
                                     void **recv_buffer)
    {
        *total_length = ucx_perf_get_message_size(&m_perf.params);

        if (CMD == UCX_PERF_CMD_PUT) {
            ucs_assert(*total_length >= sizeof(psn_t));
        }

        ucp_perf_test_prepare_iov_buffers();

        *send_length = *recv_length = *total_length;

        *send_dt = ucp_perf_test_get_datatype(m_perf.params.ucp.send_datatype,
                                              m_perf.ucp.send_iov, send_length,
                                              send_buffer);
        *recv_dt = ucp_perf_test_get_datatype(m_perf.params.ucp.recv_datatype,
                                              m_perf.ucp.recv_iov, recv_length,
                                              recv_buffer);
        if (CMD == UCX_PERF_CMD_AM) {
            m_am_rx_params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                          UCP_OP_ATTR_FIELD_USER_DATA |
                                          UCP_OP_ATTR_FIELD_DATATYPE |
                                          UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
            m_am_rx_params.datatype     = *recv_dt;
            m_am_rx_params.cb.recv_am   = am_data_recv_cb;
            m_am_rx_params.user_data    = this;
            m_am_rx_buffer              = *recv_buffer;
            m_am_rx_length              = *recv_length;
        }

        if ((CMD == UCX_PERF_CMD_AM) || (CMD == UCX_PERF_CMD_ADD) ||
            (CMD == UCX_PERF_CMD_FADD) || (CMD == UCX_PERF_CMD_SWAP) ||
            (CMD == UCX_PERF_CMD_CSWAP)) {
            m_send_params.op_attr_mask  = UCP_OP_ATTR_FIELD_DATATYPE |
                                          UCP_OP_ATTR_FIELD_CALLBACK;
            m_send_params.datatype      = *send_dt;
            m_send_params.cb.send       = send_nbx_cb;
        }

        if ((CMD == UCX_PERF_CMD_FADD) || (CMD == UCX_PERF_CMD_SWAP) ||
            (CMD == UCX_PERF_CMD_CSWAP)) {
            m_send_params.op_attr_mask |= UCP_OP_ATTR_FIELD_REPLY_BUFFER;
            m_send_params.reply_buffer  = *send_buffer;
        }
    }

    void UCS_F_ALWAYS_INLINE blocking_progress() {
        if (ucp_worker_progress(m_perf.ucp.worker) == 0) {
            ucp_worker_wait(m_perf.ucp.worker);
        }
    }

    void UCS_F_ALWAYS_INLINE progress() {
        if (ucs_unlikely(UCX_PERF_WAIT_MODE_SLEEP == m_perf.params.wait_mode)) {
            blocking_progress();
        } else {
            ucp_worker_progress(m_perf.ucp.worker);
        }
    }

    void UCS_F_ALWAYS_INLINE progress_responder() {
        if (!(FLAGS & UCX_PERF_TEST_FLAG_ONE_SIDED) &&
            !(m_perf.params.flags & UCX_PERF_TEST_FLAG_ONE_SIDED))
        {
            progress();
        }
    }

    void UCS_F_ALWAYS_INLINE progress_requestor() {
        progress();
    }

    ssize_t UCS_F_ALWAYS_INLINE wait_stream_recv(void *request)
    {
        size_t       length;
        ucs_status_t status;

        ucs_assert(UCS_PTR_IS_PTR(request));

        while ((status = ucp_stream_recv_request_test(request, &length)) ==
                UCS_INPROGRESS) {
            progress_responder();
        }
        ucp_request_release(request);

        return ucs_likely(status == UCS_OK) ? length : status;
    }

    ucs_status_t am_rndv_recv(void *data, size_t length,
                              const ucp_am_recv_param_t *rx_params)
    {
        ucs_assert(!(rx_params->recv_attr & UCP_AM_RECV_ATTR_FLAG_DATA));
        ucs_assert(length == ucx_perf_get_message_size(&m_perf.params));

        ucs_status_ptr_t sp = ucp_am_recv_data_nbx(m_perf.ucp.worker, data,
                                                   m_am_rx_buffer,
                                                   m_am_rx_length,
                                                   &m_am_rx_params);
        ucs_assert(UCS_PTR_IS_PTR(sp));
        ucp_request_release(sp);

        return UCS_INPROGRESS;
    }


    static void send_cb(void *request, ucs_status_t status)
    {
        ucp_perf_request_t *r      = reinterpret_cast<ucp_perf_request_t*>(
                                          request);
        ucp_perf_test_runner *test = (ucp_perf_test_runner*)r->context;

        test->send_completed();
        r->context = NULL;
        ucp_request_free(request);
    }

    static void send_nbx_cb(void *request, ucs_status_t status, void *user_data)
    {
        send_cb(request, status);
    }

    static void tag_recv_cb(void *request, ucs_status_t status,
                            ucp_tag_recv_info_t *info)
    {
        ucp_perf_request_t *r = reinterpret_cast<ucp_perf_request_t*>(request);
        ucp_perf_test_runner *test;

        /* if the request is completed during tag_recv_nb(), the context is
         * still NULL */
        if (r->context == NULL) {
            return;
        }

        test = (ucp_perf_test_runner*)r->context;
        test->recv_completed();
        r->context = NULL;
        ucp_request_free(request);
    }

    static void am_data_recv_cb(void *request, ucs_status_t status,
                                size_t length, void *user_data)
    {
        ucp_perf_test_runner *test = (ucp_perf_test_runner*)user_data;
        test->recv_completed();
    }

    static ucs_status_t
    am_data_handler(void *arg, const void *header, size_t header_length,
                    void *data, size_t length, const ucp_am_recv_param_t *param)
    {
        ucp_perf_test_runner *test = (ucp_perf_test_runner*)arg;

        if (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) {
            return test->am_rndv_recv(data, length, param);
        }

        /* TODO: Add option to do memcopy here */
        test->recv_completed();
        return UCS_OK;
    }

    void UCS_F_ALWAYS_INLINE wait_send_window(unsigned n)
    {
        while (m_sends_outstanding >= (m_max_outstanding - n + 1)) {
            progress_requestor();
        }
    }

    void UCS_F_ALWAYS_INLINE wait_recv_window(unsigned n)
    {
        while (m_recvs_outstanding >= (m_max_outstanding - n + 1)) {
            progress_responder();
        }
    }

    ucs_status_t UCS_F_ALWAYS_INLINE
    send(ucp_ep_h ep, void *buffer, unsigned length, ucp_datatype_t datatype,
         uint8_t sn, uint64_t remote_addr, ucp_rkey_h rkey)
    {
        uint64_t value = 0;
        void *request;

        /* coverity[switch_selector_expr_is_constant] */
        switch (CMD) {
        case UCX_PERF_CMD_TAG:
        case UCX_PERF_CMD_TAG_SYNC:
        case UCX_PERF_CMD_STREAM:
        case UCX_PERF_CMD_AM:
            wait_send_window(1);
            /* coverity[switch_selector_expr_is_constant] */
            switch (CMD) {
            case UCX_PERF_CMD_TAG:
                request = ucp_tag_send_nb(ep, buffer, length, datatype, TAG,
                                          send_cb);
                break;
            case UCX_PERF_CMD_TAG_SYNC:
                request = ucp_tag_send_sync_nb(ep, buffer, length, datatype, TAG,
                                               send_cb);
                break;
            case UCX_PERF_CMD_STREAM:
                request = ucp_stream_send_nb(ep, buffer, length, datatype,
                                             send_cb, 0);
                break;
            case UCX_PERF_CMD_AM:
                request = ucp_am_send_nbx(ep, AM_ID, m_perf.ucp.am_hdr,
                                          m_perf.params.ucp.am_hdr_size, buffer,
                                          length, &m_send_params);
                break;
            default:
                request = UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM);
                break;
            }
            if (ucs_likely(!UCS_PTR_IS_PTR(request))) {
                return UCS_PTR_STATUS(request);
            }
            reinterpret_cast<ucp_perf_request_t*>(request)->context = this;
            send_started();
            return UCS_OK;
        case UCX_PERF_CMD_PUT:
            /* coverity[switch_selector_expr_is_constant] */
            switch (TYPE) {
            case UCX_PERF_TEST_TYPE_PINGPONG:
            case UCX_PERF_TEST_TYPE_PINGPONG_WAIT_MEM:
                *((uint8_t*)buffer + length - 1) = sn;
                break;
            case UCX_PERF_TEST_TYPE_STREAM_UNI:
                *((uint8_t*)buffer + length - 1) = 0;
                break;
            default:
                return UCS_ERR_INVALID_PARAM;
            }
            return ucp_put(ep, buffer, length, remote_addr, rkey);
        case UCX_PERF_CMD_GET:
            return ucp_get(ep, buffer, length, remote_addr, rkey);
        case UCX_PERF_CMD_ADD:
        case UCX_PERF_CMD_FADD:
        case UCX_PERF_CMD_SWAP:
        case UCX_PERF_CMD_CSWAP:
            wait_send_window(1);
            request = ucp_atomic_op_nbx(ep, m_atomic_op, &value, 1,
                                        remote_addr, rkey, &m_send_params);
            if (ucs_likely(!UCS_PTR_IS_PTR(request))) {
                return UCS_PTR_STATUS(request);
            }
            reinterpret_cast<ucp_perf_request_t*>(request)->context = this;
            send_started();
            return UCS_OK;
        default:
            return UCS_ERR_INVALID_PARAM;
        }
    }

    ucs_status_t UCS_F_ALWAYS_INLINE
    recv(ucp_worker_h worker, ucp_ep_h ep, void *buffer, unsigned length,
         ucp_datatype_t datatype, uint8_t sn)
    {
        volatile uint8_t *ptr;
        void *request;

        /* coverity[switch_selector_expr_is_constant] */
        switch (CMD) {
        case UCX_PERF_CMD_TAG:
        case UCX_PERF_CMD_TAG_SYNC:
            wait_recv_window(1);
            if (FLAGS & UCX_PERF_TEST_FLAG_TAG_UNEXP_PROBE) {
                ucp_tag_recv_info_t tag_info;
                while (ucp_tag_probe_nb(worker, TAG, TAG_MASK, 0, &tag_info) == NULL) {
                    progress_responder();
                }
            }
            request = ucp_tag_recv_nb(worker, buffer, length, datatype, TAG, TAG_MASK,
                                      tag_recv_cb);
            if (ucs_likely(!UCS_PTR_IS_PTR(request))) {
                return UCS_PTR_STATUS(request);
            }
            if (ucp_request_is_completed(request)) {
                /* request is already completed and callback was called */
                ucp_request_free(request);
                return UCS_OK;
            }
            reinterpret_cast<ucp_perf_request_t*>(request)->context = this;
            recv_started();
            return UCS_OK;
        case UCX_PERF_CMD_AM:
            recv_started();
            return UCS_OK;
        case UCX_PERF_CMD_PUT:
            /* coverity[switch_selector_expr_is_constant] */
            switch (TYPE) {
            case UCX_PERF_TEST_TYPE_PINGPONG:
                ptr = (volatile uint8_t*)buffer + length - 1;
                while (*ptr != sn) {
                    progress_responder();
                }
                return UCS_OK;
            case UCX_PERF_TEST_TYPE_PINGPONG_WAIT_MEM:
                ptr = (volatile uint8_t*)buffer + length - 1;
                while (*ptr != sn) {
                    ucp_worker_wait_mem(worker, (void *)ptr);
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
            /* coverity[switch_selector_expr_is_constant] */
            switch (TYPE) {
            case UCX_PERF_TEST_TYPE_STREAM_UNI:
                progress_responder();
                return UCS_OK;
            default:
                return UCS_ERR_INVALID_PARAM;
            }
        case UCX_PERF_CMD_STREAM:
            if (FLAGS & UCX_PERF_TEST_FLAG_STREAM_RECV_DATA) {
                return recv_stream_data(ep, length, datatype);
            } else {
                return recv_stream(ep, buffer, length, datatype);
            }
        default:
            return UCS_ERR_INVALID_PARAM;
        }
    }

    /* wait for the last iteration to be completed in case of
     * unidirectional PUT test, since it need to progress responder
     * for SW-based RMA implementations */
    void wait_last_iter(void *buffer)
    {
        volatile uint8_t *ptr = (uint8_t*)buffer;

        if (use_psn()) {
            while (*ptr != UCP_PERF_LAST_ITER_SN) {
                progress_responder();
            }
        }
    }

    static void nop_cb(void *request, ucs_status_t status)
    {
        ucp_request_free(request);
    }

    /* send the special flag as a last iteration in case of
     * unidirectional PUT test, since responder is waiting for
     * this message */
    ucs_status_t send_last_iter(ucp_ep_h ep, void *buffer, size_t size,
                                uint64_t remote_addr, ucp_rkey_h rkey)
    {
        ucs_status_ptr_t status_p;

        if (!use_psn()) {
            return UCS_OK;
        }

        fence();
        *(uint8_t*)buffer = UCP_PERF_LAST_ITER_SN;
        /* coverity[switch_selector_expr_is_constant] */
        switch (CMD) {
        case UCX_PERF_CMD_PUT:
            return ucp_put(ep, buffer, sizeof(uint8_t), remote_addr, rkey);
        case UCX_PERF_CMD_ADD:
            return ucp_atomic_post(ep, UCP_ATOMIC_POST_OP_ADD, 1, size,
                                   remote_addr, rkey);
        case UCX_PERF_CMD_FADD:
            status_p = ucp_atomic_fetch_nb(ep, UCP_ATOMIC_FETCH_OP_FADD, 1,
                                           buffer, size, remote_addr, rkey,
                                           nop_cb);
            return UCS_PTR_STATUS(status_p);
        case UCX_PERF_CMD_SWAP:
            status_p = ucp_atomic_fetch_nb(ep, UCP_ATOMIC_FETCH_OP_SWAP, 1,
                                           buffer, size, remote_addr, rkey,
                                           nop_cb);
            return UCS_PTR_STATUS(status_p);
        case UCX_PERF_CMD_CSWAP:
            status_p = ucp_atomic_fetch_nb(ep, UCP_ATOMIC_FETCH_OP_CSWAP, 0,
                                           buffer, size, remote_addr, rkey,
                                           nop_cb);
            return UCS_PTR_STATUS(status_p);
        default:
            return UCS_OK;
        }
    }

    void flush()
    {
        if (m_perf.params.flags & UCX_PERF_TEST_FLAG_FLUSH_EP) {
            ucp_ep_flush(m_perf.ucp.ep);
        } else {
            ucp_worker_flush(m_perf.ucp.worker);
        }
    }

    void fence()
    {
        ucp_worker_fence(m_perf.ucp.worker);
    }


    int use_psn() {
        return ((CMD == UCX_PERF_CMD_PUT) || (CMD == UCX_PERF_CMD_ADD) ||
                (CMD == UCX_PERF_CMD_FADD) || (CMD == UCX_PERF_CMD_SWAP) ||
                (CMD == UCX_PERF_CMD_CSWAP));
    }

    void reset_buffers(const psn_t psn, size_t offset)
    {
        psn_t src = psn;

        if (use_psn()) {
            m_perf.allocator->memcpy(UCS_PTR_BYTE_OFFSET(m_perf.recv_buffer, offset),
                                     m_perf.allocator->mem_type,
                                     &src, UCS_MEMORY_TYPE_HOST,
                                     sizeof(src));
            m_perf.allocator->memcpy(UCS_PTR_BYTE_OFFSET(m_perf.send_buffer, offset),
                                     m_perf.allocator->mem_type,
                                     &src, UCS_MEMORY_TYPE_HOST,
                                     sizeof(src));
        }
    }

    ucs_status_t run_pingpong()
    {
        const psn_t unknown_psn = std::numeric_limits<psn_t>::max();
        unsigned my_index;
        ucp_worker_h worker;
        ucp_ep_h ep;
        void *send_buffer, *recv_buffer;
        ucp_datatype_t send_datatype, recv_datatype;
        uint64_t remote_addr;
        uint8_t sn;
        ucp_rkey_h rkey;
        size_t length, send_length, recv_length;

        send_buffer = m_perf.send_buffer;
        recv_buffer = m_perf.recv_buffer;
        worker      = m_perf.ucp.worker;
        ep          = m_perf.ucp.ep;
        remote_addr = m_perf.ucp.remote_addr;
        rkey        = m_perf.ucp.rkey;
        sn          = 0;

        ucp_perf_init_common_params(&length, &send_length, &send_datatype,
                                    &send_buffer, &recv_length, &recv_datatype,
                                    &recv_buffer);

        reset_buffers(unknown_psn, length - 1);

        ucp_perf_barrier(&m_perf);

        my_index = rte_call(&m_perf, group_index);

        ucx_perf_test_start_clock(&m_perf);

        ucx_perf_omp_barrier(&m_perf);

        if (my_index == 0) {
            UCX_PERF_TEST_FOREACH(&m_perf) {
                send(ep, send_buffer, send_length, send_datatype, sn, remote_addr, rkey);
                recv(worker, ep, recv_buffer, recv_length, recv_datatype, sn);
                ucx_perf_update(&m_perf, 1, length);
                ++sn;
            }
        } else if (my_index == 1) {
            UCX_PERF_TEST_FOREACH(&m_perf) {
                recv(worker, ep, recv_buffer, recv_length, recv_datatype, sn);
                send(ep, send_buffer, send_length, send_datatype, sn, remote_addr, rkey);
                ucx_perf_update(&m_perf, 1, length);
                ++sn;
            }
        }

        wait_recv_window(m_max_outstanding);
        wait_send_window(m_max_outstanding);
        flush();

        ucx_perf_omp_barrier(&m_perf);

        ucx_perf_get_time(&m_perf);
        ucp_perf_barrier(&m_perf);
        return UCS_OK;
    }

    ucs_status_t run_stream_uni()
    {
        unsigned my_index;
        ucp_worker_h worker;
        ucp_ep_h ep;
        void *send_buffer, *recv_buffer;
        ucp_datatype_t send_datatype, recv_datatype;
        uint64_t remote_addr;
        ucp_rkey_h rkey;
        size_t length, send_length, recv_length;
        uint8_t sn;

        send_buffer = m_perf.send_buffer;
        recv_buffer = m_perf.recv_buffer;
        worker      = m_perf.ucp.worker;
        ep          = m_perf.ucp.ep;
        remote_addr = m_perf.ucp.remote_addr;
        rkey        = m_perf.ucp.rkey;
        sn          = 0;

        ucp_perf_init_common_params(&length, &send_length, &send_datatype,
                                    &send_buffer, &recv_length, &recv_datatype,
                                    &recv_buffer);

        reset_buffers(0, 0);

        ucp_perf_barrier(&m_perf);

        my_index = rte_call(&m_perf, group_index);

        ucx_perf_test_start_clock(&m_perf);

        ucx_perf_omp_barrier(&m_perf);

        if (m_perf.params.flags & UCX_PERF_TEST_FLAG_LOOPBACK) {
            UCX_PERF_TEST_FOREACH(&m_perf) {
                send(ep, send_buffer, send_length, send_datatype,
                     sn, remote_addr, rkey);
                recv(worker, ep, recv_buffer, recv_length, recv_datatype, sn);
                ucx_perf_update(&m_perf, 1, length);
                ++sn;
            }

            wait_send_window(m_max_outstanding);
            wait_recv_window(m_max_outstanding);
        } else if (my_index == 0) {
            UCX_PERF_TEST_FOREACH(&m_perf) {
                recv(worker, ep, recv_buffer, recv_length, recv_datatype, sn);
                ucx_perf_update(&m_perf, 1, length);
                ++sn;
            }

            wait_last_iter(recv_buffer);
            wait_recv_window(m_max_outstanding);
        } else if (my_index == 1) {
            UCX_PERF_TEST_FOREACH(&m_perf) {
                send(ep, send_buffer, send_length, send_datatype, sn,
                     remote_addr, rkey);
                ucx_perf_update(&m_perf, 1, length);
                ++sn;
            }

            send_last_iter(ep, send_buffer, send_length, remote_addr, rkey);
            wait_send_window(m_max_outstanding);
        }

        flush();

        ucx_perf_omp_barrier(&m_perf);

        ucx_perf_get_time(&m_perf);

        ucp_perf_barrier(&m_perf);
        return UCS_OK;
    }

    ucs_status_t run()
    {
        /* coverity[switch_selector_expr_is_constant] */
        switch (TYPE) {
        case UCX_PERF_TEST_TYPE_PINGPONG:
        case UCX_PERF_TEST_TYPE_PINGPONG_WAIT_MEM:
            return run_pingpong();
        case UCX_PERF_TEST_TYPE_STREAM_UNI:
            return run_stream_uni();
        case UCX_PERF_TEST_TYPE_STREAM_BI:
        default:
            return UCS_ERR_INVALID_PARAM;
        }
    }

private:
    ucs_status_t UCS_F_ALWAYS_INLINE
    recv_stream_data(ucp_ep_h ep, unsigned length, ucp_datatype_t datatype)
    {
        void *data;
        size_t data_length;
        size_t total = 0;

        do {
            progress_responder();
            data = ucp_stream_recv_data_nb(ep, &data_length);
            if (ucs_likely(UCS_PTR_IS_PTR(data))) {
                total += data_length;
                ucp_stream_data_release(ep, data);
            }
        } while ((total < length) && !UCS_PTR_IS_ERR(data));

        return UCS_PTR_IS_ERR(data) ? UCS_PTR_STATUS(data) : UCS_OK;
    }

    ucs_status_t UCS_F_ALWAYS_INLINE
    recv_stream(ucp_ep_h ep, void *buf, unsigned length, ucp_datatype_t datatype)
    {
        ssize_t  total = 0;
        void    *rreq;
        size_t   rlength;
        ssize_t  rlength_s;

        do {
            rreq = ucp_stream_recv_nb(ep, (char *)buf + total, length - total,
                                      datatype,
                                      (ucp_stream_recv_callback_t)ucs_empty_function,
                                      &rlength, 0);
            if (ucs_likely(rreq == NULL)) {
                total += rlength;
            } else if (UCS_PTR_IS_PTR(rreq)) {
                rlength_s = wait_stream_recv(rreq);
                if (ucs_unlikely(rlength_s < 0)) {
                    return ucs_status_t(rlength_s);
                }
                total += rlength_s;
            } else {
                return UCS_PTR_STATUS(rreq);
            }
        } while (total < length);

        return UCS_OK;
    }

    void UCS_F_ALWAYS_INLINE send_started()
    {
        ++m_sends_outstanding;
    }

    void UCS_F_ALWAYS_INLINE recv_started()
    {
        ++m_recvs_outstanding;
    }

    void UCS_F_ALWAYS_INLINE send_completed()
    {
        --m_sends_outstanding;
    }

    void UCS_F_ALWAYS_INLINE recv_completed()
    {
        --m_recvs_outstanding;
    }

    ucx_perf_context_t  &m_perf;
    unsigned            m_recvs_outstanding;
    unsigned            m_sends_outstanding;
    const unsigned      m_max_outstanding;
    /*
     * These fields are used by UCP AM flow only, because receive operation is
     * initiated from the data receive callback.
     */
    void                *m_am_rx_buffer;
    size_t              m_am_rx_length;
    ucp_request_param_t m_am_rx_params;
    ucp_request_param_t m_send_params;
    ucp_atomic_op_t     m_atomic_op;
};


#define TEST_CASE(_perf, _cmd, _type, _flags, _mask) \
    if (((_perf)->params.command == (_cmd)) && \
        ((_perf)->params.test_type == (_type)) && \
        (((_perf)->params.flags & (_mask)) == (_flags))) \
    { \
        ucp_perf_test_runner<_cmd, _type, _flags> r(*_perf); \
        return r.run(); \
    }

#define TEST_CASE_ALL_STREAM(_perf, _case) \
    TEST_CASE(_perf, UCS_PP_TUPLE_0 _case, UCS_PP_TUPLE_1 _case, \
              0, \
              UCX_PERF_TEST_FLAG_STREAM_RECV_DATA) \
    TEST_CASE(_perf, UCS_PP_TUPLE_0 _case, UCS_PP_TUPLE_1 _case, \
              UCX_PERF_TEST_FLAG_STREAM_RECV_DATA, \
              UCX_PERF_TEST_FLAG_STREAM_RECV_DATA)

#define TEST_CASE_ALL_TAG(_perf, _case) \
    TEST_CASE(_perf, UCS_PP_TUPLE_0 _case, UCS_PP_TUPLE_1 _case, \
              0, \
              UCX_PERF_TEST_FLAG_TAG_WILDCARD|UCX_PERF_TEST_FLAG_TAG_UNEXP_PROBE) \
    TEST_CASE(_perf, UCS_PP_TUPLE_0 _case, UCS_PP_TUPLE_1 _case, \
              UCX_PERF_TEST_FLAG_TAG_WILDCARD, \
              UCX_PERF_TEST_FLAG_TAG_WILDCARD|UCX_PERF_TEST_FLAG_TAG_UNEXP_PROBE) \
    TEST_CASE(_perf, UCS_PP_TUPLE_0 _case, UCS_PP_TUPLE_1 _case, \
              UCX_PERF_TEST_FLAG_TAG_UNEXP_PROBE, \
              UCX_PERF_TEST_FLAG_TAG_WILDCARD|UCX_PERF_TEST_FLAG_TAG_UNEXP_PROBE) \
    TEST_CASE(_perf, UCS_PP_TUPLE_0 _case, UCS_PP_TUPLE_1 _case, \
              UCX_PERF_TEST_FLAG_TAG_WILDCARD|UCX_PERF_TEST_FLAG_TAG_UNEXP_PROBE, \
              UCX_PERF_TEST_FLAG_TAG_WILDCARD|UCX_PERF_TEST_FLAG_TAG_UNEXP_PROBE)

#define TEST_CASE_ALL_OSD(_perf, _case) \
    TEST_CASE(_perf, UCS_PP_TUPLE_0 _case, UCS_PP_TUPLE_1 _case, \
              0, UCX_PERF_TEST_FLAG_ONE_SIDED) \
    TEST_CASE(_perf, UCS_PP_TUPLE_0 _case, UCS_PP_TUPLE_1 _case, \
              UCX_PERF_TEST_FLAG_ONE_SIDED, UCX_PERF_TEST_FLAG_ONE_SIDED)

#define TEST_CASE_ALL_AM(_perf, _case) \
    TEST_CASE(_perf, UCS_PP_TUPLE_0 _case, UCS_PP_TUPLE_1 _case, 0, 0)

ucs_status_t ucp_perf_test_dispatch(ucx_perf_context_t *perf)
{
    UCS_PP_FOREACH(TEST_CASE_ALL_OSD, perf,
        (UCX_PERF_CMD_PUT,   UCX_PERF_TEST_TYPE_PINGPONG),
        (UCX_PERF_CMD_PUT,   UCX_PERF_TEST_TYPE_PINGPONG_WAIT_MEM),
        (UCX_PERF_CMD_PUT,   UCX_PERF_TEST_TYPE_STREAM_UNI),
        (UCX_PERF_CMD_GET,   UCX_PERF_TEST_TYPE_STREAM_UNI),
        (UCX_PERF_CMD_ADD,   UCX_PERF_TEST_TYPE_STREAM_UNI),
        (UCX_PERF_CMD_FADD,  UCX_PERF_TEST_TYPE_STREAM_UNI),
        (UCX_PERF_CMD_SWAP,  UCX_PERF_TEST_TYPE_STREAM_UNI),
        (UCX_PERF_CMD_CSWAP, UCX_PERF_TEST_TYPE_STREAM_UNI)
        );

    UCS_PP_FOREACH(TEST_CASE_ALL_TAG, perf,
        (UCX_PERF_CMD_TAG,      UCX_PERF_TEST_TYPE_PINGPONG),
        (UCX_PERF_CMD_TAG,      UCX_PERF_TEST_TYPE_STREAM_UNI),
        (UCX_PERF_CMD_TAG_SYNC, UCX_PERF_TEST_TYPE_PINGPONG),
        (UCX_PERF_CMD_TAG_SYNC, UCX_PERF_TEST_TYPE_STREAM_UNI)
        );

    UCS_PP_FOREACH(TEST_CASE_ALL_STREAM, perf,
        (UCX_PERF_CMD_STREAM,   UCX_PERF_TEST_TYPE_STREAM_UNI),
        (UCX_PERF_CMD_STREAM,   UCX_PERF_TEST_TYPE_PINGPONG)
        );

    UCS_PP_FOREACH(TEST_CASE_ALL_AM, perf,
        (UCX_PERF_CMD_AM,       UCX_PERF_TEST_TYPE_PINGPONG),
        (UCX_PERF_CMD_AM,       UCX_PERF_TEST_TYPE_STREAM_UNI)
        );

    ucs_error("Invalid test case: %d/%d/0x%x",
              perf->params.command, perf->params.test_type,
              perf->params.flags);
    return UCS_ERR_INVALID_PARAM;
}
