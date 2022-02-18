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
#include <ucs/sys/string.h>
#include <limits>


template <ucx_perf_cmd_t CMD, ucx_perf_test_type_t TYPE, unsigned FLAGS>
class ucp_perf_test_runner {
public:
    typedef uint8_t psn_t;

    static const unsigned AM_ID     = 1;
    static const ucp_tag_t TAG      = 0x1337a880u;
    static const ucp_tag_t TAG_MASK = (FLAGS & UCX_PERF_TEST_FLAG_TAG_WILDCARD) ?
                                      0 : (ucp_tag_t)-1;
    static const psn_t INITIAL_SN   = 0;
    static const psn_t LAST_ITER_SN = 1;
    static const psn_t UNKNOWN_SN   = std::numeric_limits<psn_t>::max();

    ucp_perf_test_runner(ucx_perf_context_t &perf) :
        m_perf(perf),
        m_recvs_outstanding(0),
        m_sends_outstanding(0),
        m_max_outstanding(m_perf.params.max_outstanding),
        m_am_rx_buffer(NULL),
        m_am_rx_length(0ul)

    {
        memset(&m_am_rx_params, 0, sizeof(m_am_rx_params));
        memset(&m_send_params, 0, sizeof(m_send_params));
        memset(&m_send_get_info_params, 0, sizeof(m_send_get_info_params));
        memset(&m_recv_params, 0, sizeof(m_recv_params));

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
        if (is_atomic()) {
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
            fill_prereg_params(m_am_rx_params, m_perf.ucp.recv_memh);
        }

        fill_send_params(m_send_params, *send_buffer, *send_dt, send_cb, 0);
        fill_send_params(m_send_get_info_params, *send_buffer, *send_dt,
                         send_get_info_cb, UCP_OP_ATTR_FLAG_NO_IMM_CMPL);

        m_recv_params.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE |
                                     UCP_OP_ATTR_FIELD_CALLBACK |
                                     UCP_OP_ATTR_FIELD_USER_DATA;
        m_recv_params.datatype     = *recv_dt;
        m_recv_params.cb.recv      = tag_recv_cb;
        m_recv_params.user_data    = this;
        fill_prereg_params(m_recv_params, m_perf.ucp.recv_memh);
    }

    void fill_send_params(ucp_request_param_t &params, void *reply_buffer,
                          ucp_datatype_t send_dt, ucp_send_nbx_callback_t cb,
                          uint32_t op_attr_mask)
    {
        params.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE |
                              UCP_OP_ATTR_FIELD_CALLBACK |
                              UCP_OP_ATTR_FIELD_USER_DATA | op_attr_mask;
        params.datatype     = send_dt;
        params.cb.send      = cb;
        params.user_data    = this;

        if (TYPE == UCX_PERF_TEST_TYPE_STREAM_UNI) {
            params.op_attr_mask |= UCP_OP_ATTR_FLAG_MULTI_SEND;
        }

        if ((CMD == UCX_PERF_CMD_FADD) || (CMD == UCX_PERF_CMD_SWAP) ||
            (CMD == UCX_PERF_CMD_CSWAP)) {
            params.op_attr_mask |= UCP_OP_ATTR_FIELD_REPLY_BUFFER;
            params.reply_buffer  = reply_buffer;
        }

        fill_prereg_params(params, m_perf.ucp.send_memh);
    }

    void fill_prereg_params(ucp_request_param_t &params, ucp_mem_h memh)
    {
        if (m_perf.params.flags & UCX_PERF_TEST_FLAG_PREREG) {
            params.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMH;
            params.memh          = memh;
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

    static void send_cb(void *request, ucs_status_t status, void *user_data)
    {
        ucp_perf_test_runner *test = (ucp_perf_test_runner*)user_data;
        test->send_completed();
        ucp_request_free(request);
    }

    static void
    send_get_info_cb(void *request, ucs_status_t status, void *user_data)
    {
        ucp_perf_test_runner *test = (ucp_perf_test_runner*)user_data;
        test->send_completed();
    }

    static void tag_recv_cb(void *request, ucs_status_t status,
                            const ucp_tag_recv_info_t *info, void *user_data)
    {
        ucp_perf_test_runner *test = (ucp_perf_test_runner*)user_data;
        test->recv_completed();
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

    void UCS_F_ALWAYS_INLINE wait_send_window(unsigned n)
    {
        ucs_assert(m_sends_outstanding >= 0);
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

    UCS_F_ALWAYS_INLINE void *sn_ptr(void *buffer, size_t length)
    {
        return UCS_PTR_BYTE_OFFSET(buffer, length - sizeof(psn_t));
    }

    UCS_F_ALWAYS_INLINE psn_t read_sn(void *buffer, size_t length)
    {
        ucs_memory_type_t mem_type = m_perf.params.recv_mem_type;
        const void *ptr            = sn_ptr(buffer, length);
        psn_t sn;

        if (mem_type == UCS_MEMORY_TYPE_HOST) {
            return *(const volatile psn_t*)ptr;
        } else {
            m_perf.recv_allocator->memcpy(&sn, UCS_MEMORY_TYPE_HOST, ptr,
                                          mem_type, sizeof(sn));
            return sn;
        }
    }

    UCS_F_ALWAYS_INLINE void
    write_sn(void *buffer, ucs_memory_type_t mem_type,
             size_t length, psn_t sn,
             const ucx_perf_allocator_t *allocator)
    {
        void *ptr = sn_ptr(buffer, length);

        if (mem_type == UCS_MEMORY_TYPE_HOST) {
            *(volatile psn_t*)ptr = sn;
        } else {
            allocator->memcpy(ptr, mem_type, &sn, UCS_MEMORY_TYPE_HOST,
                              sizeof(sn));
        }
    }

    ucs_status_t UCS_F_ALWAYS_INLINE
    send(ucp_ep_h ep, void *buffer, unsigned length, ucp_datatype_t datatype,
         psn_t sn, uint64_t remote_addr, ucp_rkey_h rkey, bool get_info = false)
    {
        ucp_request_param_t *param = get_info ? &m_send_get_info_params :
                                                &m_send_params;
        uint64_t value             = 0;
        void *request;

        wait_send_window(1);

        /* coverity[switch_selector_expr_is_constant] */
        switch (CMD) {
        case UCX_PERF_CMD_TAG:
        case UCX_PERF_CMD_TAG_SYNC:
        case UCX_PERF_CMD_STREAM:
        case UCX_PERF_CMD_AM:
            /* coverity[switch_selector_expr_is_constant] */
            switch (CMD) {
            case UCX_PERF_CMD_TAG:
                request = ucp_tag_send_nbx(ep, buffer, length, TAG, param);
                break;
            case UCX_PERF_CMD_TAG_SYNC:
                request = ucp_tag_send_sync_nbx(ep, buffer, length, TAG, param);
                break;
            case UCX_PERF_CMD_STREAM:
                request = ucp_stream_send_nbx(ep, buffer, length, param);
                break;
            case UCX_PERF_CMD_AM:
                request = ucp_am_send_nbx(ep, AM_ID, m_perf.ucp.am_hdr,
                                          m_perf.params.ucp.am_hdr_size, buffer,
                                          length, param);
                break;
            default:
                return UCS_ERR_INVALID_PARAM;
            }
            break;
        case UCX_PERF_CMD_PUT:
            /* coverity[switch_selector_expr_is_constant] */
            switch (TYPE) {
            case UCX_PERF_TEST_TYPE_PINGPONG:
            case UCX_PERF_TEST_TYPE_PINGPONG_WAIT_MEM:
                write_sn(buffer, m_perf.params.send_mem_type, length, sn,
                         m_perf.send_allocator);
                break;
            case UCX_PERF_TEST_TYPE_STREAM_UNI:
                break;
            default:
                return UCS_ERR_INVALID_PARAM;
            }
            request = ucp_put_nbx(ep, buffer, length, remote_addr, rkey, param);
            break;
        case UCX_PERF_CMD_GET:
            request = ucp_get_nbx(ep, buffer, length, remote_addr, rkey, param);
            break;
        case UCX_PERF_CMD_ADD:
        case UCX_PERF_CMD_FADD:
        case UCX_PERF_CMD_SWAP:
        case UCX_PERF_CMD_CSWAP:
            request = ucp_atomic_op_nbx(ep, m_atomic_op, &value, 1, remote_addr,
                                        rkey, param);
            break;
        default:
            return UCS_ERR_INVALID_PARAM;
        }

        if (!UCS_PTR_IS_PTR(request)) {
            /* coverity[overflow] */
            return UCS_PTR_STATUS(request);
        }

        if (get_info) {
            get_request_info(request);
            ucp_request_release(request);
        }

        send_started();
        return UCS_OK;
    }

    ucs_status_t UCS_F_ALWAYS_INLINE
    recv(ucp_worker_h worker, ucp_ep_h ep, void *buffer, unsigned length,
         ucp_datatype_t datatype, psn_t sn)
    {
        void *request;
        void *ptr;

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
            request = ucp_tag_recv_nbx(worker, buffer, length, TAG, TAG_MASK,
                                       &m_recv_params);
            if (ucs_likely(!UCS_PTR_IS_PTR(request))) {
                return UCS_PTR_STATUS(request);
            }
            recv_started();
            return UCS_OK;
        case UCX_PERF_CMD_AM:
            recv_started();
            return UCS_OK;
        case UCX_PERF_CMD_PUT:
            /* coverity[switch_selector_expr_is_constant] */
            switch (TYPE) {
            case UCX_PERF_TEST_TYPE_PINGPONG:
                while (read_sn(buffer, length) != sn) {
                    progress_responder();
                }
                return UCS_OK;
            case UCX_PERF_TEST_TYPE_PINGPONG_WAIT_MEM:
                ptr = sn_ptr(buffer, length);
                while (read_sn(buffer, length) != sn) {
                    ucp_worker_wait_mem(worker, ptr);
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
    void wait_last_iter(void *buffer, size_t size)
    {
        if (use_psn()) {
            while (read_sn(buffer, size) != LAST_ITER_SN) {
                progress_responder();
            }
        }
    }

    /* send the special flag as a last iteration in case of
     * unidirectional PUT test, since responder is waiting for
     * this message */
    void send_last_iter(ucp_ep_h ep, void *buffer, size_t size,
                        uint64_t remote_addr, ucp_rkey_h rkey)
    {
        uint64_t atomic_value = 0;
        ucs_status_ptr_t status_p;
        ucp_request_param_t atomic_param;

        if (use_psn()) {
            fence();
        }

        /* Make sure that doing the last opetarion will write 1 to the end of
           the remote buffer */
        if (CMD == UCX_PERF_CMD_PUT) {
            write_sn(buffer, m_perf.params.send_mem_type, size, LAST_ITER_SN,
                     m_perf.send_allocator);
        } else if (is_atomic()) {
            atomic_value = 0;
            write_sn(&atomic_value, UCS_MEMORY_TYPE_HOST, size, LAST_ITER_SN,
                     NULL);
            atomic_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE |
                                        UCP_OP_ATTR_FIELD_CALLBACK |
                                        UCP_OP_ATTR_FIELD_USER_DATA;
            atomic_param.datatype     = ucp_dt_make_contig(size);
            atomic_param.cb.send      = send_cb;
            atomic_param.user_data    = this;
        }

        /* coverity[switch_selector_expr_is_constant] */
        switch (CMD) {
        case UCX_PERF_CMD_PUT:
            status_p = ucp_put_nbx(ep, buffer, size, remote_addr, rkey,
                                   &m_send_params);
            break;
        case UCX_PERF_CMD_ADD:
            status_p = ucp_atomic_op_nbx(ep, m_atomic_op, &atomic_value, 1,
                                         remote_addr, rkey, &atomic_param);
            break;
        case UCX_PERF_CMD_FADD:
        case UCX_PERF_CMD_SWAP:
            /* Atomic argument to add/swap with contains LAST_ITER_SN */
            atomic_param.op_attr_mask |= UCP_OP_ATTR_FIELD_REPLY_BUFFER;
            atomic_param.reply_buffer  = buffer;
            status_p = ucp_atomic_op_nbx(ep, m_atomic_op, &atomic_value, 1,
                                         remote_addr, rkey, &atomic_param);
            break;
        case UCX_PERF_CMD_CSWAP:
            /* Buffer to swap with contains LAST_ITER_SN */
            atomic_param.op_attr_mask |= UCP_OP_ATTR_FIELD_REPLY_BUFFER;
            atomic_param.reply_buffer  = &atomic_value;
            status_p = ucp_atomic_op_nbx(ep, m_atomic_op, buffer, 1,
                                         remote_addr, rkey, &atomic_param);
            break;
        default:
            status_p = NULL;
            break;
        }

        if (UCS_PTR_IS_PTR(status_p)) {
            send_started();
        }

        wait_send_window(m_max_outstanding);
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

    inline bool is_atomic() const
    {
        return (CMD == UCX_PERF_CMD_ADD) || (CMD == UCX_PERF_CMD_FADD) ||
               (CMD == UCX_PERF_CMD_SWAP) || (CMD == UCX_PERF_CMD_CSWAP);
    }

    inline bool use_psn() const
    {
        return (CMD == UCX_PERF_CMD_PUT) || is_atomic();
    }

    void reset_buffers(size_t length, psn_t sn)
    {
        if (!use_psn()) {
            return;
        }

        write_sn(m_perf.send_buffer, m_perf.params.send_mem_type, length, sn,
                 m_perf.send_allocator);
        write_sn(m_perf.recv_buffer, m_perf.params.recv_mem_type, length, sn,
                 m_perf.recv_allocator);
    }

    ucs_status_t run_pingpong()
    {
        unsigned my_index;
        ucp_worker_h worker;
        ucp_ep_h ep;
        void *send_buffer, *recv_buffer;
        ucp_datatype_t send_datatype, recv_datatype;
        uint64_t remote_addr;
        ucp_rkey_h rkey;
        size_t length, send_length, recv_length;
        psn_t sn;

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

        reset_buffers(length, UNKNOWN_SN);

        ucp_perf_barrier(&m_perf);

        my_index = rte_call(&m_perf, group_index);

        ucx_perf_test_start_clock(&m_perf);

        ucx_perf_omp_barrier(&m_perf);

        if (my_index == 0) {
            UCX_PERF_TEST_FOREACH(&m_perf) {
                send(ep, send_buffer, send_length, send_datatype, sn, remote_addr, rkey);
                recv(worker, ep, recv_buffer, recv_length, recv_datatype, sn);
                wait_recv_window(m_max_outstanding);
                ucx_perf_update(&m_perf, 1, length);
                ++sn;
            }
        } else if (my_index == 1) {
            UCX_PERF_TEST_FOREACH(&m_perf) {
                recv(worker, ep, recv_buffer, recv_length, recv_datatype, sn);
                wait_recv_window(m_max_outstanding);
                send(ep, send_buffer, send_length, send_datatype, sn,
                     remote_addr, rkey, m_perf.current.iters == 0);
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
        psn_t sn;

        send_buffer = m_perf.send_buffer;
        recv_buffer = m_perf.recv_buffer;
        worker      = m_perf.ucp.worker;
        ep          = m_perf.ucp.ep;
        remote_addr = m_perf.ucp.remote_addr;
        rkey        = m_perf.ucp.rkey;
        sn          = INITIAL_SN;

        ucp_perf_init_common_params(&length, &send_length, &send_datatype,
                                    &send_buffer, &recv_length, &recv_datatype,
                                    &recv_buffer);

        reset_buffers(send_length, sn);

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

            wait_last_iter(recv_buffer, send_length);
            wait_recv_window(m_max_outstanding);
        } else if (my_index == 1) {
            UCX_PERF_TEST_FOREACH(&m_perf) {
                send(ep, send_buffer, send_length, send_datatype, sn,
                     remote_addr, rkey, m_perf.current.iters == 0);
                ucx_perf_update(&m_perf, 1, length);
                ++sn;
            }

            send_last_iter(ep, send_buffer, send_length, remote_addr, rkey);
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

    void get_request_info(void *request)
    {
        ucp_request_attr_t request_attr;
        ucs_status_t status;

        request_attr.field_mask        = UCP_REQUEST_ATTR_FIELD_INFO_STRING |
                                         UCP_REQUEST_ATTR_FIELD_INFO_STRING_SIZE;
        request_attr.debug_string      = m_perf.extra_info;
        request_attr.debug_string_size = sizeof(m_perf.extra_info);

        status = ucp_request_query(request, &request_attr);
        if (status != UCS_OK) {
            ucs_snprintf_safe(m_perf.extra_info, sizeof(m_perf.extra_info),
                              "<failed to query: %s>",
                              ucs_status_string(status));
        }
    }

    ucx_perf_context_t &m_perf;
    int                m_recvs_outstanding;
    int                m_sends_outstanding;
    const int          m_max_outstanding;
    /*
     * These fields are used by UCP AM flow only, because receive operation is
     * initiated from the data receive callback.
     */
    void                *m_am_rx_buffer;
    size_t              m_am_rx_length;
    ucp_request_param_t m_am_rx_params;
    ucp_request_param_t m_send_params;
    ucp_request_param_t m_send_get_info_params;
    ucp_request_param_t m_recv_params;
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
