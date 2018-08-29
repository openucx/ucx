/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) The University of Tennessee and The University
*               of Tennessee Research Foundation. 2015-2016. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2017.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "libperf_int.h"

#include <ucs/debug/log.h>
#include <ucs/arch/bitops.h>
#include <string.h>
#include <malloc.h>
#include <unistd.h>

#define ATOMIC_OP_CONFIG(_size, _op32, _op64, _op, _msg, _params, _status)        \
    _status = __get_atomic_flag((_size), (_op32), (_op64), (_op));                \
    if (_status != UCS_OK) {                                                      \
        ucs_error("%s/%s does not support atomic %s for message size %zu bytes",  \
                  (_params)->uct.tl_name, (_params)->uct.dev_name,                \
                  (_msg)[_op], (_size));                                          \
        return _status;                                                           \
    }

#define ATOMIC_OP_CHECK(_size, _attr, _required, _params, _msg)                   \
    if (!ucs_test_all_flags(_attr, _required)) {                                  \
        if ((_params)->flags & UCX_PERF_TEST_FLAG_VERBOSE) {                      \
            ucs_error("%s/%s does not support required "#_size"-bit atomic: %s",  \
                      (_params)->uct.tl_name, (_params)->uct.dev_name,            \
                      (_msg)[ucs_ffs64(~(_attr) & (_required))]);                 \
        }                                                                         \
        return UCS_ERR_UNSUPPORTED;                                               \
    }

typedef struct {
    union {
        struct {
            size_t     dev_addr_len;
            size_t     iface_addr_len;
            size_t     ep_addr_len;
        } uct;
        struct {
            size_t     addr_len;
        } ucp;
    };
    size_t             rkey_size;
    unsigned long      recv_buffer;
} ucx_perf_ep_info_t;

static const char *perf_iface_ops[] = {
    [ucs_ilog2(UCT_IFACE_FLAG_AM_SHORT)]         = "am short",
    [ucs_ilog2(UCT_IFACE_FLAG_AM_BCOPY)]         = "am bcopy",
    [ucs_ilog2(UCT_IFACE_FLAG_AM_ZCOPY)]         = "am zcopy",
    [ucs_ilog2(UCT_IFACE_FLAG_PUT_SHORT)]        = "put short",
    [ucs_ilog2(UCT_IFACE_FLAG_PUT_BCOPY)]        = "put bcopy",
    [ucs_ilog2(UCT_IFACE_FLAG_PUT_ZCOPY)]        = "put zcopy",
    [ucs_ilog2(UCT_IFACE_FLAG_GET_SHORT)]        = "get short",
    [ucs_ilog2(UCT_IFACE_FLAG_GET_BCOPY)]        = "get bcopy",
    [ucs_ilog2(UCT_IFACE_FLAG_GET_ZCOPY)]        = "get zcopy",
    [ucs_ilog2(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE)] = "peer failure handler",
    [ucs_ilog2(UCT_IFACE_FLAG_CONNECT_TO_IFACE)] = "connect to iface",
    [ucs_ilog2(UCT_IFACE_FLAG_CONNECT_TO_EP)]    = "connect to ep",
    [ucs_ilog2(UCT_IFACE_FLAG_AM_DUP)]           = "full reliability",
    [ucs_ilog2(UCT_IFACE_FLAG_CB_SYNC)]          = "sync callback",
    [ucs_ilog2(UCT_IFACE_FLAG_CB_ASYNC)]         = "async callback",
    [ucs_ilog2(UCT_IFACE_FLAG_EVENT_SEND_COMP)]  = "send completion event",
    [ucs_ilog2(UCT_IFACE_FLAG_EVENT_RECV)]       = "tag or active message event",
    [ucs_ilog2(UCT_IFACE_FLAG_EVENT_RECV_SIG)]   = "signaled message event",
    [ucs_ilog2(UCT_IFACE_FLAG_PENDING)]          = "pending",
    [ucs_ilog2(UCT_IFACE_FLAG_TAG_EAGER_SHORT)]  = "tag eager short",
    [ucs_ilog2(UCT_IFACE_FLAG_TAG_EAGER_BCOPY)]  = "tag eager bcopy",
    [ucs_ilog2(UCT_IFACE_FLAG_TAG_EAGER_ZCOPY)]  = "tag eager zcopy",
    [ucs_ilog2(UCT_IFACE_FLAG_TAG_RNDV_ZCOPY)]   = "tag rndv zcopy"
};

static const char *perf_atomic_op[] = {
     [UCT_ATOMIC_OP_ADD]   = "add",
     [UCT_ATOMIC_OP_AND]   = "and",
     [UCT_ATOMIC_OP_OR]    = "or" ,
     [UCT_ATOMIC_OP_XOR]   = "xor"
};

static const char *perf_atomic_fop[] = {
     [UCT_ATOMIC_OP_ADD]   = "fetch-add",
     [UCT_ATOMIC_OP_AND]   = "fetch-and",
     [UCT_ATOMIC_OP_OR]    = "fetch-or",
     [UCT_ATOMIC_OP_XOR]   = "fetch-xor",
     [UCT_ATOMIC_OP_SWAP]  = "swap",
     [UCT_ATOMIC_OP_CSWAP] = "cswap"
};

/*
 *  This Quickselect routine is based on the algorithm described in
 *  "Numerical recipes in C", Second Edition,
 *  Cambridge University Press, 1992, Section 8.5, ISBN 0-521-43108-5
 *  This code by Nicolas Devillard - 1998. Public domain.
 */
static ucs_time_t __find_median_quick_select(ucs_time_t arr[], int n)
{
    int low, high ;
    int median;
    int middle, ll, hh;

#define ELEM_SWAP(a,b) { register ucs_time_t t=(a);(a)=(b);(b)=t; }

    low = 0 ; high = n-1 ; median = (low + high) / 2;
    for (;;) {
        if (high <= low) /* One element only */
            return arr[median] ;

        if (high == low + 1) {  /* Two elements only */
            if (arr[low] > arr[high])
                ELEM_SWAP(arr[low], arr[high]) ;
            return arr[median] ;
        }

        /* Find median of low, middle and high items; swap into position low */
        middle = (low + high) / 2;
        if (arr[middle] > arr[high])    ELEM_SWAP(arr[middle], arr[high]) ;
        if (arr[low] > arr[high])       ELEM_SWAP(arr[low], arr[high]) ;
        if (arr[middle] > arr[low])     ELEM_SWAP(arr[middle], arr[low]) ;

        /* Swap low item (now in position middle) into position (low+1) */
        ELEM_SWAP(arr[middle], arr[low+1]) ;

        /* Nibble from each end towards middle, swapping items when stuck */
        ll = low + 1;
        hh = high;
        for (;;) {
            do ll++; while (arr[low] > arr[ll]) ;
            do hh--; while (arr[hh]  > arr[low]) ;

            if (hh < ll)
                break;

            ELEM_SWAP(arr[ll], arr[hh]) ;
        }

        /* Swap middle item (in position low) back into correct position */
        ELEM_SWAP(arr[low], arr[hh]) ;

        /* Re-set active partition */
        if (hh <= median)
            low = ll;
        if (hh >= median)
            high = hh - 1;
    }
}

static ucs_status_t uct_perf_test_alloc_mem(ucx_perf_context_t *perf,
                                            ucx_perf_params_t *params)
{
    ucs_status_t status;
    unsigned flags;
    size_t buffer_size;

    if ((UCT_PERF_DATA_LAYOUT_ZCOPY == params->uct.data_layout) && params->iov_stride) {
        buffer_size = params->msg_size_cnt * params->iov_stride;
    } else {
        buffer_size = ucx_perf_get_message_size(params);
    }

    /* TODO use params->alignment  */

    flags = (params->flags & UCX_PERF_TEST_FLAG_MAP_NONBLOCK) ?
                    UCT_MD_MEM_FLAG_NONBLOCK : 0;
    flags |= UCT_MD_MEM_ACCESS_ALL;

    /* Allocate send buffer memory */
    status = uct_iface_mem_alloc(perf->uct.iface, 
                                 buffer_size * params->thread_count,
                                 flags, "perftest", &perf->uct.send_mem);
    if (status != UCS_OK) {
        ucs_error("Failed allocate send buffer: %s", ucs_status_string(status));
        goto err;
    }

    ucs_assert(perf->uct.send_mem.md == perf->uct.md);
    perf->send_buffer = perf->uct.send_mem.address;

    /* Allocate receive buffer memory */
    status = uct_iface_mem_alloc(perf->uct.iface, 
                                 buffer_size * params->thread_count,
                                 flags, "perftest", &perf->uct.recv_mem);
    if (status != UCS_OK) {
        ucs_error("Failed allocate receive buffer: %s", ucs_status_string(status));
        goto err_free_send;
    }

    ucs_assert(perf->uct.recv_mem.md == perf->uct.md);
    perf->recv_buffer = perf->uct.recv_mem.address;

    /* Allocate IOV datatype memory */
    perf->params.msg_size_cnt = params->msg_size_cnt;
    perf->uct.iov             = malloc(sizeof(*perf->uct.iov) *
                                       perf->params.msg_size_cnt *
                                       params->thread_count);
    if (NULL == perf->uct.iov) {
        status = UCS_ERR_NO_MEMORY;
        ucs_error("Failed allocate send IOV(%lu) buffer: %s",
                  perf->params.msg_size_cnt, ucs_status_string(status));
        goto err_free_send;
    }

    perf->offset = 0;

    ucs_debug("allocated memory. Send buffer %p, Recv buffer %p",
              perf->send_buffer, perf->recv_buffer);
    return UCS_OK;

err_free_send:
    uct_iface_mem_free(&perf->uct.send_mem);
err:
    return status;
}

static void uct_perf_test_free_mem(ucx_perf_context_t *perf)
{
    uct_iface_mem_free(&perf->uct.send_mem);
    uct_iface_mem_free(&perf->uct.recv_mem);
    free(perf->uct.iov);
}

void ucx_perf_test_start_clock(ucx_perf_context_t *perf)
{
    perf->start_time        = ucs_get_time();
    perf->prev_time         = perf->start_time;
    perf->prev.time         = perf->start_time;
}

static void ucx_perf_test_reset(ucx_perf_context_t *perf,
                                ucx_perf_params_t *params)
{
    unsigned i;

    perf->params            = *params;
    perf->start_time        = ucs_get_time();
    perf->prev_time         = perf->start_time;
    perf->end_time          = (perf->params.max_time == 0.0) ? UINT64_MAX :
                               ucs_time_from_sec(perf->params.max_time) + perf->start_time;
    perf->max_iter          = (perf->params.max_iter == 0) ? UINT64_MAX :
                               perf->params.max_iter;
    perf->report_interval   = ucs_time_from_sec(perf->params.report_interval);
    perf->current.time      = 0;
    perf->current.msgs      = 0;
    perf->current.bytes     = 0;
    perf->current.iters     = 0;
    perf->prev.time         = perf->start_time;
    perf->prev.msgs         = 0;
    perf->prev.bytes        = 0;
    perf->prev.iters        = 0;
    perf->timing_queue_head = 0;
    perf->offset            = 0;
    for (i = 0; i < TIMING_QUEUE_SIZE; ++i) {
        perf->timing_queue[i] = 0;
    }
}

void ucx_perf_calc_result(ucx_perf_context_t *perf, ucx_perf_result_t *result)
{
    double factor;
    double sec_value;

    sec_value = ucs_time_from_sec(1.0);
    if (perf->params.test_type == UCX_PERF_TEST_TYPE_PINGPONG) {
        factor = 2.0;
    } else {
        factor = 1.0;
    }

    result->iters = perf->current.iters;
    result->bytes = perf->current.bytes;
    result->elapsed_time = perf->current.time - perf->start_time;

    /* Latency */

    result->latency.typical =
        __find_median_quick_select(perf->timing_queue, TIMING_QUEUE_SIZE)
        / sec_value
        / factor;

    result->latency.moment_average =
        (double)(perf->current.time - perf->prev.time)
        / (perf->current.iters - perf->prev.iters)
        / sec_value
        / factor;

    result->latency.total_average =
        (double)(perf->current.time - perf->start_time)
        / perf->current.iters
        / sec_value
        / factor;


    /* Bandwidth */

    result->bandwidth.typical = 0.0; // Undefined

    result->bandwidth.moment_average =
        (perf->current.bytes - perf->prev.bytes) * sec_value
        / (double)(perf->current.time - perf->prev.time) * factor;

    result->bandwidth.total_average =
        perf->current.bytes * sec_value
        / (double)(perf->current.time - perf->start_time) * factor;


    /* Packet rate */

    result->msgrate.typical = 0.0; // Undefined

    result->msgrate.moment_average =
        (perf->current.msgs - perf->prev.msgs) * sec_value
        / (double)(perf->current.time - perf->prev.time) * factor;

    result->msgrate.total_average =
        perf->current.msgs * sec_value
        / (double)(perf->current.time - perf->start_time) * factor;

}

static ucs_status_t ucx_perf_test_check_params(ucx_perf_params_t *params)
{
    size_t it;

    if (ucx_perf_get_message_size(params) < 1) {
        if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
            ucs_error("Message size too small, need to be at least 1");
        }
        return UCS_ERR_INVALID_PARAM;
    }

    if (params->max_outstanding < 1) {
        if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
            ucs_error("max_outstanding, need to be at least 1");
        }
        return UCS_ERR_INVALID_PARAM;
    }

    /* check if particular message size fit into stride size */
    if (params->iov_stride) {
        for (it = 0; it < params->msg_size_cnt; ++it) {
            if (params->msg_size_list[it] > params->iov_stride) {
                if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
                    ucs_error("Buffer size %lu bigger than stride %lu",
                              params->msg_size_list[it], params->iov_stride);
                }
                return UCS_ERR_INVALID_PARAM;
            }
        }
    }

    return UCS_OK;
}

void uct_perf_iface_flush_b(ucx_perf_context_t *perf)
{
    ucs_status_t status;

    do {
        status = uct_iface_flush(perf->uct.iface, 0, NULL);
        uct_worker_progress(perf->uct.worker);
    } while (status == UCS_INPROGRESS);
}

static inline uint64_t __get_flag(uct_perf_data_layout_t layout, uint64_t short_f,
                                  uint64_t bcopy_f, uint64_t zcopy_f)
{
    return (layout == UCT_PERF_DATA_LAYOUT_SHORT) ? short_f :
           (layout == UCT_PERF_DATA_LAYOUT_BCOPY) ? bcopy_f :
           (layout == UCT_PERF_DATA_LAYOUT_ZCOPY) ? zcopy_f :
           0;
}

static inline ucs_status_t __get_atomic_flag(size_t size, uint64_t *op32,
                                             uint64_t *op64, uint64_t op)
{
    if (size == sizeof(uint32_t)) {
        *op32 = UCS_BIT(op);
        return UCS_OK;
    } else if (size == sizeof(uint64_t)) {
        *op64 = UCS_BIT(op);
        return UCS_OK;
    }
    return UCS_ERR_UNSUPPORTED;
}

static inline size_t __get_max_size(uct_perf_data_layout_t layout, size_t short_m,
                                    size_t bcopy_m, uint64_t zcopy_m)
{
    return (layout == UCT_PERF_DATA_LAYOUT_SHORT) ? short_m :
           (layout == UCT_PERF_DATA_LAYOUT_BCOPY) ? bcopy_m :
           (layout == UCT_PERF_DATA_LAYOUT_ZCOPY) ? zcopy_m :
           0;
}

static ucs_status_t uct_perf_test_check_capabilities(ucx_perf_params_t *params,
                                                     uct_iface_h iface)
{
    uint64_t required_flags = 0;
    uint64_t atomic_op32    = 0;
    uint64_t atomic_op64    = 0;
    uint64_t atomic_fop32   = 0;
    uint64_t atomic_fop64   = 0;
    uct_iface_attr_t attr;
    ucs_status_t status;
    size_t min_size, max_size, max_iov, message_size;

    status = uct_iface_query(iface, &attr);
    if (status != UCS_OK) {
        return status;
    }

    min_size = 0;
    max_iov  = 1;
    message_size = ucx_perf_get_message_size(params);
    switch (params->command) {
    case UCX_PERF_CMD_AM:
        required_flags = __get_flag(params->uct.data_layout, UCT_IFACE_FLAG_AM_SHORT,
                                    UCT_IFACE_FLAG_AM_BCOPY, UCT_IFACE_FLAG_AM_ZCOPY);
        required_flags |= UCT_IFACE_FLAG_CB_SYNC;
        min_size = __get_max_size(params->uct.data_layout, 0, 0,
                                  attr.cap.am.min_zcopy);
        max_size = __get_max_size(params->uct.data_layout, attr.cap.am.max_short,
                                  attr.cap.am.max_bcopy, attr.cap.am.max_zcopy);
        max_iov  = attr.cap.am.max_iov;
        break;
    case UCX_PERF_CMD_PUT:
        required_flags = __get_flag(params->uct.data_layout, UCT_IFACE_FLAG_PUT_SHORT,
                                    UCT_IFACE_FLAG_PUT_BCOPY, UCT_IFACE_FLAG_PUT_ZCOPY);
        min_size = __get_max_size(params->uct.data_layout, 0, 0,
                                  attr.cap.put.min_zcopy);
        max_size = __get_max_size(params->uct.data_layout, attr.cap.put.max_short,
                                  attr.cap.put.max_bcopy, attr.cap.put.max_zcopy);
        max_iov  = attr.cap.put.max_iov;
        break;
    case UCX_PERF_CMD_GET:
        required_flags = __get_flag(params->uct.data_layout, UCT_IFACE_FLAG_GET_SHORT,
                                    UCT_IFACE_FLAG_GET_BCOPY, UCT_IFACE_FLAG_GET_ZCOPY);
        min_size = __get_max_size(params->uct.data_layout, 0, 0,
                                  attr.cap.get.min_zcopy);
        max_size = __get_max_size(params->uct.data_layout, attr.cap.get.max_short,
                                  attr.cap.get.max_bcopy, attr.cap.get.max_zcopy);
        max_iov  = attr.cap.get.max_iov;
        break;
    case UCX_PERF_CMD_ADD:
        ATOMIC_OP_CONFIG(message_size, &atomic_op32, &atomic_op64, UCT_ATOMIC_OP_ADD,
                         perf_atomic_op, params, status);
        max_size = 8;
        break;
    case UCX_PERF_CMD_FADD:
        ATOMIC_OP_CONFIG(message_size, &atomic_fop32, &atomic_fop64, UCT_ATOMIC_OP_ADD,
                         perf_atomic_fop, params, status);
        max_size = 8;
        break;
    case UCX_PERF_CMD_SWAP:
        ATOMIC_OP_CONFIG(message_size, &atomic_fop32, &atomic_fop64, UCT_ATOMIC_OP_SWAP,
                         perf_atomic_fop, params, status);
        max_size = 8;
        break;
    case UCX_PERF_CMD_CSWAP:
        ATOMIC_OP_CONFIG(message_size, &atomic_fop32, &atomic_fop64, UCT_ATOMIC_OP_CSWAP,
                         perf_atomic_fop, params, status);
        max_size = 8;
        break;
    default:
        if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
            ucs_error("Invalid test command");
        }
        return UCS_ERR_INVALID_PARAM;
    }

    status = ucx_perf_test_check_params(params);
    if (status != UCS_OK) {
        return status;
    }

    /* check atomics first */
    ATOMIC_OP_CHECK(32, attr.cap.atomic32.op_flags, atomic_op32, params, perf_atomic_op);
    ATOMIC_OP_CHECK(64, attr.cap.atomic64.op_flags, atomic_op64, params, perf_atomic_op);
    ATOMIC_OP_CHECK(32, attr.cap.atomic32.fop_flags, atomic_fop32, params, perf_atomic_fop);
    ATOMIC_OP_CHECK(64, attr.cap.atomic64.fop_flags, atomic_fop64, params, perf_atomic_fop);

    /* check iface flags */
    if (!(atomic_op32 | atomic_op64 | atomic_fop32 | atomic_fop64) &&
        (!ucs_test_all_flags(attr.cap.flags, required_flags) || !required_flags)) {
        if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
            ucs_error("%s/%s does not support operation %s",
                      params->uct.tl_name, params->uct.dev_name,
                      perf_iface_ops[ucs_ffs64(~attr.cap.flags & required_flags)]);
        }
        return UCS_ERR_UNSUPPORTED;
    }

    if (message_size < min_size) {
        if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
            ucs_error("Message size (%zu) is smaller than min supported (%zu)",
                      message_size, min_size);
        }
        return UCS_ERR_UNSUPPORTED;
    }

    if (message_size > max_size) {
        if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
            ucs_error("Message size (%zu) is larger than max supported (%zu)",
                      message_size, max_size);
        }
        return UCS_ERR_UNSUPPORTED;
    }

    if (params->command == UCX_PERF_CMD_AM) {
        if ((params->uct.data_layout == UCT_PERF_DATA_LAYOUT_SHORT) &&
            (params->am_hdr_size != sizeof(uint64_t)))
        {
            if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
                ucs_error("Short AM header size must be 8 bytes");
            }
            return UCS_ERR_INVALID_PARAM;
        }

        if ((params->uct.data_layout == UCT_PERF_DATA_LAYOUT_ZCOPY) &&
            (params->am_hdr_size > attr.cap.am.max_hdr))
        {
            if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
                ucs_error("AM header size (%zu) is larger than max supported (%zu)",
                          params->am_hdr_size, attr.cap.am.max_hdr);
            }
            return UCS_ERR_UNSUPPORTED;
        }

        if (params->am_hdr_size > message_size) {
            if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
                ucs_error("AM header size (%zu) is larger than message size (%zu)",
                          params->am_hdr_size, message_size);
            }
            return UCS_ERR_INVALID_PARAM;
        }

        if (params->uct.fc_window > UCT_PERF_TEST_MAX_FC_WINDOW) {
            if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
                ucs_error("AM flow-control window (%d) too large (should be <= %d)",
                          params->uct.fc_window, UCT_PERF_TEST_MAX_FC_WINDOW);
            }
            return UCS_ERR_INVALID_PARAM;
        }

        if ((params->flags & UCX_PERF_TEST_FLAG_ONE_SIDED) &&
            (params->flags & UCX_PERF_TEST_FLAG_VERBOSE))
        {
            ucs_warn("Running active-message test with on-sided progress");
        }
    }

    if (UCT_PERF_DATA_LAYOUT_ZCOPY == params->uct.data_layout) {
        if (params->msg_size_cnt > max_iov) {
            if ((params->flags & UCX_PERF_TEST_FLAG_VERBOSE) ||
                !params->msg_size_cnt) {
                ucs_error("Wrong number of IOV entries. Requested is %lu, "
                          "should be in the range 1...%lu", params->msg_size_cnt,
                          max_iov);
            }
            return UCS_ERR_UNSUPPORTED;
        }
        /* if msg_size_cnt == 1 the message size checked above */
        if ((UCX_PERF_CMD_AM == params->command) && (params->msg_size_cnt > 1)) {
            if (params->am_hdr_size > params->msg_size_list[0]) {
                if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
                    ucs_error("AM header size (%lu) larger than the first IOV "
                              "message size (%lu)", params->am_hdr_size,
                              params->msg_size_list[0]);
                }
                return UCS_ERR_INVALID_PARAM;
            }
        }
    }

    return UCS_OK;
}

static ucs_status_t uct_perf_test_setup_endpoints(ucx_perf_context_t *perf)
{
    const size_t buffer_size = 2048;
    ucx_perf_ep_info_t info, *remote_info;
    unsigned group_size, i, group_index;
    uct_device_addr_t *dev_addr;
    uct_iface_addr_t *iface_addr;
    uct_ep_addr_t *ep_addr;
    uct_iface_attr_t iface_attr;
    uct_md_attr_t md_attr;
    void *rkey_buffer;
    ucs_status_t status;
    struct iovec vec[5];
    void *buffer;
    void *req;

    buffer = malloc(buffer_size);
    if (buffer == NULL) {
        ucs_error("Failed to allocate RTE buffer");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    status = uct_iface_query(perf->uct.iface, &iface_attr);
    if (status != UCS_OK) {
        ucs_error("Failed to uct_iface_query: %s", ucs_status_string(status));
        goto err_free;
    }

    status = uct_md_query(perf->uct.md, &md_attr);
    if (status != UCS_OK) {
        ucs_error("Failed to uct_md_query: %s", ucs_status_string(status));
        goto err_free;
    }

    if (md_attr.cap.flags & (UCT_MD_FLAG_ALLOC|UCT_MD_FLAG_REG)) {
        info.rkey_size      = md_attr.rkey_packed_size;
    } else {
        info.rkey_size      = 0;
    }
    info.uct.dev_addr_len   = iface_attr.device_addr_len;
    info.uct.iface_addr_len = iface_attr.iface_addr_len;
    info.uct.ep_addr_len    = iface_attr.ep_addr_len;
    info.recv_buffer        = (uintptr_t)perf->recv_buffer;

    rkey_buffer             = buffer;
    dev_addr                = (void*)rkey_buffer + info.rkey_size;
    iface_addr              = (void*)dev_addr    + info.uct.dev_addr_len;
    ep_addr                 = (void*)iface_addr  + info.uct.iface_addr_len;
    ucs_assert_always((void*)ep_addr + info.uct.ep_addr_len <= buffer + buffer_size);

    status = uct_iface_get_device_address(perf->uct.iface, dev_addr);
    if (status != UCS_OK) {
        ucs_error("Failed to uct_iface_get_device_address: %s",
                  ucs_status_string(status));
        goto err_free;
    }

    status = uct_iface_get_address(perf->uct.iface, iface_addr);
    if (status != UCS_OK) {
        ucs_error("Failed to uct_iface_get_address: %s", ucs_status_string(status));
        goto err_free;
    }

    if (info.rkey_size > 0) {
        status = uct_md_mkey_pack(perf->uct.md, perf->uct.recv_mem.memh, rkey_buffer);
        if (status != UCS_OK) {
            ucs_error("Failed to uct_rkey_pack: %s", ucs_status_string(status));
            goto err_free;
        }
    }

    group_size  = rte_call(perf, group_size);
    group_index = rte_call(perf, group_index);

    perf->uct.peers = calloc(group_size, sizeof(*perf->uct.peers));
    if (perf->uct.peers == NULL) {
        goto err_free;
    }

    if (iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
        for (i = 0; i < group_size; ++i) {
            if (i == group_index) {
                continue;
            }

            status = uct_ep_create(perf->uct.iface, &perf->uct.peers[i].ep);
            if (status != UCS_OK) {
                ucs_error("Failed to uct_ep_create: %s", ucs_status_string(status));
                goto err_destroy_eps;
            }
            status = uct_ep_get_address(perf->uct.peers[i].ep, ep_addr);
            if (status != UCS_OK) {
                ucs_error("Failed to uct_ep_get_address: %s", ucs_status_string(status));
                goto err_destroy_eps;
            }
        }
    }

    vec[0].iov_base         = &info;
    vec[0].iov_len          = sizeof(info);
    vec[1].iov_base         = buffer;
    vec[1].iov_len          = info.rkey_size + info.uct.dev_addr_len +
                              info.uct.iface_addr_len + info.uct.ep_addr_len;

    rte_call(perf, post_vec, vec, 2, &req);
    rte_call(perf, exchange_vec, req);

    for (i = 0; i < group_size; ++i) {
        if (i == group_index) {
            continue;
        }

        rte_call(perf, recv, i, buffer, buffer_size, req);

        remote_info = buffer;
        rkey_buffer = remote_info + 1;
        dev_addr    = (void*)rkey_buffer + remote_info->rkey_size;
        iface_addr  = (void*)dev_addr    + remote_info->uct.dev_addr_len;
        ep_addr     = (void*)iface_addr  + remote_info->uct.iface_addr_len;
        perf->uct.peers[i].remote_addr = remote_info->recv_buffer;

        if (!uct_iface_is_reachable(perf->uct.iface, dev_addr,
                                    remote_info->uct.iface_addr_len ?
                                    iface_addr : NULL)) {
            ucs_error("Destination is unreachable");
            status = UCS_ERR_UNREACHABLE;
            goto err_destroy_eps;
        }

        if (remote_info->rkey_size > 0) {
            status = uct_rkey_unpack(rkey_buffer, &perf->uct.peers[i].rkey);
            if (status != UCS_OK) {
                ucs_error("Failed to uct_rkey_unpack: %s", ucs_status_string(status));
                goto err_destroy_eps;
            }
        } else {
            perf->uct.peers[i].rkey.handle = NULL;
            perf->uct.peers[i].rkey.type   = NULL;
            perf->uct.peers[i].rkey.rkey   = UCT_INVALID_RKEY;
        }

        if (iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
            status = uct_ep_connect_to_ep(perf->uct.peers[i].ep, dev_addr, ep_addr);
        } else if (iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
            status = uct_ep_create_connected(perf->uct.iface, dev_addr, iface_addr,
                                             &perf->uct.peers[i].ep);
        } else {
            status = UCS_ERR_UNSUPPORTED;
        }
        if (status != UCS_OK) {
            ucs_error("Failed to connect endpoint: %s", ucs_status_string(status));
            goto err_destroy_eps;
        }
    }
    uct_perf_iface_flush_b(perf);

    free(buffer);
    uct_perf_barrier(perf);
    return UCS_OK;

err_destroy_eps:
    for (i = 0; i < group_size; ++i) {
        if (perf->uct.peers[i].rkey.type != NULL) {
            uct_rkey_release(&perf->uct.peers[i].rkey);
        }
        if (perf->uct.peers[i].ep != NULL) {
            uct_ep_destroy(perf->uct.peers[i].ep);
        }
    }
    free(perf->uct.peers);
err_free:
    free(buffer);
err:
    return status;
}

static void uct_perf_test_cleanup_endpoints(ucx_perf_context_t *perf)
{
    unsigned group_size, group_index, i;

    uct_perf_barrier(perf);

    uct_iface_set_am_handler(perf->uct.iface, UCT_PERF_TEST_AM_ID, NULL, NULL, UCT_CB_FLAG_SYNC);

    group_size  = rte_call(perf, group_size);
    group_index = rte_call(perf, group_index);

    for (i = 0; i < group_size; ++i) {
        if (i != group_index) {
            if (perf->uct.peers[i].rkey.rkey != UCT_INVALID_RKEY) {
                uct_rkey_release(&perf->uct.peers[i].rkey);
            }
            if (perf->uct.peers[i].ep) {
                uct_ep_destroy(perf->uct.peers[i].ep);
            }
        }
    }
    free(perf->uct.peers);
}

static ucs_status_t ucp_perf_test_fill_params(ucx_perf_params_t *params,
                                               ucp_params_t *ucp_params)
{
    ucs_status_t status, message_size;

    message_size = ucx_perf_get_message_size(params);
    switch (params->command) {
    case UCX_PERF_CMD_PUT:
    case UCX_PERF_CMD_GET:
        ucp_params->features |= UCP_FEATURE_RMA;
        break;
    case UCX_PERF_CMD_ADD:
    case UCX_PERF_CMD_FADD:
    case UCX_PERF_CMD_SWAP:
    case UCX_PERF_CMD_CSWAP:
        if (message_size == sizeof(uint32_t)) {
            ucp_params->features |= UCP_FEATURE_AMO32;
        } else if (message_size == sizeof(uint64_t)) {
            ucp_params->features |= UCP_FEATURE_AMO64;
        } else {
            if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
                ucs_error("Atomic size should be either 32 or 64 bit");
            }
            return UCS_ERR_INVALID_PARAM;
        }

        break;
    case UCX_PERF_CMD_TAG:
    case UCX_PERF_CMD_TAG_SYNC:
        ucp_params->features    |= UCP_FEATURE_TAG;
        ucp_params->field_mask  |= UCP_PARAM_FIELD_REQUEST_SIZE;
        ucp_params->request_size = sizeof(ucp_perf_request_t);
        break;
    case UCX_PERF_CMD_STREAM:
        ucp_params->features    |= UCP_FEATURE_STREAM;
        ucp_params->field_mask  |= UCP_PARAM_FIELD_REQUEST_SIZE;
        ucp_params->request_size = sizeof(ucp_perf_request_t);
        break;
    default:
        if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
            ucs_error("Invalid test command");
        }
        return UCS_ERR_INVALID_PARAM;
    }

    status = ucx_perf_test_check_params(params);
    if (status != UCS_OK) {
        return status;
    }

    return UCS_OK;
}

static ucs_status_t ucp_perf_test_alloc_iov_mem(ucp_perf_datatype_t datatype,
                                                size_t iovcnt, unsigned thread_count,
                                                ucp_dt_iov_t **iov_p)
{
    ucp_dt_iov_t *iov;

    if (UCP_PERF_DATATYPE_IOV == datatype) {
        iov = malloc(sizeof(*iov) * iovcnt * thread_count);
        if (NULL == iov) {
            ucs_error("Failed allocate IOV buffer with iovcnt=%lu", iovcnt);
            return UCS_ERR_NO_MEMORY;
        }
        *iov_p = iov;
    }

    return UCS_OK;
}

static ucs_status_t
ucp_perf_test_alloc_host(ucx_perf_context_t *perf, ucx_perf_params_t *params,
                         void **addr, size_t length, ucp_mem_h *memh,
                         int check_non_blk_flag)
{
    ucp_mem_map_params_t mem_map_params;
    ucp_mem_attr_t mem_attr;
    ucs_status_t status;

    mem_map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                                UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                                UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    mem_map_params.address    = *addr;
    mem_map_params.length     = length;
    mem_map_params.flags      = UCP_MEM_MAP_ALLOCATE;
    if (check_non_blk_flag) {
        mem_map_params.flags  |= (params->flags & UCX_PERF_TEST_FLAG_MAP_NONBLOCK) ?
                                 UCP_MEM_MAP_NONBLOCK : 0;
    }

    status = ucp_mem_map(perf->ucp.context, &mem_map_params, memh);
    if (status != UCS_OK) {
        goto err;
    }

    mem_attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS;
    status = ucp_mem_query(*memh, &mem_attr);
    if (status != UCS_OK) {
        goto err;
    }

    *addr = mem_attr.address;

    return UCS_OK;

err:
    return status;
}

static ucs_status_t
ucp_perf_test_alloc_cuda(void **addr, size_t length)
{
#if HAVE_CUDA
    cudaError_t cerr;

    cerr = cudaMalloc(addr, length);
    if (cerr != cudaSuccess) {
        return UCS_ERR_NO_MEMORY;
    }
#endif
    return UCS_OK;
}

static ucs_status_t
ucp_perf_test_alloc_contig(ucx_perf_context_t *perf, ucx_perf_params_t *params,
                           void **addr, size_t length, ucp_mem_h *memh,
                           int check_non_blk_flag)
{
    if (perf->params.mem_type == UCT_MD_MEM_TYPE_HOST) {
        return ucp_perf_test_alloc_host(perf, params, addr, length, memh,
                                        check_non_blk_flag);
    } else if (perf->params.mem_type == UCT_MD_MEM_TYPE_CUDA) {
        return ucp_perf_test_alloc_cuda(addr, length);
    }

    return UCS_ERR_UNSUPPORTED;
}

static void ucp_perf_test_free_contig(ucx_perf_context_t *perf, void *addr, ucp_mem_h memh)
{
    ucs_status_t status;

    if (perf->params.mem_type == UCT_MD_MEM_TYPE_HOST) {
        status = ucp_mem_unmap(perf->ucp.context, memh);
        if (status != UCS_OK) {
            ucs_warn("ucp_mem_unmap() failed: %s", ucs_status_string(status));
        }
    } else if (perf->params.mem_type == UCT_MD_MEM_TYPE_CUDA) {
#if HAVE_CUDA
        cudaFree(addr);
#endif
    }
}

static ucs_status_t ucp_perf_test_alloc_mem(ucx_perf_context_t *perf, ucx_perf_params_t *params)
{
    ucs_status_t status;
    size_t buffer_size;

    if (params->iov_stride) {
        buffer_size           = params->msg_size_cnt * params->iov_stride;
    } else {
        buffer_size           = ucx_perf_get_message_size(params);
    }

    /* Allocate send buffer memory */
    perf->send_buffer         = NULL;
    status = ucp_perf_test_alloc_contig(perf, params, &perf->send_buffer,
                                        buffer_size * params->thread_count,
                                        &perf->ucp.send_memh, 1);
    if (status != UCS_OK) {
        goto err;
    }

    /* Allocate receive buffer memory */
    perf->recv_buffer = NULL;
    status = ucp_perf_test_alloc_contig(perf, params, &perf->recv_buffer,
                                        buffer_size * params->thread_count,
                                        &perf->ucp.recv_memh, 0);
    if (status != UCS_OK) {
        goto err_free_send_buffer;
    }

    /* Allocate IOV datatype memory */
    perf->params.msg_size_cnt = params->msg_size_cnt;
    perf->ucp.send_iov        = NULL;
    status = ucp_perf_test_alloc_iov_mem(params->ucp.send_datatype, perf->params.msg_size_cnt,
                                         params->thread_count, &perf->ucp.send_iov);
    if (UCS_OK != status) {
        goto err_free_buffers;
    }

    perf->ucp.recv_iov        = NULL;
    status = ucp_perf_test_alloc_iov_mem(params->ucp.recv_datatype, perf->params.msg_size_cnt,
                                         params->thread_count, &perf->ucp.recv_iov);
    if (UCS_OK != status) {
        goto err_free_send_iov_buffers;
    }

    return UCS_OK;

err_free_send_iov_buffers:
    free(perf->ucp.send_iov);
err_free_buffers:
    ucp_perf_test_free_contig(perf, perf->recv_buffer, perf->ucp.recv_memh);
err_free_send_buffer:
    ucp_perf_test_free_contig(perf, perf->send_buffer, perf->ucp.send_memh);
err:
    return UCS_ERR_NO_MEMORY;
}

static void ucp_perf_test_free_mem(ucx_perf_context_t *perf)
{
    free(perf->ucp.recv_iov);
    free(perf->ucp.send_iov);
    ucp_perf_test_free_contig(perf, perf->recv_buffer, perf->ucp.recv_memh);
    ucp_perf_test_free_contig(perf, perf->send_buffer, perf->ucp.send_memh);
}

static void ucp_perf_test_destroy_eps(ucx_perf_context_t* perf,
                                      unsigned group_size)
{
    ucs_status_ptr_t    *reqs;
    ucp_tag_recv_info_t info;
    ucs_status_t        status;
    unsigned i;

    reqs = calloc(sizeof(*reqs), group_size);

    for (i = 0; i < group_size; ++i) {
        if (perf->ucp.peers[i].rkey != NULL) {
            ucp_rkey_destroy(perf->ucp.peers[i].rkey);
        }
        if (perf->ucp.peers[i].ep != NULL) {
            reqs[i] = ucp_disconnect_nb(perf->ucp.peers[i].ep);
        }
    }

    for (i = 0; i < group_size; ++i) {
        if (!UCS_PTR_IS_PTR(reqs[i])) {
            continue;
        }

        do {
            ucp_worker_progress(perf->ucp.worker);
            status = ucp_request_test(reqs[i], &info);
        } while (status == UCS_INPROGRESS);
        ucp_request_release(reqs[i]);
    }

    free(reqs);
    free(perf->ucp.peers);
}

static ucs_status_t ucp_perf_test_exchange_status(ucx_perf_context_t *perf,
                                                  ucs_status_t status)
{
    unsigned group_size  = rte_call(perf, group_size);
    ucs_status_t collective_status = status;
    struct iovec vec;
    void *req = NULL;
    unsigned i;

    vec.iov_base = &status;
    vec.iov_len  = sizeof(status);

    rte_call(perf, post_vec, &vec, 1, &req);
    rte_call(perf, exchange_vec, req);
    for (i = 0; i < group_size; ++i) {
        rte_call(perf, recv, i, &status, sizeof(status), req);
        if (status != UCS_OK) {
            collective_status = status;
        }
    }
    return collective_status;
}

static ucs_status_t ucp_perf_test_setup_endpoints(ucx_perf_context_t *perf,
                                                  uint64_t features)
{
    const size_t buffer_size = 2048;
    ucx_perf_ep_info_t info, *remote_info;
    unsigned group_size, i, group_index;
    ucp_address_t *address;
    size_t address_length = 0;
    ucp_ep_params_t ep_params;
    ucs_status_t status;
    struct iovec vec[3];
    void *rkey_buffer;
    void *req = NULL;
    void *buffer;

    group_size  = rte_call(perf, group_size);
    group_index = rte_call(perf, group_index);

    status = ucp_worker_get_address(perf->ucp.worker, &address, &address_length);
    if (status != UCS_OK) {
        if (perf->params.flags & UCX_PERF_TEST_FLAG_VERBOSE) {
            ucs_error("ucp_worker_get_address() failed: %s", ucs_status_string(status));
        }
        goto err;
    }

    info.ucp.addr_len  = address_length;
    info.recv_buffer   = (uintptr_t)perf->recv_buffer;

    vec[0].iov_base    = &info;
    vec[0].iov_len     = sizeof(info);
    vec[1].iov_base    = address;
    vec[1].iov_len     = address_length;

    if (features & (UCP_FEATURE_RMA|UCP_FEATURE_AMO32|UCP_FEATURE_AMO64)) {
        status = ucp_rkey_pack(perf->ucp.context, perf->ucp.recv_memh,
                               &rkey_buffer, &info.rkey_size);
        if (status != UCS_OK) {
            if (perf->params.flags & UCX_PERF_TEST_FLAG_VERBOSE) {
                ucs_error("ucp_rkey_pack() failed: %s", ucs_status_string(status));
            }
            ucp_worker_release_address(perf->ucp.worker, address);
            goto err;
        }

        vec[2].iov_base = rkey_buffer;
        vec[2].iov_len  = info.rkey_size;
        rte_call(perf, post_vec, vec, 3, &req);
        ucp_rkey_buffer_release(rkey_buffer);
    } else {
        info.rkey_size  = 0;
        rte_call(perf, post_vec, vec, 2, &req);
    }

    ucp_worker_release_address(perf->ucp.worker, address);
    rte_call(perf, exchange_vec, req);

    perf->ucp.peers = calloc(group_size, sizeof(*perf->uct.peers));
    if (perf->ucp.peers == NULL) {
        goto err;
    }

    buffer = malloc(buffer_size);
    if (buffer == NULL) {
        ucs_error("Failed to allocate RTE receive buffer");
        status = UCS_ERR_NO_MEMORY;
        goto err_destroy_eps;
    }

    for (i = 0; i < group_size; ++i) {
        if (i == group_index) {
            continue;
        }

        rte_call(perf, recv, i, buffer, buffer_size, req);

        remote_info = buffer;
        address     = (void*)(remote_info + 1);
        rkey_buffer = (void*)address + remote_info->ucp.addr_len;
        perf->ucp.peers[i].remote_addr = remote_info->recv_buffer;

        ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
        ep_params.address    = address;

        status = ucp_ep_create(perf->ucp.worker, &ep_params, &perf->ucp.peers[i].ep);
        if (status != UCS_OK) {
            if (perf->params.flags & UCX_PERF_TEST_FLAG_VERBOSE) {
                ucs_error("ucp_ep_create() failed: %s", ucs_status_string(status));
            }
            goto err_free_buffer;
        }

        if (remote_info->rkey_size > 0) {
            status = ucp_ep_rkey_unpack(perf->ucp.peers[i].ep, rkey_buffer,
                                        &perf->ucp.peers[i].rkey);
            if (status != UCS_OK) {
                if (perf->params.flags & UCX_PERF_TEST_FLAG_VERBOSE) {
                    ucs_fatal("ucp_rkey_unpack() failed: %s", ucs_status_string(status));
                }
                goto err_free_buffer;
            }
        } else {
            perf->ucp.peers[i].rkey = NULL;
        }
    }

    free(buffer);

    status = ucp_perf_test_exchange_status(perf, UCS_OK);
    if (status != UCS_OK) {
        ucp_perf_test_destroy_eps(perf, group_size);
    }

    /* force wireup completion */
    status = ucp_worker_flush(perf->ucp.worker);
    if (status != UCS_OK) {
        ucs_warn("ucp_worker_flush() failed: %s", ucs_status_string(status));
    }

    return status;

err_free_buffer:
    free(buffer);
err_destroy_eps:
    ucp_perf_test_destroy_eps(perf, group_size);
err:
    (void)ucp_perf_test_exchange_status(perf, status);
    return status;
}

static void ucp_perf_test_cleanup_endpoints(ucx_perf_context_t *perf)
{
    unsigned group_size;

    ucp_perf_barrier(perf);

    group_size  = rte_call(perf, group_size);

    ucp_perf_test_destroy_eps(perf, group_size);
}

static void ucx_perf_set_warmup(ucx_perf_context_t* perf, ucx_perf_params_t* params)
{
    perf->max_iter = ucs_min(params->warmup_iter, params->max_iter / 10);
    perf->report_interval = -1;
}

static ucs_status_t uct_perf_create_md(ucx_perf_context_t *perf)
{
    uct_md_resource_desc_t *md_resources;
    uct_tl_resource_desc_t *tl_resources;
    unsigned i, num_md_resources;
    unsigned j, num_tl_resources;
    ucs_status_t status;
    uct_md_h md;
    uct_md_config_t *md_config;

    status = uct_query_md_resources(&md_resources, &num_md_resources);
    if (status != UCS_OK) {
        goto out;
    }

    for (i = 0; i < num_md_resources; ++i) {
        status = uct_md_config_read(md_resources[i].md_name, NULL, NULL, &md_config);
        if (status != UCS_OK) {
            goto out_release_md_resources;
        }

        status = uct_md_open(md_resources[i].md_name, md_config, &md);
        uct_config_release(md_config);
        if (status != UCS_OK) {
            goto out_release_md_resources;
        }

        status = uct_md_query_tl_resources(md, &tl_resources, &num_tl_resources);
        if (status != UCS_OK) {
            uct_md_close(md);
            goto out_release_md_resources;
        }

        for (j = 0; j < num_tl_resources; ++j) {
            if (!strcmp(perf->params.uct.tl_name,  tl_resources[j].tl_name) &&
                !strcmp(perf->params.uct.dev_name, tl_resources[j].dev_name))
            {
                uct_release_tl_resource_list(tl_resources);
                perf->uct.md = md;
                status = UCS_OK;
                goto out_release_md_resources;
            }
        }

        uct_md_close(md);
        uct_release_tl_resource_list(tl_resources);
    }

    ucs_error("Cannot use transport %s on device %s", perf->params.uct.tl_name,
              perf->params.uct.dev_name);
    status = UCS_ERR_NO_DEVICE;

out_release_md_resources:
    uct_release_md_resource_list(md_resources);
out:
    return status;
}

void uct_perf_barrier(ucx_perf_context_t *perf)
{
    rte_call(perf, barrier, (void(*)(void*))uct_worker_progress,
             (void*)perf->uct.worker);
}

void ucp_perf_barrier(ucx_perf_context_t *perf)
{
    rte_call(perf, barrier, (void(*)(void*))ucp_worker_progress,
             (void*)perf->ucp.worker);
}

static ucs_status_t uct_perf_setup(ucx_perf_context_t *perf, ucx_perf_params_t *params)
{
    uct_iface_config_t *iface_config;
    ucs_status_t status;
    uct_iface_params_t iface_params = {
        .open_mode            = UCT_IFACE_OPEN_MODE_DEVICE,
        .mode.device.tl_name  = params->uct.tl_name,
        .mode.device.dev_name = params->uct.dev_name,
        .stats_root           = ucs_stats_get_root(),
        .rx_headroom          = 0
    };
    UCS_CPU_ZERO(&iface_params.cpu_mask);

    status = ucs_async_context_init(&perf->uct.async, params->async_mode);
    if (status != UCS_OK) {
        goto out;
    }

    status = uct_worker_create(&perf->uct.async, params->thread_mode,
                               &perf->uct.worker);
    if (status != UCS_OK) {
        goto out_cleanup_async;
    }

    status = uct_perf_create_md(perf);
    if (status != UCS_OK) {
        goto out_destroy_worker;
    }

    status = uct_md_iface_config_read(perf->uct.md, params->uct.tl_name, NULL,
                                      NULL, &iface_config);
    if (status != UCS_OK) {
        goto out_destroy_md;
    }

    status = uct_iface_open(perf->uct.md, perf->uct.worker, &iface_params,
                            iface_config, &perf->uct.iface);
    uct_config_release(iface_config);
    if (status != UCS_OK) {
        ucs_error("Failed to open iface: %s", ucs_status_string(status));
        goto out_destroy_md;
    }

    status = uct_perf_test_check_capabilities(params, perf->uct.iface);
    /* sync status across all processes */
    status = ucp_perf_test_exchange_status(perf, status);
    if (status != UCS_OK) {
        goto out_iface_close;
    }

    status = uct_perf_test_alloc_mem(perf, params);
    if (status != UCS_OK) {
        goto out_iface_close;
    }

    status = uct_perf_test_setup_endpoints(perf);
    if (status != UCS_OK) {
        ucs_error("Failed to setup endpoints: %s", ucs_status_string(status));
        goto out_free_mem;
    }

    uct_iface_progress_enable(perf->uct.iface,
                              UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);

    return UCS_OK;

out_free_mem:
    uct_perf_test_free_mem(perf);
out_iface_close:
    uct_iface_close(perf->uct.iface);
out_destroy_md:
    uct_md_close(perf->uct.md);
out_destroy_worker:
    uct_worker_destroy(perf->uct.worker);
out_cleanup_async:
    ucs_async_context_cleanup(&perf->uct.async);
out:
    return status;
}

static void uct_perf_cleanup(ucx_perf_context_t *perf)
{
    uct_perf_test_cleanup_endpoints(perf);
    uct_perf_test_free_mem(perf);
    uct_iface_close(perf->uct.iface);
    uct_md_close(perf->uct.md);
    uct_worker_destroy(perf->uct.worker);
    ucs_async_context_cleanup(&perf->uct.async);
}

static ucs_status_t ucp_perf_setup(ucx_perf_context_t *perf,
                                   ucx_perf_params_t *params)
{
    ucp_params_t ucp_params;
    ucp_worker_params_t worker_params;
    ucp_config_t *config;
    ucs_status_t status;

    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features   = 0;

    status = ucp_perf_test_fill_params(params, &ucp_params);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucp_config_read(NULL, NULL, &config);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucp_init(&ucp_params, config, &perf->ucp.context);
    ucp_config_release(config);
    if (status != UCS_OK) {
        goto err;
    }

    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = params->thread_mode;

    status = ucp_worker_create(perf->ucp.context, &worker_params,
                               &perf->ucp.worker);
    if (status != UCS_OK) {
        goto err_cleanup;
    }

    status = ucp_perf_test_alloc_mem(perf, params);
    if (status != UCS_OK) {
        ucs_warn("ucp test failed to alocate memory");
        goto err_destroy_worker;
    }

    status = ucp_perf_test_setup_endpoints(perf, ucp_params.features);
    if (status != UCS_OK) {
        if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
            ucs_error("Failed to setup endpoints: %s", ucs_status_string(status));
        }
        goto err_free_mem;
    }

    return UCS_OK;

err_free_mem:
    ucp_perf_test_free_mem(perf);
err_destroy_worker:
    ucp_worker_destroy(perf->ucp.worker);
err_cleanup:
    ucp_cleanup(perf->ucp.context);
err:
    return status;
}

static void ucp_perf_cleanup(ucx_perf_context_t *perf)
{
    ucp_perf_test_cleanup_endpoints(perf);
    ucp_perf_barrier(perf);
    ucp_perf_test_free_mem(perf);
    ucp_worker_destroy(perf->ucp.worker);
    ucp_cleanup(perf->ucp.context);
}

static struct {
    ucs_status_t (*setup)(ucx_perf_context_t *perf, ucx_perf_params_t *params);
    void         (*cleanup)(ucx_perf_context_t *perf);
    ucs_status_t (*run)(ucx_perf_context_t *perf);
    void         (*barrier)(ucx_perf_context_t *perf);
} ucx_perf_funcs[] = {
    [UCX_PERF_API_UCT] = {uct_perf_setup, uct_perf_cleanup,
                          uct_perf_test_dispatch, uct_perf_barrier},
    [UCX_PERF_API_UCP] = {ucp_perf_setup, ucp_perf_cleanup,
                          ucp_perf_test_dispatch, ucp_perf_barrier}
};

static int ucx_perf_thread_spawn(ucx_perf_context_t *perf,
                                 ucx_perf_result_t* result);

#if HAVE_CUDA
static ucs_status_t ucx_perf_init_cuda_device(ucx_perf_context_t *perf)
{
    cudaError_t cerr;
    unsigned group_index;
    int num_gpus;
    int gpu_index;

    group_index = rte_call(perf, group_index);

    cerr = cudaGetDeviceCount(&num_gpus);
    if (cerr != cudaSuccess) {
        return UCS_ERR_NO_DEVICE;
    }

    gpu_index = group_index % num_gpus;

    cerr = cudaSetDevice(gpu_index);
    if (cerr != cudaSuccess) {
        return UCS_ERR_NO_DEVICE;
    }

    return UCS_OK;
}
#endif

ucs_status_t ucx_perf_run(ucx_perf_params_t *params, ucx_perf_result_t *result)
{
    ucx_perf_context_t *perf;
    ucs_status_t status;

    if (params->command == UCX_PERF_CMD_LAST) {
        ucs_error("Test is not selected");
        status = UCS_ERR_INVALID_PARAM;
        goto out;
    }

    if ((params->api != UCX_PERF_API_UCT) && (params->api != UCX_PERF_API_UCP)) {
        ucs_error("Invalid test API parameter (should be UCT or UCP)");
        status = UCS_ERR_INVALID_PARAM;
        goto out;
    }

    perf = malloc(sizeof(*perf));
    if (perf == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    ucx_perf_test_reset(perf, params);

#if HAVE_CUDA
    if (params->mem_type == UCT_MD_MEM_TYPE_CUDA) {
        status = ucx_perf_init_cuda_device(perf);
        if (status != UCS_OK) {
            goto out_free;
        }
    }
#endif

    status = ucx_perf_funcs[params->api].setup(perf, params);
    if (status != UCS_OK) {
        goto out_free;
    }

    if (UCS_THREAD_MODE_SINGLE == params->thread_mode) {
        if (params->warmup_iter > 0) {
            ucx_perf_set_warmup(perf, params);
            status = ucx_perf_funcs[params->api].run(perf);
            if (status != UCS_OK) {
                goto out_cleanup;
            }

            ucx_perf_funcs[params->api].barrier(perf);
            ucx_perf_test_reset(perf, params);
        }

        /* Run test */
        status = ucx_perf_funcs[params->api].run(perf);
        ucx_perf_funcs[params->api].barrier(perf);
        if (status == UCS_OK) {
            ucx_perf_calc_result(perf, result);
            rte_call(perf, report, result, perf->params.report_arg, 1);
        }
    } else {
        status = ucx_perf_thread_spawn(perf, result);
    }

out_cleanup:
    ucx_perf_funcs[params->api].cleanup(perf);
out_free:
    free(perf);
out:
    return status;
}

#if _OPENMP
/* multiple threads sharing the same worker/iface */
#include <omp.h>

typedef struct {
    pthread_t           pt;
    int                 tid;
    int                 ntid;
    ucs_status_t*       statuses;
    ucx_perf_context_t  perf;
    ucx_perf_result_t   result;
} ucx_perf_thread_context_t;


static void* ucx_perf_thread_run_test(void* arg)
{
    ucx_perf_thread_context_t* tctx = (ucx_perf_thread_context_t*) arg;
    ucx_perf_result_t* result = &tctx->result;
    ucx_perf_context_t* perf = &tctx->perf;
    ucx_perf_params_t* params = &perf->params;
    ucs_status_t* statuses = tctx->statuses;
    int tid = tctx->tid;
    int i;

    if (params->warmup_iter > 0) {
        ucx_perf_set_warmup(perf, params);
        statuses[tid] = ucx_perf_funcs[params->api].run(perf);
        ucx_perf_funcs[params->api].barrier(perf);
        for (i = 0; i < tctx->ntid; i++) {
            if (UCS_OK != statuses[i]) {
                goto out;
            }
        }
#pragma omp master
        ucx_perf_test_reset(perf, params);
    }

    /* Run test */
#pragma omp barrier
    statuses[tid] = ucx_perf_funcs[params->api].run(perf);
    ucx_perf_funcs[params->api].barrier(perf);
    for (i = 0; i < tctx->ntid; i++) {
        if (UCS_OK != statuses[i]) {
            goto out;
        }
    }
#pragma omp master
    {
        /* Assuming all threads are fairly treated, reporting only tid==0
            TODO: aggregate reports */
        ucx_perf_calc_result(perf, result);
        rte_call(perf, report, result, perf->params.report_arg, 1);
    }

out:
    return &statuses[tid];
}

static int ucx_perf_thread_spawn(ucx_perf_context_t *perf,
                                 ucx_perf_result_t* result)
{
    ucx_perf_thread_context_t* tctx;
    ucs_status_t* statuses;
    size_t message_size;
    ucs_status_t status;
    int ti, nti;

    message_size = ucx_perf_get_message_size(&perf->params);
    omp_set_num_threads(perf->params.thread_count);
    nti = perf->params.thread_count;

    tctx     = calloc(nti, sizeof(ucx_perf_thread_context_t));
    statuses = calloc(nti, sizeof(ucs_status_t));
    if ((tctx == NULL) || (statuses == NULL)) {
        status = UCS_ERR_NO_MEMORY;
        goto out_free;
    }

#pragma omp parallel private(ti)
{
    ti = omp_get_thread_num();
    tctx[ti].tid = ti;
    tctx[ti].ntid = nti;
    tctx[ti].statuses = statuses;
    tctx[ti].perf = *perf;
    /* Doctor the src and dst buffers to make them thread specific */
    tctx[ti].perf.send_buffer += ti * message_size;
    tctx[ti].perf.recv_buffer += ti * message_size;
    tctx[ti].perf.offset = ti * message_size;
    ucx_perf_thread_run_test((void*)&tctx[ti]);
}

    status = UCS_OK;
    for (ti = 0; ti < nti; ti++) {
        if (UCS_OK != statuses[ti]) {
            ucs_error("Thread %d failed to run test: %s", tctx[ti].tid,
                      ucs_status_string(statuses[ti]));
            status = statuses[ti];
        }
    }

out_free:
    free(statuses);
    free(tctx);
    return status;
}
#else
static int ucx_perf_thread_spawn(ucx_perf_context_t *perf,
                                 ucx_perf_result_t* result) {
    ucs_error("Invalid test parameter (thread mode requested without OpenMP capabilities)");
    return UCS_ERR_INVALID_PARAM;
}
#endif /* _OPENMP */
