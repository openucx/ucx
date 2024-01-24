/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2019. ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) The University of Tennessee and The University
*               of Tennessee Research Foundation. 2015-2016. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2017-2021.  ALL RIGHTS RESERVED.
* Copyright (C) Huawei Technologies Co., Ltd. 2021.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/debug/log.h>
#include <ucs/arch/bitops.h>
#include <ucs/sys/module.h>
#include <ucs/sys/string.h>
#include <tools/perf/lib/libperf_int.h>

#include <string.h>
#include <unistd.h>

#if _OPENMP
#   include <omp.h>
#endif /* _OPENMP */

#define ATOMIC_OP_CONFIG(_size, _op32, _op64, _op, _msg, _params, _status) \
    _status = __get_atomic_flag((_size), (_op32), (_op64), (_op)); \
    if (_status != UCS_OK) { \
        ucs_error(UCT_PERF_TEST_PARAMS_FMT" does not support atomic %s for " \
                  "message size %zu bytes", UCT_PERF_TEST_PARAMS_ARG(_params), \
                  (_msg)[_op], (_size)); \
        return _status; \
    }

#define ATOMIC_OP_CHECK(_size, _attr, _required, _params, _msg) \
    if (!ucs_test_all_flags(_attr, _required)) { \
        if ((_params)->flags & UCX_PERF_TEST_FLAG_VERBOSE) { \
            ucs_error(UCT_PERF_TEST_PARAMS_FMT" does not support required " \
                      #_size"-bit atomic: %s", UCT_PERF_TEST_PARAMS_ARG(_params), \
                      (_msg)[ucs_ffs64(~(_attr) & (_required))]); \
        } \
        return UCS_ERR_UNSUPPORTED; \
    }

#define UCX_PERF_VERBOSE(_log_level, _params, _func, ...) \
    ({ \
        ucs_status_t _status = _func(__VA_ARGS__); \
        if (((_status) != UCS_OK) && \
            ((_params)->flags & UCX_PERF_TEST_FLAG_VERBOSE)) { \
            ucs_##_log_level("%s() failed: %s", UCS_PP_MAKE_STRING(_func), \
                             ucs_status_string(_status)); \
        } \
        _status; \
    })


typedef struct {
    union {
        struct {
            size_t     dev_addr_len;
            size_t     iface_addr_len;
            size_t     ep_addr_len;
        } uct;
        struct {
            size_t     worker_addr_len;
            size_t     total_wireup_len;
        } ucp;
    };
    size_t             rkey_size;
    unsigned long      recv_buffer;
} ucx_perf_ep_info_t;

typedef struct {
    int          num_outstanding; /* Number of outstanding flush operations */
    ucs_status_t status;          /* Cumulative status of all flush operations */
} ucp_perf_flush_context_t;


const ucx_perf_allocator_t* ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_LAST];

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
    [ucs_ilog2(UCT_IFACE_FLAG_PENDING)]          = "pending",
    [ucs_ilog2(UCT_IFACE_FLAG_TAG_EAGER_SHORT)]  = "tag eager short",
    [ucs_ilog2(UCT_IFACE_FLAG_TAG_EAGER_BCOPY)]  = "tag eager bcopy",
    [ucs_ilog2(UCT_IFACE_FLAG_TAG_EAGER_ZCOPY)]  = "tag eager zcopy",
    [ucs_ilog2(UCT_IFACE_FLAG_TAG_RNDV_ZCOPY)]   = "tag rndv zcopy",
    [ucs_ilog2(UCT_IFACE_FLAG_EP_CHECK)]         = "ep check",
    [ucs_ilog2(UCT_IFACE_FLAG_EP_KEEPALIVE)]     = "ep keepalive"
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
static ucs_time_t __find_percentile_quick_select(ucs_time_t arr[], int n, double rank)
{
    int low, high;
    int percentile_idx;
    int middle, ll, hh;

#define ELEM_SWAP(a,b) { register ucs_time_t t=(a);(a)=(b);(b)=t; }

    low = 0; high = n - 1; percentile_idx = high * (rank / 100.0);
    for (;;) {
        if (high <= low) { /* One element only */
            return arr[percentile_idx];
        }

        if (high == low + 1) {  /* Two elements only */
            if (arr[low] > arr[high]) {
                ELEM_SWAP(arr[low], arr[high]);
            }
            return arr[percentile_idx];
        }

        /* Find median of low, middle and high items; swap into position low */
        middle = (low + high) / 2;
        if (arr[middle] > arr[high])    { ELEM_SWAP(arr[middle], arr[high]); }
        if (arr[low] > arr[high])       { ELEM_SWAP(arr[low], arr[high]); }
        if (arr[middle] > arr[low])     { ELEM_SWAP(arr[middle], arr[low]); }

        /* Swap low item (now in position middle) into position (low+1) */
        ELEM_SWAP(arr[middle], arr[low + 1]);

        /* Nibble from each end towards middle, swapping items when stuck */
        ll = low + 1;
        hh = high;
        for (;;) {
            do ll++; while (arr[low] > arr[ll]);
            do hh--; while (arr[hh]  > arr[low]);

            if (hh < ll) {
                break;
            }

            ELEM_SWAP(arr[ll], arr[hh]);
        }

        /* Swap middle item (in position 'low') back into correct position */
        ELEM_SWAP(arr[low], arr[hh]);

        /* Re-set active partition */
        if (hh <= percentile_idx) {
            low = ll;
        }
        if (hh >= percentile_idx) {
            high = hh - 1;
        }
    }
}

void ucx_perf_test_start_clock(ucx_perf_context_t *perf)
{
    ucs_time_t start_time = ucs_get_time();

    perf->start_time_acc   = ucs_get_accurate_time();
    perf->end_time         = (perf->params.max_time == 0.0) ? UINT64_MAX :
                              ucs_time_from_sec(perf->params.max_time) + start_time;
    perf->prev_time        = start_time;
    perf->prev.time        = start_time;
    perf->prev.time_acc    = perf->start_time_acc;
    perf->current.time_acc = perf->start_time_acc;
}

/* Initialize/reset all parameters that could be modified by the warm-up run */
void ucx_perf_test_prepare_new_run(ucx_perf_context_t *perf,
                                   const ucx_perf_params_t *params)
{
    unsigned i;

    perf->max_iter          = (perf->params.max_iter == 0) ? UINT64_MAX :
                               perf->params.max_iter;
    perf->report_interval   = ucs_time_from_sec(perf->params.report_interval);
    perf->current.time      = 0;
    perf->current.msgs      = 0;
    perf->current.bytes     = 0;
    perf->current.iters     = 0;
    perf->prev.msgs         = 0;
    perf->prev.bytes        = 0;
    perf->prev.iters        = 0;
    perf->timing_queue_head = 0;
    perf->extra_info[0]     = '\0';

    for (i = 0; i < TIMING_QUEUE_SIZE; ++i) {
        perf->timing_queue[i] = 0;
    }
    ucx_perf_test_start_clock(perf);
}

void ucx_perf_calc_result(ucx_perf_context_t *perf, ucx_perf_result_t *result)
{
    ucs_time_t percentile;
    double factor;

    if ((perf->params.test_type == UCX_PERF_TEST_TYPE_PINGPONG) ||
        (perf->params.test_type == UCX_PERF_TEST_TYPE_PINGPONG_WAIT_MEM)) {
        factor = 2.0;
    } else {
        factor = 1.0;
    }

    result->iters = perf->current.iters;
    result->bytes = perf->current.bytes;
    result->elapsed_time = perf->current.time_acc - perf->start_time_acc;

    /* Latency */
    percentile = __find_percentile_quick_select(perf->timing_queue,
                                                ucs_min(TIMING_QUEUE_SIZE, perf->current.iters),
                                                perf->params.percentile_rank);
    result->latency.percentile = ucs_time_to_sec(percentile) / factor;

    result->latency.moment_average =
        (perf->current.time_acc - perf->prev.time_acc)
        / (perf->current.iters - perf->prev.iters)
        / factor;

    result->latency.total_average =
        (perf->current.time_acc - perf->start_time_acc)
        / perf->current.iters
        / factor;


    /* Bandwidth */

    result->bandwidth.percentile = 0.0; // Undefined

    result->bandwidth.moment_average =
        (perf->current.bytes - perf->prev.bytes) /
        (perf->current.time_acc - perf->prev.time_acc) * factor;

    result->bandwidth.total_average =
        perf->current.bytes /
        (perf->current.time_acc - perf->start_time_acc) * factor;


    /* Packet rate */

    result->msgrate.percentile = 0.0; // Undefined

    result->msgrate.moment_average =
        (perf->current.msgs - perf->prev.msgs) /
        (perf->current.time_acc - perf->prev.time_acc) * factor;

    result->msgrate.total_average =
        perf->current.msgs /
        (perf->current.time_acc - perf->start_time_acc) * factor;

}

static ucs_status_t ucx_perf_test_check_params(ucx_perf_params_t *params)
{
    size_t it;

    /* check if zero-size messages are requested and supported */
    if ((/* they are not supported by: */
         /* - UCT tests, except UCT AM Short/Bcopy */
         (params->api == UCX_PERF_API_UCT) ||
         (/* - UCP RMA and AMO tests */
          (params->api == UCX_PERF_API_UCP) &&
          (params->command != UCX_PERF_CMD_AM) &&
          (params->command != UCX_PERF_CMD_TAG) &&
          (params->command != UCX_PERF_CMD_TAG_SYNC) &&
          (params->command != UCX_PERF_CMD_STREAM))) &&
        ucx_perf_get_message_size(params) < 1) {
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

void uct_perf_ep_flush_b(ucx_perf_context_t *perf, int peer_index)
{
    uct_ep_h ep = perf->uct.peers[peer_index].ep;
    uct_completion_t comp;
    ucs_status_t status;
    int started;

    started    = 0;
    comp.func  = NULL;
    comp.count = 2;
    do {
        if (!started) {
            status = uct_ep_flush(ep, 0, &comp);
            if (status == UCS_OK) {
                --comp.count;
            } else if (status == UCS_INPROGRESS) {
                started = 1;
            } else if (status != UCS_ERR_NO_RESOURCE) {
                ucs_error("uct_ep_flush() failed: %s", ucs_status_string(status));
                return;
            }
        }
        uct_worker_progress(perf->uct.worker);
    } while (comp.count > 1);
}

void uct_perf_iface_flush_b(ucx_perf_context_t *perf)
{
    ucs_status_t status;

    do {
        status = uct_iface_flush(perf->uct.iface, 0, NULL);
        uct_worker_progress(perf->uct.worker);
    } while (status == UCS_INPROGRESS);
    if (status != UCS_OK) {
        ucs_error("uct_iface_flush() failed: %s", ucs_status_string(status));
    }
}

static inline uint64_t __get_flag(uct_perf_data_layout_t layout, uint64_t short_f,
                                  uint64_t bcopy_f, uint64_t zcopy_f)
{
    return ((layout == UCT_PERF_DATA_LAYOUT_SHORT) ||
            (layout == UCT_PERF_DATA_LAYOUT_SHORT_IOV)) ? short_f :
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
    return ((layout == UCT_PERF_DATA_LAYOUT_SHORT) ||
            (layout == UCT_PERF_DATA_LAYOUT_SHORT_IOV)) ? short_m :
           (layout == UCT_PERF_DATA_LAYOUT_BCOPY) ? bcopy_m :
           (layout == UCT_PERF_DATA_LAYOUT_ZCOPY) ? zcopy_m :
           0;
}

static ucs_status_t uct_perf_test_check_md_support(ucx_perf_params_t *params,
                                                   ucs_memory_type_t mem_type,
                                                   uct_md_attr_t *md_attr)
{
    if (!(md_attr->cap.access_mem_types & UCS_BIT(mem_type)) &&
        !(md_attr->cap.reg_mem_types & UCS_BIT(mem_type))) {
        if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
            ucs_error("Unsupported memory type %s by "UCT_PERF_TEST_PARAMS_FMT,
                      ucs_memory_type_names[mem_type],
                      UCT_PERF_TEST_PARAMS_ARG(params));
            return UCS_ERR_INVALID_PARAM;
        }
    }
    return UCS_OK;
}

static ucs_status_t uct_perf_test_check_capabilities(ucx_perf_params_t *params,
                                                     uct_iface_h iface, uct_md_h md)
{
    uint64_t required_flags = 0;
    uint64_t atomic_op32    = 0;
    uint64_t atomic_op64    = 0;
    uint64_t atomic_fop32   = 0;
    uint64_t atomic_fop64   = 0;
    uct_md_attr_t md_attr;
    uct_iface_attr_t attr;
    ucs_status_t status;
    size_t min_size, max_size, max_iov, message_size;

    status = uct_md_query(md, &md_attr);
    if (status != UCS_OK) {
        ucs_error("uct_md_query(%s) failed: %s",
                  params->uct.md_name, ucs_status_string(status));
        return status;
    }

    status = uct_iface_query(iface, &attr);
    if (status != UCS_OK) {
        ucs_error("uct_iface_query("UCT_PERF_TEST_PARAMS_FMT") failed: %s",
                  UCT_PERF_TEST_PARAMS_ARG(params),
                  ucs_status_string(status));
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
            ucs_error(UCT_PERF_TEST_PARAMS_FMT" does not support operation %s",
                      UCT_PERF_TEST_PARAMS_ARG(params),
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
            (params->uct.am_hdr_size != sizeof(uint64_t))) {
            if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
                ucs_error("Short AM header size must be 8 bytes");
            }
            return UCS_ERR_INVALID_PARAM;
        }

        if ((params->uct.data_layout == UCT_PERF_DATA_LAYOUT_ZCOPY) &&
            (params->uct.am_hdr_size > attr.cap.am.max_hdr)) {
            if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
                ucs_error("AM header size (%zu) is larger than max supported "
                          "(%zu)",
                          params->uct.am_hdr_size, attr.cap.am.max_hdr);
            }
            return UCS_ERR_UNSUPPORTED;
        }

        if (params->uct.am_hdr_size > message_size) {
            if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
                ucs_error("AM header size (%zu) is larger than message size "
                          "(%zu)",
                          params->uct.am_hdr_size, message_size);
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

    if ((UCT_PERF_DATA_LAYOUT_ZCOPY == params->uct.data_layout) ||
        (UCT_PERF_DATA_LAYOUT_SHORT_IOV == params->uct.data_layout)) {
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
        if ((UCT_PERF_DATA_LAYOUT_ZCOPY == params->uct.data_layout) &&
            (UCX_PERF_CMD_AM == params->command) && (params->msg_size_cnt > 1)) {
            if (params->uct.am_hdr_size > params->msg_size_list[0]) {
                if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
                    ucs_error("AM header size (%lu) larger than the first IOV "
                              "message size (%lu)",
                              params->uct.am_hdr_size,
                              params->msg_size_list[0]);
                }
                return UCS_ERR_INVALID_PARAM;
            }
        }
    }

    status = uct_perf_test_check_md_support(params, params->send_mem_type, &md_attr);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_perf_test_check_md_support(params, params->recv_mem_type, &md_attr);
    if (status != UCS_OK) {
        return status;
    }

    return UCS_OK;
}

static ucs_status_t uct_perf_test_setup_endpoints(ucx_perf_context_t *perf)
{
    const size_t buffer_size = ADDR_BUF_SIZE;
    ucx_perf_ep_info_t info, *remote_info;
    unsigned group_size, i, group_index;
    uct_device_addr_t *dev_addr;
    uct_iface_addr_t *iface_addr;
    uct_ep_addr_t *ep_addr;
    uct_iface_attr_t iface_attr;
    uct_md_attr_t md_attr;
    uct_ep_params_t ep_params;
    unsigned peer_index;
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
    dev_addr                = UCS_PTR_BYTE_OFFSET(rkey_buffer, info.rkey_size);
    iface_addr              = UCS_PTR_BYTE_OFFSET(dev_addr, info.uct.dev_addr_len);
    ep_addr                 = UCS_PTR_BYTE_OFFSET(iface_addr, info.uct.iface_addr_len);
    ucs_assert_always(UCS_PTR_BYTE_OFFSET(ep_addr, info.uct.ep_addr_len) <=
                      UCS_PTR_BYTE_OFFSET(buffer, buffer_size));

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

    if (md_attr.cap.flags & (UCT_MD_FLAG_ALLOC|UCT_MD_FLAG_REG)) {
        memset(rkey_buffer, 0, info.rkey_size);
        status = uct_md_mkey_pack(perf->uct.md, perf->uct.recv_mem.memh, rkey_buffer);
        if (status != UCS_OK) {
            ucs_error("Failed to uct_rkey_pack: %s", ucs_status_string(status));
            goto err_free;
        }
    }

    group_size  = rte_call(perf, group_size);
    group_index = rte_call(perf, group_index);
    peer_index  = rte_peer_index(group_size, group_index);

    perf->uct.peers = calloc(group_size, sizeof(*perf->uct.peers));
    if (perf->uct.peers == NULL) {
        goto err_free;
    }

    ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE;
    ep_params.iface      = perf->uct.iface;
    if (iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
        for (i = 0; i < group_size; ++i) {
            if (i != peer_index) {
                perf->uct.peers[i].ep = NULL;
                continue;
            }

            status = uct_ep_create(&ep_params, &perf->uct.peers[i].ep);
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
    } else if (iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
        ep_params.field_mask |= UCT_EP_PARAM_FIELD_DEV_ADDR |
                                UCT_EP_PARAM_FIELD_IFACE_ADDR;
    }

    vec[0].iov_base         = &info;
    vec[0].iov_len          = sizeof(info);
    vec[1].iov_base         = buffer;
    vec[1].iov_len          = info.rkey_size + info.uct.dev_addr_len +
                              info.uct.iface_addr_len + info.uct.ep_addr_len;

    rte_call(perf, post_vec, vec, 2, &req);
    rte_call(perf, exchange_vec, req);

    for (i = 0; i < group_size; ++i) {
        if (i != peer_index) {
            perf->uct.peers[i].rkey.handle = NULL;
            perf->uct.peers[i].rkey.rkey   = UCT_INVALID_RKEY;
            continue;
        }

        rte_call(perf, recv, i, buffer, buffer_size, req);

        remote_info = buffer;
        rkey_buffer = remote_info + 1;
        dev_addr    = UCS_PTR_BYTE_OFFSET(rkey_buffer, remote_info->rkey_size);
        iface_addr  = UCS_PTR_BYTE_OFFSET(dev_addr, remote_info->uct.dev_addr_len);
        ep_addr     = UCS_PTR_BYTE_OFFSET(iface_addr, remote_info->uct.iface_addr_len);
        perf->uct.peers[i].remote_addr = remote_info->recv_buffer;

        if (!uct_iface_is_reachable(perf->uct.iface, dev_addr,
                                    remote_info->uct.iface_addr_len ?
                                    iface_addr : NULL)) {
            ucs_error("Destination is unreachable");
            status = UCS_ERR_UNREACHABLE;
            goto err_destroy_eps;
        }

        if (md_attr.cap.flags & (UCT_MD_FLAG_ALLOC|UCT_MD_FLAG_REG)) {
            status = uct_rkey_unpack(perf->uct.cmpt, rkey_buffer,
                                     &perf->uct.peers[i].rkey);
            if (status != UCS_OK) {
                ucs_error("Failed to uct_rkey_unpack: %s", ucs_status_string(status));
                goto err_destroy_eps;
            }
        } else {
            perf->uct.peers[i].rkey.handle = NULL;
            perf->uct.peers[i].rkey.rkey   = UCT_INVALID_RKEY;
        }

        if (iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
            status = uct_ep_connect_to_ep(perf->uct.peers[i].ep, dev_addr, ep_addr);
        } else if (iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
            ep_params.dev_addr   = dev_addr;
            ep_params.iface_addr = iface_addr;
            status = uct_ep_create(&ep_params, &perf->uct.peers[i].ep);
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
        if (perf->uct.peers[i].rkey.rkey != UCT_INVALID_RKEY) {
            uct_rkey_release(perf->uct.cmpt, &perf->uct.peers[i].rkey);
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
    unsigned group_size, i;

    uct_perf_barrier(perf);

    uct_iface_set_am_handler(perf->uct.iface, UCT_PERF_TEST_AM_ID, NULL, NULL, 0);

    group_size  = rte_call(perf, group_size);

    for (i = 0; i < group_size; ++i) {
        if (perf->uct.peers[i].rkey.rkey != UCT_INVALID_RKEY) {
            uct_rkey_release(perf->uct.cmpt, &perf->uct.peers[i].rkey);
        }
        if (perf->uct.peers[i].ep) {
            uct_ep_destroy(perf->uct.peers[i].ep);
        }
    }
    free(perf->uct.peers);
}

static void ucp_perf_worker_progress(void *arg)
{
    ucx_perf_context_t *perf = arg;
    int i;

    for (i = 0; i < perf->params.thread_count; ++i) {
        ucp_worker_progress(perf->ucp.tctx[i].perf.ucp.worker);
    }
}

static int ucp_perf_test_is_host_mem_only(const ucx_perf_params_t *params)
{
    return (params->send_mem_type == UCS_MEMORY_TYPE_HOST) &&
           (params->recv_mem_type == UCS_MEMORY_TYPE_HOST);
}

static ucs_status_t ucp_perf_test_fill_params(ucx_perf_params_t *params,
                                              ucp_params_t *ucp_params)
{
    ucs_status_t status;
    size_t message_size;

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
        ucp_params->features |= UCP_FEATURE_TAG;
        break;
    case UCX_PERF_CMD_STREAM:
        ucp_params->features |= UCP_FEATURE_STREAM;
        break;
    case UCX_PERF_CMD_AM:
        ucp_params->features |= UCP_FEATURE_AM;
        break;
    default:
        if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
            ucs_error("Invalid test command");
        }
        return UCS_ERR_INVALID_PARAM;
    }

    if ((params->flags & UCX_PERF_TEST_FLAG_WAKEUP) ||
        (params->wait_mode == UCX_PERF_WAIT_MODE_SLEEP)) {
        ucp_params->features |= UCP_FEATURE_WAKEUP;
    }

    if (!ucp_perf_test_is_host_mem_only(params)) {
        /* Required as sn is being copied over put / get operations */
        ucp_params->features |= UCP_FEATURE_RMA;
    }

    status = ucx_perf_test_check_params(params);
    if (status != UCS_OK) {
        return status;
    }

    return UCS_OK;
}

static ucs_status_ptr_t
ucp_perf_test_destroy_ep(ucp_ep_h ep, const ucp_rkey_h *rkeys, unsigned num_rkeys, unsigned index)
{
    ucp_request_param_t ep_close_params = {0};
    ucs_status_ptr_t *req;
    unsigned i;

    for (i = 0; i < num_rkeys; ++i) {
        if (rkeys[i] != NULL) {
            ucp_rkey_destroy(rkeys[i]);
        }
    }

    if (ep != NULL) {
        req = ucp_ep_close_nbx(ep, &ep_close_params);
        if (UCS_PTR_IS_PTR(req)) {
            return req;
        } else if (UCS_PTR_STATUS(req) != UCS_OK) {
            ucs_warn("failed to close ep %p on thread %d: %s\n", ep, index,
                     ucs_status_string(UCS_PTR_STATUS(req)));
        }
    }

    return NULL;
}

static void ucp_perf_release_requests_in_progress(ucx_perf_context_t *perf,
                                                  ucs_status_ptr_t **reqs,
                                                  unsigned num_in_prog)
{
    ucs_status_t status;
    unsigned i;

    while (num_in_prog != 0) {
        ucp_perf_worker_progress(perf);
        for (i = 0; i < num_in_prog; ++i) {
            status = ucp_request_check_status(reqs[i]);
            if (status != UCS_INPROGRESS) {
                ucp_request_release(reqs[i]);
                reqs[i] = reqs[--num_in_prog];
            }
        }
    }
}

static void ucp_perf_test_destroy_self_eps(ucx_perf_context_t *perf)
{
    unsigned thread_count   = perf->params.thread_count;
    unsigned num_in_prog    = 0;
    ucs_status_ptr_t **reqs = ucs_alloca(thread_count * sizeof(*reqs));
    ucs_status_ptr_t *req;
    unsigned i;

    for (i = 0; i < thread_count; ++i) {
        ucp_rkey_h self_rkeys[] = {perf->ucp.tctx[i].perf.ucp.self_send_rkey,
                                   perf->ucp.tctx[i].perf.ucp.self_recv_rkey};
        req = ucp_perf_test_destroy_ep(perf->ucp.tctx[i].perf.ucp.self_ep,
                                       self_rkeys, 2, i);
        if (req != NULL) {
            reqs[num_in_prog++] = req;
        }
    }

    ucp_perf_release_requests_in_progress(perf, reqs, num_in_prog);
}

static void ucp_perf_test_destroy_eps(ucx_perf_context_t *perf)
{
    unsigned thread_count   = perf->params.thread_count;
    unsigned num_in_prog    = 0;
    ucs_status_ptr_t **reqs = ucs_alloca(thread_count * sizeof(*reqs));
    ucs_status_ptr_t *req;
    unsigned i;

    for (i = 0; i < thread_count; ++i) {
        req = ucp_perf_test_destroy_ep(perf->ucp.tctx[i].perf.ucp.ep,
                                       &perf->ucp.tctx[i].perf.ucp.rkey, 1, i);
        if (req != NULL) {
            reqs[num_in_prog++] = req;
        }
    }

    ucp_perf_release_requests_in_progress(perf, reqs, num_in_prog);
}

static ucs_status_t
ucx_perf_test_exchange_status(ucx_perf_context_t *perf, ucs_status_t status)
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

static void ucp_perf_test_err_handler(void *arg, ucp_ep_h ep,
                                      ucs_status_t status)
{
    ucs_error("error handler called with status %d (%s)\n", status,
              ucs_status_string(status));
    exit(EXIT_FAILURE);
}

static ucs_status_t ucp_perf_test_rkey_pack(ucx_perf_context_t *perf,
                                            uint64_t features,
                                            void **rkey_buffer,
                                            size_t *rkey_size)
{
    ucs_status_t status;

    if (features & (UCP_FEATURE_RMA | UCP_FEATURE_AMO32 | UCP_FEATURE_AMO64)) {
        status = UCX_PERF_VERBOSE(error, &perf->params, ucp_rkey_pack,
                                  perf->ucp.context, perf->ucp.recv_memh,
                                  rkey_buffer, rkey_size);
        if (status != UCS_OK) {
            return status;
        }
    } else {
        *rkey_size = 0;
    }

    return UCS_OK;
}

static ucs_status_t
ucp_perf_test_rkey_unpack(ucx_perf_thread_context_t *thread,
                          ucx_perf_params_t *params, void *rkey_buffer,
                          size_t rkey_size)
{
    ucs_status_t status;

    if (rkey_size > 0) {
        status = UCX_PERF_VERBOSE(fatal, params, ucp_ep_rkey_unpack,
                                  thread->perf.ucp.ep, rkey_buffer,
                                  &thread->perf.ucp.rkey);
        if (status != UCS_OK) {
            return status;
        }
    } else {
        thread->perf.ucp.rkey = NULL;
    }

    return UCS_OK;
}

static ucs_status_t ucp_perf_test_create_self_rkey(ucx_perf_context_t *perf,
                                                   ucp_mem_h memh,
                                                   ucp_rkey_h *rkey)
{
    void *rkey_buffer                                      = NULL;
    size_t rkey_size                                       = 0;
    ucp_memh_buffer_release_params_t buffer_release_params = {0};
    ucs_status_t status;

    status = UCX_PERF_VERBOSE(fatal, &perf->params, ucp_rkey_pack,
                              perf->ucp.context, memh, &rkey_buffer,
                              &rkey_size);
    if (status != UCS_OK) {
        return status;
    }

    status = UCX_PERF_VERBOSE(fatal, &perf->params, ucp_ep_rkey_unpack,
                              perf->ucp.self_ep, rkey_buffer, rkey);
    ucp_memh_buffer_release(rkey_buffer, &buffer_release_params);
    return status;
}

static ucs_status_t ucp_perf_test_receive_remote_data(ucx_perf_context_t *perf,
                                                      unsigned peer_index)
{
    unsigned thread_count = perf->params.thread_count;
    void *rkey_buffer     = NULL;
    void *req             = NULL;
    ucx_perf_ep_info_t *remote_info;
    ucp_ep_params_t ep_params;
    ucp_address_t *address;
    ucs_status_t status;
    size_t buffer_size;
    void *buffer;
    unsigned i;

    buffer_size = ADDR_BUF_SIZE * thread_count;

    buffer = malloc(buffer_size);
    if (buffer == NULL) {
        ucs_error("failed to allocate RTE receive buffer");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    /* Initialize all endpoints and rkeys to NULL to handle error flow */
    for (i = 0; i < thread_count; i++) {
        perf->ucp.tctx[i].perf.ucp.ep             = NULL;
        perf->ucp.tctx[i].perf.ucp.rkey           = NULL;
        perf->ucp.tctx[i].perf.ucp.self_ep        = NULL;
        perf->ucp.tctx[i].perf.ucp.self_send_rkey = NULL;
        perf->ucp.tctx[i].perf.ucp.self_recv_rkey = NULL;
    }

    /* Receive the data from the remote peer, extract the address from it
     * (along with additional wireup info) and create an endpoint to the peer */
    rte_call(perf, recv, peer_index, buffer, buffer_size, req);

    remote_info = buffer;
    for (i = 0; i < thread_count; i++) {
        address                                = (ucp_address_t*)(remote_info + 1);
        rkey_buffer                            = UCS_PTR_BYTE_OFFSET(address,
                                                                     remote_info->ucp.worker_addr_len);
        perf->ucp.tctx[i].perf.ucp.remote_addr = remote_info->recv_buffer;

        ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
        ep_params.address    = address;

        if (perf->params.flags & UCX_PERF_TEST_FLAG_ERR_HANDLING) {
            ep_params.field_mask     |= UCP_EP_PARAM_FIELD_ERR_HANDLER |
                                        UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
            ep_params.err_handler.cb  = ucp_perf_test_err_handler;
            ep_params.err_handler.arg = NULL;
            ep_params.err_mode        = UCP_ERR_HANDLING_MODE_PEER;
        }

        status = UCX_PERF_VERBOSE(error, &perf->params, ucp_ep_create,
                                  perf->ucp.tctx[i].perf.ucp.worker, &ep_params,
                                  &perf->ucp.tctx[i].perf.ucp.ep);
        if (status != UCS_OK) {
            goto err_free_eps_buffer;
        }

        status = ucp_perf_test_rkey_unpack(&perf->ucp.tctx[i], &perf->params,
                                           rkey_buffer, remote_info->rkey_size);
        if (status != UCS_OK) {
            goto err_free_eps_buffer;
        }

        remote_info = UCS_PTR_BYTE_OFFSET(remote_info,
                                          remote_info->ucp.total_wireup_len);
    }

    free(buffer);
    return UCS_OK;

err_free_eps_buffer:
    ucp_perf_test_destroy_eps(perf);
    free(buffer);
err:
    return status;
}

static ucs_status_t ucp_perf_test_send_local_data(ucx_perf_context_t *perf,
                                                  uint64_t features)
{
    unsigned i, j, thread_count = perf->params.thread_count;
    size_t address_length       = 0;
    void *rkey_buffer           = NULL;
    void *req                   = NULL;
    ucx_perf_ep_info_t *info;
    ucp_address_t *address;
    ucs_status_t status;
    struct iovec *vec;
    size_t rkey_size;

    status = ucp_perf_test_rkey_pack(perf, features, &rkey_buffer, &rkey_size);
    if (status != UCS_OK) {
        goto err;
    }

    /* each thread has an iovec with 3 entries to send to the remote peer:
     * ep_info, worker_address and rkey buffer */
    vec = calloc(3 * thread_count, sizeof(struct iovec));
    if (vec == NULL) {
        ucs_error("failed to allocate iovec");
        status = UCS_ERR_NO_MEMORY;
        goto err_rkey_release;
    }

    /* get the worker address created for every thread and send it to the remote
     * peer */
    for (i = 0; i < thread_count; i++) {
        status = UCX_PERF_VERBOSE(error, &perf->params, ucp_worker_get_address,
                                  perf->ucp.tctx[i].perf.ucp.worker, &address,
                                  &address_length);
        if (status != UCS_OK) {
            goto err_free_workers_vec;
        }

        vec[i * 3].iov_base = malloc(sizeof(*info));
        if (vec[i * 3].iov_base == NULL) {
            ucs_error("failed to allocate vec entry for info");
            status = UCS_ERR_NO_MEMORY;
            ucp_worker_destroy(perf->ucp.tctx[i].perf.ucp.worker);
            goto err_free_workers_vec;
        }

        info                       = vec[i * 3].iov_base;
        info->ucp.worker_addr_len  = address_length;
        info->ucp.total_wireup_len = sizeof(*info) + address_length + rkey_size;
        info->rkey_size            = rkey_size;
        info->recv_buffer          = (uintptr_t)perf->ucp.tctx[i].perf.recv_buffer;

        vec[(i * 3) + 0].iov_len  = sizeof(*info);
        vec[(i * 3) + 1].iov_base = address;
        vec[(i * 3) + 1].iov_len  = address_length;
        vec[(i * 3) + 2].iov_base = rkey_buffer;
        vec[(i * 3) + 2].iov_len  = info->rkey_size;

        address_length = 0;
    }

    /* send to the remote peer */
    rte_call(perf, post_vec, vec, 3 * thread_count, &req);
    rte_call(perf, exchange_vec, req);

    if (features & (UCP_FEATURE_RMA|UCP_FEATURE_AMO32|UCP_FEATURE_AMO64)) {
        ucp_rkey_buffer_release(rkey_buffer);
    }

    for (i = 0; i < thread_count; i++) {
        free(vec[i * 3].iov_base);
        ucp_worker_release_address(perf->ucp.tctx[i].perf.ucp.worker,
                                   vec[(i * 3) + 1].iov_base);
    }

    free(vec);

    return UCS_OK;

err_free_workers_vec:
    for (j = 0; j < i; j++) {
        ucp_worker_destroy(perf->ucp.tctx[j].perf.ucp.worker);
    }
    free(vec);
err_rkey_release:
    if (features & (UCP_FEATURE_RMA|UCP_FEATURE_AMO32|UCP_FEATURE_AMO64)) {
        ucp_rkey_buffer_release(rkey_buffer);
    }
err:
    return status;
}

static void ucp_perf_worker_flush_callback(void *request, ucs_status_t status,
                                           void *user_data)
{
    ucp_perf_flush_context_t *ctx = user_data;

    --ctx->num_outstanding;
    if (status != UCS_OK) {
        ucs_error("worker flush callback got status %s",
                  ucs_status_string(status));
        ctx->status = status;
    }
    ucp_request_free(request);
}

static ucs_status_t ucp_perf_test_flush_workers(ucx_perf_context_t *perf)
{
    ucp_perf_flush_context_t ctx = {
        .num_outstanding = 0,
        .status          = UCS_OK
    };
    ucp_request_param_t param    = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                        UCP_OP_ATTR_FIELD_USER_DATA,
        .cb.send      = ucp_perf_worker_flush_callback,
        .user_data    = &ctx
    };
    void *flush_req;
    unsigned i;

    /* Initiate flush operation on all workers */
    for (i = 0; i < perf->params.thread_count; i++) {
        flush_req = ucp_worker_flush_nbx(perf->ucp.tctx[i].perf.ucp.worker,
                                         &param);
        if (UCS_PTR_IS_ERR(flush_req)) {
            ctx.status = UCS_PTR_STATUS(flush_req);
            ucs_error("ucp_worker_flush_nbx() failed on thread %d: %s", i,
                      ucs_status_string(ctx.status));
            break;
        }

        if (UCS_PTR_IS_PTR(flush_req)) {
            ++ctx.num_outstanding;
        }
    }

    /* Progress all workers in parallel to avoid deadlocks */
    while (ctx.num_outstanding > 0) {
        for (i = 0; i < perf->params.thread_count; i++) {
            ucp_worker_progress(perf->ucp.tctx[i].perf.ucp.worker);
        }
    }

    return ctx.status;
}

static ucs_status_t ucp_perf_test_setup_self_endpoints(ucx_perf_context_t *perf)
{
    ucp_worker_h worker = perf->ucp.tctx[0].perf.ucp.worker;
    ucs_status_t status;
    ucp_ep_params_t ep_params;
    ucx_perf_context_t *thread_perf;
    ucp_worker_attr_t worker_attr = {
        .field_mask = UCP_WORKER_ATTR_FIELD_ADDRESS
    };
    unsigned i;

    status = UCX_PERF_VERBOSE(error, &perf->params, ucp_worker_query, worker,
                              &worker_attr);
    if (status != UCS_OK) {
        return status;
    }

    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address    = worker_attr.address;

    for (i = 0; i < perf->params.thread_count; ++i) {
        thread_perf = &perf->ucp.tctx[i].perf;
        status = UCX_PERF_VERBOSE(error, &perf->params, ucp_ep_create, worker,
                                  &ep_params, &thread_perf->ucp.self_ep);
        if (status != UCS_OK) {
            goto err_destroy_eps;
        }

        status = ucp_perf_test_create_self_rkey(thread_perf, thread_perf->ucp.send_memh,
                                                &thread_perf->ucp.self_send_rkey);
        if (status != UCS_OK) {
            goto err_destroy_eps;
        }

        status = ucp_perf_test_create_self_rkey(thread_perf, thread_perf->ucp.recv_memh,
                                                &thread_perf->ucp.self_recv_rkey);
        if (status != UCS_OK) {
            goto err_destroy_eps;
        }
    }

    /* Global default ep will point to the ep of the 1st thread */
    perf->ucp.self_ep        = perf->ucp.tctx[0].perf.ucp.self_ep;
    perf->ucp.self_send_rkey = perf->ucp.tctx[0].perf.ucp.self_send_rkey;
    perf->ucp.self_recv_rkey = perf->ucp.tctx[0].perf.ucp.self_recv_rkey;

    status = UCS_OK;
    goto out;

err_destroy_eps:
    ucp_perf_test_destroy_self_eps(perf);
out:
    ucp_worker_release_address(worker, worker_attr.address);
    return status;
}

static ucs_status_t ucp_perf_test_setup_endpoints(ucx_perf_context_t *perf,
                                                  uint64_t features)
{
    unsigned group_size  = rte_call(perf, group_size);
    unsigned group_index = rte_call(perf, group_index);
    unsigned peer_index  = rte_peer_index(group_size, group_index);
    ucs_status_t status;

    if ((perf->params.flags & UCX_PERF_TEST_FLAG_LOOPBACK) &&
        (group_size != 1)) {
        ucs_error("perftest loopback requires group size to be 1 "
                  "(actual group size: %u)", group_size);
        return UCS_ERR_UNSUPPORTED;
    }

    if (!(perf->params.flags & UCX_PERF_TEST_FLAG_LOOPBACK) &&
        (group_size != 2)) {
        ucs_error("perftest p2p requires group size to be exactly 2 "
                  "(actual group size: %u)", group_size);
        return UCS_ERR_UNSUPPORTED;
    }

    /* Pack the local endpoints data and send to the remote peer */
    status = ucp_perf_test_send_local_data(perf, features);
    if (status != UCS_OK) {
        goto err;
    }

    /* Receive remote peer's endpoints' data and connect to them */
    status = ucp_perf_test_receive_remote_data(perf, peer_index);
    if (status != UCS_OK) {
        goto err;
    }

    /* Sync status across all processes */
    status = ucx_perf_test_exchange_status(perf, UCS_OK);
    if (status != UCS_OK) {
        goto err_destroy_eps;
    }

    /* Create self_ep used for copying into non-host memory types*/
    if (!ucp_perf_test_is_host_mem_only(&perf->params)) {
        status = ucp_perf_test_setup_self_endpoints(perf);
        if (status != UCS_OK) {
            goto err_destroy_eps;
        }
    }

    /* Force wireup completion */
    return ucp_perf_test_flush_workers(perf);

err_destroy_eps:
    ucp_perf_test_destroy_eps(perf);
err:
    (void)ucx_perf_test_exchange_status(perf, status);
    return status;
}

static void ucp_perf_test_cleanup_endpoints(ucx_perf_context_t *perf)
{
    ucp_perf_barrier(perf);
    ucp_perf_test_destroy_self_eps(perf);
    ucp_perf_test_destroy_eps(perf);
}

static void
ucp_perf_test_destroy_workers(ucx_perf_context_t *perf, unsigned count)
{
    unsigned i;

    for (i = 0; i < count; i++) {
        ucp_worker_destroy(perf->ucp.tctx[i].perf.ucp.worker);
    }
}

ucs_status_t
ucx_perf_do_warmup(ucx_perf_context_t *perf, const ucx_perf_params_t *params)
{
    ucs_time_t deadline = ucs_get_time() +
                          ucs_time_from_sec(params->warmup_time);
    ucx_perf_counter_t warmup_iter, total_warmup_iter;
    ucs_status_t status, stop_status;

    /* Perform no more than 'params->warmup_iter' iterations but try to not
       exceed 'params->warmup_time' */
    warmup_iter       = 1;
    total_warmup_iter = 0;
    while (total_warmup_iter < params->warmup_iter) {
        perf->max_iter        = warmup_iter;
        perf->report_interval = ULONG_MAX;

        status = ucx_perf_funcs[params->api].run(perf);
        if (status != UCS_OK) {
            return status;
        }

        ucx_perf_funcs[params->api].barrier(perf);
        ucx_perf_test_prepare_new_run(perf, params);

        /* Stop when reaching the deadline */
        stop_status = (ucs_get_time() > deadline) ? UCS_OK : UCS_INPROGRESS;

        if (params->thread_count == 1) {
            status = ucx_perf_test_exchange_status(perf, stop_status);
        } else {
#pragma omp barrier
#pragma omp single copyprivate(status)
            /* Synchronize on whether to continue or stop the warmup phase */
            status = ucx_perf_test_exchange_status(perf, stop_status);
#pragma omp barrier
        }

        if (status != UCS_INPROGRESS) {
            return status;
        }

        total_warmup_iter += warmup_iter;
        warmup_iter       *= 2;
    }

    return UCS_OK;
}

static ucs_status_t uct_perf_create_md(ucx_perf_context_t *perf)
{
    uct_component_h *uct_components;
    uct_component_attr_t component_attr;
    uct_tl_resource_desc_t *tl_resources;
    unsigned md_index, num_components;
    unsigned tl_index, num_tl_resources;
    unsigned cmpt_index;
    ucs_status_t status;
    uct_md_h md;
    uct_md_config_t *md_config;


    status = uct_query_components(&uct_components, &num_components);
    if (status != UCS_OK) {
        goto out;
    }

    for (cmpt_index = 0; cmpt_index < num_components; ++cmpt_index) {

        component_attr.field_mask = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCE_COUNT;
        status = uct_component_query(uct_components[cmpt_index], &component_attr);
        if (status != UCS_OK) {
            goto out_release_components_list;
        }

        component_attr.field_mask   = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCES;
        component_attr.md_resources = alloca(sizeof(*component_attr.md_resources) *
                                             component_attr.md_resource_count);
        status = uct_component_query(uct_components[cmpt_index], &component_attr);
        if (status != UCS_OK) {
            goto out_release_components_list;
        }

        for (md_index = 0; md_index < component_attr.md_resource_count; ++md_index) {
            status = uct_md_config_read(uct_components[cmpt_index], NULL, NULL,
                                        &md_config);
            if (status != UCS_OK) {
                goto out_release_components_list;
            }

            ucs_strncpy_zero(perf->params.uct.md_name,
                             component_attr.md_resources[md_index].md_name,
                             UCT_MD_NAME_MAX);

            status = uct_md_open(uct_components[cmpt_index],
                                 component_attr.md_resources[md_index].md_name,
                                 md_config, &md);
            uct_config_release(md_config);
            if (status != UCS_OK) {
                goto out_release_components_list;
            }

            status = uct_md_query_tl_resources(md, &tl_resources, &num_tl_resources);
            if (status != UCS_OK) {
                uct_md_close(md);
                goto out_release_components_list;
            }

            for (tl_index = 0; tl_index < num_tl_resources; ++tl_index) {
                if (!strcmp(perf->params.uct.tl_name,  tl_resources[tl_index].tl_name) &&
                    !strcmp(perf->params.uct.dev_name, tl_resources[tl_index].dev_name))
                {
                    uct_release_tl_resource_list(tl_resources);
                    perf->uct.cmpt = uct_components[cmpt_index];
                    perf->uct.md   = md;
                    status         = UCS_OK;
                    goto out_release_components_list;
                }
            }

            uct_md_close(md);
            uct_release_tl_resource_list(tl_resources);
        }
    }

    ucs_error("Cannot use "UCT_PERF_TEST_PARAMS_FMT,
              UCT_PERF_TEST_PARAMS_ARG(&perf->params));
    status = UCS_ERR_NO_DEVICE;

out_release_components_list:
    uct_release_component_list(uct_components);
out:
    return status;
}

void uct_perf_barrier(ucx_perf_context_t *perf)
{
    rte_call(perf, barrier, (void(*)(void*))uct_worker_progress,
             (void*)perf->uct.worker);
}

void ucp_perf_thread_barrier(ucx_perf_context_t *perf)
{
    rte_call(perf, barrier, (void(*)(void*))ucp_worker_progress,
#if _OPENMP
             (void*)perf->ucp.tctx[omp_get_thread_num()].perf.ucp.worker);
#else
             (void*)perf->ucp.tctx[0].perf.ucp.worker);
#endif
}

void ucp_perf_barrier(ucx_perf_context_t *perf)
{
    rte_call(perf, barrier, ucp_perf_worker_progress, perf);
}

static ucs_status_t uct_perf_setup(ucx_perf_context_t *perf)
{
    ucx_perf_params_t *params = &perf->params;
    uct_iface_config_t *iface_config;
    ucs_status_t status;
    uct_iface_params_t iface_params = {
        .field_mask           = UCT_IFACE_PARAM_FIELD_OPEN_MODE   |
                                UCT_IFACE_PARAM_FIELD_STATS_ROOT  |
                                UCT_IFACE_PARAM_FIELD_RX_HEADROOM |
                                UCT_IFACE_PARAM_FIELD_DEVICE,
        .open_mode            = UCT_IFACE_OPEN_MODE_DEVICE,
        .mode.device.tl_name  = params->uct.tl_name,
        .mode.device.dev_name = params->uct.dev_name,
        .stats_root           = ucs_stats_get_root(),
        .rx_headroom          = 0
    };

    if (params->thread_count > 1) {
        ucs_error("UCT tests do not support multi-thread mode");
        status = UCS_ERR_UNSUPPORTED;
        goto out;
    }

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

    status = uct_perf_test_check_capabilities(params, perf->uct.iface,
                                              perf->uct.md);

    status = ucx_perf_test_exchange_status(perf, status);
    if (status != UCS_OK) {
        goto out_iface_close;
    }

    status = uct_perf_test_alloc_mem(perf);
    if (status != UCS_OK) {
        goto out_iface_close;
    }

    /* Enable progress before `uct_iface_flush` and `uct_worker_progress` called
     * to give a chance to finish connection for some transports (ib/ud, tcp).
     * They may return UCS_INPROGRESS from `uct_iface_flush` when connections are
     * in progress */
    uct_iface_progress_enable(perf->uct.iface,
                              UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);

    status = uct_perf_test_setup_endpoints(perf);
    if (status != UCS_OK) {
        ucs_error("Failed to setup endpoints: %s", ucs_status_string(status));
        goto out_free_mem;
    }

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

static void ucp_perf_request_init(void *req)
{
    ucp_perf_request_t *request = req;

    request->context = NULL;
}

static ucs_status_t ucp_perf_setup(ucx_perf_context_t *perf)
{
    ucp_params_t ucp_params;
    ucp_worker_params_t worker_params;
    ucp_worker_attr_t worker_attr;
    ucp_config_t *config;
    ucs_status_t status;
    unsigned i, thread_count;
    size_t message_size;

    ucp_params.field_mask   = UCP_PARAM_FIELD_FEATURES |
                              UCP_PARAM_FIELD_REQUEST_SIZE |
                              UCP_PARAM_FIELD_REQUEST_INIT |
                              UCP_PARAM_FIELD_NAME;
    ucp_params.features     = 0;
    ucp_params.request_size = sizeof(ucp_perf_request_t);
    ucp_params.request_init = ucp_perf_request_init;
    ucp_params.name         = "perftest";

    if (perf->params.thread_count > 1) {
        /* when there is more than one thread, a ucp_worker would be created for
         * each. all of them will share the same ucp_context */
        ucp_params.field_mask        |= UCP_PARAM_FIELD_MT_WORKERS_SHARED;
        ucp_params.mt_workers_shared  = 1;
    }

    status = ucp_perf_test_fill_params(&perf->params, &ucp_params);
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

    thread_count = perf->params.thread_count;
    message_size = ucx_perf_get_message_size(&perf->params);

    status = ucp_perf_test_alloc_mem(perf);
    if (status != UCS_OK) {
        ucs_warn("ucp test failed to allocate memory");
        goto err_cleanup;
    }

    perf->ucp.tctx = calloc(thread_count, sizeof(ucx_perf_thread_context_t));
    if (perf->ucp.tctx == NULL) {
        ucs_warn("ucp test failed to allocate memory for thread contexts");
        goto err_free_mem;
    }

    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = perf->params.thread_mode;

    for (i = 0; i < thread_count; i++) {
        perf->ucp.tctx[i].tid              = i;
        perf->ucp.tctx[i].perf             = *perf;
        /* Doctor the src and dst buffers to make them thread specific */
        perf->ucp.tctx[i].perf.send_buffer =
                        UCS_PTR_BYTE_OFFSET(perf->send_buffer, i * message_size);
        perf->ucp.tctx[i].perf.recv_buffer =
                        UCS_PTR_BYTE_OFFSET(perf->recv_buffer, i * message_size);

        status = ucp_worker_create(perf->ucp.context, &worker_params,
                                   &perf->ucp.tctx[i].perf.ucp.worker);
        if (status != UCS_OK) {
            ucp_perf_test_destroy_workers(perf, i);
            goto err_free_tctx;
        }
    }

    if (perf->params.command == UCX_PERF_CMD_AM) {
        /* Check that requested AM header size is not larger than max supported. */
        worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_MAX_AM_HEADER;
        status = ucp_worker_query(perf->ucp.tctx[0].perf.ucp.worker,
                                  &worker_attr);
        if (status != UCS_OK) {
            goto err_destroy_workers;
        }

        if (worker_attr.max_am_header < perf->params.ucp.am_hdr_size) {
            ucs_error("AM header size (%zu) is larger than max supported (%zu)",
                      perf->params.ucp.am_hdr_size, worker_attr.max_am_header);
            status = UCS_ERR_INVALID_PARAM;
            goto err_destroy_workers;
        }
    }

    status = UCX_PERF_VERBOSE(error, &perf->params,
                              ucp_perf_test_setup_endpoints, perf,
                              ucp_params.features);
    if (status != UCS_OK) {
        goto err_destroy_workers;
    }

    return UCS_OK;

err_destroy_workers:
    ucp_perf_test_destroy_workers(perf, thread_count);
err_free_tctx:
    free(perf->ucp.tctx);
err_free_mem:
    ucp_perf_test_free_mem(perf);
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
    ucp_perf_test_destroy_workers(perf, perf->params.thread_count);
    free(perf->ucp.tctx);
    ucp_cleanup(perf->ucp.context);
}

ucx_perf_funcs_t ucx_perf_funcs[] = {
    [UCX_PERF_API_UCT] = {uct_perf_setup, uct_perf_cleanup,
                          uct_perf_test_dispatch, uct_perf_barrier},
    [UCX_PERF_API_UCP] = {ucp_perf_setup, ucp_perf_cleanup,
                          ucp_perf_test_dispatch, ucp_perf_thread_barrier}
};

ucs_status_t ucx_perf_allocators_init(ucx_perf_context_t *perf,
                                      const ucx_perf_params_t *params)
{
    if ((params->send_mem_type == UCS_MEMORY_TYPE_RDMA) ||
        (params->recv_mem_type == UCS_MEMORY_TYPE_RDMA)) {
        return UCS_OK;
    }

    ucs_debug("set send allocator by send mem type %s",
              ucs_memory_type_names[params->send_mem_type]);
    perf->send_allocator = ucx_perf_mem_type_allocators[params->send_mem_type];

    ucs_debug("set recv allocator by recv mem type %s",
              ucs_memory_type_names[params->recv_mem_type]);
    perf->recv_allocator = ucx_perf_mem_type_allocators[params->recv_mem_type];

    if ((perf->send_allocator == NULL) || (perf->recv_allocator == NULL)) {
        ucs_error("Unsupported memory types %s<->%s",
                  ucs_memory_type_names[params->send_mem_type],
                  ucs_memory_type_names[params->recv_mem_type]);
        return UCS_ERR_UNSUPPORTED;
    }

    if (perf->send_allocator->mem_type != UCS_MEMORY_TYPE_HOST) {
        ucs_diag("UCT tests also copy one-byte value from %s memory to "
                 "%s send memory, which may impact performance results",
                 ucs_memory_type_names[UCS_MEMORY_TYPE_HOST],
                 ucs_memory_type_names[perf->send_allocator->mem_type]);
    }

    if (perf->recv_allocator->mem_type != UCS_MEMORY_TYPE_HOST) {
        ucs_diag("UCT tests also copy one-byte value from %s recv memory "
                 "to %s memory, which may impact performance results",
                 ucs_memory_type_names[perf->recv_allocator->mem_type],
                 ucs_memory_type_names[UCS_MEMORY_TYPE_HOST]);
    }

    return ucx_perf_allocators_init_thread(perf);
}

ucs_status_t ucx_perf_run(const ucx_perf_params_t *params,
                          ucx_perf_result_t *result)
{
    ucx_perf_context_t *perf;
    ucs_status_t status;

    ucx_perf_global_init();

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

    perf->params = *params;

    status = ucx_perf_allocators_init(perf, params);
    if (status != UCS_OK) {
        goto out_free;
    }

    ucx_perf_test_prepare_new_run(perf, params);

    status = ucx_perf_funcs[params->api].setup(perf);
    if (status != UCS_OK) {
        goto out_free;
    }

    if (params->thread_count == 1) {
        if (params->api == UCX_PERF_API_UCP) {
            perf->ucp.worker         = perf->ucp.tctx[0].perf.ucp.worker;
            perf->ucp.ep             = perf->ucp.tctx[0].perf.ucp.ep;
            perf->ucp.remote_addr    = perf->ucp.tctx[0].perf.ucp.remote_addr;
            perf->ucp.rkey           = perf->ucp.tctx[0].perf.ucp.rkey;
            perf->ucp.self_ep        = perf->ucp.tctx[0].perf.ucp.self_ep;
            perf->ucp.self_send_rkey = perf->ucp.tctx[0].perf.ucp.self_send_rkey;
            perf->ucp.self_recv_rkey = perf->ucp.tctx[0].perf.ucp.self_recv_rkey;
        }

        status = ucx_perf_do_warmup(perf, params);
        if (status != UCS_OK) {
            goto out_cleanup;
        }

        /* Run test */
        status = ucx_perf_funcs[params->api].run(perf);
        ucx_perf_funcs[params->api].barrier(perf);
        if (status == UCS_OK) {
            ucx_perf_calc_result(perf, result);
            rte_call(perf, report, result, perf->params.report_arg,
                     perf->extra_info, 1, 0);
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

size_t ucx_perf_get_message_size(const ucx_perf_params_t *params)
{
    size_t length, it;

    ucs_assert(params->msg_size_list != NULL);

    length = 0;
    for (it = 0; it < params->msg_size_cnt; ++it) {
        length += params->msg_size_list[it];
    }

    return length;
}

unsigned rte_peer_index(unsigned group_size, unsigned group_index)
{
    unsigned peer_index = group_size - 1 - group_index;

    ucs_assert(group_index < group_size);
    return peer_index;
}

void ucx_perf_report(ucx_perf_context_t *perf)
{
    ucx_perf_result_t result;

    ucx_perf_get_time(perf);
    ucx_perf_calc_result(perf, &result);
    rte_call(perf, report, &result, perf->params.report_arg, "", 0, 0);
    perf->prev = perf->current;
}
