/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) The University of Tennessee and The University 
*               of Tennessee Research Foundation. 2015. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "libperf_int.h"

#include <ucs/debug/log.h>
#include <malloc.h>
#include <unistd.h>


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

    /* TODO use params->alignment  */

    status = uct_iface_mem_alloc(perf->uct.iface, 
                                 params->message_size * params->thread_count,
                                 "perftest", &perf->uct.send_mem);
    if (status != UCS_OK) {
        ucs_error("Failed allocate send buffer: %s", ucs_status_string(status));
        goto err;
    }

    ucs_assert(perf->uct.send_mem.pd == perf->uct.pd);
    perf->send_buffer = perf->uct.send_mem.address;

    status = uct_iface_mem_alloc(perf->uct.iface, 
                                 params->message_size * params->thread_count,
                                 "perftest", &perf->uct.recv_mem);
    if (status != UCS_OK) {
        ucs_error("Failed allocate receive buffer: %s", ucs_status_string(status));
        goto err_free_send;
    }

    ucs_assert(perf->uct.recv_mem.pd == perf->uct.pd);
    perf->recv_buffer = perf->uct.recv_mem.address;

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
    for (i = 0; i < TIMING_QUEUE_SIZE; ++i) {
        perf->timing_queue[i] = 0;
    }
}

void ucx_perf_test_cleanup(ucx_perf_context_t *perf)
{
    free(perf->send_buffer);
    free(perf->recv_buffer);
}

void ucx_perf_calc_result(ucx_perf_context_t *perf, ucx_perf_result_t *result)
{
    double latency_factor;
    double sec_value;

    sec_value = ucs_time_from_sec(1.0);
    if (perf->params.test_type == UCX_PERF_TEST_TYPE_PINGPONG) {
        latency_factor = 2.0;
    } else {
        latency_factor = 1.0;
    }

    result->iters = perf->current.iters;
    result->bytes = perf->current.bytes;
    result->elapsed_time = perf->current.time - perf->start_time;

    /* Latency */

    result->latency.typical =
        __find_median_quick_select(perf->timing_queue, TIMING_QUEUE_SIZE)
        / sec_value
        / latency_factor;

    result->latency.moment_average =
        (double)(perf->current.time - perf->prev.time)
        / (perf->current.iters - perf->prev.iters)
        / sec_value
        / latency_factor;

    result->latency.total_average =
        (double)(perf->current.time - perf->start_time)
        / perf->current.iters
        / sec_value
        / latency_factor;


    /* Bandwidth */

    result->bandwidth.typical = 0.0; // Undefined

    result->bandwidth.moment_average =
        (perf->current.bytes - perf->prev.bytes) * sec_value
        / (double)(perf->current.time - perf->prev.time);

    result->bandwidth.total_average =
        perf->current.bytes * sec_value
        / (double)(perf->current.time - perf->start_time);


    /* Packet rate */

    result->msgrate.typical = 0.0; // Undefined

    result->msgrate.moment_average =
        (perf->current.msgs - perf->prev.msgs) * sec_value
        / (double)(perf->current.time - perf->prev.time);

    result->msgrate.total_average =
        perf->current.msgs * sec_value
        / (double)(perf->current.time - perf->start_time);

}

static ucs_status_t ucx_perf_test_check_params(ucx_perf_params_t *params)
{
    if (params->message_size < 1) {
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

    return UCS_OK;
}

void uct_perf_iface_flush_b(ucx_perf_context_t *perf)
{
    while (uct_iface_flush(perf->uct.iface) == UCS_ERR_NO_RESOURCE) {
        uct_worker_progress(perf->uct.worker);
    }
}

static inline uint64_t __get_flag(uct_perf_data_layout_t layout, uint64_t short_f,
                                  uint64_t bcopy_f, uint64_t zcopy_f)
{
    return (layout == UCT_PERF_DATA_LAYOUT_SHORT) ? short_f :
           (layout == UCT_PERF_DATA_LAYOUT_BCOPY) ? bcopy_f :
           (layout == UCT_PERF_DATA_LAYOUT_ZCOPY) ? zcopy_f :
           0;
}

static inline uint64_t __get_atomic_flag(size_t size, uint64_t flag32, uint64_t flag64)
{
    return (size == 4) ? flag32 :
           (size == 8) ? flag64 :
           0;
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
    uct_iface_attr_t attr;
    ucs_status_t status;
    uint64_t required_flags;
    size_t max_size;

    status = uct_iface_query(iface, &attr);
    if (status != UCS_OK) {
        return status;
    }

    switch (params->command) {
    case UCX_PERF_CMD_AM:
        required_flags = __get_flag(params->uct.data_layout, UCT_IFACE_FLAG_AM_SHORT,
                                    UCT_IFACE_FLAG_AM_BCOPY, UCT_IFACE_FLAG_AM_ZCOPY);
        max_size = __get_max_size(params->uct.data_layout, attr.cap.am.max_short,
                                  attr.cap.am.max_bcopy, attr.cap.am.max_zcopy);
        break;
    case UCX_PERF_CMD_PUT:
        required_flags = __get_flag(params->uct.data_layout, UCT_IFACE_FLAG_PUT_SHORT,
                                    UCT_IFACE_FLAG_PUT_BCOPY, UCT_IFACE_FLAG_PUT_ZCOPY);
        max_size = __get_max_size(params->uct.data_layout, attr.cap.put.max_short,
                                  attr.cap.put.max_bcopy, attr.cap.put.max_zcopy);
        break;
    case UCX_PERF_CMD_GET:
        required_flags = __get_flag(params->uct.data_layout, 0,
                                    UCT_IFACE_FLAG_GET_BCOPY, UCT_IFACE_FLAG_GET_ZCOPY);
        max_size = __get_max_size(params->uct.data_layout, 0,
                                  attr.cap.get.max_bcopy, attr.cap.get.max_zcopy);
        break;
    case UCX_PERF_CMD_ADD:
        required_flags = __get_atomic_flag(params->message_size, UCT_IFACE_FLAG_ATOMIC_ADD32,
                                           UCT_IFACE_FLAG_ATOMIC_ADD64);
        max_size = 8;
        break;
    case UCX_PERF_CMD_FADD:
        required_flags = __get_atomic_flag(params->message_size, UCT_IFACE_FLAG_ATOMIC_FADD32,
                                           UCT_IFACE_FLAG_ATOMIC_FADD64);
        max_size = 8;
        break;
    case UCX_PERF_CMD_SWAP:
        required_flags = __get_atomic_flag(params->message_size, UCT_IFACE_FLAG_ATOMIC_SWAP32,
                                           UCT_IFACE_FLAG_ATOMIC_SWAP64);
        max_size = 8;
        break;
    case UCX_PERF_CMD_CSWAP:
        required_flags = __get_atomic_flag(params->message_size, UCT_IFACE_FLAG_ATOMIC_CSWAP32,
                                           UCT_IFACE_FLAG_ATOMIC_CSWAP64);
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

    if ((attr.cap.flags & required_flags) == 0) {
        if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
            ucs_error("Device does not support required operation");
        }
        return UCS_ERR_UNSUPPORTED;
    }

    if (params->message_size > max_size) {
        if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
            ucs_error("Message size too big");
        }
        return UCS_ERR_INVALID_PARAM;
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
                ucs_error("AM header size too big");
            }
            return UCS_ERR_INVALID_PARAM;
        }

        if (params->am_hdr_size > params->message_size) {
            if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
                ucs_error("AM header size larger than message size");
            }
            return UCS_ERR_INVALID_PARAM;
        }

        if (params->uct.fc_window > UCT_PERF_TEST_MAX_FC_WINDOW) {
            if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
                ucs_error("AM flow-control window too large (should be <= %d)",
                          UCT_PERF_TEST_MAX_FC_WINDOW);
            }
            return UCS_ERR_INVALID_PARAM;
        }

        if ((params->flags & UCX_PERF_TEST_FLAG_ONE_SIDED) &&
            (params->flags & UCX_PERF_TEST_FLAG_VERBOSE))
        {
            ucs_warn("Running active-message test with on-sided progress");
        }
    }

    return UCS_OK;
}

static ucs_status_t uct_perf_test_setup_endpoints(ucx_perf_context_t *perf)
{
    unsigned group_size, i, group_index;
    struct sockaddr *iface_addr;
    struct sockaddr *ep_addr;
    uct_iface_attr_t iface_attr;
    uct_pd_attr_t pd_attr;
    unsigned long va;
    void *rkey_buffer;
    ucs_status_t status;
    struct iovec vec[4];
    void *req;

    status = uct_iface_query(perf->uct.iface, &iface_attr);
    if (status != UCS_OK) {
        ucs_error("Failed to uct_iface_query: %s", ucs_status_string(status));
        goto err;
    }

    status = uct_pd_query(perf->uct.pd, &pd_attr);
    if (status != UCS_OK) {
        ucs_error("Failed to uct_pd_query: %s", ucs_status_string(status));
        goto err;
    }

    iface_addr  = calloc(1, iface_attr.iface_addr_len);
    ep_addr     = calloc(1, iface_attr.ep_addr_len);
    rkey_buffer = calloc(1, pd_attr.rkey_packed_size);
    if ((iface_addr == NULL) || (ep_addr == NULL) || (rkey_buffer == NULL)) {
        goto err_free;
    }

    if (iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
        status = uct_iface_get_address(perf->uct.iface, iface_addr);
        if (status != UCS_OK) {
            ucs_error("Failed to uct_iface_get_address: %s", ucs_status_string(status));
            goto err_free;
        }
    }

    status = uct_pd_mkey_pack(perf->uct.pd, perf->uct.recv_mem.memh, rkey_buffer);
    if (status != UCS_OK) {
        ucs_error("Failed to uct_rkey_pack: %s", ucs_status_string(status));
        goto err_free;
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

    va                  = (uintptr_t)perf->recv_buffer;
    vec[0].iov_base     = &va;
    vec[0].iov_len      = sizeof(va);
    vec[1].iov_base     = rkey_buffer;
    vec[1].iov_len      = pd_attr.rkey_packed_size;
    vec[2].iov_base     = iface_addr;
    vec[2].iov_len      = iface_attr.iface_addr_len;
    vec[3].iov_base     = ep_addr;
    vec[3].iov_len      = iface_attr.ep_addr_len;

    rte_call(perf, post_vec, vec , 4, &req);
    rte_call(perf, exchange_vec, req);

    for (i = 0; i < group_size; ++i) {
        if (i == group_index) {
            continue;
        }
        vec[0].iov_base     = &va;
        vec[0].iov_len      = sizeof(va);
        vec[1].iov_base     = rkey_buffer;
        vec[1].iov_len      = pd_attr.rkey_packed_size;
        vec[2].iov_base     = iface_addr;
        vec[2].iov_len      = iface_attr.iface_addr_len;
        vec[3].iov_base     = ep_addr;
        vec[3].iov_len      = iface_attr.ep_addr_len;

        rte_call(perf, recv_vec, i, vec , 4, req);

        perf->uct.peers[i].remote_addr = va;
        status = uct_rkey_unpack(rkey_buffer, &perf->uct.peers[i].rkey);
        if (status != UCS_OK) {
            ucs_error("Failed to uct_rkey_unpack: %s", ucs_status_string(status));
            return status;
        }

        if (iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
            status = uct_ep_connect_to_ep(perf->uct.peers[i].ep, ep_addr);
        } else if (iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
            status = uct_ep_create_connected(perf->uct.iface, iface_addr,
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

    rte_call(perf, barrier);

    free(iface_addr);
    free(ep_addr);
    free(rkey_buffer);

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
    free(iface_addr);
    free(ep_addr);
    free(rkey_buffer);
err:
    return status;
}

static void uct_perf_test_cleanup_endpoints(ucx_perf_context_t *perf)
{
    unsigned group_size, group_index, i;

    rte_call(perf, barrier);

    uct_iface_set_am_handler(perf->uct.iface, UCT_PERF_TEST_AM_ID, NULL, NULL);

    group_size  = rte_call(perf, group_size);
    group_index = rte_call(perf, group_index);

    for (i = 0; i < group_size; ++i) {
        if (i != group_index) {
            uct_rkey_release(&perf->uct.peers[i].rkey);
            if (perf->uct.peers[i].ep) {
                uct_ep_destroy(perf->uct.peers[i].ep);
            }
        }
    }
    free(perf->uct.peers);
}

static ucs_status_t ucp_perf_test_check_params(ucx_perf_params_t *params,
                                               uint64_t *features)
{
    ucs_status_t status;

    switch (params->command) {
    case UCX_PERF_CMD_PUT:
    case UCX_PERF_CMD_GET:
        *features = UCP_FEATURE_RMA;
        break;
    case UCX_PERF_CMD_ADD:
    case UCX_PERF_CMD_FADD:
    case UCX_PERF_CMD_SWAP:
    case UCX_PERF_CMD_CSWAP:
        if ((params->message_size != sizeof(uint32_t)) &&
            (params->message_size != sizeof(uint64_t)))
        {
            if (params->flags & UCX_PERF_TEST_FLAG_VERBOSE) {
                ucs_error("Atomic size should be either 32 or 64 bit");
            }
            return UCS_ERR_INVALID_PARAM;
        }

        *features = UCP_FEATURE_AMO;
        break;
    case UCX_PERF_CMD_TAG:
        *features = UCP_FEATURE_TAG;
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

static ucs_status_t ucp_perf_test_alloc_mem(ucx_perf_context_t *perf, ucx_perf_params_t *params)
{
    ucs_status_t status;

    perf->send_buffer = NULL;
    status = ucp_mem_map(perf->ucp.context, &perf->send_buffer,
                         params->message_size * params->thread_count,
                         0, &perf->ucp.send_memh);
    if (status != UCS_OK) {
        goto err;
    }

    perf->recv_buffer = NULL;
    status = ucp_mem_map(perf->ucp.context, &perf->recv_buffer,
                         params->message_size * params->thread_count,
                         0, &perf->ucp.recv_memh);
    if (status != UCS_OK) {
        goto err_free_send_buffer;
    }

    return UCS_OK;

err_free_send_buffer:
    ucp_mem_unmap(perf->ucp.context, perf->ucp.send_memh);
err:
    return UCS_ERR_NO_MEMORY;
}

static void ucp_perf_test_free_mem(ucx_perf_context_t *perf)
{
    ucp_mem_unmap(perf->ucp.context, perf->ucp.recv_memh);
    ucp_mem_unmap(perf->ucp.context, perf->ucp.send_memh);
}

static void ucp_perf_test_destroy_eps(ucx_perf_context_t* perf,
                                      unsigned group_size)
{
    unsigned i;

    for (i = 0; i < group_size; ++i) {
        if (perf->ucp.peers[i].rkey != NULL) {
            ucp_rkey_destroy(perf->ucp.peers[i].rkey);
        }
        if (perf->ucp.peers[i].ep != NULL) {
            ucp_ep_destroy(perf->ucp.peers[i].ep);
        }
    }
    free(perf->ucp.peers);
}

static ucs_status_t ucp_perf_test_setup_endpoints(ucx_perf_context_t *perf)
{
    unsigned group_size, i, group_index;
    ucp_address_t *address;
    size_t address_length = 0;
    ucs_status_t status;
    struct iovec vec[3];
    void *rkey_buffer;
    size_t rkey_size;
    void *req = NULL;

    group_size  = rte_call(perf, group_size);
    group_index = rte_call(perf, group_index);

    status = ucp_worker_get_address(perf->ucp.worker, &address, &address_length);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucp_rkey_pack(perf->ucp.context, perf->ucp.recv_memh, &rkey_buffer,
                           &rkey_size);
    if (status != UCS_OK) {
        ucp_worker_release_address(perf->ucp.worker, address);
        goto err;
    }

    vec[0].iov_base = address;
    vec[0].iov_len  = address_length;
    vec[1].iov_base = &perf->recv_buffer;
    vec[1].iov_len  = sizeof(uintptr_t);
    vec[2].iov_base = rkey_buffer;
    vec[2].iov_len  = rkey_size;

    rte_call(perf, post_vec, vec, 3, &req);

    ucp_rkey_buffer_release(rkey_buffer);
    ucp_worker_release_address(perf->ucp.worker, address);

    rte_call(perf, exchange_vec, req);

    perf->ucp.peers = calloc(group_size, sizeof(*perf->uct.peers));
    if (perf->ucp.peers == NULL) {
        goto err;
    }

    for (i = 0; i < group_size; ++i) {
        if (i == group_index) {
            continue;
        }

        address     = malloc(address_length);
        rkey_buffer = malloc(rkey_size);

        vec[0].iov_base = address;
        vec[0].iov_len  = address_length;
        vec[1].iov_base = &perf->ucp.peers[i].remote_addr;
        vec[1].iov_len  = sizeof(uintptr_t);
        vec[2].iov_base = rkey_buffer;
        vec[2].iov_len  = rkey_size;

        rte_call(perf, recv_vec, i, vec, 3, req);

        status = ucp_ep_create(perf->ucp.worker, address, &perf->ucp.peers[i].ep);
        if (status != UCS_OK) {
            if (perf->params.flags & UCX_PERF_TEST_FLAG_VERBOSE) {
                ucs_error("ucp_ep_create() failed: %s", ucs_status_string(status));
            }
            free(rkey_buffer);
            free(address);
            goto err_destroy_eps;
        }

        free(address);

        status = ucp_ep_rkey_unpack(perf->ucp.peers[i].ep, rkey_buffer,
                                    &perf->ucp.peers[i].rkey);
        if (status != UCS_OK) {
            if (perf->params.flags & UCX_PERF_TEST_FLAG_VERBOSE) {
                ucs_error("ucp_rkey_unpack() failed: %s", ucs_status_string(status));
            }
            free(rkey_buffer);
            goto err_destroy_eps;
        }

        free(rkey_buffer);
    }

    rte_call(perf, barrier);
    return UCS_OK;

err_destroy_eps:
    ucp_perf_test_destroy_eps(perf, group_size);
err:
    return status;
}

static void ucp_perf_test_cleanup_endpoints(ucx_perf_context_t *perf)
{
    unsigned group_size;

    rte_call(perf, barrier);

    group_size  = rte_call(perf, group_size);

    ucp_perf_test_destroy_eps(perf, group_size);
}

static void ucx_perf_set_warmup(ucx_perf_context_t* perf, ucx_perf_params_t* params)
{
    perf->max_iter = ucs_min(params->warmup_iter, params->max_iter / 10);
    perf->report_interval = -1;
}

static ucs_status_t uct_perf_create_pd(ucx_perf_context_t *perf)
{
    uct_pd_resource_desc_t *pd_resources;
    uct_tl_resource_desc_t *tl_resources;
    unsigned i, num_pd_resources;
    unsigned j, num_tl_resources;
    ucs_status_t status;
    uct_pd_h pd;
    uct_pd_config_t *pd_config;

    status = uct_query_pd_resources(&pd_resources, &num_pd_resources);
    if (status != UCS_OK) {
        goto out;
    }

    for (i = 0; i < num_pd_resources; ++i) {
        status = uct_pd_config_read(pd_resources[i].pd_name, NULL, NULL, &pd_config);
        if (status != UCS_OK) {
            goto out_release_pd_resources;
        }

        status = uct_pd_open(pd_resources[i].pd_name, pd_config, &pd);
        uct_config_release(pd_config);
        if (status != UCS_OK) {
            goto out_release_pd_resources;
        }

        status = uct_pd_query_tl_resources(pd, &tl_resources, &num_tl_resources);
        if (status != UCS_OK) {
            uct_pd_close(pd);
            goto out_release_pd_resources;
        }

        for (j = 0; j < num_tl_resources; ++j) {
            if (!strcmp(perf->params.uct.tl_name,  tl_resources[j].tl_name) &&
                !strcmp(perf->params.uct.dev_name, tl_resources[j].dev_name))
            {
                uct_release_tl_resource_list(tl_resources);
                perf->uct.pd = pd;
                status = UCS_OK;
                goto out_release_pd_resources;
            }
        }

        uct_pd_close(pd);
        uct_release_tl_resource_list(tl_resources);
    }

    ucs_error("Cannot use transport %s on device %s", perf->params.uct.tl_name,
              perf->params.uct.dev_name);
    status = UCS_ERR_NO_DEVICE;

out_release_pd_resources:
    uct_release_pd_resource_list(pd_resources);
out:
    return status;
}

static ucs_status_t uct_perf_setup(ucx_perf_context_t *perf, ucx_perf_params_t *params)
{
    uct_iface_config_t *iface_config;
    ucs_status_t status;

    status = ucs_async_context_init(&perf->uct.async, UCS_ASYNC_MODE_THREAD);
    if (status != UCS_OK) {
        goto out;
    }

    status = uct_worker_create(&perf->uct.async, params->thread_mode,
                               &perf->uct.worker);
    if (status != UCS_OK) {
        goto out_cleanup_async;
    }

    status = uct_perf_create_pd(perf);
    if (status != UCS_OK) {
        goto out_destroy_worker;
    }

    status = uct_iface_config_read(params->uct.tl_name, NULL, NULL, &iface_config);
    if (status != UCS_OK) {
        goto out_destroy_pd;
    }

    status = uct_iface_open(perf->uct.pd, perf->uct.worker, params->uct.tl_name,
                            params->uct.dev_name, 0, iface_config, &perf->uct.iface);
    uct_config_release(iface_config);
    if (status != UCS_OK) {
        ucs_error("Failed to open iface: %s", ucs_status_string(status));
        goto out_destroy_pd;
    }

    status = uct_perf_test_check_capabilities(params, perf->uct.iface);
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

    return UCS_OK;

out_free_mem:
    uct_perf_test_free_mem(perf);
out_iface_close:
    uct_iface_close(perf->uct.iface);
out_destroy_pd:
    uct_pd_close(perf->uct.pd);
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
    uct_pd_close(perf->uct.pd);
    uct_worker_destroy(perf->uct.worker);
    ucs_async_context_cleanup(&perf->uct.async);
}

static ucs_status_t ucp_perf_setup(ucx_perf_context_t *perf, ucx_perf_params_t *params)
{
    ucp_params_t ucp_params;
    ucp_config_t *config;
    ucs_status_t status;
    uint64_t features;

    status = ucp_perf_test_check_params(params, &features);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucp_config_read(NULL, NULL, &config);
    if (status != UCS_OK) {
        goto err;
    }

    ucp_params.features        = features;
    ucp_params.request_size    = 0;
    ucp_params.request_init    = NULL;
    ucp_params.request_cleanup = NULL;

    status = ucp_init(&ucp_params, config, &perf->ucp.context);
    ucp_config_release(config);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucp_worker_create(perf->ucp.context, params->thread_mode,
                               &perf->ucp.worker);
    if (status != UCS_OK) {
        goto err_cleanup;
    }

    status = ucp_perf_test_alloc_mem(perf, params);
    if (status != UCS_OK) {
        ucs_warn("ucp test failed to alocate memory");
        goto err_destroy_worker;
    }

    status = ucp_perf_test_setup_endpoints(perf);
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
    ucp_perf_test_free_mem(perf);
    ucp_worker_destroy(perf->ucp.worker);
    ucp_cleanup(perf->ucp.context);
}

static struct {
    ucs_status_t (*setup)(ucx_perf_context_t *perf, ucx_perf_params_t *params);
    void         (*cleanup)(ucx_perf_context_t *perf);
    ucs_status_t (*run)(ucx_perf_context_t *perf);
} ucx_perf_funcs[] = {
    [UCX_PERF_API_UCT] = {uct_perf_setup, uct_perf_cleanup, uct_perf_test_dispatch},
    [UCX_PERF_API_UCP] = {ucp_perf_setup, ucp_perf_cleanup, ucp_perf_test_dispatch}
};

static int ucx_perf_thread_spawn(ucx_perf_params_t* params, 
                                 ucx_perf_result_t* result);

ucs_status_t ucx_perf_run(ucx_perf_params_t *params, ucx_perf_result_t *result)
{
    ucx_perf_context_t perf;
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

    if (UCS_THREAD_MODE_SINGLE != params->thread_mode) {
        return ucx_perf_thread_spawn(params, result);
    }
        
    ucx_perf_test_reset(&perf, params);

    status = ucx_perf_funcs[params->api].setup(&perf, params);
    if (status != UCS_OK) {
        goto out;
    }

    if (params->warmup_iter > 0) {
        ucx_perf_set_warmup(&perf, params);
        status = ucx_perf_funcs[params->api].run(&perf);
        if (status != UCS_OK) {
            goto out_cleanup;
        }

        rte_call(&perf, barrier);
        ucx_perf_test_reset(&perf, params);
    }

    /* Run test */
    status = ucx_perf_funcs[params->api].run(&perf);
    rte_call(&perf, barrier);
    if (status == UCS_OK) {
        ucx_perf_calc_result(&perf, result);
        rte_call(&perf, report, result, 1);
    }

out_cleanup:
    ucx_perf_funcs[params->api].cleanup(&perf);
out:
    return status;
}


/* multiple threads sharing the same worker/iface */

typedef struct {
    pthread_t           pt;
    int                 tid;
    int                 ntid;
    pthread_barrier_t*  tbarrier;
    ucs_status_t*       statuses;
    ucx_perf_context_t  perf;
    ucx_perf_params_t   params;
    ucx_perf_result_t   result;
} ucx_perf_thread_context_t;

static void* ucx_perf_thread_run_test(void* arg) {
    ucx_perf_thread_context_t* tctx = (ucx_perf_thread_context_t*) arg;
    ucx_perf_params_t* params = &tctx->params;
    ucx_perf_result_t* result = &tctx->result;
    ucx_perf_context_t* perf = &tctx->perf;
    ucs_status_t* statuses = tctx->statuses;
    pthread_barrier_t* tbarrier = tctx->tbarrier;
    int tid = tctx->tid;
    int i;

    if (params->warmup_iter > 0) {
        ucx_perf_set_warmup(perf, params);
        statuses[tid] = ucx_perf_funcs[params->api].run(perf);
        pthread_barrier_wait(tbarrier);
        for (i = 0; i < tctx->ntid; i++) {
            if (UCS_OK != statuses[i]) {
                goto out;
            }
        }
        if (0 == tid) {
            rte_call(perf, barrier);
            ucx_perf_test_reset(perf, params);
        }
    }

    /* Run test */
    pthread_barrier_wait(tbarrier);
    statuses[tid] = ucx_perf_funcs[params->api].run(perf);
    pthread_barrier_wait(tbarrier);
    for (i = 0; i < tctx->ntid; i++) {
        if (UCS_OK != statuses[i]) {
            goto out;
        }
    }
    if (0 == tid) {
        rte_call(perf, barrier);
        /* Assuming all threads are fairly treated, reporting only tid==0
            TODO: aggregate reports */
        ucx_perf_calc_result(perf, result);
        rte_call(perf, report, result, 1);
    }

out:
    return &statuses[tid];
}

static int ucx_perf_thread_spawn(ucx_perf_params_t* params, 
                                 ucx_perf_result_t* result) {
    ucx_perf_context_t perf;
    ucs_status_t status;
    int ti;
    int nti = params->thread_count;

    ucx_perf_thread_context_t* tctx = 
        calloc(nti, sizeof(ucx_perf_thread_context_t));
    ucs_status_t* statuses = 
        calloc(nti, sizeof(ucs_status_t));
    pthread_barrier_t tbarrier;
    pthread_barrier_init(&tbarrier, NULL, nti);

    ucx_perf_test_reset(&perf, params);
    status = ucx_perf_funcs[params->api].setup(&perf, params);
    if (UCS_OK != status) {
        goto out_cleanup;
    }

    for (ti = 0; ti < nti; ti++) {
        tctx[ti].tid = ti;
        tctx[ti].ntid = nti;
        tctx[ti].tbarrier = &tbarrier;
        tctx[ti].statuses = statuses;
        tctx[ti].params = *params;
        tctx[ti].perf = perf;
        /* Doctor the src and dst buffers to make them thread specific */
        tctx[ti].perf.send_buffer += ti * params->message_size;
        tctx[ti].perf.recv_buffer += ti * params->message_size;
        pthread_create(&tctx[ti].pt, NULL, 
                       ucx_perf_thread_run_test, (void*)&tctx[ti]);
    }
    
    for (ti = 0; ti < nti; ti++) {
        pthread_join(tctx[ti].pt, NULL);
        if (UCS_OK != statuses[ti]) {
            ucs_error("Thread %d failed to run test: %s", tctx[ti].tid, ucs_status_string(statuses[ti]));
            status = statuses[ti];
        }
    }
    
    ucx_perf_funcs[params->api].cleanup(&perf);

out_cleanup:
    pthread_barrier_destroy(&tbarrier);
    free(statuses);
    free(tctx);

    return status;
}

