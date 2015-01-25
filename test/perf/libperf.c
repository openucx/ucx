/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
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

static ucs_status_t ucx_perf_test_init(ucx_perf_context_t *perf,
                                          ucx_perf_test_params_t *params)
{
    perf->send_buffer = memalign(params->alignment, params->message_size);
    perf->recv_buffer = memalign(params->alignment, params->message_size);
    if (perf->send_buffer == NULL || perf->recv_buffer == NULL) {
        return UCS_ERR_NO_MEMORY;
    }
    ucs_debug("Allocating memory. Send buffer %p, Recv buffer %p",
              perf->send_buffer, perf->recv_buffer);
    return UCS_OK;
}

void ucx_perf_test_start_clock(ucx_perf_context_t *perf)
{
    perf->start_time        = ucs_get_time();
    perf->prev_time         = perf->start_time;
    perf->prev.time         = perf->start_time;
}

static void ucx_perf_test_reset(ucx_perf_context_t *perf,
                                ucx_perf_test_params_t *params)
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

static void uct_perf_iface_flush_b(uct_perf_context_t *perf)
{
    while (uct_iface_flush(perf->iface) == UCS_ERR_WOULD_BLOCK) {
        uct_progress(perf->context);
    }
}

ucs_status_t ucx_perf_am_hander(void *data, unsigned length, void *arg)
{
    *(uint64_t*)arg = *(uint64_t*)data; /* Only copy out the header */
    return UCS_OK;
}

ucs_status_t uct_perf_test_setup_endpoints(uct_perf_context_t *perf)
{
    unsigned group_size, i, group_index;
    uct_iface_addr_t *iface_addr;
    uct_ep_addr_t *ep_addr;
    uct_iface_attr_t iface_attr;
    uct_pd_attr_t pd_attr;
    unsigned long address;
    void *rkey_buffer;
    ucs_status_t status;
    struct iovec vec[4];
    void *req;

    status = uct_iface_query(perf->iface, &iface_attr);
    if (status != UCS_OK) {
        ucs_error("Failed to uct_iface_query: %s", ucs_status_string(status));
        goto err;
    }

    status = uct_pd_query(perf->iface->pd, &pd_attr);
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

    status = uct_iface_get_address(perf->iface, iface_addr);
    if (status != UCS_OK) {
        ucs_error("Failed to uct_iface_get_address: %s", ucs_status_string(status));
        goto err_free;
    }

    status = uct_rkey_pack(perf->iface->pd, perf->recv_lkey, rkey_buffer);
    if (status != UCS_OK) {
        ucs_error("Failed to uct_rkey_pack: %s", ucs_status_string(status));
        goto err_free;
    }

    address = (uintptr_t)perf->super.recv_buffer;

    group_size  = rte_call(&perf->super, group_size);
    group_index = rte_call(&perf->super, group_index);

    perf->peers = calloc(group_size, sizeof(*perf->peers));
    if (perf->peers == NULL) {
        goto err_free;
    }

    for (i = 0; i < group_size; ++i) {
        if (i != group_index) {
            status = uct_ep_create(perf->iface, &perf->peers[i].ep);
            if (status != UCS_OK) {
                ucs_error("Failed to uct_ep_create: %s", ucs_status_string(status));
                goto err_destroy_eps;
            }
            status = uct_ep_get_address(perf->peers[i].ep, ep_addr);
            if (status != UCS_OK) {
                ucs_error("Failed to uct_ep_get_address: %s", ucs_status_string(status));
                goto err_destroy_eps;
            }
        }
    }

    vec[0].iov_base     = &address;
    vec[0].iov_len      = sizeof(address);
    vec[1].iov_base     = rkey_buffer;
    vec[1].iov_len      = pd_attr.rkey_packed_size;
    vec[2].iov_base     = iface_addr;
    vec[2].iov_len      = iface_attr.iface_addr_len;
    vec[3].iov_base     = ep_addr;
    vec[3].iov_len      = iface_attr.ep_addr_len;

    rte_call(&perf->super, post_vec, vec , 4, &req);
    rte_call(&perf->super, exchange_vec, req);

    for (i = 0; i < group_size; ++i) {
        if (i != group_index) {
            vec[0].iov_base     = &address;
            vec[0].iov_len      = sizeof(address);
            vec[1].iov_base     = rkey_buffer;
            vec[1].iov_len      = pd_attr.rkey_packed_size;
            vec[2].iov_base     = iface_addr;
            vec[2].iov_len      = iface_attr.iface_addr_len;
            vec[3].iov_base     = ep_addr;
            vec[3].iov_len      = iface_attr.ep_addr_len;

            rte_call(&perf->super, recv_vec, i, vec , 4, req);

            perf->peers[i].remote_addr = address;
            status = uct_rkey_unpack(perf->context, rkey_buffer, &perf->peers[i].rkey);
            if (status != UCS_OK) {
                ucs_error("Failed to uct_rkey_unpack: %s", ucs_status_string(status));
                return status;
            }

            status = uct_ep_connect_to_ep(perf->peers[i].ep, iface_addr, ep_addr);
            if (status != UCS_OK) {
                ucs_error("Failed to uct_ep_connect_to_ep: %s", ucs_status_string(status));
                goto err_destroy_eps;
            }
        }
    }
    uct_perf_iface_flush_b(perf);

    status = uct_set_am_handler(perf->iface, UCT_PERF_TEST_AM_ID,
                                ucx_perf_am_hander, perf->super.recv_buffer);
    if (status != UCS_OK) {
        ucs_error("Failed to uct_set_am_handler: %s", ucs_status_string(status));
        goto err_destroy_eps;
    }

    rte_call(&perf->super, barrier);

    free(iface_addr);
    free(ep_addr);
    free(rkey_buffer);

    return UCS_OK;

err_destroy_eps:
    for (i = 0; i < group_size; ++i) {
        if (perf->peers[i].rkey.type != NULL) {
            uct_rkey_release(perf->context, &perf->peers[i].rkey);
        }
        if (perf->peers[i].ep != NULL) {
            uct_ep_destroy(perf->peers[i].ep);
        }
    }
    free(perf->peers);
err_free:
    free(iface_addr);
    free(ep_addr);
    free(rkey_buffer);
err:
    return status;
}

void uct_perf_test_cleanup_endpoints(uct_perf_context_t *perf)
{
    unsigned group_size, i;

    rte_call(&perf->super, barrier);

    uct_set_am_handler(perf->iface, UCT_PERF_TEST_AM_ID, NULL, NULL);

    group_size = rte_call(&perf->super, group_size);
    for (i = 0; i < group_size; ++i) {
        uct_rkey_release(perf->context, &perf->peers[i].rkey);
        if (perf->peers[i].ep) {
            uct_ep_destroy(perf->peers[i].ep);
        }
    }
    free(perf->peers);
}


ucs_status_t uct_perf_test_run(uct_context_h context,ucx_perf_test_params_t *params,
                               const char *tl_name, const char *dev_name,
                               uct_iface_config_t *iface_config,
                               ucx_perf_result_t *result)
{
    uct_perf_context_t perf;
    ucs_status_t status;
    void *address;
    size_t length;

    perf.context = context;

    status = ucx_perf_test_init(&perf.super, params);
    if (status != UCS_OK) {
        goto out;
    }

    ucx_perf_test_reset(&perf.super, params);

    status = uct_iface_open(perf.context, tl_name, dev_name, 0, iface_config,
                            &perf.iface);
    if (status != UCS_OK) {
        ucs_error("Failed to open iface: %s", ucs_status_string(status));
        goto out_test_cleanup;
    }

    address = perf.super.send_buffer;
    length  = perf.super.params.message_size;
    status = uct_mem_map(perf.iface->pd, &address, &length, 0, &perf.send_lkey);
    if (status != UCS_OK) {
        goto out_iface_close;
    }

    address = perf.super.recv_buffer;
    length  = perf.super.params.message_size;
    status = uct_mem_map(perf.iface->pd, &address, &length, 0, &perf.recv_lkey);
    if (status != UCS_OK) {
        goto out_mem_unmap_send;
    }

    status = uct_perf_test_setup_endpoints(&perf);
    if (status != UCS_OK) {
        ucs_error("Failed to setup endpoints: %s", ucs_status_string(status));
        goto out_mem_unmap_recv;
    }

    if (params->warmup_iter > 0) {
        perf.super.params.max_iter        = params->warmup_iter;
        perf.super.params.report_interval = 1e10;
        uct_perf_test_dispatch(&perf);
        uct_perf_iface_flush_b(&perf);
        rte_call(&perf.super, barrier);
        ucx_perf_test_reset(&perf.super, params);
    }

    /* Run test */
    status = uct_perf_test_dispatch(&perf);
    uct_perf_iface_flush_b(&perf);
    rte_call(&perf.super, barrier);

    if (status == UCS_OK) {
        ucx_perf_calc_result(&perf.super, result);
    }

    uct_perf_test_cleanup_endpoints(&perf);
out_mem_unmap_recv:
    uct_mem_unmap(perf.iface->pd, perf.recv_lkey);
out_mem_unmap_send:
    uct_mem_unmap(perf.iface->pd, perf.send_lkey);
out_iface_close:
    uct_iface_close(perf.iface);
out_test_cleanup:
    ucx_perf_test_cleanup(&perf.super);
out:
    return status;
}

