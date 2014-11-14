/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "libperf.h"

#include <ucs/debug/log.h>
#include <ucs/time/time.h>
#include <malloc.h>
#include <unistd.h>

#define TIMING_QUEUE_SIZE   2048


typedef struct ucx_perf_context {
    ucx_perf_test_params_t       params;

    /* Buffers */
    void                         *send_buffer;
    void                         *recv_buffer;

    /* Measurements */
    ucs_time_t                   start_time;
    ucs_time_t                   prev_time;
    ucs_time_t                   end_time;
    ucs_time_t                   report_interval;
    ucx_perf_counter_t           max_iter;
    struct {
        ucx_perf_counter_t       msgs;
        ucx_perf_counter_t       bytes;
        ucx_perf_counter_t       iters;
        ucs_time_t               time;
    } current, prev;

    ucs_time_t                   timing_queue[TIMING_QUEUE_SIZE];
    unsigned                     timing_queue_head;
} ucx_perf_context_t;


typedef struct uct_peer {
    uct_ep_h                     ep;
    unsigned long                remote_addr;
    uct_rkey_bundle_t            rkey;
} uct_peer_t;


typedef struct uct_perf_context {
    ucx_perf_context_t           super;
    uct_context_h                context;
    uct_iface_h                  iface;
    uct_peer_t                   *peers;
    uct_lkey_t                   lkey;
} uct_perf_context_t;


#define UCX_PERF_TEST_FOREACH(perf) \
    while (!ucx_perf_context_done(perf))

#define rte_call(_perf, _func, ...) \
    (_perf)->params.rte->_func((_perf)->params.rte_group, ## __VA_ARGS__)


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
    unsigned i;

    perf->params = *params;

    perf->send_buffer = memalign(perf->params.alignment, perf->params.message_size);
    perf->recv_buffer = memalign(perf->params.alignment, perf->params.message_size);
    if (perf->send_buffer == NULL || perf->recv_buffer == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

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

    return UCS_OK;
}

void ucx_perf_test_cleanup(ucx_perf_context_t *perf)
{
    free(perf->send_buffer);
    free(perf->recv_buffer);
}

static inline int ucx_perf_context_done(ucx_perf_context_t *perf) {
    return ucs_unlikely((perf->current.iters > perf->max_iter) ||
                        (perf->current.time  > perf->end_time));
}

static void ucx_perf_calc_result(ucx_perf_context_t *perf,
                                 ucx_perf_result_t *result)
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

static inline void ucx_perf_update(ucx_perf_context_t *perf, ucx_perf_counter_t iters,
                                   size_t bytes)
{
    ucx_perf_result_t result;

    perf->current.time   = ucs_get_time();
    perf->current.iters += iters;
    perf->current.bytes += bytes;
    perf->current.msgs  += 1;

    perf->timing_queue[perf->timing_queue_head++] = perf->current.time - perf->prev_time;
    perf->timing_queue_head %= TIMING_QUEUE_SIZE;
    perf->prev_time = perf->current.time;

    if (perf->current.time - perf->prev.time >= perf->report_interval) {
        ucx_perf_calc_result(perf, &result);
        rte_call(perf, report, &result);
        perf->prev = perf->current;
    }
}

static ucs_status_t ucx_perf_run_put_lat(uct_perf_context_t *perf)
{
    volatile uint8_t *send_sn, *recv_sn;
    unsigned my_index;
    uct_ep_h ep;
    void *buffer;
    size_t length;
    unsigned long remote_addr;
    uct_rkey_t rkey;
    uint8_t sn;

    recv_sn = (uint8_t*)perf->super.recv_buffer + perf->super.params.message_size - 1;
    send_sn = (uint8_t*)perf->super.send_buffer + perf->super.params.message_size - 1;

    *recv_sn = -1;
    rte_call(&perf->super, barrier);

    my_index = rte_call(&perf->super, group_index);

    buffer = perf->super.send_buffer;
    length = perf->super.params.message_size;

    sn = 0;
    if (my_index == 0) {
        ep          = perf->peers[1].ep;
        remote_addr = perf->peers[1].remote_addr;
        rkey        = perf->peers[1].rkey.rkey;
        UCX_PERF_TEST_FOREACH(&perf->super) {
            while (*recv_sn != sn);
            uct_ep_put_short(ep, buffer,length, remote_addr, rkey, NULL, NULL);
            *send_sn = ++sn;
            ucx_perf_update(&perf->super, 1, length);
        }
    } else if (my_index == 1) {
        ep          = perf->peers[0].ep;
        remote_addr = perf->peers[0].remote_addr;
        rkey        = perf->peers[0].rkey.rkey;
        UCX_PERF_TEST_FOREACH(&perf->super) {
            uct_ep_put_short(ep, buffer,length, remote_addr, rkey, NULL, NULL);
            while (*recv_sn != sn);
            *send_sn = ++sn;
            ucx_perf_update(&perf->super, 1, length);
        }
    }
    return UCS_OK;
}

ucs_status_t uct_perf_test_setup_endpoints(uct_perf_context_t *perf)
{
    unsigned group_size, i;
    uct_iface_addr_t *iface_addr;
    uct_ep_addr_t *ep_addr;
    uct_iface_attr_t iface_attr;
    uct_pd_attr_t pd_attr;
    unsigned long address;
    void *rkey_buffer;
    ucs_status_t status;

    status = uct_iface_query(perf->iface, &iface_attr);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_pd_query(perf->iface->pd, &pd_attr);
    if (status != UCS_OK) {
        goto err;
    }

    iface_addr  = malloc(iface_attr.iface_addr_len);
    ep_addr     = malloc(iface_attr.ep_addr_len);
    rkey_buffer = malloc(pd_attr.rkey_packed_size);
    if ((iface_addr == NULL) || (ep_addr == NULL) || (rkey_buffer == NULL)) {
        goto err_free;
    }

    status = uct_iface_get_address(perf->iface, iface_addr);
    if (status != UCS_OK) {
        goto err_free;
    }

    status = uct_rkey_pack(perf->iface->pd, perf->lkey, rkey_buffer);
    if (status != UCS_OK) {
        goto err_free;
    }

    address = (uintptr_t)perf->super.recv_buffer;

    group_size  = rte_call(&perf->super, group_size);

    perf->peers = calloc(group_size, sizeof(*perf->peers));
    if (perf->peers == NULL) {
        goto err_free;
    }

    for (i = 0; i < group_size; ++i) {
        status = uct_ep_create(perf->iface, &perf->peers[i].ep);
        if (status != UCS_OK) {
            goto err_destroy_eps;
        }

        status = uct_ep_get_address(perf->peers[i].ep, ep_addr);
        if (status != UCS_OK) {
            goto err_destroy_eps;
        }

        rte_call(&perf->super, send, i, &address, sizeof(address));
        rte_call(&perf->super, send, i, rkey_buffer, pd_attr.rkey_packed_size);
        rte_call(&perf->super, send, i, iface_addr, iface_attr.iface_addr_len);
        rte_call(&perf->super, send, i, ep_addr, iface_attr.ep_addr_len);
    }

    for (i = 0; i < group_size; ++i) {
        rte_call(&perf->super, recv, i, &address, sizeof(address));
        rte_call(&perf->super, recv, i, rkey_buffer, pd_attr.rkey_packed_size);
        rte_call(&perf->super, recv, i, iface_addr, iface_attr.iface_addr_len);
        rte_call(&perf->super, recv, i, ep_addr, iface_attr.ep_addr_len);

        perf->peers[i].remote_addr = address;
        status = uct_rkey_unpack(perf->context, rkey_buffer, &perf->peers[i].rkey);
        if (status != UCS_OK) {
            return status;
        }

        status = uct_ep_connect_to_ep(perf->peers[i].ep, iface_addr, ep_addr);
        if (status != UCS_OK) {
            goto err_destroy_eps;
        }
    }

    free(iface_addr);
    free(ep_addr);
    free(rkey_buffer);

    rte_call(&perf->super, barrier);

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

    group_size = rte_call(&perf->super, group_size);
    for (i = 0; i < group_size; ++i) {
        uct_rkey_release(perf->context, &perf->peers[i].rkey);
        uct_ep_destroy(perf->peers[i].ep);
    }
    free(perf->peers);
}

ucs_status_t uct_perf_test_run(uct_context_h context,
                               ucx_perf_test_params_t *params, const char *hw_name,
                               const char *tl_name, ucx_perf_result_t *result)
{
    uct_perf_context_t perf;
    ucs_status_t status;

    status = ucx_perf_test_init(&perf.super, params);
    if (status != UCS_OK) {
        goto out;
    }

    perf.context = context;

    status = uct_iface_open(perf.context, tl_name, hw_name, &perf.iface);
    if (status != UCS_OK) {
        goto out_test_cleanup;
    }

    status = uct_mem_map(perf.iface->pd, perf.super.recv_buffer,
                         perf.super.params.message_size, 0, &perf.lkey);
    if (status != UCS_OK) {
        goto out_iface_close;
    }

    status = uct_perf_test_setup_endpoints(&perf);
    if (status != UCS_OK) {
        goto out_mem_unmap;
    }

    /* Run test */
    if (perf.super.params.command == UCX_PERF_TEST_CMD_PUT_SHORT &&
        perf.super.params.test_type == UCX_PERF_TEST_TYPE_PINGPONG &&
        perf.super.params.data_layout == UCX_PERF_DATA_LAYOUT_BUFFER)
    {
        status = ucx_perf_run_put_lat(&perf);
    } else {
        status = UCS_ERR_INVALID_PARAM;
    }

    if (status == UCS_OK) {
        ucx_perf_calc_result(&perf.super, result);
    }

    uct_perf_test_cleanup_endpoints(&perf);
out_mem_unmap:
    uct_mem_unmap(perf.iface->pd, perf.lkey);
out_iface_close:
    uct_iface_close(perf.iface);
out_test_cleanup:
    ucx_perf_test_cleanup(&perf.super);
out:
    return status;
}

