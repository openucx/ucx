/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef LIBPERF_INT_H_
#define LIBPERF_INT_H_

#include "libperf.h"

BEGIN_C_DECLS

#include <ucs/time/time.h>


#define TIMING_QUEUE_SIZE    2048
#define UCT_PERF_TEST_AM_ID  5


typedef struct ucx_perf_context  ucx_perf_context_t;
typedef struct uct_peer          uct_peer_t;
typedef struct uct_perf_context  uct_perf_context_t;


struct ucx_perf_context {
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
};


struct uct_peer {
    uct_ep_h                     ep;
    unsigned long                remote_addr;
    uct_rkey_bundle_t            rkey;
};


struct uct_perf_context {
    ucx_perf_context_t           super;
    uct_context_h                context;
    uct_iface_h                  iface;
    uct_peer_t                   *peers;
    uct_lkey_t                   send_lkey;
    uct_lkey_t                   recv_lkey;
};


#define UCX_PERF_TEST_FOREACH(perf) \
    while (!ucx_perf_context_done(perf))

#define rte_call(_perf, _func, ...) \
    (_perf)->params.rte->_func((_perf)->params.rte_group, ## __VA_ARGS__)


void ucx_perf_test_start_clock(ucx_perf_context_t *perf);


void uct_perf_iface_flush_b(uct_perf_context_t *perf);


ucs_status_t uct_perf_test_dispatch(uct_perf_context_t *perf);


void ucx_perf_calc_result(ucx_perf_context_t *perf, ucx_perf_result_t *result);


static UCS_F_ALWAYS_INLINE int ucx_perf_context_done(ucx_perf_context_t *perf)
{
    return ucs_unlikely((perf->current.iters >= perf->max_iter) ||
                        (perf->current.time  > perf->end_time));
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
        rte_call(perf, report, &result, 0);
        perf->prev = perf->current;
    }
}

END_C_DECLS

#endif
