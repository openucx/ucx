/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) The University of Tennessee and The University
*               of Tennessee Research Foundation. 2016. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef LIBPERF_INT_H_
#define LIBPERF_INT_H_

#include <tools/perf/api/libperf.h>

BEGIN_C_DECLS

/** @file libperf_int.h */

#include <ucs/async/async.h>
#include <ucs/time/time.h>
#include <ucs/sys/math.h>


#if _OPENMP
#include <omp.h>
#endif


#define TIMING_QUEUE_SIZE    2048
#define UCT_PERF_TEST_AM_ID  5
#define ADDR_BUF_SIZE        2048
#define EXTRA_INFO_SIZE      256

#define UCX_PERF_TEST_FOREACH(perf) \
    while (!ucx_perf_context_done(perf))

#define rte_call(_perf, _func, ...) \
    ((_perf)->params.rte->_func((_perf)->params.rte_group, ## __VA_ARGS__))


typedef struct ucx_perf_context        ucx_perf_context_t;
typedef struct uct_peer                uct_peer_t;
typedef struct ucp_perf_request        ucp_perf_request_t;
typedef struct ucx_perf_thread_context ucx_perf_thread_context_t;


struct ucx_perf_allocator {
    ucs_memory_type_t mem_type;
    ucs_status_t (*init)(ucx_perf_context_t *perf);
    ucs_status_t (*uct_alloc)(const ucx_perf_context_t *perf, size_t length,
                              unsigned flags, uct_allocated_memory_t *alloc_mem);
    void         (*uct_free)(const ucx_perf_context_t *perf,
                             uct_allocated_memory_t *alloc_mem);
    void         (*memcpy)(void *dst, ucs_memory_type_t dst_mem_type,
                           const void *src, ucs_memory_type_t src_mem_type,
                           size_t count);
    void*        (*memset)(void *dst, int value, size_t count);
};

struct ucx_perf_context {
    ucx_perf_params_t            params;

    /* Buffers */
    void                         *send_buffer;
    void                         *recv_buffer;

    /* Measurements */
    double                       start_time_acc;  /* accurate start time */
    ucs_time_t                   end_time;        /* inaccurate end time (upper bound) */
    ucs_time_t                   prev_time;       /* time of previous iteration */
    ucs_time_t                   report_interval; /* interval of showing report */
    ucx_perf_counter_t           max_iter;

    /* Measurements of current/previous **report** */
    struct {
        ucx_perf_counter_t       msgs;    /* number of messages */
        ucx_perf_counter_t       bytes;   /* number of bytes */
        ucx_perf_counter_t       iters;   /* number of iterations */
        ucs_time_t               time;    /* inaccurate time (for median and report interval) */
        double                   time_acc; /* accurate time (for avg latency/bw/msgrate) */
    } current, prev;

    ucs_time_t                   timing_queue[TIMING_QUEUE_SIZE];
    unsigned                     timing_queue_head;

    const ucx_perf_allocator_t   *send_allocator;
    const ucx_perf_allocator_t   *recv_allocator;

    char                         extra_info[EXTRA_INFO_SIZE];

    union {
        struct {
            ucs_async_context_t    async;
            uct_component_h        cmpt;
            uct_md_h               md;
            uct_worker_h           worker;
            uct_iface_h            iface;
            uct_peer_t             *peers;
            uct_allocated_memory_t send_mem;
            uct_allocated_memory_t recv_mem;
            uct_iov_t              *iov;
        } uct;

        struct {
            ucp_context_h              context;
            ucx_perf_thread_context_t* tctx;
            ucp_worker_h               worker;
            ucp_ep_h                   ep;
            ucp_rkey_h                 rkey;
            unsigned long              remote_addr;
            ucp_mem_h                  send_memh;
            ucp_mem_h                  recv_memh;
            ucp_dt_iov_t               *send_iov;
            ucp_dt_iov_t               *recv_iov;
            void                       *am_hdr;
        } ucp;
    };
};

struct ucx_perf_thread_context {
    pthread_t           pt;
    int                 tid;
    ucs_status_t        status;
    ucx_perf_context_t  perf;
    ucx_perf_result_t   result;
};

struct uct_peer {
    uct_ep_h                     ep;
    unsigned long                remote_addr;
    uct_rkey_bundle_t            rkey;
};

struct ucp_perf_request {
    void                         *context;
};

typedef struct {
    ucs_status_t (*setup)(ucx_perf_context_t *perf);
    void         (*cleanup)(ucx_perf_context_t *perf);
    ucs_status_t (*run)(ucx_perf_context_t *perf);
    void         (*barrier)(ucx_perf_context_t *perf);
} ucx_perf_funcs_t;

extern ucx_perf_funcs_t ucx_perf_funcs[];

unsigned rte_peer_index(unsigned group_size, unsigned group_index);
void ucx_perf_test_start_clock(ucx_perf_context_t *perf);
void uct_perf_ep_flush_b(ucx_perf_context_t *perf, int peer_index);
void uct_perf_iface_flush_b(ucx_perf_context_t *perf);
ucs_status_t uct_perf_test_dispatch(ucx_perf_context_t *perf);
ucs_status_t ucp_perf_test_dispatch(ucx_perf_context_t *perf);
void ucx_perf_calc_result(ucx_perf_context_t *perf, ucx_perf_result_t *result);
void uct_perf_barrier(ucx_perf_context_t *perf);
void ucp_perf_thread_barrier(ucx_perf_context_t *perf);
void ucp_perf_barrier(ucx_perf_context_t *perf);

ucs_status_t ucp_perf_test_alloc_mem(ucx_perf_context_t *perf);
void ucp_perf_test_free_mem(ucx_perf_context_t *perf);
ucs_status_t uct_perf_test_alloc_mem(ucx_perf_context_t *perf);
void uct_perf_test_free_mem(ucx_perf_context_t *perf);
ucs_status_t ucx_perf_thread_spawn(ucx_perf_context_t *perf,
                                   ucx_perf_result_t* result);
void ucx_perf_test_prepare_new_run(ucx_perf_context_t *perf,
                                   const ucx_perf_params_t *params);
ucs_status_t
ucx_perf_do_warmup(ucx_perf_context_t *perf, const ucx_perf_params_t *params);

/**
 * Get the total length of the message size given by parameters
 */
size_t ucx_perf_get_message_size(const ucx_perf_params_t *params);

void ucx_perf_report(ucx_perf_context_t *perf);


static UCS_F_ALWAYS_INLINE int ucx_perf_context_done(ucx_perf_context_t *perf)
{
    return ucs_unlikely((perf->current.iters >= perf->max_iter) ||
                        (perf->current.time  > perf->end_time));
}

static inline void ucx_perf_get_time(ucx_perf_context_t *perf)
{
    perf->current.time_acc = ucs_get_accurate_time();
}

static inline void ucx_perf_omp_barrier(ucx_perf_context_t *perf)
{
#if _OPENMP
    if (perf->params.thread_count > 1) {
#pragma omp barrier
    }
#endif
}

static UCS_F_ALWAYS_INLINE void ucx_perf_update(ucx_perf_context_t *perf,
                                                ucx_perf_counter_t iters,
                                                size_t bytes)
{
    perf->current.time   = ucs_get_time();
    perf->current.iters += iters;
    perf->current.bytes += bytes;
    perf->current.msgs  += 1;

    perf->timing_queue[perf->timing_queue_head] =
                    perf->current.time - perf->prev_time;
    ++perf->timing_queue_head;
    if (perf->timing_queue_head == TIMING_QUEUE_SIZE) {
        perf->timing_queue_head = 0;
    }

    perf->prev_time = perf->current.time;

    if (ucs_unlikely((perf->current.time - perf->prev.time) >=
                     perf->report_interval)) {
        ucx_perf_report(perf);
    }
}

END_C_DECLS

#endif
