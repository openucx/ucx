/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2021. ALL RIGHTS RESERVED.
*
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


static ucs_status_t ucx_perf_thread_run_test(void* arg)
{
    ucx_perf_thread_context_t* tctx = (ucx_perf_thread_context_t*) arg; /* a single thread context */
    ucx_perf_result_t* result       = &tctx->result;
    ucx_perf_context_t* perf        = &tctx->perf;
    ucx_perf_params_t* params       = &perf->params;
    ucs_status_t status;

    status = ucx_perf_allocators_init_thread(perf);
    if (status != UCS_OK) {
        goto out;
    }

    status = ucx_perf_do_warmup(perf, params);
    if (UCS_OK != status) {
        goto out;
    }

    /* Run test */
#pragma omp barrier
    status = ucx_perf_funcs[params->api].run(perf);
    ucx_perf_funcs[params->api].barrier(perf);
    if (UCS_OK != status) {
        goto out;
    }

    ucx_perf_calc_result(perf, result);

out:
    return status;
}

static void ucx_perf_thread_report_aggregated_results(ucx_perf_context_t *perf)
{
    ucx_perf_thread_context_t* tctx = perf->ucp.tctx;  /* all the thread contexts on perf */
    unsigned i, thread_count        = perf->params.thread_count;
    double lat_sum_total_avegare    = 0.0;
    ucx_perf_result_t agg_result;

    agg_result.iters        = tctx[0].result.iters;
    agg_result.bytes        = tctx[0].result.bytes;
    agg_result.elapsed_time = tctx[0].result.elapsed_time;

    agg_result.bandwidth.total_average  = 0.0;
    agg_result.bandwidth.percentile     = 0.0; /* Undefined since used only for latency calculations */
    agg_result.latency.total_average    = 0.0;
    agg_result.msgrate.total_average    = 0.0;
    agg_result.msgrate.percentile       = 0.0; /* Undefined since used only for latency calculations */

    /* when running with multiple threads, the moment average value is
     * undefined since we don't capture the values of the last iteration */
    agg_result.msgrate.moment_average   = 0.0;
    agg_result.bandwidth.moment_average = 0.0;
    agg_result.latency.moment_average   = 0.0;
    agg_result.latency.percentile       = 0.0;

    /* in case of multiple threads, we have to aggregate the results so that the
     * final output of the result would show the performance numbers that were
     * collected from all the threads.
     * BW and message rate values will be the sum of their values from all
     * the threads, while the latency value is the average latency from the
     * threads. */

    for (i = 0; i < thread_count; i++) {
        agg_result.bandwidth.total_average  += tctx[i].result.bandwidth.total_average;
        agg_result.msgrate.total_average    += tctx[i].result.msgrate.total_average;
        lat_sum_total_avegare               += tctx[i].result.latency.total_average;
    }

    agg_result.latency.total_average = lat_sum_total_avegare / thread_count;

    perf->params.report_func(perf->params.rte_group, &agg_result,
                             perf->params.report_arg, "", 1, 1);
}

ucs_status_t ucx_perf_thread_spawn(ucx_perf_context_t *perf,
                                   ucx_perf_result_t* result)
{
    ucx_perf_thread_context_t* tctx = perf->ucp.tctx;   /* all the thread contexts on perf */
    int ti, thread_count            = perf->params.thread_count;
    ucs_status_t* statuses;
    ucs_status_t status;

    omp_set_num_threads(thread_count);

    statuses = calloc(thread_count, sizeof(ucs_status_t));
    if (statuses == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

#pragma omp parallel private(ti)
{
    ti              = omp_get_thread_num();
    tctx[ti].status = ucx_perf_thread_run_test((void*)&tctx[ti]);
}

    status = UCS_OK;
    for (ti = 0; ti < thread_count; ti++) {
        if (UCS_OK != tctx[ti].status) {
            ucs_error("Thread %d failed to run test: %s", tctx[ti].tid,
                      ucs_status_string(tctx[ti].status));
            status = tctx[ti].status;
        }
    }

    ucx_perf_thread_report_aggregated_results(perf);

    free(statuses);
out:
    return status;
}

#else

ucs_status_t ucx_perf_thread_spawn(ucx_perf_context_t *perf,
                                   ucx_perf_result_t* result)
{
    ucs_error("Invalid test parameter (thread mode requested without OpenMP capabilities)");
    return UCS_ERR_INVALID_PARAM;
}

#endif /* _OPENMP */

static ucs_status_t
ucx_perf_get_device_id(int conf_device_id, int num_devices,
                       unsigned group_index, int *device_id_p)
{
    int device_id = conf_device_id;

    if ((device_id == UCX_PERF_MEM_DEV_DEFAULT) && (num_devices > 0)) {
        device_id = group_index % num_devices;
    }

    if (device_id >= num_devices) {
        ucs_error("Illegal device id %d, number of devices: %d", device_id,
                  num_devices);
        return UCS_ERR_INVALID_PARAM;
    }

    *device_id_p = device_id;
    return UCS_OK;
}

static ucs_status_t
ucx_perf_get_device_ids(ucx_perf_context_t *perf, int num_devices,
                        int *send_device_id_p, int *recv_device_id_p)
{
    int send_device_id, recv_device_id;
    unsigned group_index;
    ucs_status_t status;

    group_index = rte_call(perf, group_index);

    status = ucx_perf_get_device_id(perf->params.send_mem_device, num_devices,
                                    group_index, &send_device_id);
    if (status != UCS_OK) {
        return status;
    }

    status = ucx_perf_get_device_id(perf->params.recv_mem_device, num_devices,
                                    group_index, &recv_device_id);
    if (status != UCS_OK) {
        return status;
    }

    /* If perftest is not running in a loopback mode, then keep the same device
     * for send and recv. */
    if (!(perf->params.flags & UCX_PERF_TEST_FLAG_LOOPBACK)) {
        if (group_index == 0) {
            send_device_id = recv_device_id;
        } else {
            recv_device_id = send_device_id;
        }
    }

    *send_device_id_p = send_device_id;
    *recv_device_id_p = recv_device_id;
    return UCS_OK;
}

ucs_status_t ucx_perf_allocators_init_thread(ucx_perf_context_t *perf)
{
    ucs_status_t status;
    int send_device_id, recv_device_id, num_devices;

    status = perf->send_allocator.dev_count(&num_devices);
    if (status != UCS_OK) {
        return status;
    }

    status = ucx_perf_get_device_ids(perf, num_devices, &send_device_id,
                                     &recv_device_id);
    if (status != UCS_OK) {
        return status;
    }

    /* New threads need explicit device association */
    status = perf->send_allocator.init(&perf->send_allocator, send_device_id);
    if (status != UCS_OK) {
        return status;
    }

    return perf->recv_allocator.init(&perf->recv_allocator, recv_device_id);
}
