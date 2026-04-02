/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 *
 * Pure host dispatcher: params handler, run logic, calls host entry points
 * from ucp_cuda_kernel_*.cuh.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucp_cuda_impl.h"
#include "ucp_cuda_kernel_bw.cuh"
#include "ucp_cuda_kernel_latency.cuh"
#include "ucp_cuda_kernel_wait.cuh"
#include "cuda_common.h"

#include <ucs/sys/compiler.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/sys/device_code.h>
#include <ucs/time/time.h>
#include <unistd.h>

static uint64_t *ucx_perf_cuda_get_sn(const void *address, size_t length)
{
    return (uint64_t *)UCS_PTR_BYTE_OFFSET(address, length);
}

static unsigned ucx_perf_cuda_thread_index(unsigned tid, ucs_device_level_t level)
{
    switch (level) {
    case UCS_DEVICE_LEVEL_THREAD:
        return tid;
    case UCS_DEVICE_LEVEL_WARP:
        return tid / 32;
    default:
        return 0;
    }
}

static void wait_for_kernel(ucx_perf_context_t *perf,
                           ucx_perf_cuda_context_t *cpu_ctx)
{
    size_t msg_length        = ucx_perf_get_message_size(&perf->params);
    ucx_perf_counter_t last_completed = 0;
    ucx_perf_counter_t completed      = cpu_ctx->completed_iters;
    unsigned thread_count             = perf->params.device_thread_count;
    unsigned block_count              = perf->params.device_block_count;
    ucs_device_level_t level         = perf->params.device_level;
    unsigned msgs_per_iter            = ucx_perf_cuda_thread_index(thread_count,
                                                                  level);

    while (1) {
        ucx_perf_counter_t delta = completed - last_completed;
        if (delta > 0) {
            ucx_perf_update(perf, delta, delta * msgs_per_iter, msg_length);
        } else if (completed >= (ucx_perf_counter_t)(perf->max_iter * block_count)) {
            break;
        }
        last_completed = completed;
        completed      = cpu_ctx->completed_iters;
        usleep(100);
    }
}

ucs_status_t ucp_perf_cuda_params_init(const ucx_perf_context_t *perf,
                                        ucp_perf_cuda_params_t *params)
{
    size_t data_count = perf->params.msg_size_cnt;
    size_t offset     = 0;
    size_t i;
    ucp_device_mem_list_elem_t *elems;
    ucp_device_mem_list_params_t list_params;
    ucs_status_t status;
    ucs_time_t deadline;

    params->local_mem_list  = NULL;
    params->remote_mem_list = NULL;

    elems = ucs_alloca(data_count * sizeof(ucp_device_mem_list_elem_t));

    for (i = 0; i < data_count; ++i) {
        elems[i].field_mask  = UCP_DEVICE_MEM_LIST_ELEM_FIELD_MEMH |
                              UCP_DEVICE_MEM_LIST_ELEM_FIELD_RKEY |
                              UCP_DEVICE_MEM_LIST_ELEM_FIELD_LOCAL_ADDR |
                              UCP_DEVICE_MEM_LIST_ELEM_FIELD_REMOTE_ADDR |
                              UCP_DEVICE_MEM_LIST_ELEM_FIELD_EP |
                              UCP_DEVICE_MEM_LIST_ELEM_FIELD_LENGTH;
        elems[i].memh        = perf->ucp.send_memh;
        elems[i].rkey        = perf->ucp.rkey;
        elems[i].local_addr  = UCS_PTR_BYTE_OFFSET(perf->send_buffer, offset);
        elems[i].remote_addr = perf->ucp.remote_addr + offset;
        elems[i].ep          = perf->ucp.ep;
        elems[i].length      = perf->params.msg_size_list[i];

        offset += perf->params.msg_size_list[i];
    }

    list_params.field_mask   = UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENTS |
                              UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENT_SIZE |
                              UCP_DEVICE_MEM_LIST_PARAMS_FIELD_NUM_ELEMENTS |
                              UCP_DEVICE_MEM_LIST_PARAMS_FIELD_WORKER;
    list_params.element_size = sizeof(ucp_device_mem_list_elem_t);
    list_params.num_elements = data_count;
    list_params.elements     = elems;
    list_params.worker       = perf->ucp.worker;

    status = ucp_device_local_mem_list_create(&list_params,
                                              &params->local_mem_list);
    if (status != UCS_OK) {
        ucs_warn("Failed to create local memory list: %s",
                 ucs_status_string(status));
        return status;
    }

    deadline = ucs_get_time() + ucs_time_from_sec(60.0);
    do {
        if (ucs_get_time() > deadline) {
            ucs_warn("timeout creating memory list");
            deadline = ULONG_MAX;
        }

        ucp_worker_progress(perf->ucp.worker);
        status = ucp_device_remote_mem_list_create(&list_params,
                                                   &params->remote_mem_list);
    } while (status == UCS_ERR_NOT_CONNECTED);

    if (status != UCS_OK) {
        ucp_device_mem_list_release(params->local_mem_list);
        params->local_mem_list = NULL;
        ucs_warn("Failed to create remote memory list: %s",
                 ucs_status_string(status));
        return status;
    }

    params->length       = ucx_perf_get_message_size(&perf->params);
    params->counter_send = ucx_perf_cuda_get_sn(perf->send_buffer,
                                               params->length);
    params->counter_recv = ucx_perf_cuda_get_sn(perf->recv_buffer,
                                               params->length);

    return UCS_OK;
}

void ucp_perf_cuda_params_cleanup(ucp_perf_cuda_params_t *params)
{
    if (params->local_mem_list != NULL) {
        ucp_device_mem_list_release(params->local_mem_list);
        params->local_mem_list = NULL;
    }
    if (params->remote_mem_list != NULL) {
        ucp_device_mem_list_release(params->remote_mem_list);
        params->remote_mem_list = NULL;
    }
}

ucs_status_t ucp_perf_cuda_run_pingpong(ucx_perf_context_t *perf,
                                         ucx_perf_cuda_context_t *cpu_ctx,
                                         ucx_perf_cuda_context_t *gpu_ctx)
{
    ucp_perf_cuda_params_t params;
    unsigned my_index;
    ucs_status_t status;

    status = ucp_perf_cuda_params_init(perf, &params);
    if (status != UCS_OK) {
        return status;
    }

    my_index = rte_call(perf, group_index);

    ucp_perf_barrier(perf);
    ucx_perf_test_start_clock(perf);

    ucp_perf_cuda_launch_pingpong(perf, gpu_ctx, &params, my_index);
    CUDA_CALL_RET(UCS_ERR_NO_DEVICE, cudaGetLastError);

    wait_for_kernel(perf, cpu_ctx);

    CUDA_CALL_RET(UCS_ERR_IO_ERROR, cudaDeviceSynchronize);

    ucp_perf_cuda_params_cleanup(&params);

    ucx_perf_get_time(perf);
    ucp_perf_barrier(perf);
    return cpu_ctx->status;
}

ucs_status_t ucp_perf_cuda_run_stream_uni(ucx_perf_context_t *perf,
                                           ucx_perf_cuda_context_t *cpu_ctx,
                                           ucx_perf_cuda_context_t *gpu_ctx)
{
    ucp_perf_cuda_params_t params;
    unsigned my_index;
    ucs_status_t status;

    status = ucp_perf_cuda_params_init(perf, &params);
    if (status != UCS_OK) {
        return status;
    }

    my_index = rte_call(perf, group_index);

    ucp_perf_barrier(perf);
    ucx_perf_test_start_clock(perf);

    if (my_index == 1) {
        ucp_perf_cuda_launch_bw(perf, gpu_ctx, &params);
        CUDA_CALL_RET(UCS_ERR_NO_DEVICE, cudaGetLastError);
        wait_for_kernel(perf, cpu_ctx);
    } else if (my_index == 0) {
        ucp_perf_cuda_launch_wait(gpu_ctx, &params);
    }

    CUDA_CALL_RET(UCS_ERR_IO_ERROR, cudaDeviceSynchronize);

    ucp_perf_cuda_params_cleanup(&params);

    ucx_perf_get_time(perf);
    ucp_perf_barrier(perf);
    return cpu_ctx->status;
}
