/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "test_cuda_check_def.h"
#include "test_mpi_tags_def.h"
#include "test_ucp.h"
#include "test_ucx_check_def.h"

#include <ucp/api/device/ucp_host.h>

#include <cuda.h>
#include <mpi.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MPI_COMM_SIZE 2
#define SIZE          (1024 * 1024 * 1024)

static void mpi_barrier(void)
{
    MPI_Request request;

    MPI_Ibarrier(MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
}

/* Progress the worker while waiting, to serve the peer's wireup messages */
static void mpi_barrier_progress(ucp_worker_h worker)
{
    int done = 0;
    MPI_Request request;

    MPI_Ibarrier(MPI_COMM_WORLD, &request);
    while (!done) {
        ucp_worker_progress(worker);
        MPI_Test(&request, &done, MPI_STATUS_IGNORE);
    }
}

static ucp_device_remote_mem_list_h
create_remote_mem_list(ucp_worker_h worker, ucp_ep_h ep, rkey_t rkey)
{
    ucp_device_mem_list_elem_t elem;
    ucp_device_mem_list_params_t params;
    ucp_device_remote_mem_list_h mem_list_h;
    ucs_status_t status;

    elem.field_mask  = UCP_DEVICE_MEM_LIST_ELEM_FIELD_EP |
                       UCP_DEVICE_MEM_LIST_ELEM_FIELD_REMOTE_ADDR |
                       UCP_DEVICE_MEM_LIST_ELEM_FIELD_RKEY;
    elem.ep          = ep;
    elem.remote_addr = rkey.remote_address;
    elem.rkey        = rkey.rkey;

    params.field_mask   = UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENTS |
                          UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENT_SIZE |
                          UCP_DEVICE_MEM_LIST_PARAMS_FIELD_NUM_ELEMENTS;
    params.element_size = sizeof(elem);
    params.num_elements = 1;
    params.elements     = &elem;

    do {
        ucp_worker_progress(worker);
        status = ucp_device_remote_mem_list_create(&params, &mem_list_h);
    } while (status == UCS_ERR_NOT_CONNECTED);

    if (status != UCS_OK) {
        fprintf(stderr, "ucp_device_remote_mem_list_create failed: %s\n",
                ucs_status_string(status));
        exit(status);
    }

    return mem_list_h;
}

static int rank0(ucp_t ucp)
{
    CUdeviceptr ptr;
    ucp_mem_h ucp_mem;
    size_t free_bytes_before, total_bytes, free_bytes_after, unreleased_memory;

    CUDA_CHECK(cuMemGetInfo(&free_bytes_before, &total_bytes));
    CUDA_CHECK(cuMemAlloc(&ptr, SIZE));

    ucp_mem = send_rkey(1, (void*)ptr, SIZE, ucp.context);

    mpi_barrier_progress(ucp.worker);

    UCX_CHECK(ucp_mem_unmap(ucp.context, ucp_mem));
    CUDA_CHECK(cuMemFree(ptr));
    CUDA_CHECK(cuCtxSynchronize());

    CUDA_CHECK(cuMemGetInfo(&free_bytes_after, &total_bytes));
    unreleased_memory = free_bytes_before - free_bytes_after;
    fprintf(stdout, "Unreleased memory: %zu bytes: %s\n", unreleased_memory,
            (unreleased_memory == 0) ? "PASS" : "FAIL");
    return (unreleased_memory == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

static void rank1(ucp_t ucp)
{
    rkey_t rkey;
    ucp_device_remote_mem_list_h remote_mem_list;
    void *ptr;

    rkey            = recv_rkey(0, ucp.ep);
    remote_mem_list = create_remote_mem_list(ucp.worker, ucp.ep, rkey);

    UCX_CHECK(ucp_rkey_ptr(rkey.rkey, rkey.remote_address, &ptr));

    ucp_device_mem_list_release(remote_mem_list);
    ucp_rkey_destroy(rkey.rkey);

    mpi_barrier();
}

int main(int argc, char **argv)
{
    int comm_size, rank;
    int cu_dev_count;
    CUdevice cu_dev;
    CUcontext cu_ctx;
    ucp_t ucp;
    int exit_status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (comm_size != MPI_COMM_SIZE) {
        if (rank == 0) {
            fprintf(stderr, "This test requires exactly %d MPI processes\n",
                    MPI_COMM_SIZE);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cuInit(0));
    CUDA_CHECK(cuDeviceGetCount(&cu_dev_count));
    CUDA_CHECK(cuDeviceGet(&cu_dev, rank % cu_dev_count));
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev));
    CUDA_CHECK(cuCtxSetCurrent(cu_ctx));

    ucp = create_ucp(UCP_FEATURE_RMA | UCP_FEATURE_DEVICE,
                     UCP_EP_PARAMS_FLAGS_RKEY_BOUND_LIFETIME);

    if (rank == 0) {
        exit_status = rank0(ucp);
    } else {
        rank1(ucp);
    }

    MPI_Bcast(&exit_status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    destroy_ucp(ucp);

    CUDA_CHECK(cuCtxPopCurrent(NULL));
    CUDA_CHECK(cuDevicePrimaryCtxRelease(cu_dev));

    MPI_Finalize();
    return exit_status;
}
