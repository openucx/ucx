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

#include <cuda.h>
#include <mpi.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MPI_COMM_SIZE 2
#define SIZE          1024 * 1024 * 1024

static void rank0(ucp_context_h ucp_context)
{
    CUdeviceptr ptr;
    ucp_mem_h ucp_mem;
    size_t free_bytes_before, total_bytes, free_bytes_after, unreleased_memory;

    CUDA_CHECK(cuMemGetInfo(&free_bytes_before, &total_bytes));
    CUDA_CHECK(cuMemAlloc(&ptr, SIZE));

    ucp_mem = send_rkey(1, (void*)ptr, SIZE, ucp_context);

    MPI_Barrier(MPI_COMM_WORLD);

    UCX_CHECK(ucp_mem_unmap(ucp_context, ucp_mem));
    CUDA_CHECK(cuMemFree(ptr));
    CUDA_CHECK(cuCtxSynchronize());

    CUDA_CHECK(cuMemGetInfo(&free_bytes_after, &total_bytes));
    unreleased_memory = free_bytes_before - free_bytes_after;
    fprintf(stdout, "Unreleased memory: %zu bytes: %s\n", unreleased_memory,
            (unreleased_memory == 0) ? "PASS" : "FAIL");
}

static void rank1(ucp_ep_h ucp_ep)
{
    rkey_t rkey;
    void *ptr;

    rkey = recv_rkey(0, ucp_ep);

    UCX_CHECK(ucp_rkey_ptr(rkey.rkey, rkey.remote_address, &ptr));
    ucp_rkey_destroy(rkey.rkey);

    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char **argv)
{
    int comm_size, rank;
    int cu_dev_count;
    CUdevice cu_dev;
    CUcontext cu_ctx;
    ucp_t ucp;

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

    ucp = create_ucp();

    if (rank == 0) {
        rank0(ucp.context);
    } else {
        rank1(ucp.ep);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    destroy_ucp(ucp);

    CUDA_CHECK(cuCtxPopCurrent(NULL));
    CUDA_CHECK(cuDevicePrimaryCtxRelease(cu_dev));

    MPI_Finalize();
    return EXIT_SUCCESS;
}
