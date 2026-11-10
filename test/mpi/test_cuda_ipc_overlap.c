/**
 * Copyright (c) 2026 Jiacheng Xu. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "test_cuda_check_def.h"

#include <ucs/debug/log_def.h>

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Run on one node with two GPUs, a logging-enabled UCX build, and an MPI
 * configured to use that build. For Open MPI:
 * OMPI_MCA_pml=ucx OMPI_MCA_pml_ucx_tls=any \
 * UCX_TLS=sm,self,cuda_copy,cuda_ipc UCX_RNDV_THRESH=1 \
 * UCX_RNDV_SCHEME=get_zcopy UCX_LOG_LEVEL=trace \
 * mpirun -np 2 ./test_cuda_ipc_overlap
 * Exit 77 means that the GPU/allocator/transport prerequisites were not met. */

#define BLOCK_SIZE 32768
#define ITERATIONS 30

static unsigned import_errors;
static unsigned imported_regions;

static int check_ipc(int rank, CUdeviceptr ptr)
{
    CUipcMemHandle handle;
    CUdeviceptr mapped;
    CUresult result = CUDA_SUCCESS;

    if (rank == 0) {
        result = cuIpcGetMemHandle(&handle, ptr);
    }
    MPI_Bcast(&result, sizeof(result), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (result == CUDA_SUCCESS) {
        MPI_Bcast(&handle, sizeof(handle), MPI_BYTE, 0, MPI_COMM_WORLD);
        if (rank == 1) {
            result = cuIpcOpenMemHandle(&mapped, handle,
                                       CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
            if (result == CUDA_SUCCESS) {
                result = cuIpcCloseMemHandle(mapped);
            }
        }
        MPI_Bcast(&result, sizeof(result), MPI_BYTE, 1, MPI_COMM_WORLD);
    }

    if ((result == CUDA_ERROR_PEER_ACCESS_UNSUPPORTED) ||
        (result == CUDA_ERROR_TOO_MANY_PEERS) ||
        (result == CUDA_ERROR_NOT_SUPPORTED)) {
        printf("SKIP: CUDA IPC unavailable: %s\n", _CUDA_ERROR_STRING(result));
        return 77;
    }

    CUDA_CHECK(result);
    return 0;
}

static ucs_log_func_rc_t
count_import_errors(const char *file, unsigned line, const char *function,
                    ucs_log_level_t level,
                    const ucs_log_component_config_t *comp, const char *format,
                    va_list ap)
{
    if (strstr(format, "cuIpcOpenMemHandle() failed") != NULL) {
        ++import_errors;
    }

    if (strstr(format, "cuda_ipc cache new region:") != NULL) {
        ++imported_regions;
    }

    return UCS_LOG_FUNC_RC_CONTINUE;
}

static void transfer(int rank, CUdeviceptr send, CUdeviceptr recv, size_t size,
                     unsigned char value, unsigned char *host)
{
    size_t i;

    if (rank == 0) {
        CUDA_CHECK(cuMemsetD8(send, value, size));
        CUDA_CHECK(cuCtxSynchronize());
        MPI_Send((void*)send, size, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv((void*)recv, size, MPI_BYTE, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        CUDA_CHECK(cuMemcpyDtoH(host, recv, size));
        for (i = 0; i < size; ++i) {
            if (host[i] != value) {
                fprintf(stderr, "data mismatch at byte %zu: %u != %u\n",
                        i, host[i], value);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    /* The sender must not free an allocation until its receiver is done. */
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char **argv)
{
    unsigned overlaps = 0, total_overlaps, total_errors, total_regions;
    CUdeviceptr lower = 0, old = 0, upper = 0, replacement = 0, recv;
    CUuuid uuid, uuids[2];
    CUdevice device;
    CUcontext context;
    MPI_Comm local_comm;
    unsigned char *host;
    CUresult result;
    int rank, size, local_size, available, all_available, iteration, ret;
    int device_count = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        printf("SKIP: requires exactly two MPI ranks\n");
        MPI_Finalize();
        return 77;
    }

    if (!ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE)) {
        printf("SKIP: requires logging support and UCX_LOG_LEVEL=trace\n");
        MPI_Finalize();
        return 77;
    }

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                       MPI_INFO_NULL, &local_comm);
    MPI_Comm_size(local_comm, &local_size);
    MPI_Comm_free(&local_comm);
    if (local_size != 2) {
        printf("SKIP: both ranks must be on the same node\n");
        MPI_Finalize();
        return 77;
    }

    result = cuInit(0);
    if (result != CUDA_ERROR_NO_DEVICE) {
        CUDA_CHECK(result);
        CUDA_CHECK(cuDeviceGetCount(&device_count));
    }
    available = device_count > 0;
    MPI_Allreduce(&available, &all_available, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    if (!all_available) {
        printf("SKIP: a rank has no CUDA device\n");
        MPI_Finalize();
        return 77;
    }

    CUDA_CHECK(cuDeviceGet(&device, rank % device_count));
    CUDA_CHECK(cuDeviceGetUuid(&uuid, device));
    MPI_Allgather(&uuid, sizeof(uuid), MPI_BYTE, uuids, sizeof(uuid), MPI_BYTE,
                  MPI_COMM_WORLD);
    if (memcmp(&uuids[0], &uuids[1], sizeof(uuid)) == 0) {
        printf("SKIP: requires two distinct GPUs\n");
        MPI_Finalize();
        return 77;
    }

    CUDA_CHECK(cuDevicePrimaryCtxRetain(&context, device));
    CUDA_CHECK(cuCtxPushCurrent(context));
    CUDA_CHECK(cuMemAlloc(&recv, 2 * BLOCK_SIZE));
    ret = check_ipc(rank, recv);
    if (ret != 0) {
        goto out_free_recv;
    }

    host = malloc(2 * BLOCK_SIZE);
    if (host == NULL) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    ucs_log_push_handler(count_import_errors);

    for (iteration = 0; iteration < ITERATIONS; ++iteration) {
        if (rank == 0) {
            CUDA_CHECK(cuMemAlloc(&lower, BLOCK_SIZE));
            CUDA_CHECK(cuMemAlloc(&old, BLOCK_SIZE));
            CUDA_CHECK(cuMemAlloc(&upper, BLOCK_SIZE));
        }

        transfer(rank, old, recv, BLOCK_SIZE, iteration + 1, host);
        if (iteration == 0) {
            /* Check transport selection before recycling any allocation. */
            MPI_Allreduce(&imported_regions, &total_regions, 1, MPI_UNSIGNED,
                          MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&import_errors, &total_errors, 1, MPI_UNSIGNED,
                          MPI_SUM, MPI_COMM_WORLD);
            if ((total_errors != 0) || (total_regions == 0)) {
                ret = (total_errors != 0) ? 1 : 77;
                printf("%s: IPC preflight errors=%u, regions=%u\n",
                       (ret == 77) ? "SKIP" : "FAIL", total_errors,
                       total_regions);
                if (rank == 0) {
                    CUDA_CHECK(cuMemFree(old));
                    CUDA_CHECK(cuMemFree(lower));
                    CUDA_CHECK(cuMemFree(upper));
                }
                ucs_log_pop_handler();
                goto out_free_host;
            }
        }

        if (rank == 0) {
            CUDA_CHECK(cuMemFree(old));
            CUDA_CHECK(cuMemFree(lower));
            CUDA_CHECK(cuMemAlloc(&replacement, 2 * BLOCK_SIZE));
            overlaps += (replacement < old) &&
                        ((replacement + 2 * BLOCK_SIZE) > old);
        }

        transfer(rank, replacement, recv, 2 * BLOCK_SIZE, iteration + 101,
                 host);
        if (rank == 0) {
            CUDA_CHECK(cuMemFree(replacement));
            CUDA_CHECK(cuMemFree(upper));
        }
    }

    ucs_log_pop_handler();
    MPI_Allreduce(&overlaps, &total_overlaps, 1, MPI_UNSIGNED, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&import_errors, &total_errors, 1, MPI_UNSIGNED, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&imported_regions, &total_regions, 1, MPI_UNSIGNED, MPI_SUM,
                  MPI_COMM_WORLD);
    ret = (total_errors != 0) ? 1 :
          (((total_overlaps == 0) || (total_regions == 0)) ? 77 : 0);
    if (rank == 0) {
        printf("%s: lower overlaps=%u, IPC import errors=%u, regions=%u; "
               "data checks passed\n",
               (ret == 0) ? "PASS" : ((ret == 77) ? "SKIP" : "FAIL"),
               total_overlaps, total_errors, total_regions);
    }

out_free_host:
    free(host);
out_free_recv:
    CUDA_CHECK(cuMemFree(recv));
    MPI_Finalize();
    CUDA_CHECK(cuCtxPopCurrent(NULL));
    CUDA_CHECK(cuDevicePrimaryCtxRelease(device));
    return ret;
}
