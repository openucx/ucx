/**
 * Copyright (c) 2026 Jiacheng Xu. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "test_cuda_check_def.h"
#include "test_cuda_common.h"
#include "test_ucx_check_def.h"

#include <uct/api/uct.h>
#include <uct/api/v2/uct_v2.h>

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Run on one node with two GPUs and UCX_LOG_LEVEL=warn. MPI exchanges only
 * host-side metadata; GPU data is accessed through the cuda_ipc MD so an MPI
 * transport fallback cannot bypass the cache being tested. Exit 77 means
 * that a GPU/allocator/component prerequisite was not met. */

#define BLOCK_SIZE 32768
#define ITERATIONS 30

typedef struct {
    uct_component_h component;
    uct_md_h        md;
    void            *rkey_buffer;
    size_t          rkey_size;
} ipc_context_t;

static int open_ipc(ipc_context_t *ipc)
{
    uct_component_h *components;
    uct_component_attr_t attr;
    uct_md_config_t *config;
    uct_md_attr_t md_attr;
    unsigned count, i;

    UCX_CHECK(uct_query_components(&components, &count));
    for (i = 0; i < count; ++i) {
        attr.field_mask = UCT_COMPONENT_ATTR_FIELD_NAME;
        UCX_CHECK(uct_component_query(components[i], &attr));
        if (strcmp(attr.name, "cuda_ipc") == 0) {
            ipc->component = components[i];
            break;
        }
    }

    uct_release_component_list(components);
    if (ipc->component == NULL) {
        return 0;
    }

    attr.field_mask = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCE_COUNT;
    UCX_CHECK(uct_component_query(ipc->component, &attr));
    if (attr.md_resource_count == 0) {
        return 0;
    }

    attr.md_resources = calloc(attr.md_resource_count,
                               sizeof(*attr.md_resources));
    if (attr.md_resources == NULL) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    attr.field_mask = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCES;
    UCX_CHECK(uct_component_query(ipc->component, &attr));
    UCX_CHECK(uct_md_config_read(ipc->component, NULL, NULL, &config));
    UCX_CHECK(uct_config_modify(config, "CUDA_IPC_CACHE", "y"));
    UCX_CHECK(uct_config_modify(config, "CUDA_IPC_CACHE_MAX_REGIONS", "inf"));
    UCX_CHECK(uct_config_modify(config, "CUDA_IPC_CACHE_MAX_SIZE", "inf"));
    UCX_CHECK(uct_md_open(ipc->component, attr.md_resources[0].md_name,
                          config, &ipc->md));
    uct_config_release(config);
    free(attr.md_resources);

    UCX_CHECK(uct_md_query(ipc->md, &md_attr));
    ipc->rkey_size   = md_attr.rkey_packed_size;
    ipc->rkey_buffer = malloc(ipc->rkey_size);
    if (ipc->rkey_buffer == NULL) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return 1;
}

static int check_legacy_ipc(int rank, CUdeviceptr ptr)
{
    CUresult result = CUDA_SUCCESS;
    CUipcMemHandle handle;
    CUdeviceptr mapped;

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

static ucs_status_t transfer(ipc_context_t *ipc, int rank, CUdeviceptr send,
                             CUdeviceptr recv, size_t size, unsigned char value,
                             unsigned char *host)
{
    uct_md_mkey_pack_params_t pack_params = {0};
    ucs_status_t status                   = UCS_OK;
    uct_mem_h memh;
    uct_rkey_bundle_t rkey;
    void *mapped;
    int rkey_valid;

    if (rank == 0) {
        CUDA_CHECK(cuMemsetD8(send, value, size));
        CUDA_CHECK(cuCtxSynchronize());
        UCX_CHECK(uct_md_mem_reg(ipc->md, (void*)send, size, 0, &memh));
        UCX_CHECK(uct_md_mkey_pack_v2(ipc->md, memh, (void*)send, size,
                                      &pack_params, ipc->rkey_buffer));
    }

    MPI_Bcast(&send, sizeof(send), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(ipc->rkey_buffer, ipc->rkey_size, MPI_BYTE, 0, MPI_COMM_WORLD);
    if (rank == 1) {
        /* Count diagnostics only while importing the remote mapping. */
        ucs_log_push_handler(test_cuda_errors_and_warnings_counter);
        status     = uct_rkey_unpack(ipc->component, ipc->rkey_buffer, &rkey);
        rkey_valid = (status == UCS_OK);
        if (rkey_valid) {
            status = uct_rkey_ptr(ipc->component, &rkey, send, &mapped);
        }

        ucs_log_pop_handler();

        if (status == UCS_OK) {
            CUDA_CHECK(cuMemcpyDtoD(recv, (CUdeviceptr)mapped, size));
            memset(host, value, size);
            if (test_cuda_check_buffer(recv, host, size) != 0) {
                fprintf(stderr, "data mismatch in %zu-byte transfer\n", size);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        if (rkey_valid) {
            UCX_CHECK(uct_rkey_release(ipc->component, &rkey));
        }
    }

    /* The sender must not free an allocation until its receiver is done. */
    MPI_Bcast(&status, sizeof(status), MPI_BYTE, 1, MPI_COMM_WORLD);
    if (rank == 0) {
        UCX_CHECK(uct_md_mem_dereg(ipc->md, memh));
    }

    return status;
}

int main(int argc, char **argv)
{
    ipc_context_t ipc = {0};
    unsigned overlaps = 0, total_overlaps, errors_and_warnings, total_errors;
    CUdeviceptr lower = 0, old = 0, upper = 0, replacement = 0, recv;
    int device_count  = 0;
    CUuuid uuid, uuids[2];
    CUdevice device;
    CUcontext context;
    MPI_Comm local_comm;
    unsigned char *host;
    CUresult result;
    ucs_status_t status;
    int rank, size, local_size, available, all_available, iteration, ret;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        printf("SKIP: requires exactly two MPI ranks\n");
        MPI_Finalize();
        return 77;
    }

    if (!ucs_log_is_enabled(UCS_LOG_LEVEL_WARN)) {
        printf("SKIP: requires UCX_LOG_LEVEL=warn or more verbose\n");
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
    host = malloc(2 * BLOCK_SIZE);
    if (host == NULL) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    available = open_ipc(&ipc);
    MPI_Allreduce(&available, &all_available, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    if (!all_available) {
        printf("SKIP: a rank has no cuda_ipc memory domain\n");
        ret = 77;
        goto out_free_ipc;
    }

    for (iteration = 0; iteration < ITERATIONS; ++iteration) {
        if (rank == 0) {
            CUDA_CHECK(cuMemAlloc(&lower, BLOCK_SIZE));
            CUDA_CHECK(cuMemAlloc(&old, BLOCK_SIZE));
            CUDA_CHECK(cuMemAlloc(&upper, BLOCK_SIZE));
        }

        status = transfer(&ipc, rank, old, recv, BLOCK_SIZE, iteration + 1,
                          host);
        if ((iteration == 0) && (status == UCS_ERR_UNREACHABLE)) {
            /* Unpack also reports unexpected import errors as unreachable.
             * Diagnose unsupported legacy IPC only after UCT has failed. */
            ret = check_legacy_ipc(rank, old);
            if (ret == 77) {
                if (rank == 0) {
                    CUDA_CHECK(cuMemFree(old));
                    CUDA_CHECK(cuMemFree(lower));
                    CUDA_CHECK(cuMemFree(upper));
                }

                goto out_free_ipc;
            }
        }

        UCX_CHECK(status);

        if (rank == 0) {
            CUDA_CHECK(cuMemFree(old));
            CUDA_CHECK(cuMemFree(lower));
            CUDA_CHECK(cuMemAlloc(&replacement, 2 * BLOCK_SIZE));
            overlaps += (replacement < old) &&
                        ((replacement + 2 * BLOCK_SIZE) > old);
        }

        UCX_CHECK(transfer(&ipc, rank, replacement, recv, 2 * BLOCK_SIZE,
                           iteration + 101, host));
        if (rank == 0) {
            CUDA_CHECK(cuMemFree(replacement));
            CUDA_CHECK(cuMemFree(upper));
        }
    }

    MPI_Allreduce(&overlaps, &total_overlaps, 1, MPI_UNSIGNED, MPI_SUM,
                  MPI_COMM_WORLD);
    errors_and_warnings = test_cuda_get_errors_and_warnings();
    MPI_Allreduce(&errors_and_warnings, &total_errors, 1, MPI_UNSIGNED, MPI_SUM,
                  MPI_COMM_WORLD);
    ret = (total_errors != 0) ? 1 :
          ((total_overlaps == 0) ? 77 : 0);
    if (rank == 0) {
        printf("%s: lower overlaps=%u, UCX errors/warnings=%u; "
               "data checks passed\n",
               (ret == 0) ? "PASS" : ((ret == 77) ? "SKIP" : "FAIL"),
               total_overlaps, total_errors);
    }

out_free_ipc:
    free(ipc.rkey_buffer);
    if (ipc.md != NULL) {
        uct_md_close(ipc.md);
    }

    free(host);
    CUDA_CHECK(cuMemFree(recv));
    MPI_Finalize();
    CUDA_CHECK(cuCtxPopCurrent(NULL));
    CUDA_CHECK(cuDevicePrimaryCtxRelease(device));
    return ret;
}
