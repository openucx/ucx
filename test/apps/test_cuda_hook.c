/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <ucp/api/ucp.h>
#include <ucm/api/ucm.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <getopt.h>
#include <cuda.h>


static void event_cb(ucm_event_type_t event_type, ucm_event_t *event, void *arg)
{
    int *count_p = arg;
    const char *title;

    if (event_type == UCM_EVENT_MEM_TYPE_ALLOC) {
        title = "allocate";
    } else if (event_type == UCM_EVENT_MEM_TYPE_FREE) {
        title = "free";
    } else {
        printf("unexpected memory event type %d\n", event_type);
        return;
    }

    printf("%s %s address %p size %zu\n", title,
           ucs_memory_type_names[event->mem_type.mem_type],
           event->mem_type.address, event->mem_type.size);
    ++(*count_p);
}

static void alloc_driver_api()
{
    CUdeviceptr dptr = 0;
    CUcontext context;
    CUdevice device;
    CUresult res;

    res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        printf("cuInit() failed: %d\n", res);
        return;
    }

    res = cuDeviceGet(&device, 0);
    if (res != CUDA_SUCCESS) {
        printf("cuDeviceGet(0) failed: %d\n", res);
        return;
    }

    res = cuCtxCreate(&context, 0, device);
    if (res != CUDA_SUCCESS) {
        printf("cuCtxCreate() failed: %d\n", res);
        return;
    }

    res = cuMemAlloc(&dptr, 4096);
    printf("cuMemAlloc() returned 0x%lx result %d\n", (uintptr_t)dptr, res);
    cuMemFree(dptr);

    cuCtxDetach(context);
}

static void alloc_runtime_api()
{
    void *dptr = NULL;
    cudaError_t res;

    res = cudaMalloc(&dptr, 4096);
    printf("cudaMalloc() returned %p result %d\n", dptr, res);
    cudaFree(dptr);
}

int main(int argc, char **argv)
{
    static const ucm_event_type_t memtype_events = UCM_EVENT_MEM_TYPE_ALLOC |
                                                   UCM_EVENT_MEM_TYPE_FREE;
    static const size_t dummy_va_size            = 4 * (1ul << 30); /* 4 GB */
    static const int num_expected_events         = 2;
    ucp_context_h context;
    ucs_status_t status;
    ucp_params_t params;
    int use_driver_api;
    void *dummy_ptr;
    int num_events;
    int c;

    use_driver_api = 0;
    while ((c = getopt(argc, argv, "d")) != -1) {
        switch (c) {
        case 'd':
            use_driver_api = 1;
            break;
        default:
            printf("Usage: test_cuda_hook [options]\n");
            printf("Options are:\n");
            printf("  -d :   Use Cuda driver API (Default: use runtime API)\n");
            printf("\n");
            return -1;
        }
    }

    /* In order to test long jumps in bistro hooks code, increase address space
     * separation by allocaing a large VA space segment.
     */
    dummy_ptr = mmap(NULL, dummy_va_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (dummy_ptr == MAP_FAILED) {
        printf("failed to allocate dummy VA space: %m\n");
        return -1;
    }

    printf("allocated dummy VA space at %p\n", dummy_ptr);

    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features   = UCP_FEATURE_TAG | UCP_FEATURE_STREAM;
    status            = ucp_init(&params, NULL, &context);
    if (status != UCS_OK) {
        printf("failed to create context\n");
        return -1;
    }

    num_events = 0;
    ucm_set_event_handler(memtype_events, 1000, event_cb, &num_events);

    if (use_driver_api) {
        alloc_driver_api();
    } else {
        alloc_runtime_api();
    }

    ucm_unset_event_handler(memtype_events, event_cb, &num_events);
    printf("got %d/%d memory events\n", num_events, num_expected_events);

    ucp_cleanup(context);

    munmap(dummy_ptr, dummy_va_size);
    return (num_events >= num_expected_events) ? 0 : -1;
}
