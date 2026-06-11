/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <ucp/api/ucp.h>
#include <ucm/api/ucm.h>
#include <cuda_runtime.h>


__device__ int device_int;


typedef struct {
    int num_events;
    int num_errors;
} event_ctx_t;


static void event_cb(ucm_event_type_t event_type, ucm_event_t *event, void *arg)
{
    event_ctx_t *ctx = (event_ctx_t *)arg;
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
    ++ctx->num_events;

    if ((event_type == UCM_EVENT_MEM_TYPE_ALLOC) &&
        (event->mem_type.mem_type != UCS_MEMORY_TYPE_CUDA_MANAGED)) {
        printf("unexpected symbol memory type %s, expected %s\n",
               ucs_memory_type_names[event->mem_type.mem_type],
               ucs_memory_type_names[UCS_MEMORY_TYPE_CUDA_MANAGED]);
        ++ctx->num_errors;
    }
}

int main(int argc, char **argv)
{
    const int memtype_events      = UCM_EVENT_MEM_TYPE_ALLOC |
                                    UCM_EVENT_MEM_TYPE_FREE;
    const int num_expected_events = 1;
    ucp_context_h context;
    ucs_status_t status;
    ucp_params_t params;
    event_ctx_t event_ctx;
    void *dptr;
    cudaError_t res;

    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features   = UCP_FEATURE_TAG;
    status            = ucp_init(&params, NULL, &context);
    if (status != UCS_OK) {
        printf("failed to create UCP context: %s\n", ucs_status_string(status));
        return -1;
    }

    event_ctx.num_events = 0;
    event_ctx.num_errors = 0;
    ucm_set_event_handler(memtype_events, 1000, event_cb, &event_ctx);

    res = cudaGetSymbolAddress(&dptr, device_int);
    printf("cudaGetSymbolAddress() returned %p result %d\n", dptr, res);

    ucm_unset_event_handler(memtype_events, event_cb, &event_ctx);
    printf("got %d/%d memory events, %d errors\n", event_ctx.num_events,
           num_expected_events, event_ctx.num_errors);

    ucp_cleanup(context);
    return ((event_ctx.num_events == num_expected_events) &&
            (event_ctx.num_errors == 0)) ? 0 : -1;
}
