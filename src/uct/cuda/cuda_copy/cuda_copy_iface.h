/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_COPY_IFACE_H
#define UCT_CUDA_COPY_IFACE_H

#include <uct/base/uct_iface.h>
#include <ucs/arch/cpu.h>
#include <ucs/sys/preprocessor.h>
#include <cuda_runtime.h>
#include <cuda.h>


#define UCT_CUDA_COPY_TL_NAME    "cuda_copy"
#define UCT_CUDA_DEV_NAME   "cudacopy0"

#define UCT_CUDA_FUNC(_func)  ({                        \
ucs_status_t _status = UCS_OK;                          \
do {                                                    \
    cudaError_t _result = (_func);                      \
    if (cudaSuccess != _result) {                       \
        ucs_error("%s failed with %d \n",               \
                  UCS_PP_MAKE_STRING(_func), _result);  \
        _status = UCS_ERR_IO_ERROR;                     \
    }                                                   \
} while (0);                                            \
_status;                                                \
})

typedef struct uct_cuda_copy_iface {
    uct_base_iface_t        super;
    ucs_mpool_t             cuda_event_desc;
    ucs_queue_head_t        outstanding_d2h_cuda_event_q;
    ucs_queue_head_t        outstanding_h2d_cuda_event_q;
    cudaStream_t            stream_d2h;
    cudaStream_t            stream_h2d;
    struct {
        unsigned            max_poll;
    } config;
} uct_cuda_copy_iface_t;


typedef struct uct_cuda_copy_iface_config {
    uct_iface_config_t      super;
    unsigned                max_poll;
} uct_cuda_copy_iface_config_t;

typedef struct uct_cuda_copy_event_desc {
    cudaEvent_t event;
    uct_completion_t *comp;
    ucs_queue_elem_t  queue;
} uct_cuda_copy_event_desc_t;

#endif
