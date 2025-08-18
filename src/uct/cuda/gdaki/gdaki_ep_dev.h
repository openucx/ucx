/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_GDAKI_EP_DEV_H
#define UCT_GDAKI_EP_DEV_H

#include <uct/api/cuda/uct.h>
#include <uct/api/uct_def.h>
#include <uct/api/uct.h>
#include <doca_gpunetio.h>

#define UCT_DEV_TL_GDAKI 1

#define UCT_DEV_CUDA_NAME "cuda"
#define UCT_DEV_CUDA_NAME_LEN 4

typedef struct {
    uct_dev_completion_t *comp;
} uct_gdaki_op_t;

typedef struct {
    uct_dev_ep_t super;
    struct doca_gpu_dev_verbs_qp *qp;
    uct_gdaki_op_t ops[0];
} uct_gdaki_dev_ep_t;

typedef struct {
    int e_op;
    size_t size;
    uint64_t src;
    uint32_t lkey;
    uint64_t dst;
    uint32_t rkey;
} uct_gdaki_batch_elem_t;

typedef struct {
    uct_batch_t super;
    int op;
    size_t num;
    uct_gdaki_dev_ep_t *ep;
    struct ibv_mr *mr;
    uint64_t atomic_buff;
    uct_gdaki_batch_elem_t list[0];
} uct_gdaki_batch_t;

#endif /* UCT_GDAKI_EP_DEV_H */
