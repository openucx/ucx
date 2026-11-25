/*
 * Copyright (C) Advanced Micro Devices, Inc. 2023. ALL RIGHTS RESERVED.
 */

#include <ucs/arch/cpu.h>
#include <ucs/type/class.h>
#include <ucs/sys/string.h>


typedef struct uct_rocm_base_signal_desc {
    hsa_signal_t     signal;
    void             *mapped_addr;
    uct_completion_t *comp;
    ucs_queue_elem_t queue;
} uct_rocm_base_signal_desc_t;

extern ucs_mpool_ops_t uct_rocm_base_signal_desc_mpool_ops;

unsigned uct_rocm_base_progress(ucs_queue_head_t *signal_queue);
