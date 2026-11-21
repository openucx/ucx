/*
 * Copyright (C) Advanced Micro Devices, Inc. 2023. ALL RIGHTS RESERVED.
 */

#ifndef UCT_ROCM_SIGNAL_H_
#define UCT_ROCM_SIGNAL_H_

#include <ucs/arch/cpu.h>
#include <ucs/type/class.h>
#include <ucs/sys/string.h>


typedef void (*uct_rocm_base_signal_release_cb_t)(void *arg);

typedef struct uct_rocm_base_signal_desc {
    hsa_signal_t                       signal;
    void                               *mapped_addr;
    uct_completion_t                   *comp;
    ucs_queue_elem_t                   queue;
    uct_rocm_base_signal_release_cb_t  release;     /**< Called when signal completes */
    void                               *release_arg;
} uct_rocm_base_signal_desc_t;

extern ucs_mpool_ops_t uct_rocm_base_signal_desc_mpool_ops;

/* Fill the fields consumed by @ref uct_rocm_base_progress. @a release runs
 * after the completion callback and before an uncached @a mapped_addr is
 * detached. */
void uct_rocm_base_signal_desc_set(uct_rocm_base_signal_desc_t *desc,
                                   uct_completion_t *comp, void *mapped_addr,
                                   uct_rocm_base_signal_release_cb_t release,
                                   void *release_arg);

unsigned uct_rocm_base_progress(ucs_queue_head_t *signal_queue);

#endif
