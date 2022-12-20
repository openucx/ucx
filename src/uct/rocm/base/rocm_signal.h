/*
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */


#ifndef ROCM_SIGNAL_H
#define ROCM_SIGNAL_H

#include "rocm_base.h"

#include <hsa.h>

typedef struct uct_rocm_base_signal_desc {
    hsa_signal_t signal;
    void *mapped_addr;
    uct_completion_t *comp;
    ucs_queue_elem_t queue;
} uct_rocm_base_signal_desc_t;

ucs_status_t uct_rocm_base_signal_init(void);
void uct_rocm_base_signal_finalize();
uct_rocm_base_signal_desc_t* uct_rocm_base_get_signal();
unsigned uct_rocm_base_iface_progress(uct_iface_h tl_iface);
ucs_status_t uct_rocm_base_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                       uct_completion_t *comp);
void uct_rocm_base_signal_push(uct_rocm_base_signal_desc_t* rocm_signal);

#endif /* ROCM_SIGNAL_H */
