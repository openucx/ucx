/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCM_UTIL_CONFIG_H_
#define UCM_UTIL_CONFIG_H_

#include <ucs/config/types.h>
#include <ucs/debug/log.h>
#include <stdio.h>


typedef struct ucm_config {
    ucs_log_level_t log_level;
    int             enable_events;
    int             enable_mmap_reloc;
    int             enable_malloc_hooks;
    int             enable_malloc_reloc;
    int             enable_dynamic_mmap_thresh;
#if HAVE_CUDA
    int             enable_cuda_hooks;
#endif
    size_t          alloc_alignment;
} ucm_config_t;


extern ucm_config_t ucm_global_config;

#endif
