/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCM_MALLOC_HOOK_H_
#define UCM_MALLOC_HOOK_H_

#include <ucs/type/status.h>

ucs_status_t ucm_malloc_install(int events);

void ucm_malloc_state_reset(int default_mmap_thresh, int default_trim_thresh);

#endif
