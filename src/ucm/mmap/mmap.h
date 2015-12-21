/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCM_MMAP_H_
#define UCM_MMAP_H_

#include <ucm/api/ucm.h>

ucs_status_t ucm_mmap_install(int events);

void ucm_mmap_event_test_callback(ucm_event_type_t event_type,
                                  ucm_event_t *event, void *arg);

#endif
