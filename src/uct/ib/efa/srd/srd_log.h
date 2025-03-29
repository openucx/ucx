/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef SRD_LOG_H_
#define SRD_LOG_H_

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <uct/ib/base/ib_iface.h>
#include <uct/ib/base/ib_log.h>


void uct_srd_dump_packet(uct_base_iface_t *iface, uct_am_trace_type_t type,
                         void *data, size_t length, size_t valid_length,
                         char *buffer, size_t max);


#endif
