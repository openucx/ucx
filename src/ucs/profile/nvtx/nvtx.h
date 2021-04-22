/*
 * Copyright (C) NVIDIA Corporation. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <nvToolsExt.h>
#include <ucs/datastruct/khash.h>
#include <ucs/profile/profile.h>
#include <stdarg.h>

#define NVTX_RANGE_NAME_MAX_LEN 256

#define ucs_profile_set_str_from_args(_str, _format, _args) \
    va_start(_args, _format); \
    vsprintf(_str, _format, _args); \
    va_end(_args);

uint64_t ucs_nvtx_profile_range_start(const char *format, ...);

void ucs_nvtx_profile_range_stop(uint64_t id);

void ucs_nvtx_profile_range_add_marker(const char *format, ...);

void ucs_nvtx_profile_range_push(const char *format, ...);

void ucs_nvtx_profile_range_pop();
