/**
 * Copyright (C) NVIDIA Corporation. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#include "profile.h"

#ifndef HAVE_NVTX
uint64_t ucs_profile_range_start(const char *format, ...)
{
    return 0;
}

void ucs_profile_range_stop(uint64_t id)
{
}

void ucs_profile_range_add_marker(const char *format, ...)
{
}

void ucs_profile_range_push(const char *format, ...)
{
}

void ucs_profile_range_pop()
{
}
#endif
