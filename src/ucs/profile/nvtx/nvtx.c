/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "nvtx.h"

uint32_t nvtx_colors[] = { 0xff00ff00,
                           0xff0000ff,
                           0xffffff00,
                           0xffff00ff,
                           0xff00ffff,
                           0xffff0000,
                           0xffffffff };

uint64_t ucs_profile_range_start(const char *format, ...)
{
    va_list args;
    char str[NVTX_RANGE_NAME_MAX_LEN];

    ucs_profile_set_str_from_args(str, format, args);

    return (uint64_t)nvtxRangeStartA(str);
}

void ucs_profile_range_stop(uint64_t id)
{
    nvtxRangeEnd(id);
}

void ucs_profile_range_add_marker(const char *format, ...)
{
    va_list args;
    char str[NVTX_RANGE_NAME_MAX_LEN];

    ucs_profile_set_str_from_args(str, format, args);

    nvtxMarkA(str);
}

void ucs_profile_range_push(const char *format, ...)
{
    nvtxEventAttributes_t attrib = {0};
    unsigned num_colors          = sizeof(nvtx_colors)/sizeof(uint32_t);
    va_list args;
    char str[NVTX_RANGE_NAME_MAX_LEN];

    ucs_profile_set_str_from_args(str, format, args);

    attrib.version       = NVTX_VERSION;
    attrib.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attrib.colorType     = NVTX_COLOR_ARGB;
    attrib.color         = nvtx_colors[kh_str_hash_func(str) % num_colors];
    attrib.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    attrib.message.ascii = str;

    nvtxRangePushEx(&attrib);
}

void ucs_profile_range_pop()
{
    nvtxRangePop();
}
