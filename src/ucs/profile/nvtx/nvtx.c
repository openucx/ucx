/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */
#include "nvtx.h"

uint32_t nvtx_colors[] = { 0xff00ff00,
                           0xff0000ff,
                           0xffffff00,
                           0xffff00ff,
                           0xff00ffff,
                           0xffff0000,
                           0xffffffff };

uint64_t ucs_profile_range_start(const char *name)
{
    return (uint64_t)nvtxRangeStartA(name);
}

void ucs_profile_range_stop(uint64_t id)
{
    nvtxRangeEnd(id);
}

void ucs_profile_range_push(const char *name)
{
    nvtxEventAttributes_t attrib = {0};
    unsigned num_colors          = sizeof(nvtx_colors)/sizeof(uint32_t);

    attrib.version       = NVTX_VERSION;
    attrib.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attrib.colorType     = NVTX_COLOR_ARGB;
    attrib.color         = nvtx_colors[kh_str_hash_func(name) % num_colors];
    attrib.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    attrib.message.ascii = name;

    nvtxRangePushEx(&attrib);
}

void ucs_profile_range_pop()
{
    nvtxRangePop();
}
