/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */
#include "nvtx.h"

void ucs_start_profile_range(const char *name)
{
    nvtxEventAttributes_t event_attrib = {0};
    uint32_t nvtx_colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
    unsigned cu_num_colors = sizeof(nvtx_colors)/sizeof(uint32_t);
    unsigned color_id;

    color_id                   = kh_str_hash_func(name) % cu_num_colors;
    event_attrib.version       = NVTX_VERSION;
    event_attrib.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    event_attrib.colorType     = NVTX_COLOR_ARGB;
    event_attrib.color         = nvtx_colors[color_id];
    event_attrib.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    event_attrib.message.ascii = name;

    nvtxRangePushEx(&event_attrib);
}

void ucs_stop_profile_range()
{
    nvtxRangePop();
}
