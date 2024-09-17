/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2017. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_GDR_COPY_IFACE_H
#define UCT_GDR_COPY_IFACE_H

#include <uct/base/uct_iface.h>


typedef uint64_t uct_gdr_copy_iface_addr_t;


typedef struct uct_gdr_copy_iface {
    uct_base_iface_t            super;
    uct_gdr_copy_iface_addr_t   id;
    /* bandwidth and latency values used for perf estimate */
    struct {
        double            get_bw;
        double            put_bw;
        ucs_linear_func_t get_lat;
        ucs_linear_func_t put_lat;
    } bw_lat;

} uct_gdr_copy_iface_t;


typedef struct uct_gdr_copy_iface_config {
    uct_iface_config_t      super;
} uct_gdr_copy_iface_config_t;

#endif
