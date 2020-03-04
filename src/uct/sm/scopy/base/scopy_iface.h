/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_SCOPY_IFACE_H
#define UCT_SCOPY_IFACE_H

#include <uct/base/uct_iface.h>
#include <uct/sm/base/sm_iface.h>


extern ucs_config_field_t uct_scopy_iface_config_table[];


typedef struct uct_scopy_iface_config {
    uct_sm_iface_config_t         super;
    size_t                        max_iov;
} uct_scopy_iface_config_t;

typedef struct uct_scopy_iface {
    uct_sm_iface_t                super;
    struct {
        size_t                    max_iov;
    } config;
} uct_scopy_iface_t;

typedef struct uct_scopy_iface_ops {
    uct_iface_ops_t               super;
} uct_scopy_iface_ops_t;

#define uct_scopy_trace_data(_remote_addr, _rkey, _fmt, ...) \
    ucs_trace_data(_fmt " to %"PRIx64"(%+ld)", ## __VA_ARGS__, \
                   (_remote_addr), (_rkey))


void uct_scopy_iface_query(uct_scopy_iface_t *iface, uct_iface_attr_t *iface_attr);

UCS_CLASS_DECLARE(uct_scopy_iface_t, uct_scopy_iface_ops_t*, uct_md_h, uct_worker_h,
                  const uct_iface_params_t*, const uct_iface_config_t*);

#endif
