/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_SCOPY_IFACE_H
#define UCT_SCOPY_IFACE_H

#include "scopy_ep.h"

#include <uct/base/uct_iface.h>
#include <uct/sm/base/sm_iface.h>

#define uct_scopy_trace_data(_tx) \
    ucs_trace_data("%s [tx %p iov %zu/%zu length %zu/%zu] to %" PRIx64 "(%+ld)", \
                   uct_scopy_tx_op_str[(_tx)->op], (_tx), \
                   (_tx)->iov_iter.iov_index, (_tx)->iov_cnt, \
                   uct_iov_iter_flat_offset((_tx)->iov, (_tx)->iov_cnt, \
                                            &(_tx)->iov_iter), \
                   uct_iov_total_length((_tx)->iov, (_tx)->iov_cnt), \
                   (_tx)->remote_addr, (_tx)->rkey)


extern ucs_config_field_t uct_scopy_iface_config_table[];


typedef struct uct_scopy_iface_config {
    uct_sm_iface_config_t         super;
    size_t                        max_iov;    /* Maximum supported IOVs */
    size_t                        seg_size;   /* Segment size that is used to perfrom
                                               * data transfer for RMA operations */
    unsigned                      tx_quota;   /* How many TX segments can be dispatched
                                               * during iface progress */
    uct_iface_mpool_config_t      tx_mpool;   /* TX memory pool configuration */
} uct_scopy_iface_config_t;


typedef struct uct_scopy_iface {
    uct_sm_iface_t                super;
    ucs_arbiter_t                 arbiter;     /* TX arbiter */
    ucs_mpool_t                   tx_mpool;    /* TX memory pool */
    uct_scopy_ep_tx_func_t        tx;          /* TX function */
    struct {
        size_t                    max_iov;     /* Maximum supported IOVs limited by
                                                * user configuration and system
                                                * settings */
        size_t                    seg_size;    /* Maximal size of the segments
                                                * that has to be used in GET/PUT
                                                * Zcopy transfers */
        unsigned                  tx_quota;    /* How many TX segments can be dispatched
                                                * during iface progress */
    } config;
} uct_scopy_iface_t;


typedef struct uct_scopy_iface_ops {
    uct_iface_internal_ops_t super;
    uct_scopy_ep_tx_func_t   ep_tx;
} uct_scopy_iface_ops_t;


void uct_scopy_iface_query(uct_scopy_iface_t *iface, uct_iface_attr_t *iface_attr);

UCS_CLASS_DECLARE(uct_scopy_iface_t, uct_iface_ops_t*, uct_scopy_iface_ops_t*,
                  uct_md_h, uct_worker_h, const uct_iface_params_t*,
                  const uct_iface_config_t*);

unsigned uct_scopy_iface_progress(uct_iface_h tl_iface);

ucs_status_t uct_scopy_iface_event_arm(uct_iface_h tl_iface, unsigned events);

ucs_status_t uct_scopy_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                   uct_completion_t *comp);

#endif
