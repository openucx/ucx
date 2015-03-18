/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/


#ifndef UCT_UD_IFACE_H
#define UCT_UD_IFACE_H

#include <uct/ib/base/ib_iface.h>
#include <ucs/datastruct/sglib_wrapper.h>
#include <ucs/datastruct/ptr_array.h>

#include "ud_def.h"
#include "ud_ep.h"


#define UCT_UD_MIN_INLINE   48


typedef struct uct_ud_iface_addr {
    uct_iface_addr_t     super;
    uint32_t             qp_num;
    uint32_t             lid;
    /* TODO: add mtu */
} uct_ud_iface_addr_t;


/* TODO: maybe tx_moderation can be defined at compile-time since tx completions are used only to know how much space is there in tx qp */

typedef struct uct_ud_iface_config {
    uct_ib_iface_config_t    super;
} uct_ud_iface_config_t;


struct uct_ud_iface {
    uct_ib_iface_t           super;
    struct ibv_qp           *qp;
    struct {
        ucs_mpool_h          mp;
        unsigned             available;
    } rx;
    struct {
        ucs_mpool_h          mp;
        unsigned             available;
        /* TODO: move to base class as this is common with rc */
        unsigned             unsignaled;
        ucs_queue_head_t     pending_ops;
    } tx;
    struct {
        unsigned             tx_qp_len;
        unsigned             rx_max_batch;
    } config;
    ucs_ptr_array_t eps;
};
UCS_CLASS_DECLARE(uct_ud_iface_t, uct_iface_ops_t*, uct_worker_h, const char *,
                  unsigned, unsigned, uct_ud_iface_config_t*)


extern ucs_config_field_t uct_ud_iface_config_table[];

void uct_ud_iface_query(uct_ud_iface_t *iface, uct_iface_attr_t *iface_attr);

ucs_status_t uct_ud_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr);

#if 0
uct_ud_ep_t *uct_ud_iface_lookup_ep(uct_ud_iface_t *iface, unsigned qp_num);
#endif

void uct_ud_iface_add_ep(uct_ud_iface_t *iface, uct_ud_ep_t *ep);
void uct_ud_iface_remove_ep(uct_ud_iface_t *iface, uct_ud_ep_t *ep);

ucs_status_t uct_ud_iface_flush(uct_iface_h tl_iface);

static inline int uct_ud_iface_can_tx(uct_ud_iface_t *iface)
{
    if (iface->tx.available == 0) {
        ucs_trace_data("iface=%p out of tx wqe", iface);
        return 0;
    }
    return 1;
}

#endif

