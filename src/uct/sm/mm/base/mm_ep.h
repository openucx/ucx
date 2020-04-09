/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCT_MM_EP_H
#define UCT_MM_EP_H

#include "mm_iface.h"

#include <ucs/datastruct/khash.h>


KHASH_INIT(uct_mm_remote_seg, uintptr_t, uct_mm_remote_seg_t, 1,
           kh_int64_hash_func, kh_int64_hash_equal)


/**
 * MM transport endpoint
 */
typedef struct uct_mm_ep {
    uct_base_ep_t              super;

    /* Remote peer */
    uct_mm_fifo_ctl_t          *fifo_ctl;   /* pointer to the destination's ctl struct in the receive fifo */
    void                       *fifo_elems; /* fifo elements (destination's receive fifo) */

    uint64_t                   cached_tail; /* the sender's own copy of the remote FIFO's tail.
                                               it is not always updated with the actual remote tail value */

    /* mapped remote memory chunks to which remote descriptors belong to.
     * (after attaching to them) */
    khash_t(uct_mm_remote_seg) remote_segs;

    void                       *remote_iface_addr; /* remote md-specific address, can be NULL */

    ucs_arbiter_group_t        arb_group;   /* the group that holds this ep's pending operations */

    /* Used for signaling remote side wakeup */
    struct {
        struct sockaddr_un     sockaddr;  /* address of signaling socket */
        socklen_t              addrlen;   /* address length of signaling socket */
    } signal;
} uct_mm_ep_t;


UCS_CLASS_DECLARE_NEW_FUNC(uct_mm_ep_t, uct_ep_t,const uct_ep_params_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_mm_ep_t, uct_ep_t);

ucs_status_t uct_mm_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t header,
                                const void *payload, unsigned length);
ssize_t uct_mm_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id, uct_pack_callback_t pack_cb,
                           void *arg, unsigned flags);

ucs_status_t uct_mm_ep_flush(uct_ep_h tl_ep, unsigned flags,
                             uct_completion_t *comp);

ucs_status_t uct_mm_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *n,
                                   unsigned flags);

void uct_mm_ep_pending_purge(uct_ep_h ep, uct_pending_purge_callback_t cb,
                             void *arg);

ucs_arbiter_cb_result_t uct_mm_ep_process_pending(ucs_arbiter_t *arbiter,
                                                  ucs_arbiter_group_t *group,
                                                  ucs_arbiter_elem_t *elem,
                                                  void *arg);

#endif
