/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_RC_VERBS_H
#define UCT_RC_VERBS_H

#include "rc_iface.h"
#include "rc_ep.h"

#include <ucs/type/class.h>


/**
 * RC verbs communication context.
 */
typedef struct uct_rc_verbs_iface {
    uct_rc_ep_t        super;

    struct {
        unsigned       available;
        unsigned       unsignaled;
    } tx;
} uct_rc_verbs_ep_t;


/**
 * RC verbs remote endpoint.
 */
typedef struct uct_rc_verbs_ep {
    uct_rc_iface_t     super;

    struct ibv_send_wr inl_am_wr;
    struct ibv_send_wr inl_rwrite_wr;
    struct ibv_sge     inl_sge[2];

    struct {
        unsigned       available;
    } rx;
} uct_rc_verbs_iface_t;


UCS_CLASS_DECLARE_NEW_FUNC(uct_rc_verbs_ep_t, uct_ep_t, uct_iface_h);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_rc_verbs_ep_t, uct_ep_t);

ucs_status_t uct_rc_verbs_ep_put_short(uct_ep_h tl_ep, void *buffer,
                                       unsigned length, uint64_t remote_addr,
                                       uct_rkey_t rkey);

ucs_status_t uct_rc_verbs_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                      void *buffer, unsigned length);

#endif
