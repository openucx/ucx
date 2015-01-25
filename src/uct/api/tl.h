/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_TL_H_
#define UCT_TL_H_

#if !defined(UCT_H_)
#  error "You should not include this header directly. Include uct.h instead."
#endif

#include "uct_def.h"

#include <ucs/type/status.h>
#include <stddef.h>


/**
 * Transport iface operations.
 */
typedef struct uct_iface_ops {

    ucs_status_t (*iface_query)(uct_iface_h iface,
                                uct_iface_attr_t *iface_attr);
    ucs_status_t (*iface_get_address)(uct_iface_h iface,
                                      uct_iface_addr_t *iface_addr);

    ucs_status_t (*iface_flush)(uct_iface_h iface);

    void         (*iface_close)(uct_iface_h iface);

    ucs_status_t (*ep_create)(uct_iface_h iface, uct_ep_h *ep_p);
    void         (*ep_destroy)(uct_ep_h ep);

    /* Connection management */

    ucs_status_t (*ep_get_address)(uct_ep_h ep, uct_ep_addr_t *ep_addr);
    ucs_status_t (*ep_connect_to_iface)(uct_ep_h ep, uct_iface_addr_t *iface_addr);
    ucs_status_t (*ep_connect_to_ep)(uct_ep_h ep, uct_iface_addr_t *iface_addr,
                                     uct_ep_addr_t *ep_addr);

    /* Put */

    ucs_status_t (*ep_put_short)(uct_ep_h ep, void *buffer, unsigned length,
                                 uint64_t remote_addr, uct_rkey_t rkey);

    ucs_status_t (*ep_put_bcopy)(uct_ep_h ep, uct_pack_callback_t pack_cb,
                                 void *arg, size_t length, uint64_t remote_addr,
                                 uct_rkey_t rkey);

    ucs_status_t (*ep_put_zcopy)(uct_ep_h ep, void *buffer, size_t length,
                                 uct_lkey_t lkey, uint64_t remote_addr,
                                 uct_rkey_t rkey, uct_completion_t *comp);

    /* Get */

    ucs_status_t (*ep_get_bcopy)(uct_ep_h ep, size_t length, uint64_t remote_addr,
                                 uct_rkey_t rkey, uct_bcopy_recv_callback_t cb,
                                 void *arg);

    ucs_status_t (*ep_get_zcopy)(uct_ep_h ep, void *buffer, size_t length,
                                 uct_lkey_t lkey, uint64_t remote_addr,
                                 uct_rkey_t rkey, uct_completion_t *comp);

    /* Active message */

    ucs_status_t (*ep_am_short)(uct_ep_h ep, uint8_t id, uint64_t header,
                                void *payload, unsigned length);

    ucs_status_t (*ep_am_bcopy)(uct_ep_h ep, uint8_t id,
                                uct_pack_callback_t pack_cb, void *arg,
                                size_t length);

    ucs_status_t (*ep_am_zcopy)(uct_ep_h ep, uint8_t id, void *header,
                                unsigned header_length, void *payload,
                                size_t length, uct_lkey_t lkey,
                                uct_completion_t *comp);

    /* Atomics */

    ucs_status_t (*ep_atomic_add64)(uct_ep_h ep, uint64_t add,
                                    uint64_t remote_addr, uct_rkey_t rkey,
                                    uct_completion_t *comp);

    ucs_status_t (*ep_atomic_fadd64)(uct_ep_h ep, uint64_t add,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uct_imm_recv_callback_t cb, void *arg);

    ucs_status_t (*ep_atomic_swap64)(uct_ep_h ep, uint64_t swap,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uct_imm_recv_callback_t cb, void *arg);

    ucs_status_t (*ep_atomic_cswap64)(uct_ep_h ep, uint64_t compare, uint64_t swap,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_imm_recv_callback_t cb, void *arg);

    /* Synchronization */

    ucs_status_t (*ep_flush)(uct_ep_h ep);

} uct_iface_ops_t;


/**
 * Protection domain
 */
typedef struct uct_pd {
    uct_pd_ops_t             *ops;
    uct_context_h            context;
} uct_pd_t;


/**
 * Communication interface context
 */
typedef struct uct_iface {
    uct_iface_ops_t          ops;
    uct_pd_h                 pd;
} uct_iface_t;


/**
 * Remote endpoint
 */
typedef struct uct_ep {
    uct_iface_h              iface;
} uct_ep_t;


#endif
