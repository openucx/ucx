/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_TL_H_
#define UCT_TL_H_

#if !defined(UCT_H_)
#  error "You should not include this header directly. Include uct.h instead."
#endif

#include "uct_def.h"

#include <ucs/type/status.h>
#include <sys/types.h>
#include <stddef.h>


/**
 * Transport iface operations.
 */
typedef struct uct_iface_ops {

    void         (*iface_close)(uct_iface_h iface);

    ucs_status_t (*iface_query)(uct_iface_h iface,
                                uct_iface_attr_t *iface_attr);

    ucs_status_t (*iface_flush)(uct_iface_h iface, unsigned flags,
                                uct_completion_t *comp);

    ucs_status_t (*iface_fence)(uct_iface_h iface, unsigned flags);

    void         (*iface_release_am_desc)(uct_iface_h iface, void *desc);

    ucs_status_t (*iface_wakeup_open)(uct_iface_h iface, unsigned events,
                                     uct_wakeup_h wakeup);

    ucs_status_t (*iface_wakeup_get_fd)(uct_wakeup_h wakeup, int *fd_p);

    ucs_status_t (*iface_wakeup_arm)(uct_wakeup_h wakeup);

    ucs_status_t (*iface_wakeup_wait)(uct_wakeup_h wakeup);

    ucs_status_t (*iface_wakeup_signal)(uct_wakeup_h wakeup);

    void         (*iface_wakeup_close)(uct_wakeup_h wakeup);

    /* Connection establishment */

    ucs_status_t (*ep_create)(uct_iface_h iface, uct_ep_h *ep_p);

    ucs_status_t (*ep_create_connected)(uct_iface_h iface,
                                        const uct_device_addr_t *dev_addr,
                                        const uct_iface_addr_t *iface_addr,
                                        uct_ep_h* ep_p);

    void         (*ep_destroy)(uct_ep_h ep);

    ucs_status_t (*ep_get_address)(uct_ep_h ep, uct_ep_addr_t *addr);

    ucs_status_t (*ep_connect_to_ep)(uct_ep_h ep,
                                     const uct_device_addr_t *dev_addr,
                                     const uct_ep_addr_t *ep_addr);

    ucs_status_t (*iface_get_device_address)(uct_iface_h iface,
                                             uct_device_addr_t *addr);

    ucs_status_t (*iface_get_address)(uct_iface_h iface, uct_iface_addr_t *addr);

    int          (*iface_is_reachable)(const uct_iface_h iface,
                                       const uct_device_addr_t *dev_addr,
                                       const uct_iface_addr_t *iface_addr);

    /* Put */

    ucs_status_t (*ep_put_short)(uct_ep_h ep, const void *buffer, unsigned length,
                                 uint64_t remote_addr, uct_rkey_t rkey);

    ssize_t      (*ep_put_bcopy)(uct_ep_h ep, uct_pack_callback_t pack_cb,
                                 void *arg, uint64_t remote_addr, uct_rkey_t rkey);

    ucs_status_t (*ep_put_zcopy)(uct_ep_h ep, const uct_iov_t *iov, size_t iovcnt,
                                 uint64_t remote_addr, uct_rkey_t rkey,
                                 uct_completion_t *comp);

    /* Get */

    ucs_status_t (*ep_get_bcopy)(uct_ep_h ep, uct_unpack_callback_t unpack_cb,
                                 void *arg, size_t length,
                                 uint64_t remote_addr, uct_rkey_t rkey,
                                 uct_completion_t *comp);

    ucs_status_t (*ep_get_zcopy)(uct_ep_h ep, const uct_iov_t *iov, size_t iovcnt,
                                 uint64_t remote_addr, uct_rkey_t rkey,
                                 uct_completion_t *comp);

    /* Active message */

    ucs_status_t (*ep_am_short)(uct_ep_h ep, uint8_t id, uint64_t header,
                                const void *payload, unsigned length);

    ssize_t      (*ep_am_bcopy)(uct_ep_h ep, uint8_t id,
                                uct_pack_callback_t pack_cb, void *arg);

    ucs_status_t (*ep_am_zcopy)(uct_ep_h ep, uint8_t id, const void *header,
                                unsigned header_length, const uct_iov_t *iov,
                                size_t iovcnt, uct_completion_t *comp);

    /* Atomics */

    ucs_status_t (*ep_atomic_add64)(uct_ep_h ep, uint64_t add,
                                    uint64_t remote_addr, uct_rkey_t rkey);

    ucs_status_t (*ep_atomic_fadd64)(uct_ep_h ep, uint64_t add,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uint64_t *result, uct_completion_t *comp);

    ucs_status_t (*ep_atomic_swap64)(uct_ep_h ep, uint64_t swap,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uint64_t *result, uct_completion_t *comp);

    ucs_status_t (*ep_atomic_cswap64)(uct_ep_h ep, uint64_t compare, uint64_t swap,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uint64_t *result, uct_completion_t *comp);

    ucs_status_t (*ep_atomic_add32)(uct_ep_h ep, uint32_t add,
                                    uint64_t remote_addr, uct_rkey_t rkey);

    ucs_status_t (*ep_atomic_fadd32)(uct_ep_h ep, uint32_t add,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uint32_t *result, uct_completion_t *comp);

    ucs_status_t (*ep_atomic_swap32)(uct_ep_h ep, uint32_t swap,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uint32_t *result, uct_completion_t *comp);

    ucs_status_t (*ep_atomic_cswap32)(uct_ep_h ep, uint32_t compare, uint32_t swap,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uint32_t *result, uct_completion_t *comp);

    /* Pending queue */

    ucs_status_t (*ep_pending_add)(uct_ep_h ep, uct_pending_req_t *n);

    void         (*ep_pending_purge)(uct_ep_h ep, uct_pending_purge_callback_t cb,
                                     void * arg);

    /* TODO purge per iface */

    /* Synchronization */

    ucs_status_t (*ep_flush)(uct_ep_h ep, unsigned flags,
                             uct_completion_t *comp);

    ucs_status_t (*ep_fence)(uct_ep_h ep, unsigned flags);

} uct_iface_ops_t;


/**
 * Communication interface context
 */
typedef struct uct_iface {
    uct_iface_ops_t          ops;
} uct_iface_t;


/**
 * Remote endpoint
 */
typedef struct uct_ep {
    uct_iface_h              iface;
} uct_ep_t;


/**
 * Receive descriptor
 */
typedef struct uct_am_recv_desc {
    uct_iface_h              iface;
} uct_am_recv_desc_t;


#define uct_recv_desc_iface(_desc) \
    ((((uct_am_recv_desc_t*)(_desc)) - 1)->iface)



#endif
