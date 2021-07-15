/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
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
#include <ucs/datastruct/callbackq.h>
#include <ucs/config/types.h>
#include <sys/types.h>
#include <stddef.h>

BEGIN_C_DECLS

/** @file tl.h */

/* endpoint - put */

typedef ucs_status_t (*uct_ep_put_short_func_t)(uct_ep_h ep,
                                                const void *buffer,
                                                unsigned length,
                                                uint64_t remote_addr,
                                                uct_rkey_t rkey);

typedef ssize_t      (*uct_ep_put_bcopy_func_t)(uct_ep_h ep,
                                                uct_pack_callback_t pack_cb,
                                                void *arg,
                                                uint64_t remote_addr,
                                                uct_rkey_t rkey);

typedef ucs_status_t (*uct_ep_put_zcopy_func_t)(uct_ep_h ep,
                                                const uct_iov_t *iov,
                                                size_t iovcnt,
                                                uint64_t remote_addr,
                                                uct_rkey_t rkey,
                                                uct_completion_t *comp);

/* endpoint - get */

typedef ucs_status_t (*uct_ep_get_short_func_t)(uct_ep_h ep,
                                                void *buffer,
                                                unsigned length,
                                                uint64_t remote_addr,
                                                uct_rkey_t rkey);

typedef ucs_status_t (*uct_ep_get_bcopy_func_t)(uct_ep_h ep,
                                                uct_unpack_callback_t unpack_cb,
                                                void *arg,
                                                size_t length,
                                                uint64_t remote_addr,
                                                uct_rkey_t rkey,
                                                uct_completion_t *comp);

typedef ucs_status_t (*uct_ep_get_zcopy_func_t)(uct_ep_h ep,
                                                const uct_iov_t *iov,
                                                size_t iovcnt,
                                                uint64_t remote_addr,
                                                uct_rkey_t rkey,
                                                uct_completion_t *comp);

/* endpoint - active message */

typedef ucs_status_t (*uct_ep_am_short_func_t)(uct_ep_h ep,
                                               uint8_t id,
                                               uint64_t header,
                                               const void *payload,
                                               unsigned length);

typedef ucs_status_t (*uct_ep_am_short_iov_func_t)(uct_ep_h ep, uint8_t id,
                                                   const uct_iov_t *iov,
                                                   size_t iovcnt);

typedef ssize_t      (*uct_ep_am_bcopy_func_t)(uct_ep_h ep,
                                               uint8_t id,
                                               uct_pack_callback_t pack_cb,
                                               void *arg,
                                               unsigned flags);

typedef ucs_status_t (*uct_ep_am_zcopy_func_t)(uct_ep_h ep,
                                               uint8_t id,
                                               const void *header,
                                               unsigned header_length,
                                               const uct_iov_t *iov,
                                               size_t iovcnt,
                                               unsigned flags,
                                               uct_completion_t *comp);

/* endpoint - atomics */

typedef ucs_status_t (*uct_ep_atomic_cswap64_func_t)(uct_ep_h ep,
                                                     uint64_t compare,
                                                     uint64_t swap,
                                                     uint64_t remote_addr,
                                                     uct_rkey_t rkey,
                                                     uint64_t *result,
                                                     uct_completion_t *comp);

typedef ucs_status_t (*uct_ep_atomic_cswap32_func_t)(uct_ep_h ep,
                                                     uint32_t compare,
                                                     uint32_t swap,
                                                     uint64_t remote_addr,
                                                     uct_rkey_t rkey,
                                                     uint32_t *result,
                                                     uct_completion_t *comp);

typedef ucs_status_t (*uct_ep_atomic32_post_func_t)(uct_ep_h ep,
                                                    unsigned opcode,
                                                    uint32_t value,
                                                    uint64_t remote_addr,
                                                    uct_rkey_t rkey);

typedef ucs_status_t (*uct_ep_atomic64_post_func_t)(uct_ep_h ep,
                                                    unsigned opcode,
                                                    uint64_t value,
                                                    uint64_t remote_addr,
                                                    uct_rkey_t rkey);

typedef ucs_status_t (*uct_ep_atomic32_fetch_func_t)(uct_ep_h ep,
                                                     unsigned opcode,
                                                     uint32_t value,
                                                     uint32_t *result,
                                                     uint64_t remote_addr,
                                                     uct_rkey_t rkey,
                                                     uct_completion_t *comp);

typedef ucs_status_t (*uct_ep_atomic64_fetch_func_t)(uct_ep_h ep,
                                                     unsigned opcode,
                                                     uint64_t value,
                                                     uint64_t *result,
                                                     uint64_t remote_addr,
                                                     uct_rkey_t rkey,
                                                     uct_completion_t *comp);

/* endpoint - tagged operations */

typedef ucs_status_t (*uct_ep_tag_eager_short_func_t)(uct_ep_h ep,
                                                      uct_tag_t tag,
                                                      const void *data,
                                                      size_t length);

typedef ssize_t      (*uct_ep_tag_eager_bcopy_func_t)(uct_ep_h ep,
                                                      uct_tag_t tag,
                                                      uint64_t imm,
                                                      uct_pack_callback_t pack_cb,
                                                      void *arg,
                                                      unsigned flags);

typedef ucs_status_t (*uct_ep_tag_eager_zcopy_func_t)(uct_ep_h ep,
                                                      uct_tag_t tag,
                                                      uint64_t imm,
                                                      const uct_iov_t *iov,
                                                      size_t iovcnt,
                                                      unsigned flags,
                                                      uct_completion_t *comp);

typedef ucs_status_ptr_t (*uct_ep_tag_rndv_zcopy_func_t)(uct_ep_h ep,
                                                         uct_tag_t tag,
                                                         const void *header,
                                                         unsigned header_length,
                                                         const uct_iov_t *iov,
                                                         size_t iovcnt,
                                                         unsigned flags,
                                                         uct_completion_t *comp);

typedef ucs_status_t (*uct_ep_tag_rndv_cancel_func_t)(uct_ep_h ep, void *op);

typedef ucs_status_t (*uct_ep_tag_rndv_request_func_t)(uct_ep_h ep,
                                                       uct_tag_t tag,
                                                       const void* header,
                                                       unsigned header_length,
                                                       unsigned flags);

/* interface - tagged operations */

typedef ucs_status_t (*uct_iface_tag_recv_zcopy_func_t)(uct_iface_h iface,
                                                        uct_tag_t tag,
                                                        uct_tag_t tag_mask,
                                                        const uct_iov_t *iov,
                                                        size_t iovcnt,
                                                        uct_tag_context_t *ctx);

typedef ucs_status_t (*uct_iface_tag_recv_cancel_func_t)(uct_iface_h iface,
                                                         uct_tag_context_t *ctx,
                                                         int force);

/* endpoint - pending queue */

typedef ucs_status_t (*uct_ep_pending_add_func_t)(uct_ep_h ep,
                                                  uct_pending_req_t *n,
                                                  unsigned flags);

typedef void         (*uct_ep_pending_purge_func_t)(uct_ep_h ep,
                                                    uct_pending_purge_callback_t cb,
                                                    void *arg);

/* endpoint - synchronization */

typedef ucs_status_t (*uct_ep_flush_func_t)(uct_ep_h ep,
                                            unsigned flags,
                                            uct_completion_t *comp);

typedef ucs_status_t (*uct_ep_fence_func_t)(uct_ep_h ep, unsigned flags);

typedef ucs_status_t (*uct_ep_check_func_t)(uct_ep_h ep,
                                            unsigned flags,
                                            uct_completion_t *comp);

/* endpoint - connection establishment */

typedef ucs_status_t (*uct_ep_create_func_t)(const uct_ep_params_t *params,
                                             uct_ep_h *ep_p);

typedef ucs_status_t (*uct_ep_connect_func_t)(
        uct_ep_h ep, const uct_ep_connect_params_t *params);

typedef ucs_status_t (*uct_ep_disconnect_func_t)(uct_ep_h ep, unsigned flags);

typedef ucs_status_t (*uct_cm_ep_conn_notify_func_t)(uct_ep_h ep);

typedef ucs_status_t (*uct_ep_query_func_t)(uct_ep_h ep, uct_ep_attr_t *ep_attr);

typedef void         (*uct_ep_destroy_func_t)(uct_ep_h ep);

typedef ucs_status_t (*uct_ep_get_address_func_t)(uct_ep_h ep,
                                                  uct_ep_addr_t *addr);

typedef ucs_status_t (*uct_ep_connect_to_ep_func_t)(uct_ep_h ep,
                                                    const uct_device_addr_t *dev_addr,
                                                    const uct_ep_addr_t *ep_addr);

typedef ucs_status_t (*uct_iface_accept_func_t)(uct_iface_h iface,
                                                uct_conn_request_h conn_request);

typedef ucs_status_t (*uct_iface_reject_func_t)(uct_iface_h iface,
                                                uct_conn_request_h conn_request);

/* interface - synchronization */

typedef ucs_status_t (*uct_iface_flush_func_t)(uct_iface_h iface,
                                               unsigned flags,
                                               uct_completion_t *comp);

typedef ucs_status_t (*uct_iface_fence_func_t)(uct_iface_h iface, unsigned flags);

/* interface - progress control */

typedef void         (*uct_iface_progress_enable_func_t)(uct_iface_h iface,
                                                         unsigned flags);

typedef void         (*uct_iface_progress_disable_func_t)(uct_iface_h iface,
                                                          unsigned flags);

typedef unsigned     (*uct_iface_progress_func_t)(uct_iface_h iface);

/* interface - events */

typedef ucs_status_t (*uct_iface_event_fd_get_func_t)(uct_iface_h iface,
                                                      int *fd_p);

typedef ucs_status_t (*uct_iface_event_arm_func_t)(uct_iface_h iface,
                                                   unsigned events);

/* interface - management */

typedef void         (*uct_iface_close_func_t)(uct_iface_h iface);

typedef ucs_status_t (*uct_iface_query_func_t)(uct_iface_h iface,
                                               uct_iface_attr_t *iface_attr);

/* interface - connection establishment */

typedef ucs_status_t (*uct_iface_get_device_address_func_t)(uct_iface_h iface,
                                                            uct_device_addr_t *addr);

typedef ucs_status_t (*uct_iface_get_address_func_t)(uct_iface_h iface,
                                                     uct_iface_addr_t *addr);

typedef int          (*uct_iface_is_reachable_func_t)(const uct_iface_h iface,
                                                      const uct_device_addr_t *dev_addr,
                                                      const uct_iface_addr_t *iface_addr);


/**
 * Transport interface operations.
 * Every operation exposed in the API must appear in the table below, to allow
 * creating interface/endpoint with custom operations.
 */
typedef struct uct_iface_ops {

    /* endpoint - put */
    uct_ep_put_short_func_t             ep_put_short;
    uct_ep_put_bcopy_func_t             ep_put_bcopy;
    uct_ep_put_zcopy_func_t             ep_put_zcopy;

    /* endpoint - get */
    uct_ep_get_short_func_t             ep_get_short;
    uct_ep_get_bcopy_func_t             ep_get_bcopy;
    uct_ep_get_zcopy_func_t             ep_get_zcopy;

    /* endpoint - active message */
    uct_ep_am_short_func_t              ep_am_short;
    uct_ep_am_short_iov_func_t          ep_am_short_iov;
    uct_ep_am_bcopy_func_t              ep_am_bcopy;
    uct_ep_am_zcopy_func_t              ep_am_zcopy;

    /* endpoint - atomics */
    uct_ep_atomic_cswap64_func_t        ep_atomic_cswap64;
    uct_ep_atomic_cswap32_func_t        ep_atomic_cswap32;
    uct_ep_atomic32_post_func_t         ep_atomic32_post;
    uct_ep_atomic64_post_func_t         ep_atomic64_post;
    uct_ep_atomic32_fetch_func_t        ep_atomic32_fetch;
    uct_ep_atomic64_fetch_func_t        ep_atomic64_fetch;

    /* endpoint - tagged operations */
    uct_ep_tag_eager_short_func_t       ep_tag_eager_short;
    uct_ep_tag_eager_bcopy_func_t       ep_tag_eager_bcopy;
    uct_ep_tag_eager_zcopy_func_t       ep_tag_eager_zcopy;
    uct_ep_tag_rndv_zcopy_func_t        ep_tag_rndv_zcopy;
    uct_ep_tag_rndv_cancel_func_t       ep_tag_rndv_cancel;
    uct_ep_tag_rndv_request_func_t      ep_tag_rndv_request;

    /* interface - tagged operations */
    uct_iface_tag_recv_zcopy_func_t     iface_tag_recv_zcopy;
    uct_iface_tag_recv_cancel_func_t    iface_tag_recv_cancel;

    /* endpoint - pending queue */
    uct_ep_pending_add_func_t           ep_pending_add;
    uct_ep_pending_purge_func_t         ep_pending_purge;

    /* endpoint - synchronization */
    uct_ep_flush_func_t                 ep_flush;
    uct_ep_fence_func_t                 ep_fence;
    uct_ep_check_func_t                 ep_check;

    /* endpoint - connection establishment */
    uct_ep_create_func_t                ep_create;
    uct_ep_connect_func_t               ep_connect;
    uct_ep_disconnect_func_t            ep_disconnect;
    uct_cm_ep_conn_notify_func_t        cm_ep_conn_notify;
    uct_ep_query_func_t                 ep_query;
    uct_ep_destroy_func_t               ep_destroy;
    uct_ep_get_address_func_t           ep_get_address;
    uct_ep_connect_to_ep_func_t         ep_connect_to_ep;
    uct_iface_accept_func_t             iface_accept;
    uct_iface_reject_func_t             iface_reject;

    /* interface - synchronization */
    uct_iface_flush_func_t              iface_flush;
    uct_iface_fence_func_t              iface_fence;

    /* interface - progress control */
    uct_iface_progress_enable_func_t    iface_progress_enable;
    uct_iface_progress_disable_func_t   iface_progress_disable;
    uct_iface_progress_func_t           iface_progress;

    /* interface - events */
    uct_iface_event_fd_get_func_t       iface_event_fd_get;
    uct_iface_event_arm_func_t          iface_event_arm;

    /* interface - management */
    uct_iface_close_func_t              iface_close;
    uct_iface_query_func_t              iface_query;

    /* interface - connection establishment */
    uct_iface_get_device_address_func_t iface_get_device_address;
    uct_iface_get_address_func_t        iface_get_address;
    uct_iface_is_reachable_func_t       iface_is_reachable;

} uct_iface_ops_t;


/**
 *  A progress engine and a domain for allocating communication resources.
 *  Different workers are progressed independently.
 */
typedef struct uct_worker {
    ucs_callbackq_t        progress_q;
} uct_worker_t;


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
 * Listener for incoming connections
 */
typedef struct uct_listener {
    uct_cm_h                 cm;
} uct_listener_t;


typedef struct uct_recv_desc uct_recv_desc_t;
typedef void (*uct_desc_release_callback_t)(uct_recv_desc_t *self, void * desc);


/**
 * Receive descriptor
 */
struct uct_recv_desc {
    uct_desc_release_callback_t cb;
};


#define uct_recv_desc(_desc) \
    ( *( ( (uct_recv_desc_t**)(_desc) ) - 1) )

END_C_DECLS

#endif
