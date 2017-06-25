/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_IB_CM_H_
#define UCT_IB_CM_H_

#include <uct/base/uct_worker.h>
#include <uct/ib/base/ib_iface.h>
#include <ucs/datastruct/queue.h>
#include <ucs/sys/compiler.h>
#include <ucs/type/class.h>
#include <infiniband/cm.h>


/**
 * IB CM configuration
 */
typedef struct uct_cm_iface_config {
    uct_ib_iface_config_t  super;
    ucs_async_mode_t       async_mode;
    double                 timeout;
    unsigned               retry_count;
    unsigned               max_outstanding;
} uct_cm_iface_config_t;


/**
 * Outstanding operation - can be either a send or flush request.
 */
typedef struct uct_cm_iface_op {
    ucs_queue_elem_t       queue;    /* queue element */
    int                    is_id;    /* 1: id field is valid. 0: comp field is valid */
    union {
        struct ib_cm_id    *id;      /* send operation: cm id */
        uct_completion_t   *comp;    /* flush request: user completion */
    };
} uct_cm_iface_op_t;


/**
 * IB CM interface/
 */
typedef struct uct_cm_iface {
    uct_ib_iface_t            super;
    uint32_t                  service_id;      /* Service ID we're listening to */
    struct ib_cm_device      *cmdev;           /* CM device */
    struct ib_cm_id          *listen_id;       /* Listening "socket" */
    ucs_queue_head_t          notify_q;        /* Notification queue */
    uint32_t                  num_outstanding; /* Number of outstanding sends */
    ucs_queue_head_t          outstanding_q;   /* Outstanding operations queue */
    uct_worker_cb_id_t        slow_prog_id;    /* Callback id for slowpath progress */

    struct {
        int                timeout_ms;
        uint32_t           max_outstanding;
        uint8_t            retry_count;
    } config;
} uct_cm_iface_t;


/**
 * CM endpoint - container for destination address
 */
typedef struct uct_cm_ep {
    uct_base_ep_t          super;
    uint16_t               dlid;
    uint8_t                is_global;
    uint32_t               dest_service_id;
    union ibv_gid          dgid;
} uct_cm_ep_t;


/**
 * CM network header
 */
typedef struct uct_cm_hdr {
    uint8_t                am_id;   /* Active message ID */
    uint8_t                length;  /* Payload length */
} UCS_S_PACKED uct_cm_hdr_t;


/**
 * CM pending request private data
 */
typedef struct {
    uct_pending_req_priv_t super;
    uct_cm_ep_t            *ep;
} uct_cm_pending_req_priv_t;


UCS_CLASS_DECLARE_NEW_FUNC(uct_cm_ep_t, uct_ep_t, uct_iface_h,
                           const uct_device_addr_t *, const uct_iface_addr_t*);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_cm_ep_t, uct_ep_t);

ucs_status_t uct_cm_ep_connect_to_iface(uct_ep_h ep, const uct_iface_addr_t *iface_addr);
ucs_status_t uct_cm_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                uct_completion_t *comp);

ucs_status_t uct_cm_iface_flush_do(uct_cm_iface_t *iface, uct_completion_t *comp);

ssize_t uct_cm_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id, uct_pack_callback_t pack_cb,
                           void *arg, unsigned flags);

ucs_status_t uct_cm_ep_pending_add(uct_ep_h ep, uct_pending_req_t *req);
void uct_cm_ep_pending_purge(uct_ep_h ep, uct_pending_purge_callback_t cb,
                             void *arg);

ucs_status_t uct_cm_ep_flush(uct_ep_h tl_ep, unsigned flags,
                             uct_completion_t *comp);

static inline int uct_cm_iface_has_tx_resources(uct_cm_iface_t *iface)
{
    return iface->num_outstanding < iface->config.max_outstanding;
}

#define uct_cm_iface_trace_data(_iface, _type, _hdr, _fmt, ...) \
    uct_iface_trace_am(&(_iface)->super.super, _type, (_hdr)->am_id, \
                       (_hdr) + 1, (_hdr)->length, _fmt, ## __VA_ARGS__)

#define uct_cm_iface_worker(_iface) \
    ((_iface)->super.super.worker)

#define uct_cm_enter(_iface) \
    UCS_ASYNC_BLOCK(uct_cm_iface_worker(_iface)->async);

#define uct_cm_leave(_iface) \
    UCS_ASYNC_UNBLOCK(uct_cm_iface_worker(_iface)->async); \
    ucs_async_check_miss(uct_cm_iface_worker(_iface)->async);

#endif
