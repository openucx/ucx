/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_IB_CM_H_
#define UCT_IB_CM_H_

#include <uct/ib/base/ib_iface.h>
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
} uct_cm_iface_config_t;


/**
 * IB CM interface/
 */
typedef struct uct_cm_iface {
    uct_ib_iface_t         super;
    uint32_t               service_id;  /* Service ID we're listening to */
    struct ib_cm_device    *cmdev;      /* CM device */
    struct ib_cm_id        *listen_id;  /* Listening "socket" */
    volatile uint32_t      inflight;    /* Atomic: number of inflight sends */

    struct {
        int                timeout_ms;
        uint8_t            retry_count;
    } config;
} uct_cm_iface_t;


/**
 * CM interface address - consists of IB address, and service ID.
 */
typedef struct uct_cm_iface_addr {
    uct_iface_addr_t       super;
    uint16_t               lid;
    union ibv_gid          gid;
    uint64_t               service_id;
} uct_cm_iface_addr_t;


/**
 * CM endpoint - container for destination address
 */
typedef struct uct_cm_ep {
    uct_base_ep_t          super;
    uct_cm_iface_addr_t    dest_addr;
} uct_cm_ep_t;


/**
 * CM network header
 */
typedef struct uct_cm_hdr {
    uint8_t                am_id;   /* Active message ID */
    uint8_t                length;  /* Payload length */
} UCS_S_PACKED uct_cm_hdr_t;


UCS_CLASS_DECLARE_NEW_FUNC(uct_cm_ep_t, uct_ep_t, uct_iface_h);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_cm_ep_t, uct_ep_t);

ucs_status_t uct_cm_ep_connect_to_iface(uct_ep_h ep, const uct_iface_addr_t *iface_addr);
ucs_status_t uct_cm_iface_flush(uct_iface_h tl_iface);

ucs_status_t uct_cm_ep_am_short(uct_ep_h ep, uint8_t id, uint64_t header,
                                const void *payload, unsigned length);

ucs_status_t uct_cm_iface_get_addr(uct_cm_iface_t *iface, uct_cm_iface_addr_t *addr);
ucs_status_t uct_cm_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                uct_pack_callback_t pack_cb, void *arg,
                                size_t length);


ucs_status_t uct_cm_ep_flush(uct_ep_h tl_ep);

#endif
