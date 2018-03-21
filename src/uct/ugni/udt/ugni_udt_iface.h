/**
 * Copyright (c) UT-Battelle, LLC. 2014-2017. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_UGNI_UDT_IFACE_H
#define UCT_UGNI_UDT_IFACE_H

#include "ugni_udt_ep.h"
#include <uct/ugni/base/ugni_types.h>
#include <uct/ugni/base/ugni_iface.h>
#include <ucs/datastruct/list.h>
#include <uct/base/uct_md.h>
#include <ucs/async/async.h>
#include <ucs/async/pipe.h>

typedef void uct_ugni_udt_desc_t;

typedef struct uct_ugni_udt_iface {
    uct_ugni_iface_t        super;        /**< Super type */
    ucs_mpool_t             free_desc;    /**< Pool of FMA descriptors for
                                               requests without bouncing buffers */
    gni_ep_handle_t         ep_any;       /**< Unbound endpoint that accept any datagram
                                               messages */
    uct_ugni_udt_desc_t     *desc_any;    /**< Segment that accepts datagram from any source */
    uct_recv_desc_t         release_desc; /**< Callback for receive desc release */
    struct {
        unsigned            udt_seg_size; /**< Max UDT size */
        size_t              rx_headroom;  /**< The size of user defined header for am */
    } config;

    pthread_t event_thread;
    pthread_mutex_t device_lock;
    pthread_cond_t device_condition;
    int events_ready;
    ucs_async_pipe_t event_pipe;
} uct_ugni_udt_iface_t;

enum {
    UCT_UGNI_UDT_EMPTY    = 0,
    UCT_UGNI_UDT_PAYLOAD  = 1
};

typedef struct uct_ugni_udt_header {
    uint8_t type;
    uint8_t am_id;
    uint8_t length;
} uct_ugni_udt_header_t;

unsigned uct_ugni_udt_progress(void *arg);

#define uct_ugni_udt_get_offset(i) ((size_t)(ucs_max(sizeof(uct_ugni_udt_header_t), ((i)->config.rx_headroom  + \
                 sizeof(uct_recv_desc_t)))))

#define uct_ugni_udt_get_diff(i) ((size_t)(uct_ugni_udt_get_offset(i) - sizeof(uct_ugni_udt_header_t)))

#define uct_ugni_udt_get_rheader(d, i) ((uct_ugni_udt_header_t *)((char *)(d) + uct_ugni_udt_get_diff(i)))
#define uct_ugni_udt_get_sheader(d, i) ((uct_ugni_udt_header_t *)((char *)uct_ugni_udt_get_rheader(d, i) + GNI_DATAGRAM_MAXSIZE))

#define uct_ugni_udt_get_rpayload(d, i) (uct_ugni_udt_get_rheader(d, i) + 1)
#define uct_ugni_udt_get_spayload(d, i) (uct_ugni_udt_get_sheader(d, i) + 1)
#define uct_ugni_udt_get_user_desc(d, i) ((char *)uct_ugni_udt_get_rpayload(d, i) - (i)->config.rx_headroom)

#define UCT_UGNI_UDT_CHECK_RC(rc, desc)                            \
if (ucs_unlikely(GNI_RC_SUCCESS != rc)) {                          \
    if (GNI_RC_ERROR_RESOURCE == rc || GNI_RC_ERROR_NOMEM == rc) { \
        ucs_debug("GNI_EpPostDataWId failed, Error status: %s %d", \
                  gni_err_str[rc], rc);                            \
        ucs_mpool_put(desc);                                       \
        return UCS_ERR_NO_RESOURCE;                                \
    } else {                                                       \
        ucs_error("GNI_EpPostDataWId failed, Error status: %s %d", \
                  gni_err_str[rc], rc);                            \
        ucs_mpool_put(desc);                                       \
        return UCS_ERR_IO_ERROR;                                   \
    }                                                              \
}

#define uct_ugni_udt_iface_nic_handle(_iface) uct_ugni_iface_nic_handle(&(_iface)->super)

static inline void uct_ugni_udt_reset_desc(uct_ugni_udt_desc_t *desc, uct_ugni_udt_iface_t *iface)
{
    uct_ugni_udt_header_t *sheader = uct_ugni_udt_get_sheader(desc, iface);
    uct_ugni_udt_header_t *rheader = uct_ugni_udt_get_rheader(desc, iface);

    memset(sheader, 0, sizeof(*sheader));
    memset(rheader, 0, sizeof(*rheader));
}

static inline int uct_ugni_udt_ep_any_post(uct_ugni_udt_iface_t *iface)
{
    gni_return_t ugni_rc;

    uct_ugni_udt_reset_desc(iface->desc_any, iface);
    uct_ugni_cdm_lock(&iface->super.cdm);
    ugni_rc = GNI_EpPostDataWId(iface->ep_any,
                                uct_ugni_udt_get_sheader(iface->desc_any, iface),
                                iface->config.udt_seg_size,
                                uct_ugni_udt_get_rheader(iface->desc_any, iface),
                                iface->config.udt_seg_size,
                                UCT_UGNI_UDT_ANY);
    uct_ugni_cdm_unlock(&iface->super.cdm);
    UCT_UGNI_UDT_CHECK_RC(ugni_rc, iface->desc_any);
    return UCS_OK;
}

#endif
