/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef TL_BASE_H_
#define TL_BASE_H_

#include "tl_def.h"

#include <services/sys/error.h>
#include <stddef.h>


/**
 * Communication interface context
 */
typedef struct mopsy_tl_iface {
} mopsy_tl_iface_t;


/**
 * Remote endpoint
 */
typedef struct mopsy_tl_ep {
    mopsy_tl_ops_t           *ops;
} mopsy_tl_ep_t;


/**
 * Send completion callback.
 */
typedef void (*mopsy_tl_completion_cb_t)(mopsy_tl_req_h req,
                                         mopsy_status_t status);


/**
 * Interface attributes: capabilities and limitations.
 */
typedef struct mopsy_tl_iface_attr {
    size_t                   max_short;
    size_t                   max_bcopy;
    size_t                   max_zcopy;
    size_t                   iface_addr_len;
    size_t                   ep_addr_len;
    unsigned                 flags;
} mopsy_tl_iface_attr_t;


/**
 * Transport operations.
 */
struct mopsy_tl_ops {

    mopsy_status_t (*iface_open)(mopsy_tl_context_h *context, mopsy_tl_iface_h *iface_p);
    void           (*iface_close)(mopsy_tl_iface_h iface);

    mopsy_status_t (*iface_query)(mopsy_tl_iface_h iface,
                                  mopsy_tl_iface_attr_t *iface_attr);
    mopsy_status_t (*iface_get_address)(mopsy_tl_iface_h iface,
                                        mopsy_tl_iface_addr_t *iface_addr);

    mopsy_status_t (*ep_create)(mopsy_tl_ep_h *ep_p);
    void           (*ep_destroy)(mopsy_tl_ep_h ep);

    mopsy_status_t (*ep_get_address)(mopsy_tl_ep_h *ep,
                                     mopsy_tl_ep_addr_t *ep_addr);
    mopsy_status_t (*ep_connect_to_iface)(mopsy_tl_iface_addr_t *iface_addr);
    mopsy_status_t (*ep_connect_to_ep)(mopsy_tl_iface_addr_t *iface_addr,
                                       mopsy_tl_ep_addr_t *ep_addr);

    mopsy_status_t (*ep_put_short)(mopsy_tl_ep_h ep, void *buffer, unsigned length,
                                   mopsy_tl_rkey_t rkey, mopsy_tl_req_h *req_p,
                                   mopsy_tl_completion_cb_t cb);

    mopsy_status_t (*iface_flush)(mopsy_tl_iface_h iface, mopsy_tl_req_h *req_p,
                                  mopsy_tl_completion_cb_t cb);

};


#endif
