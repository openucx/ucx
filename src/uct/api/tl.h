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
#include <sys/socket.h>
#include <stddef.h>
#include <sched.h>


/**
 * Communication resource.
 */
typedef struct uct_resource_desc {
    char                     tl_name[UCT_MAX_NAME_LEN];   /* Transport name */
    char                     dev_name[UCT_MAX_NAME_LEN];  /* Hardware device name */
    uint64_t                 latency;      /* Latency, nanoseconds */
    size_t                   bandwidth;    /* Bandwidth, bytes/second */
    cpu_set_t                local_cpus;   /* Mask of CPUs near the resource */
    socklen_t                addrlen;      /* Size of address */
    struct sockaddr_storage  subnet_addr;  /* Subnet address. Devices which can
                                              reach each other have same address */
} uct_resource_desc_t;


struct uct_iface_addr {
};

struct uct_ep_addr {
};


/**
 * Send completion callback.
 */
typedef void (*uct_completion_cb_t)(uct_req_h req, ucs_status_t status);


typedef struct uct_callback uct_callback_t;
struct uct_callback {
    void (*cb)(uct_callback_t *self, ucs_status_t status);
};


/**
 * Remote key release function.
 */
typedef void (*uct_rkey_release_func_t)(uct_context_h context, uct_rkey_t rkey);

/**
 * Interface attributes: capabilities and limitations.
 */
typedef struct uct_iface_attr {
    size_t                   max_short;
    size_t                   max_bcopy;
    size_t                   max_zcopy;
    size_t                   iface_addr_len;
    size_t                   ep_addr_len;
    unsigned                 flags;
} uct_iface_attr_t;


/**
 * Protection domain attributes
 */
typedef struct uct_pd_attr {
    size_t                   rkey_packed_size; /* Size of buffer needed for packed rkey */
} uct_pd_attr_t;


/**
 * Remote key with its type
 */
typedef struct uct_rkey_bundle {
    uct_rkey_t               rkey;   /**< Remote key descriptor, passed to RMA functions */
    void                     *type;  /**< Remote key type */
} uct_rkey_bundle_t;


/**
 * Transport "global" operations
 */
typedef struct uct_tl_ops {

    ucs_status_t (*query_resources)(uct_context_h context,
                                    uct_resource_desc_t **resources_p,
                                    unsigned *num_resources_p);

    ucs_status_t (*iface_open)(uct_context_h context, const char *dev_name,
                               uct_iface_config_t *config, uct_iface_h *iface_p);

    ucs_status_t (*rkey_unpack)(uct_context_h context, void *rkey_buffer,
                                uct_rkey_bundle_t *rkey_ob);

} uct_tl_ops_t;


/**
 * Transport memory operations
 */
typedef struct uct_pd_ops {
    ucs_status_t (*query)(uct_pd_h pd, uct_pd_attr_t *pd_attr);

    /* TODO
     * - support "mem attach", MPI-3 style, e.g by passing rkey
     * - support allocation, e.g by returning an address
     */
    ucs_status_t (*mem_map)(uct_pd_h pd, void *address, size_t length,
                            unsigned flags, uct_lkey_t *lkey_p);

    ucs_status_t (*mem_unmap)(uct_pd_h pd, uct_lkey_t lkey);

    ucs_status_t (*rkey_pack)(uct_pd_h pd, uct_lkey_t lkey, void *rkey_buffer);

} uct_pd_ops_t;


/**
 * Transport iface operations.
 */
typedef struct uct_iface_ops {

    ucs_status_t (*iface_query)(uct_iface_h iface,
                                uct_iface_attr_t *iface_attr);
    ucs_status_t (*iface_get_address)(uct_iface_h iface,
                                      uct_iface_addr_t *iface_addr);

    ucs_status_t (*iface_flush)(uct_iface_h iface, uct_req_h *req_p,
                                uct_completion_cb_t cb);

    void         (*iface_close)(uct_iface_h iface);

    ucs_status_t (*ep_create)(uct_iface_h iface, uct_ep_h *ep_p);
    void         (*ep_destroy)(uct_ep_h ep);

    ucs_status_t (*ep_get_address)(uct_ep_h ep,
                                     uct_ep_addr_t *ep_addr);
    ucs_status_t (*ep_connect_to_iface)(uct_ep_h ep, uct_iface_addr_t *iface_addr);
    ucs_status_t (*ep_connect_to_ep)(uct_ep_h ep, uct_iface_addr_t *iface_addr,
                                     uct_ep_addr_t *ep_addr);

    ucs_status_t (*ep_put_short)(uct_ep_h ep, void *buffer, unsigned length,
                                 uint64_t remote_addr, uct_rkey_t rkey);

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
