/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef TL_BASE_H_
#define TL_BASE_H_


#include <uct/api/uct.h>
#include <ucs/datastruct/mpool.h>


/**
 * Transport operations
 */
struct uct_tl_ops {

    ucs_status_t (*query_resources)(uct_context_h context,
                                    uct_resource_desc_t **resources_p,
                                    unsigned *num_resources_p);

    ucs_status_t (*iface_open)(uct_context_h context, const char *dev_name,
                               uct_iface_config_t *config, uct_iface_h *iface_p);

    ucs_status_t (*rkey_unpack)(uct_context_h context, void *rkey_buffer,
                                uct_rkey_bundle_t *rkey_ob);

};


/**
 * Active message handle table entry
 */
typedef struct uct_am_handler {
    uct_am_callback_t        cb;
    void                     *arg;
} uct_am_handler_t;


/**
 * Base structure of all interfaces.
 * Includes the AM table which we don't want to expose.
 */
typedef struct uct_base_iface {
    uct_iface_t       super;
    uct_am_handler_t  am[UCT_AM_ID_MAX];
} uct_base_iface_t;


/**
 * "Base" structure which defines interface configuration options.
 * Specific transport extend this structure.
 */
struct uct_iface_config {
    size_t            max_short;
    size_t            max_bcopy;
};


#endif
