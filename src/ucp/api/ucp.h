/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCP_H_
#define UCP_H_

#include "ucp_def.h"
#include <uct/api/uct.h>
#include <ucs/type/status.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>


typedef struct ucp_context {
    uct_context_h       uct_context;
    uct_resource_desc_t *resources;    /* array of resources */
    unsigned            num_resources; /* number of the final resources for the ucp layer to use */
} ucp_context_t;


/**
 * Remote protocol layer endpoint
 */
typedef struct ucp_ep {
    ucp_iface_h       ucp_iface;
    uct_ep_h          uct_ep;       /* TODO remote eps - one per transport */
} ucp_ep_t;


/**
 * Local protocol layer interface
 */
typedef struct ucp_iface {
    ucp_context_h     context;
    uct_iface_h       uct_iface;
} ucp_iface_t;


/**
 * Device specification
 */
typedef struct ucp_device_config {
    char              **device_name;
    unsigned          count;    /* number of devices */
} ucp_device_config_t;


struct ucp_iface_config {
    ucp_device_config_t   devices;
    int                   device_policy_force;
};

/**
 * @ingroup CONTEXT
 * @brief Initialize global ucp context.
 *
 * @param [out] context_p   Filled with a ucp context handle.
 *
 * @return Error code.
 */
ucs_status_t ucp_init(ucp_context_h *context_p);


/**
 * @ingroup CONTEXT
 * @brief Destroy global ucp context.
 *
 * @param [in] context_p   Handle to the ucp context.
 */
void ucp_cleanup(ucp_context_h context_p);

/**
 * @ingroup CONTEXT
 * @brief Create and open a communication interface.
 *
 * @param [in]  ucp_context   Handle to the ucp context.
 * @param [in]  env_prefix    If non-NULL, search for environment variables
 *                            starting with this UCP_<prefix>_. Otherwise, search
 *                            for environment variables starting with just UCP_.
 * @param [out] ucp_iface     Filled with a handle to the opened communication interface.
 *
 * @return Error code.
 */
ucs_status_t ucp_iface_create(ucp_context_h ucp_context, const char *env_prefix, ucp_iface_h *ucp_iface);


/**
 * @ingroup CONTEXT
 * @brief Close the communication interface.
 *
 * @param [in]  ucp_iface   Handle to the communication interface.
 */
void ucp_iface_close(ucp_iface_h ucp_iface);


/**
 * @ingroup CONTEXT
 * @brief Create and open a remote endpoint.
 *
 * @param [in]  ucp_iface   Handle to the communication interface.
 * @param [out] ucp_ep      Filled with a handle to the opened remote endpoint.
 *
 * @return Error code.
 */
ucs_status_t ucp_ep_create(ucp_iface_h ucp_iface, ucp_ep_h *ucp_ep);


/**
 * @ingroup CONTEXT
 * @brief Close the remote endpoint.
 *
 * @param [in]  ucp_ep   Handle to the remote endpoint.
 */
void ucp_ep_destroy(ucp_ep_h ucp_ep);


#endif
