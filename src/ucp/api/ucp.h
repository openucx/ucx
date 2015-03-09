/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCP_H_
#define UCP_H_

#include <ucp/api/ucp_def.h>
#include <uct/api/uct.h>
#include <ucs/type/status.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/sys/math.h>


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

/**
 * TL specification
 */
typedef struct ucp_tl_config {
    char              **tl_name;
    unsigned          count;      /* number of tls */
} ucp_tl_config_t;

struct ucp_iface_config {
    ucp_device_config_t   devices;
    int                   device_policy_force;
    ucp_tl_config_t       tls;     /* UCTs to use */
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

/* todo:
 * atomic
 *  non blocking put/get
 *  thread safety
 *  explicit connection establishment ?
 *  upc: hint to ucp to use transport with ordered data delivery
 */

typedef struct ucp_lkey {
} ucp_lkey_t;

typedef struct ucp_rkey {
} ucp_rkey_t;

/**
 * @ingroup CONTEXT
 * @brief Map or allocate memory for zero-copy sends and remote access.
 * 
 * @param [in]     context    UCP context to map memory on.
 * @param [out]    address_p  If != NULL, memory region to map.
 *                            If == NULL, filled with a pointer to allocated region.
 * @param [inout]  length_p   How many bytes to allocate. Filled with the actual
 *                            allocated size, which is larger than or equal to the
 *                            requested size.
 * @param [in]     flags      Allocation flags (currently reserved - set to 0).
 * @param [out]    lkey_p     Filled with local access key for allocated region.
 */
ucs_status_t ucp_mem_map(ucp_context_h context, void **address_p, size_t *length_p,
                         unsigned flags, ucp_lkey_h *lkey_p);

/**
 * @ingroup CONTEXT
 * @brief Undo the operation of uct_mem_map().
 *
 * @param [in]  context     UCP context which was used to allocate/map the memory.
 * @paran [in]  lkey        Local access key to memory region.
 */
ucs_status_t ucp_mem_unmap(ucp_context_h context, ucp_lkey_h lkey);

/**
 * @ingroup CONTEXT
 * @brief Serialize memory region remote access key
 *
 * @param [in]  lkey          memory region local key.
 * @param [out] rkey_buffer   contains serialized rkey. Caller is reponsible to free() it.
 * @param [out] size          length of serialized rkey. 
 */
ucs_status_t ucp_rkey_pack(ucp_lkey_h lkey, void **rkey_buffer_p, size_t *size_p);

/**
 * @ingroup CONTEXT
 * @brief Create rkey from serialized data
 *
 * @param [in]  context       UCP context
 * @param [in]  rkey_buffer   serialized rkey
 * @param [out] rkey          filled with rkey
 */
ucs_status_t ucp_rkey_unpack(ucp_context_h context, void *rkey_buffer, ucp_rkey_h *rkey_p);

/**
 * @ingroup CONTEXT
 * @brief Destroy remote key.
 *
 * param [in] rkey
 */
ucs_status_t ucp_rkey_destroy(ucp_rkey_h rkey);

/**
 * @ingroup CONTEXT
 * @brief If possible translate remote address into local address which can be used for direct memory access
 * 
 * @param [in]  ep              endpoint address
 * @param [in]  remote_addr     address to translate
 * @param [in]  rkey            remote memory key
 * @param [out] local_addr      filled by local memory key
 */
ucs_status_t ucp_rmem_ptr(ucp_ep_h ep, void *remote_addr, ucp_rkey_h rkey, void **local_addr_p);

/**
 * @ingroup CONTEXT
 *
 * @brief Force ordering between operations
 *
 * All operations started before fence will be completed before those
 * issued after.
 *
 * @param [in] context  UCP context
 */
ucs_status_t ucp_fence(ucp_context_h context);


/**
 * @ingroup CONTEXT
 *
 * @brief Force remote completion
 *
 * All operations that were started before ucp_quiet will be completed on 
 * remote when ucp_quiet returns
 *
 * @param [in] context  UCP context
 */
ucs_status_t ucp_quiet(ucp_context_h context);

void ucp_progress(ucp_context_h context);

/**
 * @ingroup CONTEXT
 * @brief Remote put. Returns when local buffer is safe for reuse.
 */
ucs_status_t ucp_ep_put(ucp_ep_h ep, void *buffer, unsigned length,
                        uint64_t remote_addr, ucp_rkey_h rkey);

/**
 * @ingroup CONTEXT
 * @brief Remote get. Returns when data are in local buffer
 */
ucs_status_t ucp_ep_get(uct_ep_h ep, void *buffer, size_t length,
                        uint64_t remote_addr, ucp_rkey_h rkey);
#endif
