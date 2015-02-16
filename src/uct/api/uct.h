/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_H_
#define UCT_H_

#include <uct/api/uct_def.h>
#include <uct/api/tl.h>
#include <uct/api/version.h>
#include <ucs/config/types.h>
#include <ucs/datastruct/queue.h>
#include <ucs/type/callback.h>

#include <sys/socket.h>
#include <stdio.h>
#include <sched.h>


/**
 * Communication resource.
 */
typedef struct uct_resource_desc {
    char                     tl_name[UCT_MAX_NAME_LEN];   /**< Transport name */
    char                     dev_name[UCT_MAX_NAME_LEN];  /**< Hardware device name */
    uint64_t                 latency;      /**< Latency, nanoseconds */
    size_t                   bandwidth;    /**< Bandwidth, bytes/second */
    cpu_set_t                local_cpus;   /**< Mask of CPUs near the resource */
    struct sockaddr_storage  subnet_addr;  /**< Subnet address. Devices which can
                                                reach each other have same address */
} uct_resource_desc_t;


/**
 * Opaque type for interface address.
 */
struct uct_iface_addr {
};


/**
 * Opaque type for endpoint address.
 */
struct uct_ep_addr {
};


/**
 * Operation support flags.
 */
enum {
    /* Active message capabilities */
    UCT_IFACE_FLAG_AM_SHORT       = UCS_BIT(0),
    UCT_IFACE_FLAG_AM_BCOPY       = UCS_BIT(1),
    UCT_IFACE_FLAG_AM_ZCOPY       = UCS_BIT(2),

    /* PUT capabilities */
    UCT_IFACE_FLAG_PUT_SHORT      = UCS_BIT(4),
    UCT_IFACE_FLAG_PUT_BCOPY      = UCS_BIT(5),
    UCT_IFACE_FLAG_PUT_ZCOPY      = UCS_BIT(6),

    /* GET capabilities */
    UCT_IFACE_FLAG_GET_SHORT      = UCS_BIT(8),
    UCT_IFACE_FLAG_GET_BCOPY      = UCS_BIT(9),
    UCT_IFACE_FLAG_GET_ZCOPY      = UCS_BIT(10),

    /* Atomic operations capabilities */
    UCT_IFACE_FLAG_ATOMIC_ADD32   = UCS_BIT(16),
    UCT_IFACE_FLAG_ATOMIC_ADD64   = UCS_BIT(17),
    UCT_IFACE_FLAG_ATOMIC_FADD32  = UCS_BIT(18),
    UCT_IFACE_FLAG_ATOMIC_FADD64  = UCS_BIT(19),
    UCT_IFACE_FLAG_ATOMIC_SWAP32  = UCS_BIT(20),
    UCT_IFACE_FLAG_ATOMIC_SWAP64  = UCS_BIT(21),
    UCT_IFACE_FLAG_ATOMIC_CSWAP32 = UCS_BIT(22),
    UCT_IFACE_FLAG_ATOMIC_CSWAP64 = UCS_BIT(23),

    /* Error handling capabilities */
    UCT_IFACE_FLAG_ERRHANDLE_SHORT_BUF  = UCS_BIT(32), /* Invalid buffer for short */
    UCT_IFACE_FLAG_ERRHANDLE_BCOPY_BUF  = UCS_BIT(33), /* Invalid buffer for bcopy */
    UCT_IFACE_FLAG_ERRHANDLE_ZCOPY_BUF  = UCS_BIT(34), /* Invalid buffer for zcopy */
    UCT_IFACE_FLAG_ERRHANDLE_AM_ID      = UCS_BIT(35), /* Invalid AM id on remote */
    UCT_IFACE_FLAG_ERRHANDLE_REMOTE_MEM = UCS_BIT(35), /* Remote memory access */
};


/**
 * Interface attributes: capabilities and limitations.
 */
struct uct_iface_attr {
    struct {
        struct {
            size_t           max_short;
            size_t           max_bcopy;
            size_t           max_zcopy;
        } put;

        struct {
            size_t           max_bcopy;
            size_t           max_zcopy;
        } get;

        struct {
            size_t           max_short;  /* Total max. size (incl. the header) */
            size_t           max_bcopy;  /* Total max. size (incl. the header) */
            size_t           max_zcopy;  /* Total max. size (incl. the header) */
            size_t           max_hdr;    /* Max. header size for bcopy/zcopy */
        } am;

        uint64_t             flags;
    } cap;

    size_t                   iface_addr_len;
    size_t                   ep_addr_len;
    size_t                   completion_priv_len;
};


/**
 * Protection domain attributes
 */
struct uct_pd_attr {
    size_t                   rkey_packed_size; /* Size of buffer needed for packed rkey */
};


/**
 * Remote key with its type
 */
typedef struct uct_rkey_bundle {
    uct_rkey_t               rkey;   /**< Remote key descriptor, passed to RMA functions */
    void                     *type;  /**< Remote key type */
} uct_rkey_bundle_t;


/**
 * Completion handle.
 */
struct uct_completion {
    ucs_callback_t            super;
    char                      priv[0]; /**< Actual size of this field is returned
                                            in completion_priv_len by uct_iface_query() */
};


/**
 * @ingroup CONTEXT
 * @brief Initialize global context.
 *
 * @param [out] context_p   Filled with context handle.
 *
 * @return Error code.
 */
ucs_status_t uct_init(uct_context_h *context_p);


/**
 * @ingroup CONTEXT
 * @brief Destroy global context.
 *
 * @param [in] context   Handle to context.
 */
void uct_cleanup(uct_context_h context);


/**
 * @ingroup CONTEXT
 * @brief Progress all communications of the context.
 *
 * @param [in] context   Handle to context.
 */
void uct_progress(uct_context_h context);


/**
 * @ingroup CONTEXT
 * @brief Query for transport resources.
 *
 * @param [in]  context         Handle to context.
 * @param [out] resources_p     Filled with a pointer to an array of resource descriptors.
 * @param [out] num_resources_p Filled with the number of resources in the array.
 *
 * @return Error code.
 */
ucs_status_t uct_query_resources(uct_context_h context,
                                 uct_resource_desc_t **resources_p,
                                 unsigned *num_resources_p);


/**
 * @ingroup CONTEXT
 * @brief Release the list of resources returned from uct_query_resources.
 *
 * @param [in] resources  Array of resource descriptors to release.
 *
 */
void uct_release_resource_list(uct_resource_desc_t *resources);


/**
 * @ingroup CONTEXT
 * @brief Read transport-specific interface configuration.
 *
 * @param [in]  context       Handle to context.
 * @param [in]  tl_name       Transport name.
 * @param [in]  env_prefix    If non-NULL, search for environment variables
 *                            starting with this UCT_<prefix>_. Otherwise, search
 *                            for environment variables starting with just UCT_.
 * @param [in]  filename      If non-NULL, read configuration from this file. If
 *                            the file does not exist, it will be ignored.
 * @param [out] config_p      Filled with a pointer to configuration.
 *
 * @return Error code.
 */
ucs_status_t uct_iface_config_read(uct_context_h context, const char *tl_name,
                                   const char *env_prefix, const char *filename,
                                   uct_iface_config_t **config_p);


/**
 * @ingroup CONTEXT
 * @brief Release configuration memory returned from uct_iface_read_config().
 *
 * @param [in]  config        Configuration to release.
 */
void uct_iface_config_release(uct_iface_config_t *config);


/**
 * @ingroup CONTEXT
 * @brief Print interface configuration to a stream.
 *
 * @param [in]  config        Configuration to print.
 * @param [in]  stream        Output stream to print to.
 * @param [in]  title         Title to the output.
 * @param [in]  print_flags   Controls how the configuration is printed.
 */
void uct_iface_config_print(uct_iface_config_t *config, FILE *stream,
                            const char *title, ucs_config_print_flags_t print_flags);


/**
 * @ingroup CONTEXT
 * @brief Print interface configuration to a stream.
 *
 * @param [in]  config        Configuration to release.
 * @param [in]  name          Configuration variable name.
 * @param [in]  value         Value to set.
 *
 * @return Error code.
 */
ucs_status_t uct_iface_config_modify(uct_iface_config_t *config,
                                     const char *name, const char *value);


/**
 * @ingroup CONTEXT
 * @brief Open a communication interface.
 *
 * @param [in]  context       Handle to context.
 * @param [in]  tl_name       Transport name.
 * @param [in]  dev_name      Hardware device name,
 * @param [in]  rx_headroom   How much bytes to reserve before the receive segment.
 * @param [in]  config        Interface configuration options. Should be obtained
 *                            from uct_iface_read_config() function, or point to
 *                            transport-specific structure which extends uct_iface_config_t.
 * @param [out] iface_p       Filled with a handle to opened communication interface.
 *
 * @return Error code.
 */
ucs_status_t uct_iface_open(uct_context_h context, const char *tl_name,
                            const char *dev_name, size_t rx_headroom,
                            uct_iface_config_t *config, uct_iface_h *iface_p);


/**
 * @ingroup CONTEXT
 * @brief Set active message handler for the interface.
 *
 * @param [in]  iface    Interface to set the active message handler for.
 * @param [in]  id       Active message id. Must be 0..UCT_AM_ID_MAX-1.
 * @param [in]  cb       Active message callback. NULL to clear.
 * @param [in]  arg      Active message argument.
 */
ucs_status_t uct_set_am_handler(uct_iface_h iface, uint8_t id,
                                uct_bcopy_recv_callback_t cb, void *arg);


/**
 * @ingroup CONTEXT
 * @brief Query for protection domain attributes..
 *
 * @param [in]  pd       Protection domain to query.
 * @param [out] pd_attr  Filled with protection domain attributes.
 */
ucs_status_t uct_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr);


/**
 * @ingroup CONTEXT
 * @brief Map or allocate memory for zero-copy sends and remote access.
 *
 * @param [in]     pd         Protection domain to map memory on.
 * @param [out]    address_p  If != NULL, memory region to map.
 *                            If == NULL, filled with a pointer to allocated region.
 * @param [inout]  length_p   How many bytes to allocate. Filled with the actual
 *                           allocated size, which is larger than or equal to the
 *                           requested size.
 * @param [in]     flags      Allocation flags (currently reserved - set to 0).
 * @param [out]    lkey_p     Filled with local access key for allocated region.
 */
ucs_status_t uct_mem_map(uct_pd_h pd, void **address_p, size_t *length_p,
                         unsigned flags, uct_lkey_t *lkey_p);


/**
 * @ingroup CONTEXT
 * @brief Undo the operation of uct_mem_map().
 *
 * @param [in]  pd          Protection domain which was used to allocate/map the memory.
 * @paran [in]  lkey        Local access key to memory region.
 */
ucs_status_t uct_mem_unmap(uct_pd_h pd, uct_lkey_t lkey);


/**
 * @ingroup CONTEXT
 *
 * @brief Pack a remote key.
 *
 * @param [in]  pd           Handle to protection domain.
 * @param [in]  lkey         Local key, whose remote key should be packed.
 * @param [out] rkey_buffer  Filled with packed remote key.
 *
 * @return Error code.
 */
ucs_status_t uct_rkey_pack(uct_pd_h pd, uct_lkey_t lkey, void *rkey_buffer);


/**
 * @ingroup CONTEXT
 *
 * @brief Unpack a remote key.
 *
 * @param [in]  context      Handle to context.
 * @param [in]  rkey_buffer  Packed remote key buffer.
 * @param [out] rkey_ob      Filled with the unpacked remote key and its type.
 *
 * @return Error code.
 */
ucs_status_t uct_rkey_unpack(uct_context_h context, void *rkey_buffer,
                             uct_rkey_bundle_t *rkey_ob);


/**
 * @ingroup CONTEXT
 *
 * @brief Release a remote key.
 *
 * @param [in]  context      Handle to context.
 * @param [in]  rkey_ob      Remote key to release.
 */
void uct_rkey_release(uct_context_h context, uct_rkey_bundle_t *rkey_ob);


UCT_INLINE_API ucs_status_t uct_iface_query(uct_iface_h iface,
                                           uct_iface_attr_t *iface_attr)
{
    return iface->ops.iface_query(iface, iface_attr);
}

UCT_INLINE_API ucs_status_t uct_iface_get_address(uct_iface_h iface,
                                                 uct_iface_addr_t *iface_addr)
{
    return iface->ops.iface_get_address(iface, iface_addr);
}

UCT_INLINE_API ucs_status_t uct_iface_flush(uct_iface_h iface)
{
    return iface->ops.iface_flush(iface);
}

UCT_INLINE_API void uct_iface_close(uct_iface_h iface)
{
    iface->ops.iface_close(iface);
}

UCT_INLINE_API ucs_status_t uct_ep_create(uct_iface_h iface, uct_ep_h *ep_p)
{
    return iface->ops.ep_create(iface, ep_p);
}

UCT_INLINE_API void uct_ep_destroy(uct_ep_h ep)
{
    ep->iface->ops.ep_destroy(ep);
}

UCT_INLINE_API ucs_status_t uct_ep_get_address(uct_ep_h ep, uct_ep_addr_t *ep_addr)
{
    return ep->iface->ops.ep_get_address(ep, ep_addr);
}

UCT_INLINE_API ucs_status_t uct_ep_connect_to_iface(uct_ep_h ep, uct_iface_addr_t *iface_addr)
{
    return ep->iface->ops.ep_connect_to_iface(ep, iface_addr);
}

UCT_INLINE_API ucs_status_t uct_ep_connect_to_ep(uct_ep_h ep, uct_iface_addr_t *iface_addr,
                                                uct_ep_addr_t *ep_addr)
{
    return ep->iface->ops.ep_connect_to_ep(ep, iface_addr, ep_addr);
}

UCT_INLINE_API ucs_status_t uct_ep_put_short(uct_ep_h ep, void *buffer, unsigned length,
                                            uint64_t remote_addr, uct_rkey_t rkey)
{
    return ep->iface->ops.ep_put_short(ep, buffer, length, remote_addr, rkey);
}

UCT_INLINE_API ucs_status_t uct_ep_put_bcopy(uct_ep_h ep, uct_pack_callback_t pack_cb,
                                            void *arg, size_t length, uint64_t remote_addr,
                                            uct_rkey_t rkey)
{
    return ep->iface->ops.ep_put_bcopy(ep, pack_cb, arg, length, remote_addr, rkey);
}

UCT_INLINE_API ucs_status_t uct_ep_put_zcopy(uct_ep_h ep, void *buffer, size_t length,
                                            uct_lkey_t lkey, uint64_t remote_addr,
                                            uct_rkey_t rkey, uct_completion_t *comp)
{
    return ep->iface->ops.ep_put_zcopy(ep, buffer, length, lkey, remote_addr,
                                       rkey, comp);
}

UCT_INLINE_API ucs_status_t uct_ep_get_bcopy(uct_ep_h ep, size_t length,
                                            uint64_t remote_addr, uct_rkey_t rkey,
                                            uct_bcopy_recv_callback_t cb, void *arg)
{
    return ep->iface->ops.ep_get_bcopy(ep, length, remote_addr, rkey, cb, arg);
}

UCT_INLINE_API ucs_status_t uct_ep_get_zcopy(uct_ep_h ep, void *buffer, size_t length,
                                            uct_lkey_t lkey, uint64_t remote_addr,
                                            uct_rkey_t rkey, uct_completion_t *comp)
{
    return ep->iface->ops.ep_get_zcopy(ep, buffer, length, lkey, remote_addr,
                                       rkey, comp);
}

UCT_INLINE_API ucs_status_t uct_ep_am_short(uct_ep_h ep, uint8_t id, uint64_t header,
                                           void *payload, unsigned length)
{
    return ep->iface->ops.ep_am_short(ep, id, header, payload, length);
}

UCT_INLINE_API ucs_status_t uct_ep_am_bcopy(uct_ep_h ep, uint8_t id,
                                           uct_pack_callback_t pack_cb,
                                           void *arg, size_t length)
{
    return ep->iface->ops.ep_am_bcopy(ep, id, pack_cb, arg, length);
}

UCT_INLINE_API ucs_status_t uct_ep_atomic_add64(uct_ep_h ep, uint64_t add,
                                                uint64_t remote_addr, uct_rkey_t rkey)
{
    return ep->iface->ops.ep_atomic_add64(ep, add, remote_addr, rkey);
}

UCT_INLINE_API ucs_status_t uct_ep_atomic_fadd64(uct_ep_h ep, uint64_t add,
                                                 uint64_t remote_addr, uct_rkey_t rkey,
                                                 uct_imm_recv_callback_t cb, void *arg)
{
    return ep->iface->ops.ep_atomic_fadd64(ep, add, remote_addr, rkey, cb, arg);
}

UCT_INLINE_API ucs_status_t uct_ep_atomic_swap64(uct_ep_h ep, uint64_t swap,
                                                 uint64_t remote_addr, uct_rkey_t rkey,
                                                 uct_imm_recv_callback_t cb, void *arg)
{
    return ep->iface->ops.ep_atomic_swap64(ep, swap, remote_addr, rkey, cb, arg);
}

UCT_INLINE_API ucs_status_t uct_ep_atomic_cswap64(uct_ep_h ep, uint64_t compare, uint64_t swap,
                                                  uint64_t remote_addr, uct_rkey_t rkey,
                                                  uct_imm_recv_callback_t cb, void *arg)
{
    return ep->iface->ops.ep_atomic_cswap64(ep, compare, swap, remote_addr, rkey, cb, arg);
}

UCT_INLINE_API ucs_status_t uct_ep_atomic_add32(uct_ep_h ep, uint32_t add,
                                                uint64_t remote_addr, uct_rkey_t rkey)
{
    return ep->iface->ops.ep_atomic_add32(ep, add, remote_addr, rkey);
}

UCT_INLINE_API ucs_status_t uct_ep_atomic_fadd32(uct_ep_h ep, uint32_t add,
                                                 uint64_t remote_addr, uct_rkey_t rkey,
                                                 uct_imm_recv_callback_t cb, void *arg)
{
    return ep->iface->ops.ep_atomic_fadd32(ep, add, remote_addr, rkey, cb, arg);
}

UCT_INLINE_API ucs_status_t uct_ep_atomic_swap32(uct_ep_h ep, uint32_t swap,
                                                 uint64_t remote_addr, uct_rkey_t rkey,
                                                 uct_imm_recv_callback_t cb, void *arg)
{
    return ep->iface->ops.ep_atomic_swap32(ep, swap, remote_addr, rkey, cb, arg);
}

UCT_INLINE_API ucs_status_t uct_ep_atomic_cswap32(uct_ep_h ep, uint32_t compare, uint32_t swap,
                                                  uint64_t remote_addr, uct_rkey_t rkey,
                                                  uct_imm_recv_callback_t cb, void *arg)
{
    return ep->iface->ops.ep_atomic_cswap32(ep, compare, swap, remote_addr, rkey, cb, arg);
}

UCT_INLINE_API ucs_status_t uct_ep_am_zcopy(uct_ep_h ep, uint8_t id, void *header,
                                           unsigned header_length, void *payload,
                                           size_t length, uct_lkey_t lkey,
                                           uct_completion_t *comp)
{
    return ep->iface->ops.ep_am_zcopy(ep, id, header, header_length, payload,
                                      length, lkey, comp);
}

UCT_INLINE_API ucs_status_t uct_ep_flush(uct_ep_h ep)
{
    return ep->iface->ops.ep_flush(ep);
}

#endif
