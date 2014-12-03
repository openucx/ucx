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
                               size_t rx_headroom, uct_iface_config_t *config,
                               uct_iface_h *iface_p);

    ucs_status_t (*rkey_unpack)(uct_context_h context, void *rkey_buffer,
                                uct_rkey_bundle_t *rkey_ob);

};


/**
 * "Base" structure which defines interface configuration options.
 * Specific transport extend this structure.
 */
struct uct_iface_config {
    size_t            max_short;
    size_t            max_bcopy;
};


/**
 * Memory pool configuration.
 */
typedef struct uct_iface_mpool_config {
    unsigned          max_bufs;  /* Upper limit to number of buffers */
    unsigned          bufs_grow; /* How many buffers (approx.) are allocated every time */
} uct_iface_mpool_config_t;


/**
 * Define configuration fields for memory pool parameters.
 */
#define UCT_IFACE_MPOOL_CONFIG_FIELDS(_prefix, _dfl_max, _mp_name, _offset, _desc) \
    {_prefix "MAX_BUFS", UCS_PP_QUOTE(_dfl_max), \
     "Maximal number of " _mp_name " buffers for the interface. -1 is infinite." \
     _desc, \
     (_offset) + ucs_offsetof(uct_iface_mpool_config_t, max_bufs), UCS_CONFIG_TYPE_INT}, \
    \
    {_prefix "BUFS_GROW", "0", \
     "How much buffers are added every time the " _mp_name " memory pool grows.\n" \
     "0 means the value is chosen by the transport.", \
     (_offset) + ucs_offsetof(uct_iface_mpool_config_t, bufs_grow), UCS_CONFIG_TYPE_UINT}



typedef void (*uct_iface_mpool_init_obj_cb_t)(uct_iface_h iface, void *obj, uct_lkey_t lkey);

/**
 * Create a memory pool for buffers used by TL interface.
 *
 * @param elem_size
 * @param align_offset
 * @param alignment    Data will be aligned to these units.
 * @param config       Memory pool configuration.
 * @param grow         Default number of buffers added for every chunk.
 * @param init_obj_cb  Object constructor.
 * @param name         Memory pool name.
 * @param mp_p         Filled with memory pool handle.
 */
ucs_status_t uct_iface_mpool_create(uct_iface_h iface, size_t elem_size,
                                    size_t align_offset, size_t alignment,
                                    uct_iface_mpool_config_t *config, unsigned grow,
                                    uct_iface_mpool_init_obj_cb_t init_obj_cb,
                                    const char *name, ucs_mpool_h *mp_p);


static inline ucs_status_t uct_iface_invoke_am(uct_iface_h iface, uint8_t id,
                                               void *data, unsigned length)
{
    uct_am_handler_t *handler = &iface->am[id];
    return handler->cb(data, length, handler->arg);
}

#endif
