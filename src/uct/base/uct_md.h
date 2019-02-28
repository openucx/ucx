/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_MD_H_
#define UCT_MD_H_

#include "uct_component.h"
#include "uct_iface.h"

#include <uct/api/uct.h>
#include <ucs/config/parser.h>


typedef struct uct_md_rcache_config {
    size_t               alignment;    /**< Force address alignment */
    unsigned             event_prio;   /**< Memory events priority */
    double               overhead;     /**< Lookup overhead estimation */
} uct_md_rcache_config_t;

extern ucs_config_field_t uct_md_config_rcache_table[];

/**
 * "Base" structure which defines MD configuration options.
 * Specific MDs extend this structure.
 */
struct uct_md_config {
    /* C standard prohibits empty structures */
    char                   __dummy;
};


/**
 * MD->Transport
 */
typedef struct uct_md_registered_tl {
    ucs_list_link_t        list;
    uct_tl_component_t     *tl;
} uct_md_registered_tl_t;


/**
 * Add a transport component to a md component
 * (same transport component can be added to multiple md components).
 *
 * @param _mdc           Pointer to MD component to add the TL component to.
 * @param _tlc           Pointer to TL component.
 */
#define UCT_MD_REGISTER_TL(_mdc, _tlc) \
    UCS_STATIC_INIT { \
        static uct_md_registered_tl_t reg; \
        reg.tl = (_tlc); \
        ucs_list_add_tail(&(_mdc)->tl_list, &reg.list); \
    }


/**
 * Memory domain operations
 */
struct uct_md_ops {
    void         (*close)(uct_md_h md);

    ucs_status_t (*query)(uct_md_h md, uct_md_attr_t *md_attr);

    ucs_status_t (*mem_alloc)(uct_md_h md, size_t *length_p, void **address_p,
                              unsigned flags, const char *alloc_name,
                              uct_mem_h *memh_p);

    ucs_status_t (*mem_free)(uct_md_h md, uct_mem_h memh);
    ucs_status_t (*mem_advise)(uct_md_h md, uct_mem_h memh, void *addr,
                               size_t length, unsigned advice);

    ucs_status_t (*mem_reg)(uct_md_h md, void *address, size_t length,
                            unsigned flags, uct_mem_h *memh_p);

    ucs_status_t (*mem_dereg)(uct_md_h md, uct_mem_h memh);

    ucs_status_t (*mkey_pack)(uct_md_h md, uct_mem_h memh, void *rkey_buffer);

    int          (*is_sockaddr_accessible)(uct_md_h md, const ucs_sock_addr_t *sockaddr,
                                           uct_sockaddr_accessibility_t mode);

    int          (*is_mem_type_owned)(uct_md_h md, void *addr, size_t length);

    int          (*is_hugetlb)(uct_md_h md, uct_mem_h memh);
};


/**
 * Memory domain
 */
struct uct_md {
    uct_md_ops_t           *ops;
    uct_md_component_t     *component;
};


static UCS_F_ALWAYS_INLINE void*
uct_md_fill_md_name(uct_md_h md, void *buffer)
{
    memcpy(buffer, md->component->name, UCT_MD_COMPONENT_NAME_MAX);
    return (char*)buffer + UCT_MD_COMPONENT_NAME_MAX;
}


ucs_status_t uct_single_md_resource(uct_md_component_t *mdc,
                                    uct_md_resource_desc_t **resources_p,
                                    unsigned *num_resources_p);

/**
 * @brief Dummy function
 * Dummy function to emulate unpacking a remote key buffer to handle.
 *
 */
ucs_status_t uct_md_stub_rkey_unpack(uct_md_component_t *mdc,
                                     const void *rkey_buffer, uct_rkey_t *rkey_p,
                                     void **handle_p);

uct_tl_component_t *uct_find_tl_on_md(uct_md_component_t *mdc,
                                      uint64_t md_flags,
                                      const char *tl_name);


extern ucs_config_field_t uct_md_config_table[];

#endif
