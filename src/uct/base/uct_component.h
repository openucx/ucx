/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCT_COMPONENT_H_
#define UCT_COMPONENT_H_

#include <uct/api/uct.h>
#include <ucs/config/parser.h>
#include <ucs/datastruct/list.h>


typedef struct uct_md_component uct_component_t;
typedef struct uct_md_component uct_md_component_t;
struct uct_md_component {
    ucs_status_t           (*query_md_resources)(uct_component_t *component,
                                                 uct_md_resource_desc_t **resources_p,
                                                 unsigned *num_resources_p);

    ucs_status_t           (*md_open)(uct_component_t *component,
                                      const char *md_name, const uct_md_config_t *config,
                                      uct_md_h *md_p);

    ucs_status_t           (*cm_open)(uct_component_t *component, uct_worker_h worker,
                                      uct_cm_h *cm_p);

    ucs_status_t           (*rkey_unpack)(uct_md_component_t *mdc, const void *rkey_buffer,
                                          uct_rkey_t *rkey_p, void **handle_p);

    ucs_status_t           (*rkey_ptr)(uct_md_component_t *mdc, uct_rkey_t rkey, void *handle,
                                       uint64_t raddr, void **laddr_p);

    ucs_status_t           (*rkey_release)(uct_md_component_t *mdc, uct_rkey_t rkey,
                                           void *handle);

    const char             name[UCT_MD_COMPONENT_NAME_MAX];
    void                   *priv;
    ucs_config_global_list_entry_t md_config;  /**< MD configuration entry */
    ucs_list_link_t        tl_list;            /**< List of uct_md_registered_tl_t */
    ucs_list_link_t        list;
    uint64_t               flags;              /**< Flags as defined
                                                    by UCT_COMPONENT_FLAG_xx */
};


/**
 * Register a component for usage, so it will be returned from
 * @ref uct_query_components.
 *
 * @param [in] _component  Pointer to a global component structure to register.
 */
#define UCT_COMPONENT_REGISTER(_component) \
    UCS_STATIC_INIT { \
        extern ucs_list_link_t uct_md_components_list; \
        ucs_list_add_tail(&uct_md_components_list, &(_component)->list); \
    } \
    UCS_CONFIG_REGISTER_TABLE_ENTRY(&(_component)->md_config);


/**
 * Helper macro to initialize component's transport list head.
 */
#define UCT_COMPONENT_TL_LIST_INITIALIZER(_component) \
    UCS_LIST_INITIALIZER(&(_component)->tl_list, &(_component)->tl_list)


#endif
