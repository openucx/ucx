/**
 * Copyright (C) Mellanox Technologies Ltd. 2017-219.  ALL RIGHTS RESERVED.
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rdmacm_md.h"
#include "rdmacm_cm.h"


static ucs_config_field_t uct_rdmacm_md_config_table[] = {
  {"", "", NULL,
   ucs_offsetof(uct_rdmacm_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

  {"ADDR_RESOLVE_TIMEOUT", "500ms",
   "Time to wait for address resolution to complete",
    ucs_offsetof(uct_rdmacm_md_config_t, addr_resolve_timeout), UCS_CONFIG_TYPE_TIME},

  {NULL}
};

static void uct_rdmacm_md_close(uct_md_h md);

static uct_md_ops_t uct_rdmacm_md_ops = {
    .close                   = uct_rdmacm_md_close,
    .query                   = uct_rdmacm_md_query,
    .is_sockaddr_accessible  = uct_rdmacm_is_sockaddr_accessible,
    .detect_memory_type      = ucs_empty_function_return_unsupported,
};

static void uct_rdmacm_md_close(uct_md_h md)
{
    uct_rdmacm_md_t *rdmacm_md = ucs_derived_of(md, uct_rdmacm_md_t);
    ucs_free(rdmacm_md);
}

ucs_status_t uct_rdmacm_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    md_attr->cap.flags            = UCT_MD_FLAG_SOCKADDR;
    md_attr->cap.reg_mem_types    = 0;
    md_attr->cap.alloc_mem_types  = 0;
    md_attr->cap.access_mem_types = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    md_attr->cap.detect_mem_types = 0;
    md_attr->cap.max_alloc        = 0;
    md_attr->cap.max_reg          = 0;
    md_attr->rkey_packed_size     = 0;
    md_attr->reg_cost             = ucs_linear_func_make(0, 0);
    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

static enum rdma_cm_event_type
uct_rdmacm_get_event_type(struct rdma_event_channel *event_ch)
{
    enum rdma_cm_event_type event_type;
    struct rdma_cm_event *event;
    int ret;

    /* Fetch an event */
    ret = rdma_get_cm_event(event_ch, &event);
    if (ret) {
        ucs_warn("rdma_get_cm_event() failed: %m");
        return RDMA_CM_EVENT_ADDR_RESOLVED;
    }

    event_type = event->event;
    ret = rdma_ack_cm_event(event);
    if (ret) {
        ucs_warn("rdma_ack_cm_event() failed. event status: %d. %m.", event->status);
    }

    return event_type;
}

static int uct_rdmacm_is_addr_route_resolved(struct rdma_cm_id *cm_id,
                                             struct sockaddr *addr,
                                             int timeout_ms)
{
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    enum rdma_cm_event_type event_type;
    ucs_status_t status;

    status = uct_rdmacm_resolve_addr(cm_id, addr, timeout_ms, UCS_LOG_LEVEL_DEBUG);
    if (status != UCS_OK) {
        return 0;
    }

    event_type = uct_rdmacm_get_event_type(cm_id->channel);
    if (event_type != RDMA_CM_EVENT_ADDR_RESOLVED) {
        ucs_debug("failed to resolve address (addr = %s). RDMACM event %s.",
                  ucs_sockaddr_str(addr, ip_port_str, UCS_SOCKADDR_STRING_LEN),
                  rdma_event_str(event_type));
        return 0;
    }

    if (cm_id->verbs->device->transport_type == IBV_TRANSPORT_IWARP) {
        ucs_debug("%s: iWarp support is not implemented",
                  ucs_sockaddr_str(addr, ip_port_str, UCS_SOCKADDR_STRING_LEN));
        return 0;
    }

    if (rdma_resolve_route(cm_id, timeout_ms)) {
        ucs_debug("rdma_resolve_route(addr = %s) failed: %m",
                   ucs_sockaddr_str(addr, ip_port_str, UCS_SOCKADDR_STRING_LEN));
        return 0;
    }

    event_type = uct_rdmacm_get_event_type(cm_id->channel);
    if (event_type != RDMA_CM_EVENT_ROUTE_RESOLVED) {
        ucs_debug("failed to resolve route to addr = %s. RDMACM event %s.",
                  ucs_sockaddr_str(addr, ip_port_str, UCS_SOCKADDR_STRING_LEN),
                  rdma_event_str(event_type));
        return 0;
    }

    return 1;
}

int uct_rdmacm_is_sockaddr_accessible(uct_md_h md, const ucs_sock_addr_t *sockaddr,
                                      uct_sockaddr_accessibility_t mode)
{
    uct_rdmacm_md_t *rdmacm_md = ucs_derived_of(md, uct_rdmacm_md_t);
    struct rdma_event_channel *event_ch = NULL;
    struct rdma_cm_id *cm_id = NULL;
    int is_accessible = 0;
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];

    if ((mode != UCT_SOCKADDR_ACC_LOCAL) && (mode != UCT_SOCKADDR_ACC_REMOTE)) {
        ucs_error("Unknown sockaddr accessibility mode %d", mode);
        return 0;
    }

    event_ch = rdma_create_event_channel();
    if (event_ch == NULL) {
        ucs_error("rdma_create_event_channel() failed: %m");
        goto out;
    }

    if (rdma_create_id(event_ch, &cm_id, NULL, RDMA_PS_UDP)) {
        ucs_error("rdma_create_id() failed: %m");
        goto out_destroy_event_channel;
    }

    if (mode == UCT_SOCKADDR_ACC_LOCAL) {
        /* Server side to check if can bind to the given sockaddr */
        if (rdma_bind_addr(cm_id, (struct sockaddr *)sockaddr->addr)) {
            ucs_debug("rdma_bind_addr(addr = %s) failed: %m",
                      ucs_sockaddr_str((struct sockaddr *)sockaddr->addr,
                                       ip_port_str, UCS_SOCKADDR_STRING_LEN));
            goto out_destroy_id;
        }

        if (ucs_sockaddr_is_inaddr_any((struct sockaddr *)sockaddr->addr)) {
            is_accessible = 1;
            goto out_print;
        }
    }

    /* Client and server sides check if can access the given sockaddr.
     * The timeout needs to be passed in ms */
    is_accessible = uct_rdmacm_is_addr_route_resolved(cm_id,
                                                     (struct sockaddr *)sockaddr->addr,
                                                     UCS_MSEC_PER_SEC * rdmacm_md->addr_resolve_timeout);
    if (!is_accessible) {
        goto out_destroy_id;
    }

out_print:
    ucs_debug("address %s (port %d) is accessible from rdmacm_md %p with mode: %d",
              ucs_sockaddr_str((struct sockaddr *)sockaddr->addr, ip_port_str,
                               UCS_SOCKADDR_STRING_LEN),
              ntohs(rdma_get_src_port(cm_id)), rdmacm_md, mode);

out_destroy_id:
    rdma_destroy_id(cm_id);
out_destroy_event_channel:
    rdma_destroy_event_channel(event_ch);
out:
    return is_accessible;
}

static ucs_status_t
uct_rdmacm_query_md_resources(uct_component_t *component,
                              uct_md_resource_desc_t **resources_p,
                              unsigned *num_resources_p)
{
    struct rdma_event_channel *event_ch = NULL;

    /* Create a dummy event channel to check if RDMACM can be used */
    event_ch = rdma_create_event_channel();
    if (event_ch == NULL) {
        ucs_debug("could not create an RDMACM event channel. %m. "
                  "Disabling the RDMACM resource");
        return uct_md_query_empty_md_resource(resources_p, num_resources_p);

    }

    rdma_destroy_event_channel(event_ch);

    return uct_md_query_single_md_resource(component, resources_p,
                                           num_resources_p);
}

static ucs_status_t
uct_rdmacm_md_open(uct_component_t *component, const char *md_name,
                   const uct_md_config_t *uct_md_config, uct_md_h *md_p)
{
    uct_rdmacm_md_config_t *md_config = ucs_derived_of(uct_md_config,
                                                       uct_rdmacm_md_config_t);
    uct_rdmacm_md_t *md;
    ucs_status_t status;

    md = ucs_malloc(sizeof(*md), "rdmacm_md");
    if (md == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    md->super.ops            = &uct_rdmacm_md_ops;
    md->super.component      = &uct_rdmacm_component;
    md->addr_resolve_timeout = md_config->addr_resolve_timeout;

    /* cppcheck-suppress autoVariables */
    *md_p = &md->super;
    status = UCS_OK;

out:
    return status;
}

uct_component_t uct_rdmacm_component = {
    .query_md_resources = uct_rdmacm_query_md_resources,
    .md_open            = uct_rdmacm_md_open,
#if HAVE_RDMACM_QP_LESS
    .cm_open            = UCS_CLASS_NEW_FUNC_NAME(uct_rdmacm_cm_t),
#else
    .cm_open            = ucs_empty_function_return_unsupported,
#endif
    .rkey_unpack        = ucs_empty_function_return_unsupported,
    .rkey_ptr           = ucs_empty_function_return_unsupported,
    .rkey_release       = ucs_empty_function_return_success,
    .name               = "rdmacm",
    .md_config          = {
        .name           = "RDMA-CM memory domain",
        .prefix         = "RDMACM_",
        .table          = uct_rdmacm_md_config_table,
        .size           = sizeof(uct_rdmacm_md_config_t),
    },
    .cm_config          = {
        .name           = "RDMA-CM connection manager",
        .prefix         = "RDMA_CM_",
        .table          = uct_cm_config_table,
        .size           = sizeof(uct_cm_config_t),
    },
    .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_rdmacm_component),
#if HAVE_RDMACM_QP_LESS
    .flags              = UCT_COMPONENT_FLAG_CM
#else
    .flags              = 0
#endif
};
UCT_COMPONENT_REGISTER(&uct_rdmacm_component)
