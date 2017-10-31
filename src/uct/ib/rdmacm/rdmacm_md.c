/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "rdmacm_md.h"

#define UCT_RDMACM_MD_PREFIX              "rdmacm"

static ucs_config_field_t uct_rdmacm_md_config_table[] = {
  {"", "", NULL,
   ucs_offsetof(uct_rdmacm_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

  {"ADDR_RESOLVE_TIMEOUT", "500ms",
   "Time to wait for address resolution to complete",
    ucs_offsetof(uct_rdmacm_md_config_t, addr_resolve_timeout), UCS_CONFIG_TYPE_TIME},

  {NULL}
};

static uct_md_ops_t uct_rdmacm_md_ops = {
    .close                  = uct_rdmacm_md_close,
    .query                  = uct_rdmacm_md_query,
    .is_sockaddr_accessible = uct_rdmacm_is_sockaddr_accessible,
    .is_mem_type_owned      = (void *)ucs_empty_function_return_zero,
};

static void uct_rdmacm_md_close(uct_md_h md)
{
    uct_rdmacm_md_t *rdmacm_md = ucs_derived_of(md, uct_rdmacm_md_t);
    ucs_free(rdmacm_md);
}

ucs_status_t uct_rdmacm_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    md_attr->cap.flags         = UCT_MD_FLAG_SOCKADDR;
    md_attr->cap.reg_mem_types = 0;
    md_attr->cap.mem_type      = UCT_MD_MEM_TYPE_HOST;
    md_attr->cap.max_alloc     = 0;
    md_attr->cap.max_reg       = 0;
    md_attr->rkey_packed_size  = 0;
    md_attr->reg_cost.overhead = 0;
    md_attr->reg_cost.growth   = 0;
    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

int uct_rdmacm_is_sockaddr_accessible(uct_md_h md, const ucs_sock_addr_t *sockaddr,
                                      uct_sockaddr_accessibility_t mode)
{
    uct_rdmacm_md_t *rdmacm_md = ucs_derived_of(md, uct_rdmacm_md_t);
    struct rdma_event_channel *event_ch = NULL;
    struct rdma_cm_id *cm_id = NULL;
    int is_accessible;
    size_t ip_len = ucs_max(INET_ADDRSTRLEN, INET6_ADDRSTRLEN);
    char *ip_str = ucs_alloca(ip_len);

    if ((mode != UCT_SOCKADDR_ACC_LOCAL) && (mode != UCT_SOCKADDR_ACC_REMOTE)) {
        ucs_error("Unknown sockaddr accessibility mode %d", mode);
        return 0;
    }

    event_ch = rdma_create_event_channel();
    if (event_ch == NULL) {
        ucs_error("rdma_create_event_channel() failed: %m");
        is_accessible = 0;
        goto out;
    }

    if (rdma_create_id(event_ch, &cm_id, NULL, RDMA_PS_UDP)) {
        ucs_error("rdma_create_id() failed: %m");
        is_accessible = 0;
        goto out_destroy_event_channel;
    }

    if (mode == UCT_SOCKADDR_ACC_LOCAL) {
        /* Server side to check if can bind to the given sockaddr */
        if (rdma_bind_addr(cm_id, (struct sockaddr *)sockaddr->addr)) {
            ucs_debug("rdma_bind_addr(addr = %s) failed: %m",
                      ucs_sockaddr_str((struct sockaddr *)sockaddr->addr,
                                       ip_str, ip_len));
            is_accessible = 0;
            goto out_destroy_id;
        }

        is_accessible = 1;
    } else {
        /* Client side to check if can access the remote given sockaddr */
        if (rdma_resolve_addr(cm_id, NULL, (struct sockaddr *)sockaddr->addr,
                              rdmacm_md->addr_resolve_timeout)) {
            ucs_debug("rdma_resolve_addr(addr = %s) failed: %m",
                      ucs_sockaddr_str((struct sockaddr *)sockaddr->addr,
                                       ip_str, ip_len));
            is_accessible = 0;
            goto out_destroy_id;
        }

        is_accessible = 1;
    }

    ucs_debug("address %s is accessible from rdmacm_md %p with mode: %d",
              ucs_sockaddr_str((struct sockaddr *)sockaddr->addr, ip_str, ip_len),
              rdmacm_md, mode);

out_destroy_id:
    rdma_destroy_id(cm_id);
out_destroy_event_channel:
    rdma_destroy_event_channel(event_ch);
out:
    return is_accessible;
}

static ucs_status_t uct_rdmacm_query_md_resources(uct_md_resource_desc_t **resources_p,
                                                  unsigned *num_resources_p)
{
    struct rdma_event_channel *event_ch = NULL;

    /* Create a dummy event channel to check if RDMACM can be used */
    event_ch = rdma_create_event_channel();
    if (event_ch == NULL) {
        ucs_debug("Could not create an RDMACM event channel. %m. "
                  "Disabling the RDMACM resource");
        *resources_p     = NULL;
        *num_resources_p = 0;
        return UCS_OK;
    }

    rdma_destroy_event_channel(event_ch);

    return uct_single_md_resource(&uct_rdmacm_mdc, resources_p, num_resources_p);
}

static ucs_status_t
uct_rdmacm_md_open(const char *md_name, const uct_md_config_t *uct_md_config,
                   uct_md_h *md_p)
{
    uct_rdmacm_md_config_t *md_config = ucs_derived_of(uct_md_config, uct_rdmacm_md_config_t);
    uct_rdmacm_md_t *md;
    ucs_status_t status;

    md = ucs_malloc(sizeof(*md), "rdmacm_md");
    if (md == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    md->super.ops            = &uct_rdmacm_md_ops;
    md->super.component      = &uct_rdmacm_mdc;
    md->addr_resolve_timeout = md_config->addr_resolve_timeout;

    *md_p = &md->super;
    status = UCS_OK;

out:
    return status;
}

UCT_MD_COMPONENT_DEFINE(uct_rdmacm_mdc, UCT_RDMACM_MD_PREFIX,
                        uct_rdmacm_query_md_resources, uct_rdmacm_md_open, NULL,
                        ucs_empty_function_return_unsupported,
                        (void*)ucs_empty_function_return_success,
                        "RDMACM_", uct_rdmacm_md_config_table, uct_rdmacm_md_config_t);
