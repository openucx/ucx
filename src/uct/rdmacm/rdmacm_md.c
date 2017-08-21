/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "rdmacm_md.h"

#define UCT_RDMACM_MD_PREFIX              "rdmacm"
#define UCT_RDMACM_ADDR_RESOLVE_TIMEOUT    500    /* ms */

static ucs_config_field_t uct_rdmacm_md_config_table[] = {
  {"", "", NULL,
   ucs_offsetof(uct_rdmacm_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

  {NULL}
};

static void uct_rdmacm_md_close(uct_md_h md)
{
    uct_rdmacm_md_t *rdmacm_md = ucs_derived_of(md, uct_rdmacm_md_t);
    ucs_free(rdmacm_md);
}

ucs_status_t uct_rdmacm_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    md_attr->cap.flags = UCT_MD_FLAG_SOCKADDR;
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
    int rc, is_accessible;
    struct rdma_cm_id *cm_id = NULL;
    struct rdma_event_channel *event_ch = NULL;

    if ((mode != UCT_SOCKADDR_ACC_LOCAL) && (mode != UCT_SOCKADDR_ACC_REMOTE)) {
        ucs_fatal("Unknown sockaddr accessibility mode %d", mode);
        return 0;
    }

    event_ch = rdma_create_event_channel();
    if (event_ch == NULL) {
        ucs_error("Failed to create an rdmacm event channel: %m");
        is_accessible = 0;
        goto out;
    }
    rc = rdma_create_id(event_ch, &cm_id, NULL, RDMA_PS_UDP);
    if (rc) {
        ucs_error("Failed to create an rdmacm ID: %m");
        is_accessible = 0;
        goto err_destroy_event_channel;
    }

    if (mode == UCT_SOCKADDR_ACC_LOCAL) {
        /* Server side to check if can bind to the given sockaddr */
        rc = rdma_bind_addr(cm_id, (struct sockaddr *)(sockaddr->addr));
        if (rc) {
            ucs_error("Failed to bind to sockaddr: %m");
            is_accessible = 0;
            goto err_destroy_id;
        }

        is_accessible = 1;

    } else if (mode == UCT_SOCKADDR_ACC_REMOTE) {
        /* Client side to check if can access the remote given sockaddr */
        rc = rdma_resolve_addr(cm_id, NULL, (struct sockaddr *)(sockaddr->addr),
                               UCT_RDMACM_ADDR_RESOLVE_TIMEOUT);
        if (rc) {
            ucs_error("Failed to resolve address: %m");
            is_accessible = 0;
            goto err_destroy_id;
        }

        is_accessible = 1;
    }

err_destroy_id:
    rdma_destroy_id(cm_id);
err_destroy_event_channel:
    rdma_destroy_event_channel(event_ch);
out:
    return is_accessible;
}

static ucs_status_t uct_rdmacm_query_md_resources(uct_md_resource_desc_t **resources_p,
                                                  unsigned *num_resources_p)
{
    return uct_single_md_resource(&uct_rdmacm_mdc, resources_p, num_resources_p);
}

static ucs_status_t
uct_rdmacm_md_open(const char *md_name, const uct_md_config_t *uct_md_config,
                   uct_md_h *md_p)
{
    uct_rdmacm_md_t *md;
    ucs_status_t status;

    md = ucs_malloc(sizeof(*md), "rdmacm_md");
    if (md == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    md->super.ops       = &uct_rdmacm_md_ops;
    md->super.component = &uct_rdmacm_mdc;

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
