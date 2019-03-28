/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "tcpcm_md.h"

#define UCT_TCPCM_MD_PREFIX              "tcpcm"

static ucs_config_field_t uct_tcpcm_md_config_table[] = {
  {"", "", NULL,
   ucs_offsetof(uct_tcpcm_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

  /* FIXME: do we need a timeout for tcp?
  {"ADDR_RESOLVE_TIMEOUT", "500ms",
   "Time to wait for address resolution to complete",
   ucs_offsetof(uct_tcpcm_md_config_t, addr_resolve_timeout), UCS_CONFIG_TYPE_TIME},*/

  {NULL}
};

static void uct_tcpcm_md_close(uct_md_h md);

static uct_md_ops_t uct_tcpcm_md_ops = {
    .close                  = uct_tcpcm_md_close,
    .query                  = uct_tcpcm_md_query,
    .is_sockaddr_accessible = uct_tcpcm_is_sockaddr_accessible,
    .is_mem_type_owned      = (void *)ucs_empty_function_return_zero,
};

static void uct_tcpcm_md_close(uct_md_h md)
{
    uct_tcpcm_md_t *tcpcm_md = ucs_derived_of(md, uct_tcpcm_md_t);
    ucs_free(tcpcm_md);
}

ucs_status_t uct_tcpcm_md_query(uct_md_h md, uct_md_attr_t *md_attr)
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

static int uct_tcpcm_is_addr_route_resolved(int sock_id, struct sockaddr *addr,
                                            int addrlen)
{
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    char host[UCS_SOCKADDR_STRING_LEN];
    char serv[UCS_SOCKADDR_STRING_LEN];

    if (getnameinfo(addr, addrlen, host,
                    UCS_SOCKADDR_STRING_LEN, serv,
                    UCS_SOCKADDR_STRING_LEN, NI_NAMEREQD)) {
        return 0;
    }

    if (connect(sock_id, addr, addrlen)) {

        if (errno == ECONNREFUSED) {
            return 1;
        }

        ucs_debug("connect(addr = %s) failed: %m",
                   ucs_sockaddr_str(addr, ip_port_str, UCS_SOCKADDR_STRING_LEN));
        return 0;
    }

    return 1;
}

static int uct_tcpcm_is_sockaddr_inaddr_any(struct sockaddr *addr)
{
    struct sockaddr_in6 *addr_in6;
    struct sockaddr_in *addr_in;

    switch (addr->sa_family) {
    case AF_INET:
        addr_in = (struct sockaddr_in *)addr;
        return addr_in->sin_addr.s_addr == INADDR_ANY;
    case AF_INET6:
        addr_in6 = (struct sockaddr_in6 *)addr;
        return !memcmp(&addr_in6->sin6_addr, &in6addr_any, sizeof(addr_in6->sin6_addr));
    default:
        ucs_debug("Invalid address family: %d", addr->sa_family);
    }

    return 0;
}

int uct_tcpcm_is_sockaddr_accessible(uct_md_h md, const ucs_sock_addr_t *sockaddr,
                                      uct_sockaddr_accessibility_t mode)
{
    uct_tcpcm_md_t *tcpcm_md = ucs_derived_of(md, uct_tcpcm_md_t);
    int is_accessible = 0;
    int sock_id = -1;
    unsigned int port = 0;
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];

    printf("asdasd\n");

    if ((mode != UCT_SOCKADDR_ACC_LOCAL) && (mode != UCT_SOCKADDR_ACC_REMOTE)) {
        ucs_error("Unknown sockaddr accessibility mode %d", mode);
        return 0;
    }

    sock_id = socket(((struct sockaddr *)sockaddr->addr)->sa_family, SOCK_STREAM, 0);
    if (-1 == sock_id) {
        ucs_error("unable to open socket");
        return 0;
    }

    if (mode == UCT_SOCKADDR_ACC_LOCAL) {

        if (bind(sock_id, (struct sockaddr *)sockaddr->addr, sockaddr->addrlen)) {
            ucs_debug("bind(addr = %s) failed: %m",
                      ucs_sockaddr_str((struct sockaddr *)sockaddr->addr,
                                       ip_port_str, UCS_SOCKADDR_STRING_LEN));
            goto out_destroy_id;
        }

        if (uct_tcpcm_is_sockaddr_inaddr_any((struct sockaddr *)sockaddr->addr)) {
            is_accessible = 1;
            goto out_print;
        }
    }

    is_accessible = uct_tcpcm_is_addr_route_resolved(sock_id,
                                                     (struct sockaddr *)sockaddr->addr,
                                                     sockaddr->addrlen);
    if (!is_accessible) {
        goto out_destroy_id;
    }

 out_print:
    ucs_sockaddr_get_port((struct sockaddr *)sockaddr->addr, &port);
    ucs_debug("address %s (port %d) is accessible from tcpcm_md %p with mode: %d",
              ucs_sockaddr_str((struct sockaddr *)sockaddr->addr, ip_port_str,
                               UCS_SOCKADDR_STRING_LEN), port, tcpcm_md, mode);
    printf("address %s (port %d) is accessible from tcpcm_md %p with mode: %d",
              ucs_sockaddr_str((struct sockaddr *)sockaddr->addr, ip_port_str,
                               UCS_SOCKADDR_STRING_LEN), port, tcpcm_md, mode);

 out_destroy_id:
    close(sock_id);

    return is_accessible;
}

static ucs_status_t uct_tcpcm_query_md_resources(uct_md_resource_desc_t **resources_p,
                                                  unsigned *num_resources_p)
{
    return uct_single_md_resource(&uct_tcpcm_mdc, resources_p, num_resources_p);
}

static ucs_status_t
uct_tcpcm_md_open(const char *md_name, const uct_md_config_t *uct_md_config,
                   uct_md_h *md_p)
{
    uct_tcpcm_md_config_t *md_config = ucs_derived_of(uct_md_config, uct_tcpcm_md_config_t);
    uct_tcpcm_md_t *md;
    ucs_status_t status;

    md = ucs_malloc(sizeof(*md), "tcpcm_md");
    if (md == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    md->super.ops            = &uct_tcpcm_md_ops;
    md->super.component      = &uct_tcpcm_mdc;
    md->addr_resolve_timeout = md_config->addr_resolve_timeout;

    *md_p = &md->super;
    status = UCS_OK;

out:
    return status;
}

UCT_MD_COMPONENT_DEFINE(uct_tcpcm_mdc, UCT_TCPCM_MD_PREFIX,
                        uct_tcpcm_query_md_resources, uct_tcpcm_md_open, NULL,
                        ucs_empty_function_return_unsupported,
                        (void*)ucs_empty_function_return_success,
                        "TCPCM_", uct_tcpcm_md_config_table, uct_tcpcm_md_config_t);
