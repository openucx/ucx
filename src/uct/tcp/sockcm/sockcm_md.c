/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * Copyright (C) NVIDIA Corporation. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "sockcm_md.h"

#define UCT_SOCKCM_MD_PREFIX              "sockcm"

static ucs_config_field_t uct_sockcm_md_config_table[] = {
  {"", "", NULL,
   ucs_offsetof(uct_sockcm_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},
  {NULL}
};

static void uct_sockcm_md_close(uct_md_h md);

static uct_md_ops_t uct_sockcm_md_ops = {
    .close                  = uct_sockcm_md_close,
    .query                  = uct_sockcm_md_query,
    .is_sockaddr_accessible = uct_sockcm_is_sockaddr_accessible,
    .is_mem_type_owned      = (void *)ucs_empty_function_return_zero,
};

static void uct_sockcm_md_close(uct_md_h md)
{
    uct_sockcm_md_t *sockcm_md = ucs_derived_of(md, uct_sockcm_md_t);
    ucs_free(sockcm_md);
}

ucs_status_t uct_sockcm_md_query(uct_md_h md, uct_md_attr_t *md_attr)
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

static int uct_sockcm_is_addr_route_resolved(int sock_id, struct sockaddr *addr,
                                            int addrlen)
{
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    char host[UCS_SOCKADDR_STRING_LEN];
    char serv[UCS_SOCKADDR_STRING_LEN];
    int ret = -1;

    ret = getnameinfo(addr, addrlen, host, UCS_SOCKADDR_STRING_LEN, serv,
                      UCS_SOCKADDR_STRING_LEN, NI_NAMEREQD);
    if (0 != ret) {
        ucs_debug("getnameinfo error : %s\n", gai_strerror(ret));
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

static int uct_sockcm_is_sockaddr_inaddr_any(struct sockaddr *addr)
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

int uct_sockcm_is_sockaddr_accessible(uct_md_h md, const ucs_sock_addr_t *sockaddr,
                                      uct_sockaddr_accessibility_t mode)
{
    uct_sockcm_md_t *sockcm_md = ucs_derived_of(md, uct_sockcm_md_t);
    int is_accessible = 0;
    int sock_id = -1;
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    struct sockaddr *param_sockaddr = NULL;

    param_sockaddr = (struct sockaddr *) sockaddr->addr;

    if ((mode != UCT_SOCKADDR_ACC_LOCAL) && (mode != UCT_SOCKADDR_ACC_REMOTE)) {
        ucs_error("Unknown sockaddr accessibility mode %d", mode);
        return 0;
    }

    sock_id = socket(param_sockaddr->sa_family, SOCK_STREAM, 0);
    if (-1 == sock_id) {
        return 0;
    }

    if (mode == UCT_SOCKADDR_ACC_LOCAL) {

        if (bind(sock_id, param_sockaddr, sockaddr->addrlen)) {
            ucs_debug("bind(addr = %s) failed: %m",
                      ucs_sockaddr_str((struct sockaddr *)sockaddr->addr,
                                       ip_port_str, UCS_SOCKADDR_STRING_LEN));
            goto out_destroy_id;
        }

        if (uct_sockcm_is_sockaddr_inaddr_any(param_sockaddr)) {
            is_accessible = 1;
            goto out_print;
        }
    }

    is_accessible = uct_sockcm_is_addr_route_resolved(sock_id, param_sockaddr,
                                                     sockaddr->addrlen);
    if (!is_accessible) {
        goto out_destroy_id;
    }

 out_print:
    ucs_debug("address %s is accessible from sockcm_md %p with mode: %d",
              ucs_sockaddr_str(param_sockaddr, ip_port_str,
                               UCS_SOCKADDR_STRING_LEN), sockcm_md, mode);

 out_destroy_id:
    close(sock_id);

    return is_accessible;
}

static ucs_status_t uct_sockcm_query_md_resources(uct_md_resource_desc_t **resources_p,
                                                  unsigned *num_resources_p)
{
    return uct_single_md_resource(&uct_sockcm_mdc, resources_p, num_resources_p);
}

static ucs_status_t
uct_sockcm_md_open(const char *md_name, const uct_md_config_t *uct_md_config,
                   uct_md_h *md_p)
{
    uct_sockcm_md_t *md;
    ucs_status_t status;

    md = ucs_malloc(sizeof(*md), "sockcm_md");
    if (md == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    md->super.ops            = &uct_sockcm_md_ops;
    md->super.component      = &uct_sockcm_mdc;

    *md_p = &md->super;
    status = UCS_OK;

out:
    return status;
}

UCT_MD_COMPONENT_DEFINE(uct_sockcm_mdc, UCT_SOCKCM_MD_PREFIX,
                        uct_sockcm_query_md_resources, uct_sockcm_md_open, NULL,
                        ucs_empty_function_return_unsupported,
                        (void*)ucs_empty_function_return_success,
                        "SOCKCM_", uct_sockcm_md_config_table, uct_sockcm_md_config_t);
