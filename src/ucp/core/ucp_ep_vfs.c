/**
 * Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "ucp_ep.inl"
#include "ucp_ep_vfs.h"
#include "ucp_vfs.h"

#include <ucp/wireup/wireup_ep.h>
#include <ucs/datastruct/string_buffer.h>
#include <ucs/sys/sock.h>
#include <ucs/vfs/base/vfs_cb.h>
#include <ucs/vfs/base/vfs_obj.h>


typedef struct {
    const char *name;
    uint64_t   field_mask;
    size_t     offset;
} ucp_ep_vfs_attr_t;


static const ucp_ep_vfs_attr_t ucp_ep_vfs_attrs[] = {
    {"local", UCP_EP_ATTR_FIELD_LOCAL_SOCKADDR, ucs_offsetof(ucp_ep_attr_t,
                                                             local_sockaddr)},
    {"remote", UCP_EP_ATTR_FIELD_REMOTE_SOCKADDR, ucs_offsetof(ucp_ep_attr_t,
                                                               remote_sockaddr)}
};

static const char *ucp_err_handling_mode_names[] = {
    [UCP_ERR_HANDLING_MODE_NONE] = "none",
    [UCP_ERR_HANDLING_MODE_PEER] = "peer"
};

static void ucp_ep_vfs_read_peer_name(void *obj, ucs_string_buffer_t *strb,
                                      void *arg_ptr, uint64_t arg_u64)
{
    ucp_ep_h ep = obj;

    ucs_string_buffer_appendf(strb, "%s\n", ucp_ep_peer_name(ep));
}

static ucs_status_t ucp_ep_vfs_query_sockaddr(ucp_ep_h ep, ucp_ep_attr_t *attr,
                                              uint64_t field_mask,
                                              ucs_string_buffer_t *strb)
{
    ucs_status_t status;

    attr->field_mask = field_mask;
    status           = ucp_ep_query_sockaddr(ep, attr);
    if (status != UCS_OK) {
        ucs_string_buffer_appendf(strb, "<%s>", ucs_status_string(status));
    }

    return status;
}

static void ucp_ep_vfs_read_addr_ip(void *obj, ucs_string_buffer_t *strb,
                                    void *arg_ptr, uint64_t arg_u64)
{
    ucp_ep_vfs_attr_t *vfs_attr = arg_ptr;
    ucp_ep_attr_t attr;

    if (ucp_ep_vfs_query_sockaddr(obj, &attr, vfs_attr->field_mask, strb) ==
        UCS_OK) {
        ucp_vfs_read_ip(UCS_PTR_BYTE_OFFSET(&attr, vfs_attr->offset), strb);
    }
}

static void ucp_ep_vfs_read_addr_port(void *obj, ucs_string_buffer_t *strb,
                                      void *arg_ptr, uint64_t arg_u64)
{
    ucp_ep_vfs_attr_t *vfs_attr = arg_ptr;
    ucp_ep_attr_t attr;

    if (ucp_ep_vfs_query_sockaddr(obj, &attr, vfs_attr->field_mask, strb) ==
        UCS_OK) {
        ucp_vfs_read_port(UCS_PTR_BYTE_OFFSET(&attr, vfs_attr->offset), strb);
    }
}

static void ucp_ep_vfs_init_address(ucp_ep_h ep)
{
    ucp_ep_attr_t attr;
    struct sockaddr *saddr;
    size_t i;

    for (i = 0; i < ucs_static_array_size(ucp_ep_vfs_attrs); ++i) {
        attr.field_mask = ucp_ep_vfs_attrs[i].field_mask;
        if (ucp_ep_query_sockaddr(ep, &attr) != UCS_OK) {
            continue;
        }

        saddr = UCS_PTR_BYTE_OFFSET(&attr, ucp_ep_vfs_attrs[i].offset);
        ucs_vfs_obj_add_ro_file(ep, ucp_ep_vfs_read_addr_ip,
                                (void*)&ucp_ep_vfs_attrs[i], 0, "%s_address/%s",
                                ucp_ep_vfs_attrs[i].name,
                                ucs_sockaddr_address_family_str(
                                        saddr->sa_family));
        ucs_vfs_obj_add_ro_file(ep, ucp_ep_vfs_read_addr_port,
                                (void*)&ucp_ep_vfs_attrs[i], 0,
                                "%s_address/port", ucp_ep_vfs_attrs[i].name);
    }
}

void ucp_ep_vfs_init(ucp_ep_h ep)
{
    ucp_err_handling_mode_t err_mode;

#if ENABLE_DEBUG_DATA
    ucs_vfs_obj_add_dir(ep->worker, ep, "ep/%s", ep->name);
    ucs_vfs_obj_add_ro_file(ep, ucs_vfs_show_memory_address, NULL, 0,
                            "memory_address");
#else
    ucs_vfs_obj_add_dir(ep->worker, ep, "ep/%p", ep);
#endif

    ucs_vfs_obj_add_ro_file(ep, ucp_ep_vfs_read_peer_name, NULL, 0,
                            "peer_name");

    err_mode = ucp_ep_config(ep)->key.err_mode;
    ucs_vfs_obj_add_ro_file(ep, ucs_vfs_show_primitive,
                            (void*)ucp_err_handling_mode_names[err_mode],
                            UCS_VFS_TYPE_STRING, "error_mode");

    ucp_ep_vfs_init_address(ep);
}
