/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucx_info.h"

#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <ucs/debug/log.h>
#include <ucs/async/async.h>
#include <ucs/sys/string.h>


#define PRINT_CAP(_name, _cap_flags, _max) \
    if ((_cap_flags) & (UCT_IFACE_FLAG_##_name)) { \
        char *s = strduplower(#_name); \
        printf("#      %15s: %s\n", s, size_limit_to_str(0, _max)); \
        free(s); \
    }

#define PRINT_ZCAP_NO_CHECK(_name, _min, _max, _max_iov) \
    { \
        char *s = strduplower(#_name); \
        printf("#      %15s: %s, up to %zu iov\n", s, \
               size_limit_to_str((_min), (_max)), (_max_iov)); \
        free(s); \
    }

#define PRINT_ZCAP(_name, _cap_flags, _min, _max, _max_iov) \
    if ((_cap_flags) & (UCT_IFACE_FLAG_##_name)) { \
        PRINT_ZCAP_NO_CHECK(_name, _min, _max, _max_iov) \
    }

#define PRINT_ATOMIC_POST(_name, _cap)                   \
    print_atomic_info(UCT_ATOMIC_OP_##_name, #_name, "", \
                      _cap.atomic32.op_flags, _cap.atomic64.op_flags);

#define PRINT_ATOMIC_FETCH(_name, _cap, _suffix) \
    print_atomic_info(UCT_ATOMIC_OP_##_name, #_name, _suffix, \
                      _cap.atomic32.fop_flags, _cap.atomic64.fop_flags);

#define PRINT_LINEAR_FUNC_NS(_func) \
    { \
        printf("%.0f", (_func)->c * 1e9); \
        if ((_func)->m * 1e9 > 1e-3) { \
            printf(" + %.3f * N", (_func)->m * 1e9); \
        } \
        printf(" nsec\n"); \
    }


static char *strduplower(const char *str)
{
    char *s, *p;

    s = strdup(str);
    for (p = s; *p; ++p) {
        *p = tolower(*p);
    }
    return s;
}

static void print_atomic_info(uct_atomic_op_t opcode, const char *name,
                              const char *suffix, uint64_t op32, uint64_t op64)
{
    char amo[256] = "atomic_";
    char *s;

    if ((op32 & UCS_BIT(opcode)) || (op64 & UCS_BIT(opcode))) {
        s = strduplower(name);
        strncat(amo, suffix, sizeof(amo) - strlen(amo) - 1);
        strncat(amo, s, sizeof(amo) - strlen(amo) - 1);
        free(s);

        if ((op32 & UCS_BIT(opcode)) && (op64 & UCS_BIT(opcode))) {
            printf("#         %12s: 32, 64 bit\n", amo);
        } else {
            printf("#         %12s: %d bit\n", amo,
                   (op32 & UCS_BIT(opcode)) ? 32 : 64);
        }
    }
}

static const char *size_limit_to_str(size_t min_size, size_t max_size)
{
    static char buf[128];
    char *ptr, *end;

    ptr = buf;
    end = buf + sizeof(buf);

    if ((min_size == 0) && (max_size == SIZE_MAX)) {
        snprintf(ptr, end - ptr, "unlimited");
    } else {
        if (min_size == 0) {
            snprintf(ptr, end - ptr, "<= ");
            ptr += strlen(ptr);
        } else {
            ucs_memunits_to_str(min_size, ptr, end - ptr);
            ptr += strlen(ptr);

            snprintf(ptr, end - ptr, "..");
            ptr += strlen(ptr);
        }
        ucs_memunits_to_str(max_size, ptr, end - ptr);
    }

    return buf;
}

static void print_iface_info(uct_worker_h worker, uct_md_h md,
                             uct_tl_resource_desc_t *resource)
{
    char buf[256]                   = {0};
    uct_iface_params_t iface_params = {
        .field_mask            = UCT_IFACE_PARAM_FIELD_OPEN_MODE   |
                                 UCT_IFACE_PARAM_FIELD_DEVICE      |
                                 UCT_IFACE_PARAM_FIELD_STATS_ROOT  |
                                 UCT_IFACE_PARAM_FIELD_RX_HEADROOM |
                                 UCT_IFACE_PARAM_FIELD_CPU_MASK,
        .open_mode             = UCT_IFACE_OPEN_MODE_DEVICE,
        .mode.device.tl_name   = resource->tl_name,
        .mode.device.dev_name  = resource->dev_name,
        .stats_root            = ucs_stats_get_root(),
        .rx_headroom           = 0
    };
    uct_iface_config_t *iface_config;
    uct_iface_attr_t iface_attr;
    char max_eps_str[32];
    ucs_status_t status;
    uct_iface_h iface;

    UCS_CPU_ZERO(&iface_params.cpu_mask);
    status = uct_md_iface_config_read(md, resource->tl_name, NULL, NULL, &iface_config);
    if (status != UCS_OK) {
        return;
    }

    printf("#      Transport: %s\n", resource->tl_name);
    printf("#         Device: %s\n", resource->dev_name);
    printf("#           Type: %s\n", uct_device_type_names[resource->dev_type]);
    printf("#  System device: %s",
           ucs_topo_sys_device_get_name(resource->sys_device));
    if (resource->sys_device != UCS_SYS_DEVICE_ID_UNKNOWN) {
        printf(" (%d)", resource->sys_device);
    }
    printf("\n");

    status = uct_iface_open(md, worker, &iface_params, iface_config, &iface);
    uct_config_release(iface_config);

    if (status != UCS_OK) {
        printf("#   < failed to open interface >\n");
        /* coverity[leaked_storage] */
        return;
    }

    printf("#\n");
    printf("#      capabilities:\n");
    status = uct_iface_query(iface, &iface_attr);
    if (status != UCS_OK) {
        printf("#   < failed to query interface >\n");
    } else {
        printf("#            bandwidth: %-.2f/ppn + %-.2f MB/sec\n",
               iface_attr.bandwidth.shared / UCS_MBYTE,
               iface_attr.bandwidth.dedicated / UCS_MBYTE);
        printf("#              latency: ");
        PRINT_LINEAR_FUNC_NS(&iface_attr.latency);
        printf("#             overhead: %-.0f nsec\n", iface_attr.overhead * 1e9);

        PRINT_CAP(PUT_SHORT, iface_attr.cap.flags, iface_attr.cap.put.max_short);
        PRINT_CAP(PUT_BCOPY, iface_attr.cap.flags, iface_attr.cap.put.max_bcopy);
        PRINT_ZCAP(PUT_ZCOPY, iface_attr.cap.flags, iface_attr.cap.put.min_zcopy,
                   iface_attr.cap.put.max_zcopy, iface_attr.cap.put.max_iov);

        if (iface_attr.cap.flags & UCT_IFACE_FLAG_PUT_ZCOPY) {
            printf("#  put_opt_zcopy_align: %s\n",
                   size_limit_to_str(0, iface_attr.cap.put.opt_zcopy_align));
            printf("#        put_align_mtu: %s\n",
                   size_limit_to_str(0, iface_attr.cap.put.align_mtu));
        }

        PRINT_CAP(GET_SHORT, iface_attr.cap.flags, iface_attr.cap.get.max_short);
        PRINT_CAP(GET_BCOPY, iface_attr.cap.flags, iface_attr.cap.get.max_bcopy);
        PRINT_ZCAP(GET_ZCOPY, iface_attr.cap.flags, iface_attr.cap.get.min_zcopy,
                   iface_attr.cap.get.max_zcopy, iface_attr.cap.get.max_iov);
        if (iface_attr.cap.flags & UCT_IFACE_FLAG_GET_ZCOPY) {
            printf("#  get_opt_zcopy_align: %s\n",
                   size_limit_to_str(0, iface_attr.cap.get.opt_zcopy_align));
            printf("#        get_align_mtu: %s\n",
                   size_limit_to_str(0, iface_attr.cap.get.align_mtu));
        }

        PRINT_CAP(AM_SHORT,  iface_attr.cap.flags, iface_attr.cap.am.max_short);
        PRINT_CAP(AM_BCOPY,  iface_attr.cap.flags, iface_attr.cap.am.max_bcopy);
        PRINT_ZCAP(AM_ZCOPY,  iface_attr.cap.flags, iface_attr.cap.am.min_zcopy,
                   iface_attr.cap.am.max_zcopy, iface_attr.cap.am.max_iov);
        if (iface_attr.cap.flags & UCT_IFACE_FLAG_AM_ZCOPY) {
            printf("#   am_opt_zcopy_align: %s\n",
                   size_limit_to_str(0, iface_attr.cap.am.opt_zcopy_align));
            printf("#         am_align_mtu: %s\n",
                   size_limit_to_str(0, iface_attr.cap.am.align_mtu));
            printf("#            am header: %s\n",
                   size_limit_to_str(0, iface_attr.cap.am.max_hdr));
        }

        PRINT_CAP(TAG_EAGER_SHORT, iface_attr.cap.flags,
                  iface_attr.cap.tag.eager.max_short);
        PRINT_CAP(TAG_EAGER_BCOPY, iface_attr.cap.flags,
                  iface_attr.cap.tag.eager.max_bcopy);
        PRINT_ZCAP(TAG_EAGER_ZCOPY, iface_attr.cap.flags, 0,
                   iface_attr.cap.tag.eager.max_zcopy,
                   iface_attr.cap.tag.eager.max_iov);

        if (iface_attr.cap.flags & UCT_IFACE_FLAG_TAG_RNDV_ZCOPY) {
            PRINT_ZCAP_NO_CHECK(TAG_RNDV_ZCOPY, 0,
                                iface_attr.cap.tag.rndv.max_zcopy,
                                iface_attr.cap.tag.rndv.max_iov);
            printf("#  rndv private header: %s\n",
                   size_limit_to_str(0, iface_attr.cap.tag.rndv.max_hdr));
        }

        if (iface_attr.cap.flags & (UCT_IFACE_FLAG_TAG_EAGER_SHORT |
                                    UCT_IFACE_FLAG_TAG_EAGER_BCOPY |
                                    UCT_IFACE_FLAG_TAG_EAGER_ZCOPY |
                                    UCT_IFACE_FLAG_TAG_RNDV_ZCOPY)) {
            PRINT_ZCAP_NO_CHECK(TAG_RECV, iface_attr.cap.tag.recv.min_recv,
                                iface_attr.cap.tag.recv.max_zcopy,
                                iface_attr.cap.tag.recv.max_iov);
            printf("#  tag_max_outstanding: %s\n",
                   size_limit_to_str(0, iface_attr.cap.tag.recv.max_outstanding));
        }

        if (iface_attr.cap.atomic32.op_flags  ||
            iface_attr.cap.atomic64.op_flags  ||
            iface_attr.cap.atomic32.fop_flags ||
            iface_attr.cap.atomic64.fop_flags) {
            if (iface_attr.cap.flags & UCT_IFACE_FLAG_ATOMIC_DEVICE) {
                printf("#               domain: device\n");
            }
            if (iface_attr.cap.flags & UCT_IFACE_FLAG_ATOMIC_CPU) {
                printf("#               domain: cpu\n");
            }

            PRINT_ATOMIC_POST(ADD, iface_attr.cap);
            PRINT_ATOMIC_POST(AND, iface_attr.cap);
            PRINT_ATOMIC_POST(OR,  iface_attr.cap);
            PRINT_ATOMIC_POST(XOR, iface_attr.cap);

            PRINT_ATOMIC_FETCH(ADD,   iface_attr.cap, "f");
            PRINT_ATOMIC_FETCH(AND,   iface_attr.cap, "f");
            PRINT_ATOMIC_FETCH(OR,    iface_attr.cap, "f");
            PRINT_ATOMIC_FETCH(XOR,   iface_attr.cap, "f");
            PRINT_ATOMIC_FETCH(SWAP , iface_attr.cap, "");
            PRINT_ATOMIC_FETCH(CSWAP, iface_attr.cap, "");
        }

        buf[0] = '\0';
        if (iface_attr.cap.flags & (UCT_IFACE_FLAG_CONNECT_TO_EP |
                                    UCT_IFACE_FLAG_CONNECT_TO_IFACE)) {
            if (iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
                strncat(buf, " to ep,", sizeof(buf) - strlen(buf) - 1);
            }
            if (iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
                strncat(buf, " to iface,", sizeof(buf) - strlen(buf) - 1);
            }
            buf[strlen(buf) - 1] = '\0';
        } else {
            strncat(buf, " none", sizeof(buf) - strlen(buf) - 1);
        }
        printf("#           connection:%s\n", buf);

        printf("#      device priority: %d\n", iface_attr.priority);
        printf("#     device num paths: %d\n", iface_attr.dev_num_paths);
        printf("#              max eps: %s\n",
               ucs_memunits_to_str(iface_attr.max_num_eps, max_eps_str,
                                   sizeof(max_eps_str)));

        printf("#       device address: %zu bytes\n", iface_attr.device_addr_len);
        if (iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
            printf("#        iface address: %zu bytes\n", iface_attr.iface_addr_len);
        }
        if (iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
            printf("#           ep address: %zu bytes\n", iface_attr.ep_addr_len);
        }

        buf[0] = '\0';
        if (iface_attr.cap.flags & (UCT_IFACE_FLAG_ERRHANDLE_SHORT_BUF    |
                                    UCT_IFACE_FLAG_ERRHANDLE_BCOPY_BUF    |
                                    UCT_IFACE_FLAG_ERRHANDLE_ZCOPY_BUF    |
                                    UCT_IFACE_FLAG_ERRHANDLE_AM_ID        |
                                    UCT_IFACE_FLAG_ERRHANDLE_REMOTE_MEM   |
                                    UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE |
                                    UCT_IFACE_FLAG_EP_CHECK               |
                                    UCT_IFACE_FLAG_EP_KEEPALIVE)) {

            if (iface_attr.cap.flags & (UCT_IFACE_FLAG_ERRHANDLE_SHORT_BUF |
                                        UCT_IFACE_FLAG_ERRHANDLE_BCOPY_BUF |
                                        UCT_IFACE_FLAG_ERRHANDLE_ZCOPY_BUF)) {
                strncat(buf, " buffer (", sizeof(buf) - strlen(buf) - 1);
                if (iface_attr.cap.flags & UCT_IFACE_FLAG_ERRHANDLE_SHORT_BUF) {
                    strncat(buf, "short,", sizeof(buf) - strlen(buf) - 1);
                }
                if (iface_attr.cap.flags & UCT_IFACE_FLAG_ERRHANDLE_BCOPY_BUF) {
                    strncat(buf, "bcopy,", sizeof(buf) - strlen(buf) - 1);
                }
                if (iface_attr.cap.flags & UCT_IFACE_FLAG_ERRHANDLE_ZCOPY_BUF) {
                    strncat(buf, "zcopy,", sizeof(buf) - strlen(buf) - 1);
                }
                buf[strlen(buf) - 1] = '\0';
                strncat(buf, "),", sizeof(buf) - strlen(buf) - 1);
            }
            if (iface_attr.cap.flags & UCT_IFACE_FLAG_ERRHANDLE_AM_ID) {
                strncat(buf, " active-message id,", sizeof(buf) - strlen(buf) - 1);
            }
            if (iface_attr.cap.flags & UCT_IFACE_FLAG_ERRHANDLE_REMOTE_MEM) {
                strncat(buf, " remote access,", sizeof(buf) - strlen(buf) - 1);
            }
            if (iface_attr.cap.flags & UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE) {
                strncat(buf, " peer failure,", sizeof(buf) - strlen(buf) - 1);
            }
            if (iface_attr.cap.flags & UCT_IFACE_FLAG_EP_CHECK) {
                strncat(buf, " ep_check,", sizeof(buf) - strlen(buf) - 1);
            }
            if (iface_attr.cap.flags & UCT_IFACE_FLAG_EP_KEEPALIVE) {
                strncat(buf, " keepalive,", sizeof(buf) - strlen(buf) - 1);
            }
            buf[strlen(buf) - 1] = '\0';
        } else {
            strncat(buf, " none", sizeof(buf) - strlen(buf) - 1);
        }
        printf("#       error handling:%s\n", buf);
    }

    uct_iface_close(iface);
    printf("#\n");
}

static ucs_status_t print_tl_info(uct_md_h md, const char *tl_name,
                                  uct_tl_resource_desc_t *resources,
                                  unsigned num_resources,
                                  int print_opts,
                                  ucs_config_print_flags_t print_flags)
{
    ucs_async_context_t async;
    uct_worker_h worker;
    ucs_status_t status;
    unsigned i;

    status = ucs_async_context_init(&async, UCS_ASYNC_THREAD_LOCK_TYPE);
    if (status != UCS_OK) {
        return status;
    }

    /* coverity[alloc_arg] */
    status = uct_worker_create(&async, UCS_THREAD_MODE_SINGLE, &worker);
    if (status != UCS_OK) {
        goto out;
    }

    printf("#\n");

    if (num_resources == 0) {
        printf("# (No supported devices found)\n");
    }
    for (i = 0; i < num_resources; ++i) {
        ucs_assert(!strcmp(tl_name, resources[i].tl_name));
        print_iface_info(worker, md, &resources[i]);
    }

    uct_worker_destroy(worker);
out:
    ucs_async_context_cleanup(&async);
    return status;
}

static void print_md_info(uct_component_h component,
                          const uct_component_attr_t *component_attr,
                          const char *md_name, int print_opts,
                          ucs_config_print_flags_t print_flags,
                          const char *req_tl_name)
{
    uct_tl_resource_desc_t *resources, tmp;
    unsigned resource_index, j, num_resources, count;
    ucs_status_t status;
    const char *tl_name;
    uct_md_config_t *md_config;
    uct_md_attr_t md_attr;
    uct_md_h md;

    status = uct_md_config_read(component, NULL, NULL, &md_config);
    if (status != UCS_OK) {
        goto out;
    }

    status = uct_md_open(component, md_name, md_config, &md);
    uct_config_release(md_config);
    if (status != UCS_OK) {
        printf("# < failed to open memory domain %s >\n", md_name);
        goto out;
    }

    status = uct_md_query_tl_resources(md, &resources, &num_resources);
    if (status != UCS_OK) {
        printf("#   < failed to query memory domain resources >\n");
        goto out_close_md;
    }

    if (!(print_opts & PRINT_DEVICES)) {
        goto out_free_list;
    }

    if (req_tl_name != NULL) {
        resource_index = 0;
        while (resource_index < num_resources) {
            if (!strcmp(resources[resource_index].tl_name, req_tl_name)) {
                break;
            }
            ++resource_index;
        }
        if (resource_index == num_resources) {
            /* no selected transport on the MD */
            goto out_free_list;
        }
    }

    status = uct_md_query(md, &md_attr);
    if (status != UCS_OK) {
        printf("# < failed to query memory domain >\n");
        goto out_free_list;
    } else {
        printf("#\n");
        printf("# Memory domain: %s\n", md_name);
        printf("#     Component: %s\n", component_attr->name);
        if (md_attr.cap.flags & UCT_MD_FLAG_ALLOC) {
            printf("#             allocate: %s\n",
                   size_limit_to_str(0, md_attr.cap.max_alloc));
        }
        if (md_attr.cap.flags & UCT_MD_FLAG_REG) {
            printf("#             register: %s, cost: ",
                   size_limit_to_str(0, md_attr.cap.max_reg));
            PRINT_LINEAR_FUNC_NS(&md_attr.reg_cost);
        }
        if (md_attr.cap.flags & UCT_MD_FLAG_NEED_RKEY) {
            printf("#           remote key: %zu bytes\n", md_attr.rkey_packed_size);
        }
        if (md_attr.cap.flags & UCT_MD_FLAG_NEED_MEMH) {
            printf("#           local memory handle is required for zcopy\n");
        }
        if (md_attr.cap.flags & UCT_MD_FLAG_RKEY_PTR) {
            printf("#           rkey_ptr is supported\n");
        }
        if (md_attr.cap.flags & UCT_MD_FLAG_INVALIDATE) {
            printf("#           memory invalidation is supported\n");
        }
    }

    if (num_resources == 0) {
        printf("#   < no supported devices found >\n");
        goto out_free_list;
    }

    resource_index = 0;
    while (resource_index < num_resources) {
        /* Gather all resources for this transport */
        tl_name = resources[resource_index].tl_name;
        count = 1;
        for (j = resource_index + 1; j < num_resources; ++j) {
            if (!strcmp(tl_name, resources[j].tl_name)) {
                tmp = resources[count + resource_index];
                resources[count + resource_index] = resources[j];
                resources[j] = tmp;
                ++count;
            }
        }

        if ((req_tl_name == NULL) || !strcmp(tl_name, req_tl_name)) {
            print_tl_info(md, tl_name, &resources[resource_index], count,
                          print_opts, print_flags);
        }

        resource_index += count;
    }

out_free_list:
    uct_release_tl_resource_list(resources);
out_close_md:
    uct_md_close(md);
out:
    ;
}

static void print_cm_attr(uct_worker_h worker, uct_component_h component,
                          const char *comp_name)
{
    uct_cm_config_t *cm_config;
    uct_cm_attr_t cm_attr;
    ucs_status_t status;
    uct_cm_h cm;

    status = uct_cm_config_read(component, NULL, NULL, &cm_config);
    if (status != UCS_OK) {
        printf("# < failed to read the %s connection manager configuration >\n",
               comp_name);
        return;
    }

    status = uct_cm_open(component, worker, cm_config, &cm);
    uct_config_release(cm_config);
    if (status != UCS_OK) {
        printf("# < failed to open connection manager %s >\n", comp_name);
        /* coverity[leaked_storage] */
        return;
    }

    cm_attr.field_mask = UCT_CM_ATTR_FIELD_MAX_CONN_PRIV;
    status = uct_cm_query(cm, &cm_attr);
    if (status != UCS_OK) {
        printf("# < failed to query connection manager >\n");
    } else {
        printf("#\n");
        printf("# Connection manager: %s\n", comp_name);
        printf("#      max_conn_priv: %zu bytes\n", cm_attr.max_conn_priv);
    }

    uct_cm_close(cm);
}

static void print_cm_info(uct_component_h component,
                          const uct_component_attr_t *component_attr)
{
    ucs_async_context_t *async;
    uct_worker_h worker;
    ucs_status_t status;

    status = ucs_async_context_create(UCS_ASYNC_MODE_THREAD_SPINLOCK, &async);
    if (status != UCS_OK) {
        printf("# < failed to create asynchronous context >\n");
        return;
    }

    status = uct_worker_create(async, UCS_THREAD_MODE_SINGLE, &worker);
    if (status != UCS_OK) {
        printf("# < failed to create uct worker >\n");
        goto out_async_ctx_destroy;
    }

    print_cm_attr(worker, component, component_attr->name);

    uct_worker_destroy(worker);

out_async_ctx_destroy:
    ucs_async_context_destroy(async);
}

static void print_uct_component_info(uct_component_h component,
                                     int print_opts,
                                     ucs_config_print_flags_t print_flags,
                                     const char *req_tl_name)
{
    uct_component_attr_t component_attr;
    ucs_status_t status;
    unsigned i;

    component_attr.field_mask = UCT_COMPONENT_ATTR_FIELD_NAME              |
                                UCT_COMPONENT_ATTR_FIELD_MD_RESOURCE_COUNT |
                                UCT_COMPONENT_ATTR_FIELD_FLAGS;
    status = uct_component_query(component, &component_attr);
    if (status != UCS_OK) {
        printf("#   < failed to query component >\n");
        return;
    }

    component_attr.field_mask   = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCES;
    component_attr.md_resources = alloca(sizeof(*component_attr.md_resources) *
                                         component_attr.md_resource_count);
    status = uct_component_query(component, &component_attr);
    if (status != UCS_OK) {
        printf("#   < failed to query component md resources >\n");
        return;
    }

    for (i = 0; i < component_attr.md_resource_count; ++i) {
        print_md_info(component, &component_attr,
                      component_attr.md_resources[i].md_name, print_opts,
                      print_flags, req_tl_name);
    }

    if (print_opts & PRINT_DEVICES) {
        if (component_attr.flags & UCT_COMPONENT_FLAG_CM) {
            print_cm_info(component, &component_attr);
        }
    }
}

void print_uct_info(int print_opts, ucs_config_print_flags_t print_flags,
                    const char *req_tl_name)
{
    uct_component_h *components;
    unsigned i, num_components;
    ucs_status_t status;

    status = uct_query_components(&components, &num_components);
    if (status != UCS_OK) {
        printf("#   < failed to query UCT components >\n");
        return;
    }

    for (i = 0; i < num_components; ++i) {
        print_uct_component_info(components[i], print_opts, print_flags,
                                 req_tl_name);
    }

    uct_release_component_list(components);
}

