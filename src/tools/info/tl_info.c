/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

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
        printf("#         %12s: %s\n", s, size_limit_to_str(0, _max)); \
        free(s); \
    }

#define PRINT_ZCAP(_name, _cap_flags, _min, _max, _max_iov) \
    if ((_cap_flags) & (UCT_IFACE_FLAG_##_name)) { \
        char *s = strduplower(#_name); \
        printf("#         %12s: %s, up to %zu iov\n", s, \
               size_limit_to_str((_min), (_max)), (_max_iov)); \
        free(s); \
    }

#define PRINT_ATOMIC_CAP(_name, _cap_flags) \
    if ((_cap_flags) & (UCT_IFACE_FLAG_##_name##32 | UCT_IFACE_FLAG_##_name##64)) { \
        char *s = strduplower(#_name); \
        char *domain = ""; \
        if ((_cap_flags) & UCT_IFACE_FLAG_ATOMIC_CPU) { \
            domain = ", cpu"; \
        } else if ((_cap_flags) & UCT_IFACE_FLAG_ATOMIC_DEVICE) { \
            domain = ", device"; \
        } \
        if (ucs_test_all_flags(_cap_flags, \
                               UCT_IFACE_FLAG_##_name##32 | UCT_IFACE_FLAG_##_name##64)) \
        { \
            printf("#         %12s: 32, 64 bit%s\n", s, domain); \
        } else { \
            printf("#         %12s: %d bit%s\n", s, \
                   ((_cap_flags) & UCT_IFACE_FLAG_##_name##32) ? 32 : 64, domain); \
        } \
        free(s); \
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
    uct_iface_config_t *iface_config;
    uct_iface_attr_t iface_attr;
    ucs_status_t status;
    uct_iface_h iface;
    char buf[200] = {0};
    uct_iface_params_t iface_params = {
        .open_mode             = UCT_IFACE_OPEN_MODE_DEVICE,
        .mode.device.tl_name   = resource->tl_name,
        .mode.device.dev_name  = resource->dev_name,
        .stats_root            = ucs_stats_get_root(),
        .rx_headroom           = 0
    };

    UCS_CPU_ZERO(&iface_params.cpu_mask);
    status = uct_md_iface_config_read(md, resource->tl_name, NULL, NULL, &iface_config);
    if (status != UCS_OK) {
        return;
    }

    printf("#   Device: %s\n", resource->dev_name);

    status = uct_iface_open(md, worker, &iface_params, iface_config, &iface);
    uct_config_release(iface_config);

    if (status != UCS_OK) {
        printf("#   < failed to open interface >\n");
        return;
    }

    printf("#\n");
    printf("#      capabilities:\n");
    status = uct_iface_query(iface, &iface_attr);
    if (status != UCS_OK) {
        printf("#   < failed to query interface >\n");
    } else {
        printf("#            bandwidth: %-.2f MB/sec\n", iface_attr.bandwidth / UCS_MBYTE);
        printf("#              latency: %-.0f nsec", iface_attr.latency.overhead * 1e9);
        if (iface_attr.latency.growth > 0) {
            printf(" + %.0f * N\n", iface_attr.latency.growth * 1e9);
        } else {
            printf("\n");
        }
        printf("#             overhead: %-.0f nsec\n", iface_attr.overhead * 1e9);

        PRINT_CAP(PUT_SHORT, iface_attr.cap.flags, iface_attr.cap.put.max_short);
        PRINT_CAP(PUT_BCOPY, iface_attr.cap.flags, iface_attr.cap.put.max_bcopy);
        PRINT_ZCAP(PUT_ZCOPY, iface_attr.cap.flags, iface_attr.cap.put.min_zcopy,
                   iface_attr.cap.put.max_zcopy, iface_attr.cap.put.max_iov);

        if((iface_attr.cap.flags) & (UCT_IFACE_FLAG_PUT_ZCOPY)) {
            printf("#  put_opt_zcopy_align: %s\n",
                   size_limit_to_str(0, iface_attr.cap.put.opt_zcopy_align));
            printf("#        put_align_mtu: %s\n",
                   size_limit_to_str(0, iface_attr.cap.put.align_mtu));
        }
        if((iface_attr.cap.flags) & (UCT_IFACE_FLAG_GET_ZCOPY)) {
           printf("#  get_opt_zcopy_align: %s\n",
                  size_limit_to_str(0, iface_attr.cap.get.opt_zcopy_align));
           printf("#        get_align_mtu: %s\n",
                  size_limit_to_str(0, iface_attr.cap.get.align_mtu));
        }
        if((iface_attr.cap.flags) & (UCT_IFACE_FLAG_AM_ZCOPY)) {
           printf("#   am_opt_zcopy_align: %s\n",
                  size_limit_to_str(0, iface_attr.cap.am.opt_zcopy_align));
           printf("#         am_align_mtu: %s\n",
                  size_limit_to_str(0, iface_attr.cap.am.align_mtu));
        }

        PRINT_CAP(GET_BCOPY, iface_attr.cap.flags, iface_attr.cap.get.max_bcopy);
        PRINT_ZCAP(GET_ZCOPY, iface_attr.cap.flags, iface_attr.cap.get.min_zcopy,
                   iface_attr.cap.get.max_zcopy, iface_attr.cap.get.max_iov);
        PRINT_CAP(AM_SHORT,  iface_attr.cap.flags, iface_attr.cap.am.max_short);
        PRINT_CAP(AM_BCOPY,  iface_attr.cap.flags, iface_attr.cap.am.max_bcopy);
        PRINT_ZCAP(AM_ZCOPY,  iface_attr.cap.flags, iface_attr.cap.am.min_zcopy,
                   iface_attr.cap.am.max_zcopy, iface_attr.cap.am.max_iov);
        if (iface_attr.cap.flags & UCT_IFACE_FLAG_AM_ZCOPY) {
            printf("#            am header: %s\n",
                   size_limit_to_str(0, iface_attr.cap.am.max_hdr));
        }

        PRINT_ATOMIC_CAP(ATOMIC_ADD,   iface_attr.cap.flags);
        PRINT_ATOMIC_CAP(ATOMIC_FADD,  iface_attr.cap.flags);
        PRINT_ATOMIC_CAP(ATOMIC_SWAP,  iface_attr.cap.flags);
        PRINT_ATOMIC_CAP(ATOMIC_CSWAP, iface_attr.cap.flags);

        buf[0] = '\0';
        if (iface_attr.cap.flags & (UCT_IFACE_FLAG_CONNECT_TO_EP |
                                    UCT_IFACE_FLAG_CONNECT_TO_IFACE))
        {
            if (iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
                strncat(buf, " to ep,", sizeof(buf) - 1);
            }
            if (iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
                strncat(buf, " to iface,", sizeof(buf) - 1);
            }
            buf[strlen(buf) - 1] = '\0';
        } else {
            strncat(buf, " none", sizeof(buf) - 1);
        }
        printf("#           connection:%s\n", buf);

        printf("#             priority: %d\n", iface_attr.priority);

        printf("#       device address: %zu bytes\n", iface_attr.device_addr_len);
        if (iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
            printf("#        iface address: %zu bytes\n", iface_attr.iface_addr_len);
        }
        if (iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
            printf("#           ep address: %zu bytes\n", iface_attr.ep_addr_len);
        }

        buf[0] = '\0';
        if (iface_attr.cap.flags & (UCT_IFACE_FLAG_ERRHANDLE_SHORT_BUF   |
                                    UCT_IFACE_FLAG_ERRHANDLE_BCOPY_BUF   |
                                    UCT_IFACE_FLAG_ERRHANDLE_ZCOPY_BUF   |
                                    UCT_IFACE_FLAG_ERRHANDLE_AM_ID       |
                                    UCT_IFACE_FLAG_ERRHANDLE_REMOTE_MEM  |
                                    UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE))
        {
            if (iface_attr.cap.flags & (UCT_IFACE_FLAG_ERRHANDLE_SHORT_BUF |
                                        UCT_IFACE_FLAG_ERRHANDLE_BCOPY_BUF |
                                        UCT_IFACE_FLAG_ERRHANDLE_ZCOPY_BUF))
            {
                strncat(buf, " buffer (", sizeof(buf) - 1);
                if (iface_attr.cap.flags & UCT_IFACE_FLAG_ERRHANDLE_SHORT_BUF) {
                    strncat(buf, "short,", sizeof(buf) - 1);
                }
                if (iface_attr.cap.flags & UCT_IFACE_FLAG_ERRHANDLE_BCOPY_BUF) {
                    strncat(buf, "bcopy,", sizeof(buf) - 1);
                }
                if (iface_attr.cap.flags & UCT_IFACE_FLAG_ERRHANDLE_ZCOPY_BUF) {
                    strncat(buf, "zcopy,", sizeof(buf) - 1);
                }
                buf[strlen(buf) - 1] = '\0';
                strncat(buf, "),", sizeof(buf) - 1);
            }
            if (iface_attr.cap.flags & UCT_IFACE_FLAG_ERRHANDLE_AM_ID) {
                strncat(buf, " active-message id,", sizeof(buf) - 1);
            }
            if (iface_attr.cap.flags & UCT_IFACE_FLAG_ERRHANDLE_REMOTE_MEM) {
                strncat(buf, " remote access,", sizeof(buf) - 1);
            }
            if (iface_attr.cap.flags & UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE) {
                strncat(buf, " peer failure,", sizeof(buf) - 1);
            }
            buf[strlen(buf) - 1] = '\0';
        } else {
            strncat(buf, " none", sizeof(buf) - 1);
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

    status = ucs_async_context_init(&async, UCS_ASYNC_MODE_THREAD);
    if (status != UCS_OK) {
        return status;
    }

    /* coverity[alloc_arg] */
    status = uct_worker_create(&async, UCS_THREAD_MODE_MULTI, &worker);
    if (status != UCS_OK) {
        goto out;
    }

    printf("#\n");
    printf("#   Transport: %s\n", tl_name);
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

static void print_md_info(const char *md_name, int print_opts,
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

    status = uct_md_config_read(md_name, NULL, NULL, &md_config);
    if (status != UCS_OK) {
        goto out;
    }

    status = uct_md_open(md_name, md_config, &md);
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
        printf("#            component: %s\n", md_attr.component_name);
        if (md_attr.cap.flags & UCT_MD_FLAG_ALLOC) {
            printf("#             allocate: %s\n",
                   size_limit_to_str(0, md_attr.cap.max_alloc));
        }
        if (md_attr.cap.flags & UCT_MD_FLAG_REG) {
            printf("#             register: %s, cost: %.0f",
                   size_limit_to_str(0, md_attr.cap.max_reg),
                   md_attr.reg_cost.overhead * 1e9);
            if (md_attr.reg_cost.growth * 1e9 > 1e-3) {
                printf("+(%.3f*<SIZE>)", md_attr.reg_cost.growth * 1e9);
            }
            printf(" nsec\n");
        }
        if (md_attr.cap.flags & UCT_MD_FLAG_NEED_RKEY) {
            printf("#           remote key: %zu bytes\n", md_attr.rkey_packed_size);
        }
        if (md_attr.cap.flags & UCT_MD_FLAG_NEED_MEMH) {
            printf("#           local memory handle is required for zcopy\n");
        }
        if (md_attr.cap.flags & UCT_MD_FLAG_SOCKADDR) {
            printf("#           supports client-server connection establishment via sockaddr\n");
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

void print_uct_info(int print_opts, ucs_config_print_flags_t print_flags,
                    const char *req_tl_name)
{
    uct_md_resource_desc_t *resources;
    unsigned i, num_resources;
    ucs_status_t status;

    status = uct_query_md_resources(&resources, &num_resources);
    if (status != UCS_OK) {
        printf("#   < failed to query MD resources >\n");
        goto out;
    }

    for (i = 0; i < num_resources; ++i) {
        print_md_info(resources[i].md_name, print_opts, print_flags, req_tl_name);
    }

    uct_release_md_resource_list(resources);
out:
    ;
}

