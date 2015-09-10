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

#define PRINT_CAP(_name, _cap_flags, _max) \
    if ((_cap_flags) & (UCT_IFACE_FLAG_##_name)) { \
        char *s = strduplower(#_name); \
        printf("#         %12s: <= %zu\n", s, _max); \
        free(s); \
    }
#define PRINT_ATOMIC_CAP(_name, _cap_flags) \
    if ((_cap_flags) & (UCT_IFACE_FLAG_##_name##32 | UCT_IFACE_FLAG_##_name##64)) { \
        char *s = strduplower(#_name); \
        if (ucs_test_all_flags(_cap_flags, \
                               UCT_IFACE_FLAG_##_name##32 | UCT_IFACE_FLAG_##_name##64)) \
        { \
            printf("#         %12s: 32, 64 bit\n", s); \
        } else { \
            printf("#         %12s: %d bit\n", s, \
                   ((_cap_flags) & UCT_IFACE_FLAG_##_name##32) ? 32 : 64); \
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

static void print_iface_info(uct_worker_h worker, uct_pd_h pd,
                             uct_tl_resource_desc_t *resource)
{
    uct_iface_config_t *iface_config;
    uct_iface_attr_t iface_attr;
    ucs_status_t status;
    uct_iface_h iface;
    char buf[200] = {0};

    status = uct_iface_config_read(resource->tl_name, NULL, NULL, &iface_config);
    if (status != UCS_OK) {
        return;
    }

    printf("#   Device: %s\n", resource->dev_name);
    printf("#      speed:         %.2f MB/sec\n", resource->bandwidth / 1024.0 / 1024.0);
    printf("#      latency:       %.3f microsec\n", resource->latency * 1e-3);

    status = uct_iface_open(pd, worker, resource->tl_name, resource->dev_name,
                            0, iface_config, &iface);
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
        PRINT_CAP(PUT_SHORT, iface_attr.cap.flags, iface_attr.cap.put.max_short);
        PRINT_CAP(PUT_BCOPY, iface_attr.cap.flags, iface_attr.cap.put.max_bcopy);
        PRINT_CAP(PUT_ZCOPY, iface_attr.cap.flags, iface_attr.cap.put.max_zcopy);
        PRINT_CAP(GET_BCOPY, iface_attr.cap.flags, iface_attr.cap.get.max_bcopy);
        PRINT_CAP(GET_ZCOPY, iface_attr.cap.flags, iface_attr.cap.get.max_zcopy);
        PRINT_CAP(AM_SHORT,  iface_attr.cap.flags, iface_attr.cap.am.max_short);
        PRINT_CAP(AM_BCOPY,  iface_attr.cap.flags, iface_attr.cap.am.max_bcopy);
        PRINT_CAP(AM_ZCOPY,  iface_attr.cap.flags, iface_attr.cap.am.max_zcopy);
        if (iface_attr.cap.flags & (UCT_IFACE_FLAG_AM_BCOPY|UCT_IFACE_FLAG_AM_ZCOPY)) {
            printf("#            am header: <= %zu\n", iface_attr.cap.am.max_hdr);
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

        buf[0] = '\0';
        if (iface_attr.cap.flags & (UCT_IFACE_FLAG_ERRHANDLE_SHORT_BUF |
                                    UCT_IFACE_FLAG_ERRHANDLE_BCOPY_BUF |
                                    UCT_IFACE_FLAG_ERRHANDLE_ZCOPY_BUF |
                                    UCT_IFACE_FLAG_ERRHANDLE_AM_ID     |
                                    UCT_IFACE_FLAG_ERRHANDLE_REMOTE_MEM))
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
            buf[strlen(buf) - 1] = '\0';
        } else {
            strncat(buf, " none", sizeof(buf) - 1);
        }
        printf("#       error handling:%s\n", buf);
    }

    uct_iface_close(iface);
    printf("#\n");
}

static ucs_status_t print_tl_info(uct_pd_h pd, const char *tl_name,
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
        print_iface_info(worker, pd, &resources[i]);
    }

    uct_worker_destroy(worker);
out:
    ucs_async_context_cleanup(&async);
    return status;
}

static void print_pd_info(const char *pd_name, int print_opts,
                          ucs_config_print_flags_t print_flags,
                          const char *req_tl_name)
{
    uct_tl_resource_desc_t *resources, tmp;
    unsigned resource_index, j, num_resources, count;
    ucs_status_t status;
    const char *tl_name;
    uct_pd_config_t *pd_config;
    uct_pd_attr_t pd_attr;
    uct_pd_h pd;

    status = uct_pd_config_read(pd_name, NULL, NULL, &pd_config);
    if (status != UCS_OK) {
        goto out;
    }

    status = uct_pd_open(pd_name, pd_config, &pd);
    uct_config_release(pd_config);
    if (status != UCS_OK) {
        printf("# < failed to open protection domain %s >\n", pd_name);
        goto out;
    }

    status = uct_pd_query_tl_resources(pd, &resources, &num_resources);
    if (status != UCS_OK) {
        printf("#   < failed to query protection domain resources >\n");
        goto out_close_pd;
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
            /* no selected transport on the PD */
            goto out_free_list;
        }
    }

    status = uct_pd_query(pd, &pd_attr);
    if (status != UCS_OK) {
        printf("# < failed to query protection domain >\n");
        goto out_free_list;
    } else {
        printf("#\n");
        printf("# Protection domain: %s\n", pd_name);
        printf("#   component:        %s\n", pd_attr.component_name);
        if (pd_attr.cap.flags & UCT_PD_FLAG_ALLOC) {
            printf("#   allocate:         <= %zu\n", pd_attr.cap.max_alloc);
        }
        if (pd_attr.cap.flags & UCT_PD_FLAG_REG) {
            printf("#   register:         <= %zu\n", pd_attr.cap.max_reg);
        }
        printf("#   remote key:       %zu bytes\n", pd_attr.rkey_packed_size);
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
            print_tl_info(pd, tl_name, &resources[resource_index], count,
                          print_opts, print_flags);
        }

        resource_index += count;
    }

out_free_list:
    uct_release_tl_resource_list(resources);
out_close_pd:
    uct_pd_close(pd);
out:
    ;
}

void print_uct_info(int print_opts, ucs_config_print_flags_t print_flags,
                    const char *req_tl_name)
{
    uct_pd_resource_desc_t *resources;
    unsigned i, num_resources;
    ucs_status_t status;

    status = uct_query_pd_resources(&resources, &num_resources);
    if (status != UCS_OK) {
        printf("#   < failed to query PD resources >\n");
        goto out;
    }

    for (i = 0; i < num_resources; ++i) {
        print_pd_info(resources[i].pd_name, print_opts, print_flags, req_tl_name);
    }

    uct_release_pd_resource_list(resources);
out:
    ;
}

