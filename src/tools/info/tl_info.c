/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ucx_info.h"

#include <string.h>
#include <stdlib.h>
#include <ctype.h>


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

static void print_resource_info(uct_context_h context,
                                uct_resource_desc_t *resource,
                                uct_iface_config_t *iface_config)
{
    uct_iface_attr_t iface_attr;
    uct_pd_attr_t pd_attr;
    ucs_status_t status;
    uct_iface_h iface;
    char buf[200] = {0};

    printf("#   %s\n", resource->dev_name);
    printf("#      speed:         %.2f MB/sec\n", resource->bandwidth / 1024.0 / 1024.0);
    printf("#      latency:       %.3f microsec\n", resource->latency * 1e-3);

    status = uct_iface_open(context, resource->tl_name, resource->dev_name,
                            0, iface_config, &iface);
    if (status != UCS_OK) {
        printf("#   < failed to open interface >\n");
        return;
    }

    printf("#\n");
    status = uct_pd_query(iface->pd, &pd_attr);
    if (status != UCS_OK) {
        printf("#   < failed to query protection domain >\n");
    } else {
        printf("#      protection domain: %s\n", pd_attr.name);
        if (pd_attr.cap.flags & UCT_PD_FLAG_ALLOC) {
            printf("#             allocate: <= %zu\n", pd_attr.cap.max_alloc);
        }
        if (pd_attr.cap.flags & UCT_PD_FLAG_REG) {
            printf("#             register: <= %zu\n", pd_attr.cap.max_reg);
        }
        printf("#           remote key: %zu bytes\n", pd_attr.rkey_packed_size);
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

static ucs_status_t print_transport_info(uct_context_h context,
                                         const char *tl_name,
                                         uct_resource_desc_t *resources,
                                         unsigned num_resources,
                                         int print_opts,
                                         ucs_config_print_flags_t print_flags)
{
    uct_iface_config_t *iface_config;
    ucs_status_t status;
    unsigned i;

    status = uct_iface_config_read(context, tl_name, NULL, NULL, &iface_config);
    if (status != UCS_OK) {
        goto out;
    }

    printf("#\n");
    printf("# Transport: %s\n", tl_name);
    printf("#\n");

    if (print_opts & PRINT_DEVICES) {
        if (num_resources == 0) {
            printf("# (No supported devices found)\n");
        }
        for (i = 0; i < num_resources; ++i) {
            print_resource_info(context, &resources[i], iface_config);
        }
    }

    uct_iface_config_print(iface_config, stdout, "UCT interface configuration",
                           print_flags);
    uct_iface_config_release(iface_config);
out:
    return status;
}

ucs_status_t print_uct_info(int print_opts, ucs_config_print_flags_t print_flags,
                            const char *req_tl_name)
{
    uct_resource_desc_t *resources, tmp;
    unsigned resource_index, j, num_resources, count;
    uct_context_h context;
    ucs_status_t status;
    const char *tl_name;

    status = uct_init(&context);
    if (status != UCS_OK) {
        fprintf(stderr, "Failed to create context: %s\n", ucs_status_string(status));
        goto out;
    }

    status = uct_query_resources(context, &resources, &num_resources);
    if (status != UCS_OK) {
        fprintf(stderr, "Failed to query resources: %s\n", ucs_status_string(status));
        goto out_uct_cleanup;
    }

    if (req_tl_name != NULL) {
        count = 0;
        for (j = 0; j < num_resources; ++j) {
            if (!strcasecmp(req_tl_name, resources[j].tl_name)) {
                if (j != count) {
                    resources[count] = resources[j];
                }
                ++count;
            }
        }

        status = print_transport_info(context, req_tl_name, resources, count,
                                      print_opts, print_flags);
        if (status != UCS_OK) {
            fprintf(stderr, "Failed to query transport `%s': %s\n", req_tl_name,
                    ucs_status_string(status));
            goto out_free_list;
        }
    } else {
        if (num_resources == 0) {
            printf("# (No supported devices found)\n");
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

            status = print_transport_info(context, tl_name, &resources[resource_index],
                                          count, print_opts, print_flags);
            if (status != UCS_OK) {
                fprintf(stderr, "Failed to get transport info (%s): %s\n", tl_name,
                        ucs_status_string(status));
                goto out_free_list;
            }

            resource_index += count;
        }
    }

    status = UCS_OK;
out_free_list:
    uct_release_resource_list(resources);
out_uct_cleanup:
    uct_cleanup(context);
out:
    return status;
}

