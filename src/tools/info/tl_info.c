/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ucx_info.h"

#include <string.h>


static ucs_status_t print_transport_info(uct_context_h context,
                                         const char *tl_name,
                                         uct_resource_desc_t *resources,
                                         unsigned num_resources,
                                         ucs_config_print_flags_t print_flags)
{
    uct_iface_config_t *iface_config;
    ucs_status_t status;
    unsigned i;

    status = uct_iface_config_read(context, tl_name, NULL, NULL, &iface_config);
    if (status != UCS_OK) {
        goto out;
    }

    printf("# \n");
    printf("# Transport: %s\n", tl_name);
    if (num_resources == 0) {
        printf("# (No supported devices found)\n");
    }
    for (i = 0; i < num_resources; ++i) {
        printf("#   device: %s, speed: %.2f MB/sec\n",
               resources[i].dev_name, resources[i].bandwidth / 1024.0 / 1024.0);
    }
    printf("# \n");

    uct_iface_config_print(iface_config, stdout, "UCT interface configuration", print_flags);
    uct_iface_config_release(iface_config);
out:
    return status;
}

ucs_status_t print_uct_info(ucs_config_print_flags_t print_flags,
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
                                      print_flags);
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
                                          count, print_flags);
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

