/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucx_info.h"

#include <string.h>


void print_ucp_config(ucs_config_print_flags_t print_flags)
{
    ucp_config_t *config;
    ucs_status_t status;

    status = ucp_config_read(NULL, NULL, &config);
    if (status != UCS_OK) {
        printf("<Failed to read UCP configuraton>\n");
        return;
    }

    ucp_config_print(config, stdout, "protocol layer configuration", print_flags);
    ucp_config_release(config);
}

void print_uct_config(ucs_config_print_flags_t print_flags, const char *tl_name)
{
    uct_md_resource_desc_t *md_resources;
    unsigned md_rsc_index, num_md_resources;
    uct_tl_resource_desc_t *tl_resources;
    unsigned tl_rsc_index, num_tl_resources;
    uct_iface_config_t *config;
    char tl_names[UINT8_MAX][UCT_TL_NAME_MAX];
    char cfg_title[UCT_TL_NAME_MAX + 128];
    unsigned i, num_tls;
    ucs_status_t status;
    uct_md_h md;
    uct_md_config_t *md_config;

    status = uct_query_md_resources(&md_resources, &num_md_resources);
    if (status != UCS_OK) {
        return;
    }

    uct_md_component_config_print(print_flags);

    num_tls = 0;
    for (md_rsc_index = 0; md_rsc_index < num_md_resources; ++md_rsc_index) {

        status = uct_md_config_read(md_resources[md_rsc_index].md_name, NULL,
                                    NULL, &md_config);
        if (status != UCS_OK) {
            continue;
        }

        status = uct_md_open(md_resources[md_rsc_index].md_name, md_config, &md);
        uct_config_release(md_config);
        if (status != UCS_OK) {
            continue;
        }

        status = uct_md_query_tl_resources(md, &tl_resources, &num_tl_resources);
        if (status != UCS_OK) {
            uct_md_close(md);
            continue;
        }

        for (tl_rsc_index = 0; tl_rsc_index < num_tl_resources; ++tl_rsc_index) {
            i = 0;
            while (i < num_tls) {
                if (!strcmp(tl_names[i], tl_resources[tl_rsc_index].tl_name)) {
                    break;
                }
                ++i;
            }

            /* Make sure this transport is not inserted to the array before, and
             * if user selects a specific transport - also make sure this is it.
             */
            if ((i == num_tls) &&
                ((tl_name == NULL) || !strcmp(tl_name, tl_resources[tl_rsc_index].tl_name)))
            {
                strncpy(tl_names[num_tls], tl_resources[tl_rsc_index].tl_name,
                        UCT_TL_NAME_MAX);
                ++num_tls;
            }
        }

        uct_release_tl_resource_list(tl_resources);
        uct_md_close(md);
    }

    uct_release_md_resource_list(md_resources);

    for (i = 0; i < num_tls; ++i) {
        snprintf(cfg_title, sizeof(cfg_title), "%s transport configuration",
                 tl_names[i]);
        status = uct_iface_config_read(tl_names[i], NULL, NULL, &config);
        if (status != UCS_OK) {
            printf("# < Failed to read configuration >\n");
            continue;
        }

        uct_config_print(config, stdout, cfg_title, print_flags);
        uct_config_release(config);
    }

}



