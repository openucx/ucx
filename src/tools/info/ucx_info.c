/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include <uct/api/uct.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/preprocessor.h>
#include <getopt.h>
#include <string.h>


enum {
    PRINT_VERSION        = UCS_BIT(0),
    PRINT_BUILD_CONFIG   = UCS_BIT(1)
};


static void print_build_config()
{
    typedef struct {
        const char *name;
        const char *value;
    } config_var_t;
    static config_var_t config_vars[] = {
        #include "build_config.h"
        {NULL, NULL}
    };
    config_var_t *var;

    for (var = config_vars; var->name != NULL; ++var) {
        printf("#define %-25s %s\n", var->name, var->value);
    }
}

static void print_version()
{
    printf("# UCT version=%s\n", UCT_VERNO_STRING);
    printf("# configured with: %s\n", UCX_CONFIGURE_FLAGS);
}

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

static ucs_status_t print_uct_info(ucs_config_print_flags_t print_flags,
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
            if (!strcmp(req_tl_name, resources[j].tl_name)) {
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

int main(int argc, char **argv)
{
    ucs_config_print_flags_t print_flags;
    unsigned print_opts;
    char *tl_name;
    int c;

    print_opts  = 0;
    print_flags = 0;
    tl_name     = NULL;
    while ((c = getopt(argc, argv, "fahvct:")) != -1) {
        switch (c) {
        case 'f':
            print_flags |= UCS_CONFIG_PRINT_HEADER | UCS_CONFIG_PRINT_DOC;
            break;
        case 'a':
            print_flags |= UCS_CONFIG_PRINT_HIDDEN;
            break;
        case 'v':
            print_opts |= PRINT_VERSION;
            break;
        case 'c':
            print_opts |= PRINT_BUILD_CONFIG;
            break;
        case 't':
            tl_name = optarg;
            break;
        case 'h':
        default:
            printf("Usage: ucx_info [options]\n");
            printf("Options are:\n");
            printf("  -f         Fully decorated output\n");
            printf("  -a         Show also hidden options\n");
            printf("  -v         Print version\n");
            printf("  -c         Print build configuration\n");
            printf("  -t <name>  Print configuration of a specific transport\n");
            printf("\n");
            return -1;
        }
    }

    if (print_opts & PRINT_VERSION) {
        print_version();
    }

    if (print_opts & PRINT_BUILD_CONFIG) {
        print_build_config();
    }

    print_uct_info(print_flags, tl_name);
    return 0;
}
