/**
 * Copyright © 2021 NVIDIA CORPORATION & AFFILIATES.
 *
 * See file LICENSE for terms.
 */

#include <getopt.h>
#include <string.h>

#include <ucp/api/ucp.h>
#include <ucs/config/parser.h>
#include <ucs/config/global_opts.h>


static void apply_config_param(char *str, ucp_config_t *config)
{
    const char *config_param_key = strtok(str, "=");
    const char *config_param_val = strtok(NULL, "");
    ucs_status_t status;

    if ((config_param_key == NULL) || (config_param_val == NULL)) {
        fprintf(stderr, "incorrect configuration parameter: %s\n", str);
        return;
    }

    status = ucp_config_modify(config, config_param_key, config_param_val);
    if (status != UCS_OK) {
        fprintf(stderr, "ucp_config_modify(%s) failed: %s\n", str,
                ucs_status_string(status));
    }
}

int main(int argc, char **argv)
{
    ucp_config_t *config;
    ucp_params_t params;
    ucp_context_h context;
    ucp_worker_params_t worker_params;
    ucp_worker_h worker;
    ucs_status_t status;
    int c, ret;

    status = ucp_config_read(NULL, NULL, &config);
    if (status != UCS_OK) {
        fprintf(stderr, "ucp_config_read() failed: %s\n",
                ucs_status_string(status));
        ret = -1;
        goto out;
    }

    while ((c = getopt(argc, argv, "c:h")) != -1) {
        switch (c) {
        case 'c':
            apply_config_param(optarg, config);
            break;
        case 'h':
        default:
            printf("usage: %s\n", argv[0]);
            printf("\n");
            printf("supported options are:\n");
            printf("  -c <config>=<value>\n");
            printf("  -h print help\n");
            break;
        }
    }

    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features   = UCP_FEATURE_AM;

    status = ucp_init(&params, config, &context);
    if (status != UCS_OK) {
        fprintf(stderr, "ucp_init() failed: %s\n", ucs_status_string(status));
        ret = -1;
        goto out_release_config;
    }

    worker_params.field_mask = 0;

    status = ucp_worker_create(context, &worker_params, &worker);
    if (status != UCS_OK) {
        fprintf(stderr, "ucp_worker_create() failed: %s\n",
                ucs_status_string(status));
        ret = -1;
        goto out_cleanup_context;
    }

    ucp_context_print_info(context, stdout);
    ucs_config_parser_print_all_opts(stdout, UCS_DEFAULT_ENV_PREFIX,
                                     UCS_CONFIG_PRINT_CONFIG,
                                     &ucs_config_global_list);
    ucp_worker_destroy(worker);
    ret = 0;

out_cleanup_context:
    ucp_cleanup(context);
out_release_config:
    ucp_config_release(config);
out:
    return ret;
}
