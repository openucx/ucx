/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucx_info.h"

#include <ucs/config/parser.h>
#include <ucs/config/global_opts.h>
#include <ucm/api/ucm.h>
#include <getopt.h>
#include <stdlib.h>
#include <string.h>


static void usage() {
    printf("Usage: ucx_info [options]\n");
    printf("Options are:\n");
    printf("  -v              Show version information\n");
    printf("  -b              Show build configuration\n");
    printf("  -s              Show system information\n");
    printf("  -d              Show devices and transports\n");
    printf("  -t <name>       Print information for a specific transport\n");
    printf("  -c              Show UCX configuration\n");
    printf("  -a              Show also hidden configuration\n");
    printf("  -f              Display fully decorated output\n");
    printf("  -p              Show UCP context information\n");
    printf("  -w              Show UCP worker information\n");
    printf("  -e              Show UCP endpoint configuration\n");
    printf("  -u <features>   UCP context features to use. String of one or more of:\n");
    printf("                    'a' : atomic operations\n");
    printf("                    'r' : remote memory access\n");
    printf("                    't' : tag matching \n");
    printf("                    'w' : wakeup\n");
    printf("                    'e' : error handling\n");
    printf("  -n <count>      Estimated UCP endpoint count (for ucp_init)\n");
    printf("  -D <type>       Set which device types to use when creating UCP context:\n");
    printf("                    'all'  : all possible devices (default)\n");
    printf("                    'shm'  : shared memory devices only\n");
    printf("                    'net'  : network devices only\n");
    printf("                    'self' : self transport only\n");
    printf("  -y              Show type and structures information\n");
    printf("  -h              Show this help message\n");
    printf("\n");
}

int main(int argc, char **argv)
{
    ucs_config_print_flags_t print_flags;
    ucp_ep_params_t ucp_ep_params;
    unsigned dev_type_bitmap;
    uint64_t ucp_features;
    size_t ucp_num_eps;
    unsigned print_opts;
    char *tl_name;
    const char *f;
    int c;

    print_opts               = 0;
    print_flags              = 0;
    tl_name                  = NULL;
    ucp_features             = 0;
    ucp_num_eps              = 1;
    dev_type_bitmap          = -1;
    ucp_ep_params.field_mask = 0;
    while ((c = getopt(argc, argv, "fahvcydbswpet:n:u:D:")) != -1) {
        switch (c) {
        case 'f':
            print_flags |= UCS_CONFIG_PRINT_CONFIG | UCS_CONFIG_PRINT_HEADER | UCS_CONFIG_PRINT_DOC;
            break;
        case 'a':
            print_flags |= UCS_CONFIG_PRINT_HIDDEN;
            break;
        case 'c':
            print_flags |= UCS_CONFIG_PRINT_CONFIG;
            break;
        case 'v':
            print_opts |= PRINT_VERSION;
            break;
        case 'd':
            print_opts |= PRINT_DEVICES;
            break;
        case 'b':
            print_opts |= PRINT_BUILD_CONFIG;
            break;
        case 'y':
            print_opts |= PRINT_TYPES;
            break;
        case 's':
            print_opts |= PRINT_SYS_INFO;
            break;
        case 'p':
            print_opts |= PRINT_UCP_CONTEXT;
            break;
        case 'w':
            print_opts |= PRINT_UCP_WORKER;
            break;
        case 'e':
            print_opts |= PRINT_UCP_EP;
            break;
        case 't':
            tl_name = optarg;
            break;
        case 'n':
            ucp_num_eps = atol(optarg);
            break;
        case 'u':
            for (f = optarg; *f; ++f) {
                switch (*f) {
                case 'a':
                    ucp_features |= UCP_FEATURE_AMO32|UCP_FEATURE_AMO64;
                    break;
                case 'r':
                    ucp_features |= UCP_FEATURE_RMA;
                    break;
                case 't':
                    ucp_features |= UCP_FEATURE_TAG;
                    break;
                case 'w':
                    ucp_features |= UCP_FEATURE_WAKEUP;
                    break;
                case 'e':
                    ucp_ep_params.field_mask |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
                    ucp_ep_params.err_mode    = UCP_ERR_HANDLING_MODE_PEER;
                    break;
                default:
                    usage();
                    return -1;
                }
            }
            break;
        case 'D':
            if (!strcasecmp(optarg, "net")) {
                dev_type_bitmap = UCS_BIT(UCT_DEVICE_TYPE_NET);
            } else if (!strcasecmp(optarg, "shm")) {
                dev_type_bitmap = UCS_BIT(UCT_DEVICE_TYPE_SHM);
            } else if (!strcasecmp(optarg, "self")) {
                dev_type_bitmap = UCS_BIT(UCT_DEVICE_TYPE_SELF);
            } else if (!strcasecmp(optarg, "all")) {
                dev_type_bitmap = -1;
            } else {
                usage();
                return -1;
            }
            break;
        case 'h':
            usage();
            return 0;
        default:
            usage();
            return -1;
        }
    }

    if ((print_opts == 0) && (print_flags == 0)) {
        usage();
        return -2;
    }

    if (print_opts & PRINT_VERSION) {
        print_version();
    }

    if (print_opts & PRINT_SYS_INFO) {
        print_sys_info();
    }

    if (print_opts & PRINT_BUILD_CONFIG) {
        print_build_config();
    }

    if (print_opts & PRINT_TYPES) {
        print_type_info(tl_name);
    }

    if (print_flags & UCS_CONFIG_PRINT_CONFIG) {
        ucs_config_parser_print_all_opts(stdout, print_flags);
    }

    if (print_opts & PRINT_DEVICES) {
        print_uct_info(print_opts, print_flags, tl_name);
    }

    if (print_opts & (PRINT_UCP_CONTEXT|PRINT_UCP_WORKER|PRINT_UCP_EP)) {
        if (ucp_features == 0) {
            printf("Please select UCP features using -u switch\n");
            return -1;
        }
        print_ucp_info(print_opts, print_flags, ucp_features, &ucp_ep_params,
                       ucp_num_eps, dev_type_bitmap);
    }

    return 0;
}
