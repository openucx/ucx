/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucx_info.h"

#include <ucs/config/parser.h>
#include <ucs/config/global_opts.h>
#include <ucm/api/ucm.h>
#include <getopt.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>


static void usage() {
    printf("Usage: ucx_info [options]\n");
    printf("At least one of the following options has to be set:\n");
    printf("  -v              Show version information\n");
    printf("  -d              Show devices and transports\n");
    printf("  -b              Show build configuration\n");
    printf("  -y              Show type and structures information\n");
    printf("  -s              Show system information\n");
    printf("  -c              Show UCX configuration\n");
    printf("  -C              Comment-out default configuration values\n");
    printf("  -a              Show also hidden configuration\n");
    printf("  -f              Display fully decorated output\n");
    printf("\nUCP information (-u is required):\n");
    printf("  -p              Show UCP context information\n");
    printf("  -w              Show UCP worker information\n");
    printf("  -e              Show UCP endpoint configuration\n");
    printf("  -m <size>       Show UCP memory allocation method for a given size\n");
    printf("  -u <features>   UCP context features to use. String of one or more of:\n");
    printf("                    'a' : atomic operations\n");
    printf("                    'r' : remote memory access\n");
    printf("                    't' : tag matching \n");
    printf("                    'm' : active messages \n");
    printf("                    'w' : wakeup\n");
    printf("                  Modifiers to use in combination with above features:\n");
    printf("                    'e' : error handling\n");
    printf("\nOther settings:\n");
    printf("  -t <name>       Filter devices information using specified transport (requires -d)\n");
    printf("  -n <count>      Estimated UCP endpoint count (for ucp_init)\n");
    printf("  -N <count>      Estimated UCP endpoint count per node (for ucp_init)\n");
    printf("  -D <type>       Set which device types to use when creating UCP context:\n");
    printf("                    'all'  : all possible devices (default)\n");
    printf("                    'shm'  : shared memory devices only\n");
    printf("                    'net'  : network devices only\n");
    printf("                    'self' : self transport only\n");
    /* TODO: add IPv6 support */
    printf("  -A <ipv4>       Local IPv4 device address to use for creating\n"
           "                  endpoint in client/server mode");
    printf("  -h              Show this help message\n");
    printf("\n");
}

int main(int argc, char **argv)
{
    char *ip_addr = NULL;
    ucs_config_print_flags_t print_flags;
    ucp_ep_params_t ucp_ep_params;
    unsigned dev_type_bitmap;
    uint64_t ucp_features;
    size_t ucp_num_eps;
    size_t ucp_num_ppn;
    unsigned print_opts;
    char *tl_name, *mem_size;
    const char *f;
    int c;

    print_opts               = 0;
    print_flags              = (ucs_config_print_flags_t)0;
    tl_name                  = NULL;
    ucp_features             = 0;
    ucp_num_eps              = 1;
    ucp_num_ppn              = 1;
    mem_size                 = NULL;
    dev_type_bitmap          = UINT_MAX;
    ucp_ep_params.field_mask = 0;

    while ((c = getopt(argc, argv, "fahvcydbswpeCt:n:u:D:m:N:A:")) != -1) {
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
        case 'C':
            print_flags |= UCS_CONFIG_PRINT_COMMENT_DEFAULT;
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
        case 'm':
            print_opts |= PRINT_MEM_MAP;
            mem_size = optarg;
            break;
        case 't':
            tl_name = optarg;
            break;
        case 'n':
            ucp_num_eps = atol(optarg);
            break;
        case 'N':
            ucp_num_ppn = atol(optarg);
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
                case 'm':
                    ucp_features |= UCP_FEATURE_AM;
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
                dev_type_bitmap = UINT_MAX;
            } else {
                usage();
                return -1;
            }
            break;
        case 'A':
            ip_addr = optarg;
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

    if ((print_opts & PRINT_DEVICES) || (print_flags & UCS_CONFIG_PRINT_CONFIG)) {
        /* if UCS_CONFIG_PRINT_CONFIG is ON, trigger loading UCT modules by
         * calling print_uct_info()->uct_component_query()
         */
        print_uct_info(print_opts, print_flags, tl_name);
    }

    if (print_flags & UCS_CONFIG_PRINT_CONFIG) {
        ucs_config_parser_print_all_opts(stdout, UCS_DEFAULT_ENV_PREFIX,
                                         print_flags, &ucs_config_global_list);
    }

    if (print_opts & (PRINT_UCP_CONTEXT|PRINT_UCP_WORKER|PRINT_UCP_EP|PRINT_MEM_MAP)) {
        if (ucp_features == 0) {
            printf("Please select UCP features using -u switch: a|r|t|m|w\n");
            usage();
            return -1;
        }

        return print_ucp_info(print_opts, print_flags, ucp_features,
                              &ucp_ep_params, ucp_num_eps, ucp_num_ppn,
                              dev_type_bitmap, mem_size, ip_addr);
    }

    return 0;
}
