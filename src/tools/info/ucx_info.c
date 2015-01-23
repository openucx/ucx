/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ucx_info.h"

#include <getopt.h>


int main(int argc, char **argv)
{
    ucs_config_print_flags_t print_flags;
    unsigned print_opts;
    char *tl_name;
    int c;

    print_opts  = 0;
    print_flags = 0;
    tl_name     = NULL;
    while ((c = getopt(argc, argv, "fahvcyt:")) != -1) {
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
        case 'y':
            print_opts |= PRINT_TYPES;
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
            printf("  -y         Print type configuration\n");
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

    if (print_opts & PRINT_TYPES) {
        print_type_info(tl_name);
    }

    print_uct_info(print_flags, tl_name);
    return 0;
}
