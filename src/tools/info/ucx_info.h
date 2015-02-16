/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCX_INFO_H
#define UCX_INFO_H

#include <uct/api/uct.h>


enum {
    PRINT_VERSION        = UCS_BIT(0),
    PRINT_BUILD_CONFIG   = UCS_BIT(1),
    PRINT_TYPES          = UCS_BIT(2),
    PRINT_DEVICES        = UCS_BIT(3)
};


ucs_status_t print_uct_info(int print_opts, ucs_config_print_flags_t print_flags,
                            const char *req_tl_name);

void print_version();

void print_build_config();

void print_type_info(const char * tl_name);


#endif
