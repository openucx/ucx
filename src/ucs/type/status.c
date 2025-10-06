/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2014. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "status.h"

#include <stdio.h>


const char *ucs_status_string(ucs_status_t status)
{
    static char error_str[128] = {0};

    switch (status) {
    UCS_STATUS_STRING_CASES
    default:
        snprintf(error_str, sizeof(error_str) - 1, "Unknown error %d", status);
        return error_str;
    };
}
