/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <ucp/api/ucp.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/math.h>

int main(int argc, char **argv)
{
    void *ptr;

    ptr = ucs_malloc(500 * UCS_MBYTE, "test memtrack limit");
    if (ptr == NULL) {
        fprintf(stderr, "No memory\n");
        return EXIT_FAILURE;
    }

    printf("SUCCESS\n");
    ucs_free(ptr);

    return EXIT_SUCCESS;
}
