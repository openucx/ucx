/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include <uct/api/uct.h>
#include <stdio.h>


int main(int argc, char **argv)
{
    ucs_status_t status;
    uct_context_h context;

    status = uct_init(&context);
    if (status != UCS_SUCCESS) {
        fprintf(stderr, "Initialization failed\n");
        return -1;
    }

    uct_cleanup(context);
    return 0;
}

