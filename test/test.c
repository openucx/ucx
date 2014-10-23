/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include <mopsy/tl/base/tl.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    mopsy_status_t status;
    mopsy_tl_context_h context;

    status = mopsy_tl_init(&context);
    if (status != MOPSY_SUCCESS) {
        fprintf(stderr, "mopsy_tl_init() failed\n");
        return -1;
    }

    mopsy_tl_cleanup(context);
    return 0;
}

