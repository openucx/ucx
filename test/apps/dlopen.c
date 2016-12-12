/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <dlfcn.h>
#include <stdio.h>

#define _QUOTE(x) #x
#define QUOTE(x) _QUOTE(x)

int main(int argc, char **argv)
{
    const char *filename = QUOTE(UCP_LIB_PATH);
    void *handle;

    printf("opening '%s'\n", filename);
    handle = dlopen(filename, RTLD_LAZY);
    if (handle == NULL) {
        fprintf(stderr, "failed to open %s: %m\n", filename);
        return -1;
    }

    printf("done\n");
    dlclose(handle);
    return 0;
}

