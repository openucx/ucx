/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <stdlib.h>
#include <dlfcn.h>
#include <stdio.h>

#define _QUOTE(x) #x
#define QUOTE(x) _QUOTE(x)


static void* do_dlopen_or_exit(const char *filename)
{
    void *handle;

    (void)dlerror();
    printf("opening '%s'\n", filename);
    handle = dlopen(filename, RTLD_LAZY);
    if (handle == NULL) {
        fprintf(stderr, "failed to open %s: %s\n", filename,
                dlerror());
        exit(1);
    }

    return handle;
}

int main(int argc, char **argv)
{
    typedef void (*print_all_opts_func_t)(FILE*, const char *, int);

    const char *ucs_filename = QUOTE(UCS_LIB_PATH);
    const char *uct_filename = QUOTE(UCT_LIB_PATH);
    void *ucs_handle, *uct_handle;
    int i;

    /* unload and reload uct while ucs is loaded
     * would fail if uct global vars are kept on global lists in ucs */
    ucs_handle = do_dlopen_or_exit(ucs_filename);
    for (i = 0; i < 2; ++i) {
        uct_handle = do_dlopen_or_exit(uct_filename);
        dlclose(uct_handle);
    }

    /* print all config table, to force going over the global list in ucs */
    print_all_opts_func_t print_all_opts =
        (print_all_opts_func_t)dlsym(ucs_handle, "ucs_config_parser_print_all_opts");
    print_all_opts(stdout, "TEST_", 0);
    dlclose(ucs_handle);

    printf("done\n");
    return 0;
}
