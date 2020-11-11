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

typedef struct ucs_list_link {
    struct ucs_list_link *prev;
    struct ucs_list_link *next;
} ucs_list_link_t;

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
    typedef void (*print_all_opts_func_t)(FILE*, const char *, int,
                                          ucs_list_link_t *);

    const char *ucs_filename = QUOTE(UCS_LIB_PATH);
    const char *uct_filename = QUOTE(UCT_LIB_PATH);
    void *ucs_handle, *uct_handle;
    ucs_list_link_t *config_list;
    int i;
    print_all_opts_func_t print_all_opts;

    /* unload and reload uct while ucs is loaded
     * would fail if uct global vars are kept on global lists in ucs */
    ucs_handle = do_dlopen_or_exit(ucs_filename);
    for (i = 0; i < 2; ++i) {
        uct_handle = do_dlopen_or_exit(uct_filename);
        dlclose(uct_handle);
    }

    /* print all config table, to force going over the global list in ucs */
    print_all_opts =
        (print_all_opts_func_t)dlsym(ucs_handle, "ucs_config_parser_print_all_opts");
    config_list = (ucs_list_link_t*)dlsym(ucs_handle, "ucs_config_global_list");
    print_all_opts(stdout, "TEST_", 0, config_list);
    dlclose(ucs_handle);

    printf("done\n");
    return 0;
}
