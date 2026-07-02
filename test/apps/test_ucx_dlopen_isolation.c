/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#define _GNU_SOURCE

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <dlfcn.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef const char *(*test_ucx_isolation_plugin_ucp_path_func_t)(void);
typedef int (*test_ucx_isolation_plugin_init_func_t)(void);

typedef struct {
    const char *needle;
    unsigned count;
} test_ucx_dlopen_isolation_find_arg_t;


static int find_loaded_library(struct dl_phdr_info *info, size_t size,
                               void *arg)
{
    test_ucx_dlopen_isolation_find_arg_t *find_arg = arg;

    if ((info->dlpi_name != NULL) &&
        (strstr(info->dlpi_name, find_arg->needle) != NULL)) {
        ++find_arg->count;
        printf("found loaded library matching %s: %s\n",
               find_arg->needle, info->dlpi_name);
    }

    return 0;
}

static int check_loaded_library(const char *needle)
{
    test_ucx_dlopen_isolation_find_arg_t find_arg = {
        .needle = needle,
        .count  = 0
    };

    dl_iterate_phdr(find_loaded_library, &find_arg);
    if (find_arg.count == 0) {
        fprintf(stderr, "could not find loaded library matching %s\n", needle);
        return -1;
    }

    return 0;
}

static int make_soname(char *buffer, size_t max, const char *lib,
                       const char *suffix)
{
    int ret;

    ret = snprintf(buffer, max, "lib%s-%s.so.0", lib, suffix);
    if ((ret < 0) || ((size_t)ret >= max)) {
        fprintf(stderr, "suffix is too long: %s\n", suffix);
        return -1;
    }

    return 0;
}

static void usage(const char *program)
{
    fprintf(stderr, "Usage: %s <foreign-libucp-path> <plugin-path> "
            "<private-suffix> [deepbind]\n", program);
}

int main(int argc, char **argv)
{
    test_ucx_isolation_plugin_ucp_path_func_t plugin_ucp_path;
    test_ucx_isolation_plugin_init_func_t plugin_init;
    const char *foreign_libucp_path;
    const char *plugin_path;
    const char *private_suffix;
    const char *ucp_path;
    char private_libucp[64];
    char private_libuct[64];
    char private_libucs[64];
    int flags;
    void *foreign_handle;
    void *plugin_handle;

    if ((argc != 4) && (argc != 5)) {
        usage(argv[0]);
        return -1;
    }

    foreign_libucp_path = argv[1];
    plugin_path         = argv[2];
    private_suffix      = argv[3];

    flags = RTLD_NOW | RTLD_LOCAL;
    if (argc == 5) {
        if (strcmp(argv[4], "deepbind") != 0) {
            usage(argv[0]);
            return -1;
        }
#ifdef RTLD_DEEPBIND
        flags |= RTLD_DEEPBIND;
#else
        fprintf(stderr, "RTLD_DEEPBIND is not supported on this platform\n");
        return -1;
#endif
    }

    if ((make_soname(private_libucp, sizeof(private_libucp), "ucp",
                     private_suffix) != 0) ||
        (make_soname(private_libuct, sizeof(private_libuct), "uct",
                     private_suffix) != 0) ||
        (make_soname(private_libucs, sizeof(private_libucs), "ucs",
                     private_suffix) != 0)) {
        return -1;
    }

    printf("opening foreign UCX library '%s'\n", foreign_libucp_path);
    foreign_handle = dlopen(foreign_libucp_path, RTLD_NOW | RTLD_GLOBAL);
    if (foreign_handle == NULL) {
        fprintf(stderr, "failed to open %s: %s\n", foreign_libucp_path,
                dlerror());
        return -1;
    }

    if (check_loaded_library("libucp.so.0") != 0) {
        return -1;
    }

    printf("opening UCX consumer plugin '%s'\n", plugin_path);
    plugin_handle = dlopen(plugin_path, flags);
    if (plugin_handle == NULL) {
        fprintf(stderr, "failed to open %s: %s\n", plugin_path, dlerror());
        return -1;
    }

    plugin_ucp_path =
        (test_ucx_isolation_plugin_ucp_path_func_t)
        dlsym(plugin_handle, "test_ucx_isolation_plugin_ucp_path");
    plugin_init =
        (test_ucx_isolation_plugin_init_func_t)
        dlsym(plugin_handle, "test_ucx_isolation_plugin_init");
    if ((plugin_ucp_path == NULL) || (plugin_init == NULL)) {
        fprintf(stderr, "failed to resolve plugin test entry points: %s\n",
                dlerror());
        return -1;
    }

    ucp_path = plugin_ucp_path();
    if (ucp_path == NULL) {
        fprintf(stderr, "plugin could not resolve its UCX provider path\n");
        return -1;
    }

    printf("plugin UCX provider path: %s\n", ucp_path);
    if (strstr(ucp_path, private_libucp) == NULL) {
        fprintf(stderr, "plugin resolved UCX provider from %s, expected %s\n",
                ucp_path, private_libucp);
        return -1;
    }

    if ((check_loaded_library(private_libucp) != 0) ||
        (check_loaded_library(private_libuct) != 0) ||
        (check_loaded_library(private_libucs) != 0) ||
        (check_loaded_library("libucp.so.0") != 0)) {
        return -1;
    }

    if (plugin_init() != 0) {
        fprintf(stderr, "plugin UCX initialization failed\n");
        return -1;
    }

    dlclose(plugin_handle);
    dlclose(foreign_handle);

    printf("SUCCESS\n");
    return 0;
}
