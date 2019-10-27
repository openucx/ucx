/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <sys/mman.h>
#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include <ucp/api/ucp.h>

#define _QUOTE(x) #x
#define QUOTE(x) _QUOTE(x)


int test_ucp_init(void *handle)
{
    typedef ucs_status_t (*ucp_init_version_func_t)(unsigned, unsigned,
                                                    const ucp_params_t *,
                                                    const ucp_config_t *,
                                                    ucp_context_h *);
    typedef void (*ucp_context_print_info_func_t)(const ucp_context_h, FILE*);
    typedef void (*ucp_cleanup_func_t)(ucp_context_h);

    ucp_init_version_func_t ucp_init_version_f;
    ucp_context_print_info_func_t ucp_context_print_info_f;
    ucp_cleanup_func_t ucp_cleanup_f;
    ucp_params_t ucp_params;
    ucs_status_t status;
    ucp_context_h ucph;

    ucp_init_version_f       = (ucp_init_version_func_t)dlsym(handle,
                                                              "ucp_init_version");
    ucp_cleanup_f            = (ucp_cleanup_func_t)dlsym(handle, "ucp_cleanup");
    ucp_context_print_info_f = (ucp_context_print_info_func_t)dlsym(handle,
                                                                    "ucp_context_print_info");

    if (!ucp_init_version_f || !ucp_cleanup_f || !ucp_context_print_info_f) {
        fprintf(stderr, "failed to get UCP function pointers\n");
        return -1;
    }

    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features   = UCP_FEATURE_RMA;
    status = ucp_init_version_f(UCP_API_MAJOR, UCP_API_MINOR, &ucp_params,
                                NULL, &ucph);
    if (status != UCS_OK) {
        fprintf(stderr, "ucp_init_version() failed\n");
        return -1;
    }

    ucp_context_print_info_f(ucph, stdout);
    ucp_cleanup_f(ucph);

    return 0;
}

int main(int argc, char **argv)
{
    const char *filename = QUOTE(LIB_PATH);
    void *handle;
    void *ptr1, *ptr2;
    size_t alloc_size;
    long ret;

    /* get page size */
    ret = sysconf(_SC_PAGESIZE);
    if (ret < 0) {
        fprintf(stderr, "sysconf(_SC_PAGESIZE) failed: %m\n");
        return -1;
    }
    alloc_size = ret;

    /* allocate some memory */
    ptr1 = malloc(alloc_size);
    if (!ptr1) {
        fprintf(stderr, "malloc() failed\n");
        return -1;
    }

    ptr2 = mmap(NULL, alloc_size, PROT_READ|PROT_WRITE,
                MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (ptr2 == MAP_FAILED) {
        fprintf(stderr, "mmmap() failed: %m\n");
        ret = -1;
        goto failed_mmap;
    }

    /* load ucp */
    printf("opening '%s'\n", filename);
    handle = dlopen(filename, RTLD_NOW | RTLD_LOCAL);
    if (handle == NULL) {
        fprintf(stderr, "failed to open %s: %m\n", filename);
        ret = -1;
        goto failed_dlopen;
    }

    /* init ucp */
    ret = test_ucp_init(handle);

    /* unload ucp */
    dlclose(handle);

failed_dlopen:
    /* relase the memory - could break if UCM is unloaded */
    munmap(ptr2, alloc_size);
failed_mmap:
    free(ptr1);

    printf("done\n");
    return ret;
}

