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
#include <string.h>

#include <ucp/api/ucp.h>
#include <uct/api/uct.h>
#include <uct/base/uct_component.h>

#define _QUOTE(x) #x
#define QUOTE(x) _QUOTE(x)


int test_ucp_init(void *handle)
{
    typedef ucs_status_t (*ucp_init_version_func_t)(unsigned, unsigned,
                                                    const ucp_params_t *,
                                                    const ucp_config_t *,
                                                    ucp_context_h *);
    typedef void (*ucp_cleanup_func_t)(ucp_context_h);
    typedef ucs_status_t (*uct_query_components_func_t)(uct_component_h **components_p,
                                                        unsigned *num_components_p);
    typedef void (*uct_release_component_list_func_t)(uct_component_h *
                                                      components);


    ucp_init_version_func_t ucp_init_version_f;
    ucp_cleanup_func_t ucp_cleanup_f;
    ucp_params_t ucp_params;
    ucs_status_t status;
    ucp_context_h ucph;
    uct_query_components_func_t uct_query_components_f;
    uct_release_component_list_func_t uct_release_component_list_f;
    uct_component_h *uct_components;
    unsigned num_uct_components, i;


    ucp_init_version_f       = (ucp_init_version_func_t)dlsym(handle,
                                                              "ucp_init_version");
    ucp_cleanup_f            = (ucp_cleanup_func_t)dlsym(handle, "ucp_cleanup");
    uct_query_components_f   = (uct_query_components_func_t)
            dlsym(handle, "uct_query_components");
    uct_release_component_list_f = (uct_release_component_list_func_t)
            dlsym(handle, "uct_release_component_list");

    if (!ucp_init_version_f || !ucp_cleanup_f || !uct_query_components_f ||
        !uct_release_component_list_f) {
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

    status = uct_query_components_f(&uct_components, &num_uct_components);
    if (status != UCS_OK) {
        fprintf(stderr, "Could not query UCT components\n");
        return -1;
    }

    for (i = 0; i < num_uct_components; ++i) {
        printf("Found component: %s\n", uct_components[i]->name);
    }

    uct_release_component_list_f(uct_components);
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
        fprintf(stderr, "mmap() failed: %m\n");
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
    /* release the memory - could break if UCM is unloaded */
    munmap(ptr2, alloc_size);
failed_mmap:
    free(ptr1);

    printf("done\n");
    return ret;
}

