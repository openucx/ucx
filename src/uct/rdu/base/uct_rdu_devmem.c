/**
 * Copyright (C) SambaNova Systems, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "uct_rdu_devmem.h"

#include <ucs/debug/log.h>
#include <ucs/sys/math.h>
#include <ucs/type/init_once.h>
#include <dlfcn.h>

/* Names of the runtime library, found through the dynamic loader search path:
 * the versioned SONAME first, then the unversioned name of older runtimes */
static const char *uct_rdu_devmem_libs[] = {"librdu_devmem.so.1",
                                            "librdu_devmem.so"};

static ucs_init_once_t uct_rdu_devmem_init_once = UCS_INIT_ONCE_INITIALIZER;
static uct_rdu_devmem_ops_t uct_rdu_devmem_ops;
static int uct_rdu_devmem_available = 0;
static int uct_rdu_devmem_dmabuf    = 0;

static void uct_rdu_devmem_load(void)
{
    const char *lib = NULL;
    void *handle    = NULL;
    size_t i;

    for (i = 0;
         (i < ucs_static_array_size(uct_rdu_devmem_libs)) && (handle == NULL);
         ++i) {
        lib    = uct_rdu_devmem_libs[i];
        handle = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
        if (handle == NULL) {
            ucs_debug("rdu_md: failed to load %s: %s", lib, dlerror());
        }
    }

    if (handle == NULL) {
        return;
    }

    uct_rdu_devmem_ops.detect        = dlsym(handle, "rdu_detect");
    uct_rdu_devmem_ops.get_pcie_path = dlsym(handle, "rdu_get_pcie_path");
    uct_rdu_devmem_ops.foreach_alloc = dlsym(handle, "rdu_foreach_alloc");
    uct_rdu_devmem_ops.dmabuf_export = dlsym(handle, "rdu_dmabuf_export");
    uct_rdu_devmem_ops.dmabuf_export_supported =
            dlsym(handle, "rdu_dmabuf_export_supported");

    /* The dma-buf functions are optional: without them, RDU memory is not
     * exported as a dma-buf, and the IB memory domain registers it directly */
    if ((uct_rdu_devmem_ops.detect == NULL) ||
        (uct_rdu_devmem_ops.get_pcie_path == NULL) ||
        (uct_rdu_devmem_ops.foreach_alloc == NULL)) {
        ucs_debug("rdu_md: %s is missing required symbols", lib);
        dlclose(handle);
        return;
    }

    /* Ask the runtime once whether the driver can export dma-buf, so every
     * memory domain and UCP context sees the same answer */
    uct_rdu_devmem_dmabuf = (uct_rdu_devmem_ops.dmabuf_export != NULL) &&
                            (uct_rdu_devmem_ops.dmabuf_export_supported !=
                             NULL) &&
                            uct_rdu_devmem_ops.dmabuf_export_supported();
    ucs_debug("rdu_md: dma-buf export is %ssupported",
              uct_rdu_devmem_dmabuf ? "" : "not ");

    /* The library stays loaded for the lifetime of the process */
    uct_rdu_devmem_available = 1;
    ucs_debug("rdu_md: loaded %s", lib);
}

const uct_rdu_devmem_ops_t *uct_rdu_devmem_get(void)
{
    UCS_INIT_ONCE(&uct_rdu_devmem_init_once) {
        uct_rdu_devmem_load();
    }
    return uct_rdu_devmem_available ? &uct_rdu_devmem_ops : NULL;
}

int uct_rdu_devmem_dmabuf_supported(void)
{
    return (uct_rdu_devmem_get() != NULL) && uct_rdu_devmem_dmabuf;
}
