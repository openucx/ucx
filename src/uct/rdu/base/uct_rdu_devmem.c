/**
 * Copyright (C) SambaNova Systems, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "uct_rdu_devmem.h"

#include <ucs/arch/cpu.h>
#include <ucs/debug/log.h>
#include <ucs/sys/math.h>
#include <ucs/type/init_once.h>
#include <dlfcn.h>

static const char *uct_rdu_devmem_libs[] = UCX_RDU_DEVMEM_LIBS;

static ucs_init_once_t uct_rdu_devmem_init_once = UCS_INIT_ONCE_INITIALIZER;
static uct_rdu_devmem_ops_t uct_rdu_devmem_ops;
static int uct_rdu_devmem_available = 0;
static int uct_rdu_devmem_dmabuf    = 0;
static int uct_rdu_devmem_loaded    = 0;

typedef struct {
    const void *address; /* Address to look up */
    const void *base; /* Start of the allocation containing the address */
    size_t     length; /* Length of the allocation, or 0 if not found */
} uct_rdu_devmem_find_alloc_ctx_t;


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
    uct_rdu_devmem_ops.init          = dlsym(handle, "rdu_devmem_init");
    uct_rdu_devmem_ops.malloc        = dlsym(handle, "rdu_malloc");
    uct_rdu_devmem_ops.free          = dlsym(handle, "rdu_free");

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
    /* Called for every memory type detection, so take the lock only until
     * the library is loaded */
    if (ucs_likely(uct_rdu_devmem_loaded)) {
        ucs_memory_cpu_load_fence();
    } else {
        UCS_INIT_ONCE(&uct_rdu_devmem_init_once) {
            uct_rdu_devmem_load();
            ucs_memory_cpu_store_fence();
            uct_rdu_devmem_loaded = 1;
        }
    }

    return uct_rdu_devmem_available ? &uct_rdu_devmem_ops : NULL;
}

int uct_rdu_devmem_dmabuf_supported(void)
{
    return (uct_rdu_devmem_get() != NULL) && uct_rdu_devmem_dmabuf;
}

static void uct_rdu_devmem_find_alloc_cb(const void *va, size_t size,
                                         uint32_t rdu_id, void *arg)
{
    uct_rdu_devmem_find_alloc_ctx_t *ctx = arg;

    if ((ctx->address >= va) &&
        (ctx->address < UCS_PTR_BYTE_OFFSET(va, size))) {
        ctx->base   = va;
        ctx->length = size;
    }
}

int uct_rdu_devmem_find_alloc(const void *address, const void **base_p,
                              size_t *length_p)
{
    uct_rdu_devmem_find_alloc_ctx_t ctx = {address, NULL, 0};

    uct_rdu_devmem_foreach_alloc(uct_rdu_devmem_find_alloc_cb, &ctx);
    if (ctx.length == 0) {
        return 0;
    }

    *base_p   = ctx.base;
    *length_p = ctx.length;
    return 1;
}
