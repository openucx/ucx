/**
 * Copyright (C) SambaNova Systems, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "rdumem.h"

#include <ucm/event/event.h>
#include <ucm/util/log.h>
#include <ucm/util/reloc.h>
#include <ucm/util/replace.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/math.h>
#include <ucs/sys/preprocessor.h>

#include <dlfcn.h>
#include <pthread.h>


static const char *ucm_rdu_devmem_libs[] = UCM_RDU_DEVMEM_LIBS;


typedef void *(*ucm_rdu_malloc_func_t)(uint32_t rdu_id, size_t size,
                                       int mem_type);
typedef void (*ucm_rdu_free_func_t)(uint32_t rdu_id, void *ptr);
typedef void (*ucm_rdu_foreach_cb_t)(const void *va, size_t size,
                                     uint32_t rdu_id, void *arg);
typedef void (*ucm_rdu_foreach_func_t)(ucm_rdu_foreach_cb_t cb, void *arg);

typedef struct {
    const void *ptr;
    size_t     size;
} ucm_rdu_find_alloc_ctx_t;


/* Cached value of a function which the loaded runtime library does not
 * provide */
#define UCM_RDU_FUNC_NOT_FOUND ((void*)-1)


/*
 * Look up a symbol of the runtime library.  The library may have been loaded
 * with RTLD_LOCAL, so its symbols are not necessarily in the global scope; open
 * it by name without loading it, and fall back to the global scope.  Set
 * *lib_loaded_p if the runtime library is loaded.
 */
static void *ucm_rdu_dlsym(const char *symbol, int *lib_loaded_p)
{
    void *func_ptr = NULL;
    size_t i;
    void *dl;

    *lib_loaded_p = 0;
    for (i = 0;
         (i < ucs_static_array_size(ucm_rdu_devmem_libs)) && (func_ptr == NULL);
         ++i) {
        dl = dlopen(ucm_rdu_devmem_libs[i], RTLD_LAZY | RTLD_NOLOAD);
        if (dl != NULL) {
            *lib_loaded_p = 1;
            func_ptr      = dlsym(dl, symbol);
            dlclose(dl);
        }
    }

    if (func_ptr == NULL) {
        func_ptr = dlsym(RTLD_DEFAULT, symbol);
    }

    return func_ptr;
}

/*
 * Return a function of the runtime library, looking it up on first use, or
 * NULL if the library does not provide it.  The result is cached once the
 * function is found, or once the library is loaded without it, so a missing
 * optional function is not looked up again on every call.
 */
static void *ucm_rdu_get_func(const char *symbol, void **func_ptr_p)
{
    void *func_ptr = *func_ptr_p;
    int lib_loaded;

    if (ucs_unlikely(func_ptr == NULL)) {
        pthread_mutex_lock(&ucm_reloc_get_orig_lock);
        ucm_reloc_get_orig_thread = pthread_self();
        func_ptr                  = *func_ptr_p;
        if (func_ptr == NULL) {
            func_ptr = ucm_rdu_dlsym(symbol, &lib_loaded);
            if (func_ptr != NULL) {
                *func_ptr_p = func_ptr;
            } else if (lib_loaded) {
                *func_ptr_p = UCM_RDU_FUNC_NOT_FOUND;
            }
        }
        ucm_reloc_get_orig_thread = (pthread_t)-1;
        pthread_mutex_unlock(&ucm_reloc_get_orig_lock);
    }

    return (func_ptr == UCM_RDU_FUNC_NOT_FOUND) ? NULL : func_ptr;
}

/*
 * Return the original function, looking it up on first use.
 */
static void *ucm_rdu_get_orig(const char *symbol, void **func_ptr_p)
{
    void *func_ptr = ucm_rdu_get_func(symbol, func_ptr_p);

    if (func_ptr == NULL) {
        ucm_fatal("could not find address of original %s()", symbol);
    }

    return func_ptr;
}

/*
 * Return rdu_foreach_alloc(), or NULL if the runtime library does not provide
 * it.  Without it the RDU memory domain is not available, so these events are
 * not used.
 */
static ucm_rdu_foreach_func_t ucm_rdu_get_foreach_alloc(void)
{
    static void *func_ptr = NULL;

    return (ucm_rdu_foreach_func_t)ucm_rdu_get_func("rdu_foreach_alloc",
                                                    &func_ptr);
}

static void
ucm_rdu_find_alloc_cb(const void *va, size_t size, uint32_t rdu_id, void *arg)
{
    ucm_rdu_find_alloc_ctx_t *ctx = arg;

    if (va == ctx->ptr) {
        ctx->size = size;
    }
}

static size_t ucm_rdu_get_alloc_size(const void *ptr)
{
    ucm_rdu_find_alloc_ctx_t ctx = {ptr, 1}; /* minimal size by default */
    ucm_rdu_foreach_func_t foreach_alloc;

    foreach_alloc = ucm_rdu_get_foreach_alloc();
    if (foreach_alloc != NULL) {
        foreach_alloc(ucm_rdu_find_alloc_cb, &ctx);
    } else {
        ucm_debug("rdu_foreach_alloc is not found, assuming size %zu for %p",
                  ctx.size, ptr);
    }

    return ctx.size;
}

void *ucm_rdu_malloc(uint32_t rdu_id, size_t size, int mem_type)
{
    static void *orig_func_ptr = NULL;
    ucm_rdu_malloc_func_t orig_func;
    void *ptr;

    orig_func = (ucm_rdu_malloc_func_t)ucm_rdu_get_orig("rdu_malloc",
                                                        &orig_func_ptr);

    ucm_event_enter();

    ptr = orig_func(rdu_id, size, mem_type);
    if (ptr != NULL) {
        ucm_trace("ucm_rdu_malloc(rdu_id=%u size=%zu)=%p", rdu_id, size, ptr);
        /* Report the type as unknown, so the memory type cache asks the RDU
         * memory domain for the type and the device of the buffer */
        ucm_dispatch_mem_type_alloc(ptr, size, UCS_MEMORY_TYPE_UNKNOWN,
                                    UCS_MEM_FLAG_REGISTRABLE);
    }

    ucm_event_leave();
    return ptr;
}

void ucm_rdu_free(uint32_t rdu_id, void *ptr)
{
    static void *orig_func_ptr = NULL;
    ucm_rdu_free_func_t orig_func;

    orig_func = (ucm_rdu_free_func_t)ucm_rdu_get_orig("rdu_free",
                                                      &orig_func_ptr);

    ucm_event_enter();

    ucm_trace("ucm_rdu_free(rdu_id=%u ptr=%p)", rdu_id, ptr);

    /* The runtime keeps the address range mapped and later reuses it for other
     * allocations, so invalidate cached information before releasing it */
    if (ptr != NULL) {
        ucm_dispatch_mem_type_free(ptr, ucm_rdu_get_alloc_size(ptr),
                                   UCS_MEMORY_TYPE_UNKNOWN);
    }

    orig_func(rdu_id, ptr);

    ucm_event_leave();
}

void *ucm_override_rdu_malloc(uint32_t rdu_id, size_t size, int mem_type)
{
    if (ucs_unlikely(ucm_reloc_get_orig_thread == pthread_self())) {
        return NULL;
    }

    return ucm_rdu_malloc(rdu_id, size, mem_type);
}

void ucm_override_rdu_free(uint32_t rdu_id, void *ptr)
{
    if (ucs_unlikely(ucm_reloc_get_orig_thread == pthread_self())) {
        return;
    }

    ucm_rdu_free(rdu_id, ptr);
}

static ucm_reloc_patch_t patches[] = {{UCS_PP_MAKE_STRING(rdu_malloc),
                                       ucm_override_rdu_malloc},
                                      {UCS_PP_MAKE_STRING(rdu_free),
                                       ucm_override_rdu_free},
                                      {NULL, NULL}};

static ucs_status_t ucm_rdumem_install(int events)
{
    static int ucm_rdumem_installed      = 0;
    static pthread_mutex_t install_mutex = PTHREAD_MUTEX_INITIALIZER;
    ucs_status_t status                  = UCS_OK;
    ucm_reloc_patch_t *patch;

    if (!(events & (UCM_EVENT_MEM_TYPE_ALLOC | UCM_EVENT_MEM_TYPE_FREE))) {
        return UCS_OK;
    }

    pthread_mutex_lock(&install_mutex);

    if (ucm_rdumem_installed) {
        goto out_unlock;
    }

    for (patch = patches; patch->symbol != NULL; ++patch) {
        status = ucm_reloc_modify(patch);
        if (status != UCS_OK) {
            ucm_warn("failed to install relocation table entry for '%s'",
                     patch->symbol);
            goto out_unlock;
        }
    }

    ucm_debug("rdu hooks are ready");
    ucm_rdumem_installed = 1;

out_unlock:
    pthread_mutex_unlock(&install_mutex);
    return status;
}

/* RDU memory is allocated from a mapping of the RDU device, so existing
 * allocations are reported as part of that mapping */
static ucm_event_installer_t ucm_rdu_initializer = {
    .install      = ucm_rdumem_install,
    .path_pattern = "/dev/rdu"
};

UCS_STATIC_INIT
{
    ucs_list_add_tail(&ucm_event_installer_list, &ucm_rdu_initializer.list);
}

UCS_STATIC_CLEANUP
{
    ucs_list_del(&ucm_rdu_initializer.list);
}
