/*
 * Copyright (C) Intel Corporation, 2023-2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "ze_copy_md.h"

#include <uct/ze/base/ze_base.h>

#include <string.h>
#include <limits.h>
#include <dlfcn.h>
#include <pthread.h>
#include <ucm/api/ucm.h>
#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/math.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/datastruct/khash.h>
#include <ucs/type/class.h>
#include <ucs/type/init_once.h>
#include <ucs/memory/memtype_cache.h>


static ucs_config_field_t uct_ze_copy_md_config_table[] = {
    {"", "", NULL, ucs_offsetof(uct_ze_copy_md_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {"DEVICE_ORDINAL", "0",
     "Ordinal of the GPU device to allocate memory from.",
     ucs_offsetof(uct_ze_copy_md_config_t, device_ordinal),
     UCS_CONFIG_TYPE_INT},

    {"HOST_IMPORT", "y",
     "Import host memory into the Level Zero driver when it is registered, so\n"
     "that copies to and from it reuse one driver allocation instead of\n"
     "mapping the range again for every copy.",
     ucs_offsetof(uct_ze_copy_md_config_t, host_import),
     UCS_CONFIG_TYPE_BOOL},

    {NULL}
};


/* Registration handle of a range that was not imported */
#define UCT_ZE_COPY_DUMMY_MEMH ((uct_mem_h)0xdeadbeef)

/* zel_handle_type_t value of a driver handle (level_zero/loader/ze_loader.h) */
#define UCT_ZE_COPY_ZEL_HANDLE_DRIVER 0


typedef ze_result_t (*uct_ze_copy_translate_handle_func_t)(int handle_type,
                                                           void *handle_in,
                                                           void **handle_out_p);


/* Host memory range imported into the driver through this registry */
typedef struct {
    void *address;
    int  refcount; /* Registrations that resolved to this import */
} uct_ze_copy_import_t;


KHASH_INIT(ze_copy_import, uintptr_t, uct_ze_copy_import_t*, 1,
           kh_int64_hash_func, kh_int64_hash_equal)


/*
 * The driver keeps imports per driver handle and does not count references:
 * importing a range inside an existing import succeeds without recording it,
 * and releasing any address inside an import drops the whole import. All MDs
 * therefore share one registry of the imports owned by this process, keyed by
 * base address, and release an import only with its last registration.
 */
typedef struct {
    ucs_init_once_t           init_once;
    pthread_mutex_t           lock;      /* Protects the fields below; held
                                            across the driver calls, which
                                            take milliseconds for large
                                            ranges */
    khash_t(ze_copy_import)   hash;
    uct_ze_copy_host_import_t funcs;
    int                       available; /* Entry points were resolved */
    int                       enabled;   /* Driver did not reject imports */
} uct_ze_copy_import_registry_t;


static uct_ze_copy_import_registry_t uct_ze_copy_imports = {
    .init_once = UCS_INIT_ONCE_INITIALIZER,
    .lock      = PTHREAD_MUTEX_INITIALIZER,
    .hash      = KHASH_STATIC_INITIALIZER
};


static ucs_status_t
uct_ze_copy_import_get_func(ze_driver_handle_t driver, const char *name,
                            void **func_p)
{
    void *func = NULL;
    ze_result_t ret;

    /* Some drivers report success for an unknown name without setting the
       address */
    ret = zeDriverGetExtensionFunctionAddress(driver, name, &func);
    if (ret != ZE_RESULT_SUCCESS) {
        ucs_debug("zeDriverGetExtensionFunctionAddress(%s) failed: 0x%x", name,
                  ret);
        return UCS_ERR_UNSUPPORTED;
    }

    if (func == NULL) {
        ucs_debug("zeDriverGetExtensionFunctionAddress(%s) returned no address",
                  name);
        return UCS_ERR_UNSUPPORTED;
    }

    *func_p = func;
    return UCS_OK;
}

static void uct_ze_copy_import_registry_init(ze_driver_handle_t driver)
{
    uct_ze_copy_import_registry_t *registry = &uct_ze_copy_imports;
    uct_ze_copy_host_import_t *funcs        = &registry->funcs;
    uct_ze_copy_translate_handle_func_t translate;
    void *driver_handle, *import, *release, *get_base;
    ze_result_t ret;

    /* The extension bypasses the loader, so it needs the driver's own handle.
       The translation exists since loader 1.7 and is resolved at runtime to
       keep older loaders working without the import. */
    translate = (uct_ze_copy_translate_handle_func_t)
            dlsym(RTLD_DEFAULT, "zelLoaderTranslateHandle");
    if (translate == NULL) {
        ucs_debug("zelLoaderTranslateHandle not found, host memory import is "
                  "disabled");
        return;
    }

    ret = translate(UCT_ZE_COPY_ZEL_HANDLE_DRIVER, driver, &driver_handle);
    if (ret != ZE_RESULT_SUCCESS) {
        ucs_debug("zelLoaderTranslateHandle(driver %p) failed: 0x%x", driver,
                  ret);
        return;
    }

    if ((uct_ze_copy_import_get_func(driver, "zexDriverImportExternalPointer",
                                     &import) != UCS_OK) ||
        (uct_ze_copy_import_get_func(driver, "zexDriverReleaseImportedPointer",
                                     &release) != UCS_OK) ||
        (uct_ze_copy_import_get_func(driver,
                                     "zexDriverGetHostPointerBaseAddress",
                                     &get_base) != UCS_OK)) {
        return;
    }

    funcs->driver       = driver_handle;
    funcs->import       = (uct_ze_copy_import_ptr_func_t)import;
    funcs->release      = (uct_ze_copy_release_ptr_func_t)release;
    funcs->get_base     = (uct_ze_copy_get_base_func_t)get_base;
    registry->available = 1;
    registry->enabled   = 1;
}

const uct_ze_copy_host_import_t *uct_ze_copy_md_get_host_import(void)
{
    ze_driver_handle_t driver = uct_ze_base_get_driver();

    if (driver == NULL) {
        return NULL;
    }

    UCS_INIT_ONCE(&uct_ze_copy_imports.init_once) {
        uct_ze_copy_import_registry_init(driver);
    }

    return uct_ze_copy_imports.available ? &uct_ze_copy_imports.funcs : NULL;
}

/* Returns in base_p the base of the import containing ptr, or NULL */
static ucs_status_t uct_ze_copy_import_get_base(void *ptr, void **base_p)
{
    const uct_ze_copy_host_import_t *funcs = &uct_ze_copy_imports.funcs;
    void *base                             = NULL;
    ze_result_t ret;

    ret = funcs->get_base(funcs->driver, ptr, &base);
    if (ret == ZE_RESULT_ERROR_UNSUPPORTED_FEATURE) {
        return UCS_ERR_UNSUPPORTED;
    } else if ((ret != ZE_RESULT_SUCCESS) &&
               (ret != ZE_RESULT_ERROR_INVALID_ARGUMENT)) {
        /* INVALID_ARGUMENT means that the pointer is not imported */
        ucs_debug("zexDriverGetHostPointerBaseAddress(%p) failed: 0x%x", ptr,
                  ret);
        return UCS_ERR_IO_ERROR;
    }

    *base_p = (ret == ZE_RESULT_SUCCESS) ? base : NULL;
    return UCS_OK;
}

/* Called with the registry lock held */
static void uct_ze_copy_import_disable(void)
{
    uct_ze_copy_imports.enabled = 0;
    ucs_diag("level zero driver does not support host pointer import, "
             "registered host memory will not be imported");
}

/* Called with the registry lock held */
static uct_ze_copy_import_t *uct_ze_copy_import_lookup(void *base)
{
    khash_t(ze_copy_import) *hash = &uct_ze_copy_imports.hash;
    khiter_t iter;

    iter = kh_get(ze_copy_import, hash, (uintptr_t)base);
    return (iter == kh_end(hash)) ? NULL : kh_val(hash, iter);
}

/* Called with the registry lock held */
static ucs_status_t
uct_ze_copy_import_create(void *address, size_t length,
                          ucs_log_level_t log_level,
                          uct_ze_copy_import_t **import_p)
{
    const uct_ze_copy_host_import_t *funcs = &uct_ze_copy_imports.funcs;
    khash_t(ze_copy_import) *hash          = &uct_ze_copy_imports.hash;
    uct_ze_copy_import_t *import;
    ucs_status_t status;
    khiter_t iter;
    ze_result_t ret;
    int khret;

    import = ucs_malloc(sizeof(*import), "ze_copy_import");
    if (import == NULL) {
        ucs_error("failed to allocate ze_copy import record");
        return UCS_ERR_NO_MEMORY;
    }

    iter = kh_put(ze_copy_import, hash, (uintptr_t)address, &khret);
    if (khret == UCS_KH_PUT_FAILED) {
        ucs_error("failed to add host memory %p to the ze_copy import registry",
                  address);
        status = UCS_ERR_NO_MEMORY;
        goto err_free;
    }

    if (khret == UCS_KH_PUT_KEY_PRESENT) {
        /* Another user of the extension released our import of this base,
           which stays recorded until its registrations are released */
        ucs_debug("import of host memory %p was released outside ze_copy",
                  address);
        status = UCS_ERR_ALREADY_EXISTS;
        goto err_free;
    }

    ret = funcs->import(funcs->driver, address, length);
    if (ret == ZE_RESULT_ERROR_UNSUPPORTED_FEATURE) {
        status = UCS_ERR_UNSUPPORTED;
        goto err_del;
    } else if (ret != ZE_RESULT_SUCCESS) {
        ucs_log(log_level, "zexDriverImportExternalPointer(%p, %zu) failed: "
                "0x%x", address, length, ret);
        status = UCS_ERR_IO_ERROR;
        goto err_del;
    }

    import->address    = address;
    import->refcount   = 1;
    kh_val(hash, iter) = import;
    *import_p          = import;
    return UCS_OK;

err_del:
    kh_del(ze_copy_import, hash, iter);
err_free:
    ucs_free(import);
    return status;
}

/*
 * Resolve a registered host range to an import. The driver rejects a range
 * that shares only one end with an existing import, so such a range stays
 * unimported: copies still work, the driver just maps the range per copy.
 */
static void uct_ze_copy_import_register(void *address, size_t length,
                                        uint64_t flags, uct_mem_h *memh_p)
{
    uct_ze_copy_import_registry_t *registry = &uct_ze_copy_imports;
    void *end                               = UCS_PTR_BYTE_OFFSET(address,
                                                                  length - 1);
    ucs_status_t status                     = UCS_OK;
    ucs_log_level_t log_level;
    uct_ze_copy_import_t *import;
    void *base_start, *base_end;

    *memh_p   = UCT_ZE_COPY_DUMMY_MEMH;
    log_level = (flags & UCT_MD_MEM_FLAG_HIDE_ERRORS) ? UCS_LOG_LEVEL_DEBUG :
                                                        UCS_LOG_LEVEL_DIAG;

    pthread_mutex_lock(&registry->lock);
    if (!registry->enabled) {
        goto out_unlock;
    }

    status = uct_ze_copy_import_get_base(address, &base_start);
    if (status != UCS_OK) {
        goto out;
    }

    status = uct_ze_copy_import_get_base(end, &base_end);
    if (status != UCS_OK) {
        goto out;
    }

    if ((base_start == NULL) && (base_end == NULL)) {
        status = uct_ze_copy_import_create(address, length, log_level,
                                           &import);
        if (status == UCS_OK) {
            *memh_p = import;
        }
    } else if (base_start == base_end) {
        import = uct_ze_copy_import_lookup(base_start);
        if (import != NULL) {
            ++import->refcount;
            *memh_p = import;
        } else {
            /* Imported by the application: usable, but not ours to release */
            ucs_trace("host memory %p length %zu is inside a foreign import "
                      "at %p", address, length, base_start);
        }
    } else {
        ucs_debug("host memory %p length %zu partially overlaps the imports "
                  "at %p and %p, left unimported",
                  address, length, base_start, base_end);
    }

out:
    if (status == UCS_ERR_UNSUPPORTED) {
        uct_ze_copy_import_disable();
    }
out_unlock:
    pthread_mutex_unlock(&registry->lock);
}

static void uct_ze_copy_import_release(uct_ze_copy_import_t *import)
{
    uct_ze_copy_import_registry_t *registry = &uct_ze_copy_imports;
    const uct_ze_copy_host_import_t *funcs  = &registry->funcs;
    khiter_t iter;
    ze_result_t ret;

    pthread_mutex_lock(&registry->lock);
    ucs_assert(import->refcount > 0);
    if (--import->refcount > 0) {
        pthread_mutex_unlock(&registry->lock);
        return;
    }

    iter = kh_get(ze_copy_import, &registry->hash, (uintptr_t)import->address);
    ucs_assert(iter != kh_end(&registry->hash));
    kh_del(ze_copy_import, &registry->hash, iter);

    /* The driver fails a release only when no import contains the address
       (already released) or when it has no host pointer manager at all, so
       the import is gone either way and the record goes with it */
    ret = funcs->release(funcs->driver, import->address);
    if (ret != ZE_RESULT_SUCCESS) {
        ucs_diag("zexDriverReleaseImportedPointer(%p) failed: 0x%x",
                 import->address, ret);
    }
    pthread_mutex_unlock(&registry->lock);

    ucs_free(import);
}

static int
uct_ze_copy_md_is_ze_memory(uct_ze_copy_md_t *md, const void *address)
{
    ze_memory_allocation_properties_t props = {
        .stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES
    };
    ze_result_t ret;

    ret = zeMemGetAllocProperties(md->ze_context, address, &props, NULL);
    return (ret == ZE_RESULT_SUCCESS) && (props.type != ZE_MEMORY_TYPE_UNKNOWN);
}

static ucs_status_t uct_ze_copy_md_query(uct_md_h md, uct_md_attr_v2_t *md_attr)
{
    uct_md_base_md_query(md_attr);
    md_attr->flags            = UCT_MD_FLAG_REG | UCT_MD_FLAG_ALLOC;
    md_attr->reg_mem_types    = UCS_BIT(UCS_MEMORY_TYPE_HOST) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_HOST) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_DEVICE) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_MANAGED);
    md_attr->cache_mem_types  = UCS_BIT(UCS_MEMORY_TYPE_ZE_HOST) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_DEVICE) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_MANAGED);
    md_attr->alloc_mem_types  = UCS_BIT(UCS_MEMORY_TYPE_ZE_HOST) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_DEVICE) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_MANAGED);
    md_attr->access_mem_types = UCS_BIT(UCS_MEMORY_TYPE_HOST) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_HOST) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_DEVICE) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_MANAGED);
    md_attr->detect_mem_types = UCS_BIT(UCS_MEMORY_TYPE_ZE_HOST) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_DEVICE) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_MANAGED);
    md_attr->dmabuf_mem_types = UCS_BIT(UCS_MEMORY_TYPE_ZE_HOST) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_DEVICE);
    md_attr->max_alloc        = SIZE_MAX;
    return UCS_OK;
}

static ucs_status_t
uct_ze_copy_mem_alloc(uct_md_h tl_md, size_t *length_p, void **address_p,
                      ucs_memory_type_t mem_type, ucs_sys_device_t sys_dev,
                      unsigned flags, const char *alloc_name, uct_mem_h *memh_p)
{
    uct_ze_copy_md_t *md                = ucs_derived_of(tl_md, uct_ze_copy_md_t);
    ze_host_mem_alloc_desc_t host_desc  = {
        .stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC
    };
    ze_device_mem_alloc_desc_t dev_desc = {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC
    };
    size_t alignment                    = ucs_get_page_size();
    ucs_status_t status;

    switch (mem_type) {
    case UCS_MEMORY_TYPE_ZE_HOST:
        status = UCT_ZE_FUNC_LOG_ERR(zeMemAllocHost(md->ze_context, &host_desc,
                                                    *length_p, alignment,
                                                    address_p));
        break;
    case UCS_MEMORY_TYPE_ZE_DEVICE:
        status = UCT_ZE_FUNC_LOG_ERR(zeMemAllocDevice(md->ze_context, &dev_desc,
                                                      *length_p, alignment,
                                                      md->ze_device,
                                                      address_p));
        break;
    case UCS_MEMORY_TYPE_ZE_MANAGED:
        status = UCT_ZE_FUNC_LOG_ERR(zeMemAllocShared(md->ze_context, &dev_desc,
                                                      &host_desc, *length_p,
                                                      alignment, md->ze_device,
                                                      address_p));
        break;
    default:
        ucs_debug("unsupported mem_type: %d", mem_type);
        status = UCS_ERR_UNSUPPORTED;
        break;
    }

    if (status == UCS_OK) {
        *memh_p = *address_p;
    }

    return status;
}

static ucs_status_t uct_ze_copy_mem_free(uct_md_h tl_md, uct_mem_h memh)
{
    uct_ze_copy_md_t *md = ucs_derived_of(tl_md, uct_ze_copy_md_t);

    return UCT_ZE_FUNC_LOG_ERR(zeMemFree(md->ze_context, (void*)memh));
}

static ucs_status_t
uct_ze_copy_rkey_unpack(uct_component_t *component, const void *rkey_buffer,
                        const uct_rkey_unpack_params_t *params,
                        uct_rkey_t *rkey_p, void **handle_p)
{
    *handle_p = NULL;
    *rkey_p   = 0xdeadbeef;
    return UCS_OK;
}

static ucs_status_t
uct_ze_copy_mem_reg(uct_md_h tl_md, void *address, size_t length,
                    const uct_md_mem_reg_params_t *params, uct_mem_h *memh_p)
{
    uct_ze_copy_md_t *md = ucs_derived_of(tl_md, uct_ze_copy_md_t);
    uint64_t flags;

    /* Level Zero allocations are mapped for the device already; the import
       applies to host memory that comes from outside Level Zero */
    if (!md->host_import || (length == 0) ||
        uct_ze_copy_md_is_ze_memory(md, address)) {
        *memh_p = UCT_ZE_COPY_DUMMY_MEMH;
        return UCS_OK;
    }

    flags = UCT_MD_MEM_REG_FIELD_VALUE(params, flags, FIELD_FLAGS, 0);
    uct_ze_copy_import_register(address, length, flags, memh_p);
    return UCS_OK;
}

static ucs_status_t
uct_ze_copy_mem_dereg(uct_md_h md, const uct_md_mem_dereg_params_t *params)
{
    UCT_MD_MEM_DEREG_CHECK_PARAMS(params, 0);

    if (params->memh != UCT_ZE_COPY_DUMMY_MEMH) {
        uct_ze_copy_import_release(params->memh);
    }

    return UCS_OK;
}

static void uct_ze_copy_md_close(uct_md_h uct_md)
{
    uct_ze_copy_md_t *md = ucs_derived_of(uct_md, uct_ze_copy_md_t);

    zeContextDestroy(md->ze_context);
    ucs_free(md);
}

static ucs_status_t
uct_ze_copy_md_query_attributes(uct_md_h md, const void *addr, size_t length,
                                ucs_memory_info_t *mem_info, int *dmabuf_fd)
{
    uct_ze_copy_md_t *ze_md                  = ucs_derived_of(md, uct_ze_copy_md_t);
    ze_external_memory_export_fd_t export_fd = {
        .stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD,
        .flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
        .fd    = UCT_DMABUF_FD_INVALID
    };
    ze_memory_allocation_properties_t props  = {
        .stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES,
        .pNext = dmabuf_fd ? &export_fd : NULL
    };
    ze_device_handle_t ze_device             = NULL;
    ze_result_t ret;
    void *base_address;
    size_t alloc_length;

    ret = zeMemGetAllocProperties(ze_md->ze_context, addr, &props, &ze_device);
    if ((ret != ZE_RESULT_SUCCESS) || (props.type == ZE_MEMORY_TYPE_UNKNOWN)) {
        return UCS_ERR_INVALID_ADDR;
    }

    ret = zeMemGetAddressRange(ze_md->ze_context, addr, &base_address,
                               &alloc_length);
    if (ret != ZE_RESULT_SUCCESS) {
        return UCS_ERR_INVALID_ADDR;
    }

    if (props.type == ZE_MEMORY_TYPE_HOST) {
        mem_info->type = UCS_MEMORY_TYPE_ZE_HOST;
    } else if (props.type == ZE_MEMORY_TYPE_DEVICE) {
        mem_info->type = UCS_MEMORY_TYPE_ZE_DEVICE;
    } else {
        mem_info->type = UCS_MEMORY_TYPE_ZE_MANAGED;
    }
    mem_info->base_address = base_address;
    mem_info->alloc_length = alloc_length;
    mem_info->sys_dev      = uct_ze_base_get_sys_dev_from_handle(ze_device);
    if (dmabuf_fd) {
        *dmabuf_fd = export_fd.fd;
    }
    return UCS_OK;
}

static ucs_status_t uct_ze_copy_md_mem_query(uct_md_h md, const void *addr,
                                             const size_t length,
                                             uct_md_mem_attr_v2_t *mem_attr_p)
{
    int dmabuf_fd    = UCT_DMABUF_FD_INVALID;
    int *dmabuf_fd_p = NULL;
    ucs_memory_info_t mem_info;
    ucs_status_t status;

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_DMABUF_FD) {
        mem_attr_p->dmabuf_fd = UCT_DMABUF_FD_INVALID;
        dmabuf_fd_p           = &dmabuf_fd;
    }

    status = uct_ze_copy_md_query_attributes(md, addr, length, &mem_info,
                                             dmabuf_fd_p);
    if (status != UCS_OK) {
        return status;
    }

    ucs_memtype_cache_update(mem_info.base_address, mem_info.alloc_length,
                             mem_info.type, mem_info.sys_dev,
                             UCS_MEM_FLAG_REGISTRABLE);

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_MEM_TYPE) {
        mem_attr_p->mem_type = mem_info.type;
    }

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_SYS_DEV) {
        mem_attr_p->sys_dev = mem_info.sys_dev;
    }

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_BASE_ADDRESS) {
        mem_attr_p->base_address = mem_info.base_address;
    }

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_ALLOC_LENGTH) {
        mem_attr_p->alloc_length = mem_info.alloc_length;
    }

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_DMABUF_FD) {
        if (dmabuf_fd == UCT_DMABUF_FD_INVALID) {
            mem_attr_p->dmabuf_fd = UCT_DMABUF_FD_INVALID;
        } else {
            mem_attr_p->dmabuf_fd = dup(dmabuf_fd);
            if (mem_attr_p->dmabuf_fd < 0) {
                return UCS_ERR_IO_ERROR;
            }

            /* NOTE: Do not close(dmabuf_fd) here. Level Zero caches this fd
             * internally per allocation. Closing it can invalidate the cache
             * and lead to fd reuse conflicts with other transports.
             */
        }
    }

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_DMABUF_OFFSET) {
        mem_attr_p->dmabuf_offset = UCS_PTR_BYTE_DIFF(mem_info.base_address,
                                                      addr);
    }

    return UCS_OK;
}

static ucs_status_t
uct_ze_copy_md_detect_memory_type(uct_md_h md, const void *addr, size_t length,
                                  ucs_memory_type_t *mem_type_p)
{
    uct_md_mem_attr_v2_t mem_attr;
    ucs_status_t status;

    mem_attr.field_mask = UCT_MD_MEM_ATTR_V2_FIELD_MEM_TYPE;
    status = uct_ze_copy_md_mem_query(md, addr, length, &mem_attr);
    if (status != UCS_OK) {
        return status;
    }

    *mem_type_p = mem_attr.mem_type;
    return UCS_OK;
}

static uct_md_ops_t md_ops = {
    .close              = uct_ze_copy_md_close,
    .query              = uct_ze_copy_md_query,
    .mem_alloc          = uct_ze_copy_mem_alloc,
    .mem_free           = uct_ze_copy_mem_free,
    .mem_advise         = (uct_md_mem_advise_func_t)ucs_empty_function_return_unsupported,
    .mem_reg            = uct_ze_copy_mem_reg,
    .mem_dereg          = uct_ze_copy_mem_dereg,
    .mem_query          = uct_ze_copy_md_mem_query,
    .mkey_pack          = (uct_md_mkey_pack_func_t)ucs_empty_function_return_success,
    .mem_attach         = (uct_md_mem_attach_func_t)ucs_empty_function_return_unsupported,
    .detect_memory_type = uct_ze_copy_md_detect_memory_type,
    .mem_elem_pack      = (uct_md_mem_elem_pack_func_t)ucs_empty_function_return_unsupported,
    .mem_elem_release   = (uct_md_mem_elem_release_func_t)ucs_empty_function
};

static ucs_status_t
uct_ze_copy_md_open(uct_component_h component, const char *md_name,
                    const uct_md_config_t *md_config, uct_md_h *md_p)
{
    uct_ze_copy_md_config_t *config = ucs_derived_of(md_config,
                                                     uct_ze_copy_md_config_t);
    uct_ze_copy_md_t *md;
    ze_driver_handle_t ze_driver;
    ze_context_desc_t context_desc = {};
    ze_result_t ret;
    const uct_ze_subdevice_t *subdevice;

    ze_driver = uct_ze_base_get_driver();
    if (ze_driver == NULL) {
        return UCS_ERR_NO_DEVICE;
    }

    md = ucs_malloc(sizeof(uct_ze_copy_md_t), "uct_ze_copy_md_t");
    if (NULL == md) {
        ucs_error("Failed to allocate memory for uct_ze_copy_md_t");
        return UCS_ERR_NO_MEMORY;
    }

    /* Get sub-device by device_ordinal (global sub-device ID) */
    subdevice = uct_ze_base_get_subdevice_by_global_id(config->device_ordinal);
    if (subdevice == NULL) {
        ucs_error("Failed to get sub-device at ordinal %d", config->device_ordinal);
        ucs_free(md);
        return UCS_ERR_NO_DEVICE;
    }

    /* Get the actual device handle from the sub-device */
    md->ze_device = uct_ze_base_get_device_handle_from_subdevice(subdevice);
    if (md->ze_device == NULL) {
        ucs_error("Failed to get device handle for sub-device %d", config->device_ordinal);
        ucs_free(md);
        return UCS_ERR_NO_DEVICE;
    }

    ret = zeContextCreate(ze_driver, &context_desc, &md->ze_context);
    if (ret != ZE_RESULT_SUCCESS) {
        ucs_error("zeContextCreate failed with error %x", ret);
        ucs_free(md);
        return UCS_ERR_NO_DEVICE;
    }

    md->host_import     = config->host_import &&
                          (uct_ze_copy_md_get_host_import() != NULL);
    md->super.ops       = &md_ops;
    md->super.component = &uct_ze_copy_component;
    ucs_debug("ze_copy md %p: host memory import %s", md,
              md->host_import ? "enabled" : "disabled");

    *md_p = (uct_md_h)md;
    return UCS_OK;
}

uct_component_t uct_ze_copy_component = {
    .query_md_resources = uct_ze_base_query_md_resources,
    .md_open            = uct_ze_copy_md_open,
    .cm_open            = (uct_component_cm_open_func_t)ucs_empty_function_return_unsupported,
    .rkey_unpack        = uct_ze_copy_rkey_unpack,
    .rkey_ptr           = (uct_component_rkey_ptr_func_t)ucs_empty_function_return_unsupported,
    .rkey_release       = (uct_component_rkey_release_func_t)ucs_empty_function_return_success,
    .rkey_compare       = uct_base_rkey_compare,
    .name               = "ze_cpy",
    .md_config = {
        .name       = "ze-copy memory domain",
        .prefix     = "ZE_COPY_",
        .table      = uct_ze_copy_md_config_table,
        .size       = sizeof(uct_ze_copy_md_config_t),
    },
    .cm_config      = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
    .tl_list        = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_ze_copy_component),
    .flags          = 0,
    .md_vfs_init    = (uct_component_md_vfs_init_func_t)ucs_empty_function
};
UCT_COMPONENT_REGISTER(&uct_ze_copy_component);

/* Imports still held at exit end with the process. The driver is not called
   here, since its own state may already be destroyed. */
UCS_STATIC_CLEANUP
{
    uct_ze_copy_import_t *import;

    kh_foreach_value(&uct_ze_copy_imports.hash, import, ucs_free(import));
    kh_destroy_inplace(ze_copy_import, &uct_ze_copy_imports.hash);
    pthread_mutex_destroy(&uct_ze_copy_imports.lock);
}
