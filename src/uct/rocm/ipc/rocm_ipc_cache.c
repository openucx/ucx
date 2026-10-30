/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018. ALL RIGHTS RESERVED.
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rocm_ipc_cache.h"
#include "rocm_ipc_iface.h"
#include "../base/rocm_base.h"

#include <ucm/api/ucm.h>
#include <ucs/arch/atomic.h>
#include <ucs/datastruct/khash.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/profile/profile.h>
#include <ucs/sys/ptr_arith.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/type/init_once.h>
#include <ucs/type/rwlock.h>
#include <ucs/type/spinlock.h>

#include <stdbool.h>


typedef struct uct_rocm_ipc_cache_hash_key {
    pid_t        pid; /* PID of the process that owns the memory */
    ucs_sys_ns_t pid_ns; /* PID namespace of the owner process */
    int          dev_num; /* Device number the memory was allocated on */
} uct_rocm_ipc_cache_hash_key_t;

static UCS_F_ALWAYS_INLINE int
uct_rocm_ipc_cache_hash_equal(uct_rocm_ipc_cache_hash_key_t key1,
                              uct_rocm_ipc_cache_hash_key_t key2)
{
    return (key1.pid == key2.pid) && (key1.pid_ns == key2.pid_ns) &&
           (key1.dev_num == key2.dev_num);
}

static UCS_F_ALWAYS_INLINE khint32_t
uct_rocm_ipc_cache_hash_func(uct_rocm_ipc_cache_hash_key_t key)
{
    uint64_t value = key.pid ^ ((uint64_t)key.pid_ns << 32) ^
                     ((uint32_t)key.dev_num << 24);
    return kh_int64_hash_func(value);
}

KHASH_INIT(rocm_ipc_rem_cache, uct_rocm_ipc_cache_hash_key_t,
           uct_rocm_ipc_cache_t*, 1, uct_rocm_ipc_cache_hash_func,
           uct_rocm_ipc_cache_hash_equal);

/* Per-peer IPC handle caches. Caches are kept per (pid, pid_ns, dev_num). */
typedef struct uct_rocm_ipc_remote_cache {
    khash_t(rocm_ipc_rem_cache) hash;
    ucs_rw_spinlock_t           lock;
} uct_rocm_ipc_remote_cache_t;

static uct_rocm_ipc_remote_cache_t uct_rocm_ipc_remote_cache;

/* Runs before the registration cache (priority 1000). */
#define UCT_ROCM_IPC_MEM_EVENT_PRIO 100

typedef struct uct_rocm_ipc_export {
    ucs_list_link_t list;
    uintptr_t       address;
    size_t          length;
    hsa_signal_t    signal;
    hsa_amd_ipc_signal_t ipc_signal;
} uct_rocm_ipc_export_t;


KHASH_MAP_INIT_INT64(rocm_ipc_exports, uct_rocm_ipc_export_t*);

static khash_t(rocm_ipc_exports) uct_rocm_ipc_exports;
static ucs_list_link_t   uct_rocm_ipc_export_retired;
static ucs_spinlock_t    uct_rocm_ipc_export_lock;
static ucs_init_once_t   uct_rocm_ipc_export_handler_once =
        UCS_INIT_ONCE_INITIALIZER;


static int
uct_rocm_ipc_signal_is_empty(const hsa_amd_ipc_signal_t *ipc_signal)
{
    size_t i;

    for (i = 0; i < ucs_static_array_size(ipc_signal->handle); ++i) {
        if (ipc_signal->handle[i] != 0) {
            return 0;
        }
    }

    return 1;
}

static void uct_rocm_ipc_export_destroy(uct_rocm_ipc_export_t *exp)
{
    if (exp->signal.handle != 0) {
        if (hsa_signal_destroy(exp->signal) != HSA_STATUS_SUCCESS) {
            ucs_warn("failed to destroy rocm ipc export signal for %p",
                     (void*)exp->address);
        }
    }

    ucs_free(exp);
}

/* An importer attaches the IPC signal before it maps memory. Therefore, after
 * the free event has been dispatched, every mapping that can delay hipFree
 * already holds its own signal reference and any thread may reap the export. */
static void uct_rocm_ipc_export_reap(void)
{
    uct_rocm_ipc_export_t *exp, *tmp;
    ucs_list_link_t destroy_list;

    ucs_list_head_init(&destroy_list);

    ucs_spin_lock(&uct_rocm_ipc_export_lock);
    ucs_list_for_each_safe(exp, tmp, &uct_rocm_ipc_export_retired, list) {
        ucs_list_del(&exp->list);
        ucs_list_add_tail(&destroy_list, &exp->list);
    }
    ucs_spin_unlock(&uct_rocm_ipc_export_lock);

    ucs_list_for_each_safe(exp, tmp, &destroy_list, list) {
        uct_rocm_ipc_export_destroy(exp);
    }
}

static void uct_rocm_ipc_export_mem_free_cb(ucm_event_type_t event_type,
                                            ucm_event_t *event, void *arg)
{
    uct_rocm_ipc_export_t *exp;
    khiter_t iter;
    uintptr_t address;
    int reaped = 0;

    if ((event_type != UCM_EVENT_MEM_TYPE_FREE) ||
        (event->mem_type.mem_type != UCS_MEMORY_TYPE_ROCM)) {
        return;
    }

    address = (uintptr_t)event->mem_type.address;

    /* Importers acquire a signal reference before mapping memory, so the
     * exported signal can be destroyed after this callback on any thread. */
    ucs_spin_lock(&uct_rocm_ipc_export_lock);
    iter = kh_get(rocm_ipc_exports, &uct_rocm_ipc_exports, address);
    if (iter != kh_end(&uct_rocm_ipc_exports)) {
        exp = kh_val(&uct_rocm_ipc_exports, iter);
        kh_del(rocm_ipc_exports, &uct_rocm_ipc_exports, iter);
        hsa_signal_store_screlease(exp->signal, 0);
        ucs_list_add_tail(&uct_rocm_ipc_export_retired, &exp->list);
        reaped = 1;
        ucs_debug("rocm ipc signal export %p length %zu", (void*)exp->address,
                  exp->length);
    }
    ucs_spin_unlock(&uct_rocm_ipc_export_lock);

    if (reaped) {
        uct_rocm_ipc_export_reap();
    }
}

static void uct_rocm_ipc_export_install_handler(void)
{
    ucs_status_t status;

    status = ucm_set_event_handler(UCM_EVENT_MEM_TYPE_FREE,
                                   UCT_ROCM_IPC_MEM_EVENT_PRIO,
                                   uct_rocm_ipc_export_mem_free_cb, NULL);
    if (status != UCS_OK) {
        ucs_diag("rocm ipc free handler not installed: %s",
                 ucs_status_string(status));
    }
}

void uct_rocm_ipc_export_init(void)
{
    if (!uct_rocm_ipc_signal_invalidate_enabled()) {
        return;
    }

    UCS_INIT_ONCE(&uct_rocm_ipc_export_handler_once) {
        uct_rocm_ipc_export_install_handler();
    }
}

ucs_status_t uct_rocm_ipc_export_add(uintptr_t address, size_t length,
                                     hsa_amd_ipc_signal_t *ipc_signal_p)
{
    uct_rocm_ipc_export_t *exp, *existing;
    hsa_status_t hsa_status;
    khiter_t iter;
    int ret;

    if (!uct_rocm_ipc_signal_invalidate_enabled()) {
        memset(ipc_signal_p, 0, sizeof(*ipc_signal_p));
        return UCS_OK;
    }

    uct_rocm_ipc_export_reap();

    ucs_spin_lock(&uct_rocm_ipc_export_lock);
    iter = kh_get(rocm_ipc_exports, &uct_rocm_ipc_exports, address);
    if (iter != kh_end(&uct_rocm_ipc_exports)) {
        *ipc_signal_p = kh_val(&uct_rocm_ipc_exports, iter)->ipc_signal;
        ucs_spin_unlock(&uct_rocm_ipc_export_lock);
        return UCS_OK;
    }
    ucs_spin_unlock(&uct_rocm_ipc_export_lock);

    exp = ucs_malloc(sizeof(*exp), "uct_rocm_ipc_export_t");
    if (exp == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    hsa_status = hsa_amd_signal_create(1, 0, NULL, HSA_AMD_SIGNAL_IPC,
                                       &exp->signal);
    if (hsa_status != HSA_STATUS_SUCCESS) {
        ucs_debug("failed to create rocm ipc signal for %p length %zu: %d",
                  (void*)address, length, hsa_status);
        ucs_free(exp);
        return UCS_ERR_IO_ERROR;
    }

    hsa_status = hsa_amd_ipc_signal_create(exp->signal, &exp->ipc_signal);
    if (hsa_status != HSA_STATUS_SUCCESS) {
        ucs_debug("failed to export rocm ipc signal for %p length %zu: %d",
                  (void*)address, length, hsa_status);
        hsa_signal_destroy(exp->signal);
        ucs_free(exp);
        return UCS_ERR_IO_ERROR;
    }

    exp->address     = address;
    exp->length      = length;

    ucs_spin_lock(&uct_rocm_ipc_export_lock);
    iter = kh_put(rocm_ipc_exports, &uct_rocm_ipc_exports, address, &ret);
    if (ret == UCS_KH_PUT_FAILED) {
        ucs_spin_unlock(&uct_rocm_ipc_export_lock);
        uct_rocm_ipc_export_destroy(exp);
        return UCS_ERR_NO_MEMORY;
    }

    if (ret != UCS_KH_PUT_KEY_PRESENT) {
        kh_val(&uct_rocm_ipc_exports, iter) = exp;
        *ipc_signal_p = exp->ipc_signal;
    } else {
        existing = kh_val(&uct_rocm_ipc_exports, iter);
        *ipc_signal_p = existing->ipc_signal;
    }
    ucs_spin_unlock(&uct_rocm_ipc_export_lock);

    if (ret == UCS_KH_PUT_KEY_PRESENT) {
        uct_rocm_ipc_export_destroy(exp);
    }

    return UCS_OK;
}

static void uct_rocm_ipc_export_cleanup(void)
{
    uct_rocm_ipc_export_t *exp, *tmp;
    ucs_list_link_t destroy_list;

    UCS_CLEANUP_ONCE(&uct_rocm_ipc_export_handler_once) {
        ucm_unset_event_handler(UCM_EVENT_MEM_TYPE_FREE,
                                uct_rocm_ipc_export_mem_free_cb, NULL);
    }

    ucs_list_head_init(&destroy_list);
    ucs_spin_lock(&uct_rocm_ipc_export_lock);
    kh_foreach_value(&uct_rocm_ipc_exports, exp, {
        hsa_signal_store_screlease(exp->signal, 0);
        ucs_list_add_tail(&destroy_list, &exp->list);
    })
    kh_clear(rocm_ipc_exports, &uct_rocm_ipc_exports);
    ucs_list_for_each_safe(exp, tmp, &uct_rocm_ipc_export_retired, list) {
        ucs_list_del(&exp->list);
        ucs_list_add_tail(&destroy_list, &exp->list);
    }
    ucs_spin_unlock(&uct_rocm_ipc_export_lock);

    ucs_list_for_each_safe(exp, tmp, &destroy_list, list) {
        uct_rocm_ipc_export_destroy(exp);
    }
}

static ucs_pgt_dir_t *uct_rocm_ipc_cache_pgt_dir_alloc(const ucs_pgtable_t *pgtable)
{
    void *ptr;
    int ret;

    ret =  ucs_posix_memalign(&ptr,
                              ucs_max(sizeof(void *), UCS_PGT_ENTRY_MIN_ALIGN),
                              sizeof(ucs_pgt_dir_t), "rocm_ipc_cache_pgdir");
    return (ret == 0) ? ptr : NULL;
}

static void uct_rocm_ipc_cache_pgt_dir_release(const ucs_pgtable_t *pgtable,
                                               ucs_pgt_dir_t *dir)
{
    ucs_free(dir);
}

static void
uct_rocm_ipc_cache_region_collect_callback(const ucs_pgtable_t *pgtable,
                                           ucs_pgt_region_t *pgt_region,
                                           void *arg)
{
    ucs_list_link_t *list = arg;
    uct_rocm_ipc_cache_region_t *region;

    region = ucs_derived_of(pgt_region, uct_rocm_ipc_cache_region_t);
    ucs_list_add_tail(list, &region->list);
}

#ifdef HAVE_ROCM_VMM_TYPE
/* Import a cross-process VMM handle into the local address space.
 * Duplicates the sender's dmabuf fd via pidfd_open/pidfd_getfd, then
 * imports the handle, reserves a VA range, maps it, and grants RW access
 * to all local GPU agents so any copy engine can transfer to/from it. */
static void *uct_rocm_ipc_vmm_map(const uct_rocm_ipc_key_t *key)
{
    hsa_amd_vmem_alloc_handle_t vmem_handle;
    hsa_status_t status;
    void *va     = NULL;
    int local_fd = -1;
#if HAVE_DECL_SYS_PIDFD_GETFD
    int pidfd;
    int saved_errno;

    /* key->pid is meaningful only in the sender's PID namespace; a peer in a
     * different namespace would resolve to an unrelated local process. */
    if (key->pid_ns != ucs_sys_get_ns(UCS_SYS_NS_TYPE_PID)) {
        ucs_debug("cannot import vmm fd: sender pid %d is in a different pid "
                  "namespace (remote=%u local=%u)", key->pid, key->pid_ns,
                  ucs_sys_get_ns(UCS_SYS_NS_TYPE_PID));
        return NULL;
    }

    pidfd    = syscall(SYS_pidfd_open, key->pid, 0);
    local_fd = (pidfd >= 0) ?
               syscall(SYS_pidfd_getfd, pidfd, key->vmm.fd, 0) : -1;
    saved_errno = errno; /* capture before close(pidfd) clobbers errno */
    if (pidfd >= 0) {
        close(pidfd);
    }
    if (local_fd < 0) {
        errno = saved_errno;
        ucs_debug("failed to dup vmm dmabuf fd from pid %d fd %d: %m",
                  key->pid, key->vmm.fd);
        return NULL;
    }
#else
    ucs_debug("cannot import vmm dmabuf fd: SYS_pidfd_getfd is not available");
    return NULL;
#endif

    status = hsa_amd_vmem_import_shareable_handle(local_fd, &vmem_handle);
    close(local_fd);
    if (status != HSA_STATUS_SUCCESS) {
        ucs_error("failed to import vmm handle. addr:%p len:%lu",
                  (void *)key->address, key->length);
        return NULL;
    }

    status = hsa_amd_vmem_address_reserve(&va, key->length, 0, 0);
    if (status != HSA_STATUS_SUCCESS) {
        goto err_release_handle;
    }

    status = hsa_amd_vmem_map(va, key->length, 0, vmem_handle, 0);
    if (status != HSA_STATUS_SUCCESS) {
        goto err_free_va;
    }

    /* Grant RW access to all local GPU agents so any copy engine can use it */
    status = uct_rocm_base_vmm_set_gpu_access(va, key->length);
    if (status != HSA_STATUS_SUCCESS) {
        hsa_amd_vmem_unmap(va, key->length);
        goto err_free_va;
    }

    hsa_amd_vmem_handle_release(vmem_handle);
    return va;

err_free_va:
    hsa_amd_vmem_address_free(va, key->length);
err_release_handle:
    hsa_amd_vmem_handle_release(vmem_handle);
    ucs_error("failed to map vmm handle. addr:%p len:%lu",
              (void *)key->address, key->length);
    return NULL;
}
#endif

/* Release a mapping created for @a key: VMM mappings are unmapped and their
 * reserved VA freed; pool IPC handles are detached. */
static void uct_rocm_ipc_detach_mapping(const uct_rocm_ipc_key_t *key,
                                        void *mapped_addr)
{
#ifdef HAVE_ROCM_VMM_TYPE
    if (key->is_vmm) {
        if (hsa_amd_vmem_unmap(mapped_addr, key->length) != HSA_STATUS_SUCCESS) {
            ucs_warn("failed to vmem_unmap addr:%p", mapped_addr);
        }
        if (hsa_amd_vmem_address_free(mapped_addr, key->length) !=
            HSA_STATUS_SUCCESS) {
            ucs_warn("failed to free vmm addr:%p", mapped_addr);
        }
        return;
    }
#endif
    if (hsa_amd_ipc_memory_detach(mapped_addr) != HSA_STATUS_SUCCESS) {
        ucs_warn("failed to unmap addr:%p", mapped_addr);
    }
}

static UCS_F_ALWAYS_INLINE int
uct_rocm_ipc_key_equal(const uct_rocm_ipc_key_t *key1,
                       const uct_rocm_ipc_key_t *key2)
{
#ifdef HAVE_ROCM_VMM_TYPE
    /* reg_id is stable per live VMM mapping (from the vmm registry), so
     * concurrent re-registrations of the same chunk compare equal and reuse
     * the cached mapping instead of unmapping one an in-flight copy may read. */
    if (key1->is_vmm || key2->is_vmm) {
        return (key1->is_vmm == key2->is_vmm) && (key1->pid == key2->pid) &&
               (key1->vmm.reg_id == key2->vmm.reg_id);
    }
#endif
    return memcmp(&key1->ipc, &key2->ipc, sizeof(key1->ipc)) == 0;
}

static void
uct_rocm_ipc_cache_region_unref(uct_rocm_ipc_cache_region_t *region)
{
    if (ucs_atomic_fsub32(&region->refcount, 1) == 1) {
        ucs_free(region);
    }
}

/* Caller holds the remote-cache write lock. The claim winner detaches an
 * unpinned mapping immediately so the exporter's hipFree can finish. It
 * destroys the IPC signal only when no handler has taken ownership of it. */
static int
uct_rocm_ipc_cache_region_drop_locked(uct_rocm_ipc_cache_region_t *region,
                                      ucs_list_link_t *deferred)
{
    void *mapped_addr;
    ucs_status_t status;

    if (ucs_atomic_cswap32(&region->deactivate, 0, 1) != 0) {
        return 0;
    }

    /* A pinned mapping is detached by the last unpin. */
    mapped_addr = NULL;
    if (region->pin_count == 0) {
        mapped_addr         = region->mapped_addr;
        region->mapped_addr = NULL;
    }

    /* A registered handler owns the signal and destroys it when it runs. */
    if (!region->handler_armed) {
        region->retired_signal = region->signal;
        region->signal.handle = 0;
    }

    if (region->in_pgtable) {
        status = ucs_pgtable_remove(&region->cache->pgtable, &region->super);
        if (status != UCS_OK) {
            ucs_fatal("%s: failed to remove address:%p from cache (%s)",
                      region->cache->name, (void*)region->key.address,
                      ucs_status_string(status));
        }
        region->in_pgtable = 0;
    }

    if (mapped_addr != NULL) {
        uct_rocm_ipc_detach_mapping(&region->key, mapped_addr);
    }

    ucs_trace("rocm ipc cache drop addr:%p mapped:%p",
              (void*)region->key.address, mapped_addr);
    ucs_list_add_tail(deferred, &region->list);
    return 1;
}

/* Caller holds the remote-cache write lock. A different or overlapping IPC
 * handle proves that this mapping is stale, so wake its handler to release
 * the reference it holds on the region and its signal. */
static void
uct_rocm_ipc_cache_region_stale_drop_locked(uct_rocm_ipc_cache_region_t *region,
                                            ucs_list_link_t *deferred)
{
    if (region->handler_armed) {
        /* Wake the handler before the region is deactivated, so that it takes
         * the locked path and does not race with this store. */
        hsa_signal_store_screlease(region->signal, 0);
    }

    uct_rocm_ipc_cache_region_drop_locked(region, deferred);
}

static void
uct_rocm_ipc_cache_region_signal_destroy(uct_rocm_ipc_cache_region_t *region,
                                         hsa_signal_t signal)
{
    if (signal.handle == 0) {
        return;
    }

    if (hsa_signal_destroy(signal) != HSA_STATUS_SUCCESS) {
        ucs_warn("failed to destroy rocm ipc cache signal for %p",
                 (void*)region->key.address);
    }
}

static void uct_rocm_ipc_cache_region_finalize(ucs_list_link_t *deferred)
{
    uct_rocm_ipc_cache_region_t *region, *tmp;

    ucs_list_for_each_safe(region, tmp, deferred, list) {
        ucs_list_del(&region->list);
        uct_rocm_ipc_cache_region_signal_destroy(region,
                                                 region->retired_signal);
        region->retired_signal.handle = 0;
        uct_rocm_ipc_cache_region_unref(region);
    }
}

static bool uct_rocm_ipc_cache_signal_cb(hsa_signal_value_t value, void *arg)
{
    uct_rocm_ipc_cache_region_t *region = arg;
    ucs_list_link_t deferred;
    hsa_signal_t signal = {.handle = 0};
    int claimed;

    if (ucs_atomic_cswap32(&region->handler_done, 0, 1) != 0) {
        return false;
    }

    ucs_list_head_init(&deferred);
    if (ucs_atomic_cswap32(&region->deactivate, 0, 0) != 0) {
        /* A stale or teardown drop already unlinked the region. */
        signal                = region->signal;
        region->signal.handle = 0;
        claimed               = 0;
    } else {
        ucs_rw_spinlock_write_lock(&uct_rocm_ipc_remote_cache.lock);
        claimed = uct_rocm_ipc_cache_region_drop_locked(region, &deferred);
        if (region->signal.handle != 0) {
            signal                = region->signal;
            region->signal.handle = 0;
        }
        ucs_rw_spinlock_write_unlock(&uct_rocm_ipc_remote_cache.lock);
    }

    uct_rocm_ipc_cache_region_signal_destroy(region, signal);
    if (claimed) {
        uct_rocm_ipc_cache_region_finalize(&deferred);
    }
    uct_rocm_ipc_cache_region_unref(region);
    return false;
}

static void
uct_rocm_ipc_cache_purge_collect_cb(const ucs_pgtable_t *pgtable,
                                    ucs_pgt_region_t *pgt_region, void *arg)
{
    ucs_list_link_t *list = arg;
    uct_rocm_ipc_cache_region_t *region;

    /* pgtable_purge has already unlinked the region. */
    region = ucs_derived_of(pgt_region, uct_rocm_ipc_cache_region_t);
    region->in_pgtable = 0;
    ucs_list_add_tail(list, &region->list);
}

static void uct_rocm_ipc_cache_purge(uct_rocm_ipc_cache_t *cache)
{
    uct_rocm_ipc_cache_region_t *region, *tmp;
    ucs_list_link_t region_list, deferred;

    ucs_list_head_init(&region_list);
    ucs_list_head_init(&deferred);

    ucs_rw_spinlock_write_lock(&uct_rocm_ipc_remote_cache.lock);
    ucs_pgtable_purge(&cache->pgtable, uct_rocm_ipc_cache_purge_collect_cb,
                      &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        ucs_list_del(&region->list);
        uct_rocm_ipc_cache_region_stale_drop_locked(region, &deferred);
    }
    ucs_rw_spinlock_write_unlock(&uct_rocm_ipc_remote_cache.lock);

    uct_rocm_ipc_cache_region_finalize(&deferred);
    ucs_trace("%s: rocm ipc cache purged", cache->name);
}

static void
uct_rocm_ipc_cache_invalidate_regions(uct_rocm_ipc_cache_t *cache, void *from,
                                      void *to, ucs_list_link_t *deferred)
{
    ucs_list_link_t region_list;
    uct_rocm_ipc_cache_region_t *region, *tmp;

    ucs_list_head_init(&region_list);
    ucs_pgtable_search_range(&cache->pgtable, (ucs_pgt_addr_t)from,
                             (ucs_pgt_addr_t)to - 1,
                             uct_rocm_ipc_cache_region_collect_callback,
                             &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        ucs_list_del(&region->list);
        uct_rocm_ipc_cache_region_stale_drop_locked(region, deferred);
    }
    ucs_trace("%s: closed memhandles in the range [%p..%p]", cache->name, from,
              to);
}

/* Caller holds the remote-cache write lock. The mapped region is returned
 * through @a region_p. A new region is also returned through
 * @a watch_region_p so the caller can arm its IPC signal after releasing the
 * lock. @a deferred receives regions whose signals must be destroyed. */
static ucs_status_t
uct_rocm_ipc_cache_map_region(uct_rocm_ipc_cache_t *cache,
                              uct_rocm_ipc_key_t *key,
                              uct_rocm_ipc_cache_region_t **region_p,
                              uct_rocm_ipc_cache_region_t **watch_region_p,
                              ucs_list_link_t *deferred)
{
    hsa_signal_t local_signal = {.handle = 0};
    ucs_status_t status;
    ucs_pgt_region_t *pgt_region;
    uct_rocm_ipc_cache_region_t *region;
    hsa_status_t hsa_status;
    void *mapped_addr;
    int ret;

    *watch_region_p = NULL;

    /* A per-cache lock would be needed only if the caller holds no lock or
     * just a read lock on the global remote-cache lock, allowing concurrent
     * access to this page table. */
    pgt_region = UCS_PROFILE_CALL(ucs_pgtable_lookup, &cache->pgtable,
                                  key->address);
    if (ucs_likely(pgt_region != NULL)) {
        region = ucs_derived_of(pgt_region, uct_rocm_ipc_cache_region_t);
        if (uct_rocm_ipc_key_equal(key, &region->key)) {
            if ((region->signal.handle == 0) ||
                (hsa_signal_load_scacquire(region->signal) > 0)) {
                ucs_trace("%s: rocm_ipc cache hit addr:%p size:%lu region:"
                          UCS_PGT_REGION_FMT,
                          cache->name, (void*)key->address, key->length,
                          UCS_PGT_REGION_ARG(&region->super));

                *region_p = region;
                return UCS_OK;
            }

            ucs_trace("%s: rocm_ipc cache remove invalidated region:"
                      UCS_PGT_REGION_FMT,
                      cache->name, UCS_PGT_REGION_ARG(&region->super));
        } else {
            ucs_trace("%s: rocm_ipc cache remove stale region:"
                      UCS_PGT_REGION_FMT " new_addr:%p new_size:%lu",
                      cache->name, UCS_PGT_REGION_ARG(&region->super),
                      (void*)key->address, key->length);
        }

        uct_rocm_ipc_cache_region_stale_drop_locked(region, deferred);
    }

    if (uct_rocm_ipc_signal_invalidate_enabled() &&
        !uct_rocm_ipc_signal_is_empty(&key->ipc_signal)) {
        /* Acquire the signal reference before mapping memory. Once the
         * mapping exists, the exporter may be blocked in hipFree and may
         * destroy its signal reference after dispatching the free event. */
        hsa_status = hsa_amd_ipc_signal_attach(&key->ipc_signal,
                                               &local_signal);
        if (hsa_status != HSA_STATUS_SUCCESS) {
            if (hsa_status == HSA_STATUS_ERROR_INVALID_ARGUMENT) {
                ucs_debug("%s: rocm ipc buffer %p already released by the owner",
                          cache->name, (void*)key->address);
                status = UCS_ERR_INVALID_ADDR;
            } else {
                ucs_error("%s: failed to attach rocm ipc signal for %p: %d",
                          cache->name, (void*)key->address, hsa_status);
                status = UCS_ERR_IO_ERROR;
            }
            goto err;
        }

        if (hsa_signal_load_scacquire(local_signal) <= 0) {
            ucs_debug("%s: rocm ipc buffer %p already released by the owner",
                      cache->name, (void*)key->address);
            status = UCS_ERR_INVALID_ADDR;
            goto err_destroy_signal;
        }
    }

#ifdef HAVE_ROCM_VMM_TYPE
    if (key->is_vmm) {
        mapped_addr = uct_rocm_ipc_vmm_map(key);
        if (mapped_addr == NULL) {
            status = UCS_ERR_INVALID_ADDR;
            goto err_destroy_signal;
        }
    } else
#endif
    {
        hsa_status = hsa_amd_ipc_memory_attach(&key->ipc, key->length, 0, NULL,
                                               &mapped_addr);
        if (ucs_unlikely(hsa_status != HSA_STATUS_SUCCESS)) {
            ucs_error("%s: failed to open ipc mem handle. addr:%p len:%lu "
                      "hsa_status:%d", cache->name, (void*)key->address,
                      key->length, hsa_status);
            status = UCS_ERR_INVALID_ADDR;
            goto err_destroy_signal;
        }
    }

    ret = ucs_posix_memalign((void**)&region,
                             ucs_max(sizeof(void*), UCS_PGT_ENTRY_MIN_ALIGN),
                             sizeof(uct_rocm_ipc_cache_region_t),
                             "uct_rocm_ipc_cache_region");
    if (ret != 0) {
        ucs_warn("failed to allocate uct_rocm_ipc_cache region");
        uct_rocm_ipc_detach_mapping(key, mapped_addr);
        status = UCS_ERR_NO_MEMORY;
        goto err_destroy_signal;
    }

    region->super.start           =
            ucs_align_down_pow2(key->address, UCS_PGT_ADDR_ALIGN);
    region->super.end             =
            ucs_align_up_pow2(key->address + key->length, UCS_PGT_ADDR_ALIGN);
    region->key                   = *key;
    region->mapped_addr           = mapped_addr;
    region->cache                 = cache;
    region->signal                = local_signal;
    region->retired_signal.handle = 0;
    region->refcount              = 1;
    region->deactivate            = 0;
    region->handler_done          = 0;
    region->handler_armed         = 0;
    region->pin_count             = 0;
    region->in_pgtable            = 0;

    status = UCS_PROFILE_CALL(ucs_pgtable_insert, &cache->pgtable,
                              &region->super);
    if (status == UCS_ERR_ALREADY_EXISTS) {
        /* Overlap means the source freed the memory. Drop it and retry. */
        uct_rocm_ipc_cache_invalidate_regions(cache, (void*)region->super.start,
                                              (void*)region->super.end,
                                              deferred);
        status = UCS_PROFILE_CALL(ucs_pgtable_insert, &cache->pgtable,
                                  &region->super);
    }
    if (status != UCS_OK) {
        ucs_error("%s: failed to insert region:" UCS_PGT_REGION_FMT
                  " size:%lu :%s",
                  cache->name, UCS_PGT_REGION_ARG(&region->super), key->length,
                  ucs_status_string(status));
        uct_rocm_ipc_detach_mapping(key, mapped_addr);
        ucs_free(region);
        goto err_destroy_signal;
    }

    region->in_pgtable = 1;
    *region_p          = region;
    if (local_signal.handle != 0) {
        *watch_region_p = region;
    }
    ucs_trace("%s: rocm_ipc cache new region:" UCS_PGT_REGION_FMT " size:%lu",
              cache->name, UCS_PGT_REGION_ARG(&region->super), key->length);
    return UCS_OK;

err_destroy_signal:
    if ((local_signal.handle != 0) &&
        (hsa_signal_destroy(local_signal) != HSA_STATUS_SUCCESS)) {
        ucs_warn("%s: failed to destroy rocm ipc signal for %p", cache->name,
                 (void*)key->address);
    }
err:
    return status;
}

ucs_status_t uct_rocm_ipc_create_cache(uct_rocm_ipc_cache_t **cache,
                                       const char *name)
{
    ucs_status_t status;
    uct_rocm_ipc_cache_t *cache_desc;

    cache_desc = ucs_malloc(sizeof(uct_rocm_ipc_cache_t), "uct_rocm_ipc_cache_t");
    if (cache_desc == NULL) {
        ucs_error("failed to allocate memory for rocm_ipc cache");
        return UCS_ERR_NO_MEMORY;
    }

    status = ucs_pgtable_init(&cache_desc->pgtable,
                              uct_rocm_ipc_cache_pgt_dir_alloc,
                              uct_rocm_ipc_cache_pgt_dir_release);
    if (status != UCS_OK) {
        goto err;
    }

    cache_desc->name = ucs_strdup(name, "rocm_ipc_cache_name");
    if (cache_desc->name == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_cleanup_pgtable;
    }

    *cache = cache_desc;
    return UCS_OK;

err_cleanup_pgtable:
    ucs_pgtable_cleanup(&cache_desc->pgtable);
err:
    ucs_free(cache_desc);
    return status;
}

/* Look up the per-peer cache for the given hash key, creating it on first use.
 * Caller must hold the remote cache write lock. */
static ucs_status_t
uct_rocm_ipc_remote_cache_get(uct_rocm_ipc_cache_hash_key_t key,
                              uct_rocm_ipc_cache_t **cache_p)
{
    khash_t(rocm_ipc_rem_cache) *hash = &uct_rocm_ipc_remote_cache.hash;
    uct_rocm_ipc_cache_t *cache;
    char target_name[64];
    ucs_status_t status;
    khint_t it;
    int ret;

    it = kh_get(rocm_ipc_rem_cache, hash, key);
    if (ucs_likely(it != kh_end(hash))) {
        *cache_p = kh_val(hash, it);
        return UCS_OK;
    }

    it = kh_put(rocm_ipc_rem_cache, hash, key, &ret);
    if (ret == UCS_KH_PUT_FAILED) {
        ucs_error("failed to allocate rocm_ipc remote_cache hash entry");
        return UCS_ERR_NO_MEMORY;
    }

    ucs_snprintf_safe(target_name, sizeof(target_name), "dest:%d:%u:%d",
                      (int)key.pid, key.pid_ns, key.dev_num);
    status = uct_rocm_ipc_create_cache(&cache, target_name);
    if (status != UCS_OK) {
        kh_del(rocm_ipc_rem_cache, hash, it);
        ucs_error("failed to create rocm ipc cache: %s",
                  ucs_status_string(status));
        return status;
    }

    kh_val(hash, it) = cache;
    *cache_p         = cache;
    return UCS_OK;
}

static ucs_status_t
uct_rocm_ipc_cache_region_watch(uct_rocm_ipc_cache_region_t *region)
{
    hsa_signal_t signal = {.handle = 0};
    hsa_signal_value_t value;
    hsa_status_t hsa_status;
    ucs_list_link_t deferred;
    ucs_status_t status;

    ucs_list_head_init(&deferred);
    ucs_rw_spinlock_write_lock(&uct_rocm_ipc_remote_cache.lock);
    if ((ucs_atomic_cswap32(&region->deactivate, 0, 0) != 0) ||
        (region->signal.handle == 0) ||
        region->handler_armed) {
        /* The mapping was claimed before the handler existed. */
        ucs_rw_spinlock_write_unlock(&uct_rocm_ipc_remote_cache.lock);
        status = UCS_ERR_INVALID_ADDR;
        goto out_unref;
    }
    region->handler_armed = 1;
    ucs_rw_spinlock_write_unlock(&uct_rocm_ipc_remote_cache.lock);

    ucs_atomic_add32(&region->refcount, 1);
    hsa_status = hsa_amd_signal_async_handler(region->signal,
                                              HSA_SIGNAL_CONDITION_EQ, 0,
                                              uct_rocm_ipc_cache_signal_cb,
                                              region);
    if (hsa_status != HSA_STATUS_SUCCESS) {
        ucs_error("%s: failed to arm rocm ipc signal for %p: %d",
                  region->cache->name, (void*)region->key.address, hsa_status);
        ucs_atomic_fadd32(&region->refcount, (uint32_t)-1);
        status = UCS_ERR_IO_ERROR;
        goto drop_mapping;
    }

    ucs_rw_spinlock_write_lock(&uct_rocm_ipc_remote_cache.lock);
    if (ucs_atomic_cswap32(&region->deactivate, 0, 0) != 0) {
        /* The handler owns both its region reference and the signal. */
        ucs_rw_spinlock_write_unlock(&uct_rocm_ipc_remote_cache.lock);
        status = UCS_ERR_INVALID_ADDR;
        goto out_unref;
    }

    value = hsa_signal_load_scacquire(region->signal);
    if (value <= 0) {
        uct_rocm_ipc_cache_region_drop_locked(region, &deferred);
        status = UCS_ERR_INVALID_ADDR;
    } else {
        status = UCS_OK;
    }
    ucs_rw_spinlock_write_unlock(&uct_rocm_ipc_remote_cache.lock);
    uct_rocm_ipc_cache_region_finalize(&deferred);
    goto out_unref;

drop_mapping:
    ucs_rw_spinlock_write_lock(&uct_rocm_ipc_remote_cache.lock);
    signal                = region->signal;
    region->signal.handle = 0;
    uct_rocm_ipc_cache_region_drop_locked(region, &deferred);
    ucs_rw_spinlock_write_unlock(&uct_rocm_ipc_remote_cache.lock);
    uct_rocm_ipc_cache_region_signal_destroy(region, signal);
    uct_rocm_ipc_cache_region_finalize(&deferred);
out_unref:
    /* Drops the reference taken before the watcher was called. The handler
     * reference, once arming succeeds, belongs to the signal callback. */
    uct_rocm_ipc_cache_region_unref(region);
    return status;
}

ucs_status_t
uct_rocm_ipc_cache_map_memhandle(uct_rocm_ipc_key_t *key, void **mapped_addr,
                                 uct_rocm_ipc_cache_region_t **region_p)
{
    uct_rocm_ipc_cache_hash_key_t hash_key = {key->pid, key->pid_ns,
                                              key->dev_num};
    uct_rocm_ipc_cache_region_t *watch_region = NULL;
    uct_rocm_ipc_cache_region_t *pinned       = NULL;
    uct_rocm_ipc_cache_region_t *region;
    uct_rocm_ipc_cache_t *cache;
    ucs_list_link_t deferred;
    ucs_status_t status;
    void *mapped;

    ucs_list_head_init(&deferred);
    ucs_rw_spinlock_write_lock(&uct_rocm_ipc_remote_cache.lock);
    status = uct_rocm_ipc_remote_cache_get(hash_key, &cache);
    if (status == UCS_OK) {
        status = uct_rocm_ipc_cache_map_region(cache, key, &region,
                                               &watch_region, &deferred);
    }
    if (status == UCS_OK) {
        mapped = region->mapped_addr;
        if (watch_region != NULL) {
            /* Keep the region alive until its IPC signal is armed. */
            ucs_atomic_add32(&watch_region->refcount, 1);
        }
        if (region_p != NULL) {
            ++region->pin_count;
            ucs_atomic_add32(&region->refcount, 1);
            pinned = region;
        }
    }
    ucs_rw_spinlock_write_unlock(&uct_rocm_ipc_remote_cache.lock);

    uct_rocm_ipc_cache_region_finalize(&deferred);
    if ((status == UCS_OK) && (watch_region != NULL)) {
        status = uct_rocm_ipc_cache_region_watch(watch_region);
    }
    if (status != UCS_OK) {
        if (pinned != NULL) {
            uct_rocm_ipc_cache_unpin(pinned);
        }
        return status;
    }

    *mapped_addr = mapped;
    if (region_p != NULL) {
        *region_p = pinned;
    }
    return UCS_OK;
}

void uct_rocm_ipc_cache_unpin(uct_rocm_ipc_cache_region_t *region)
{
    ucs_rw_spinlock_write_lock(&uct_rocm_ipc_remote_cache.lock);
    ucs_assert(region->pin_count > 0);
    if ((--region->pin_count == 0) &&
        (ucs_atomic_cswap32(&region->deactivate, 0, 0) != 0)) {
        uct_rocm_ipc_detach_mapping(&region->key, region->mapped_addr);
        region->mapped_addr = NULL;
    }
    ucs_rw_spinlock_write_unlock(&uct_rocm_ipc_remote_cache.lock);

    uct_rocm_ipc_cache_region_unref(region);
}

void uct_rocm_ipc_destroy_cache(uct_rocm_ipc_cache_t *cache)
{
    uct_rocm_ipc_cache_purge(cache);
    ucs_pgtable_cleanup(&cache->pgtable);
    ucs_free(cache->name);
    ucs_free(cache);
}

UCS_STATIC_INIT
{
    ucs_status_t status;

    ucs_rw_spinlock_init(&uct_rocm_ipc_remote_cache.lock);
    kh_init_inplace(rocm_ipc_rem_cache, &uct_rocm_ipc_remote_cache.hash);

    status = ucs_spinlock_init(&uct_rocm_ipc_export_lock, 0);
    if (status != UCS_OK) {
        ucs_fatal("failed to initialize rocm ipc export lock");
    }
    kh_init_inplace(rocm_ipc_exports, &uct_rocm_ipc_exports);
    ucs_list_head_init(&uct_rocm_ipc_export_retired);
}

UCS_STATIC_CLEANUP
{
    uct_rocm_ipc_cache_t *cache;

    uct_rocm_ipc_export_cleanup();
    kh_foreach_value(&uct_rocm_ipc_remote_cache.hash, cache,
                     { uct_rocm_ipc_destroy_cache(cache); })
    kh_destroy_inplace(rocm_ipc_rem_cache, &uct_rocm_ipc_remote_cache.hash);
    kh_destroy_inplace(rocm_ipc_exports, &uct_rocm_ipc_exports);
    ucs_rw_spinlock_cleanup(&uct_rocm_ipc_remote_cache.lock);
    ucs_spinlock_destroy(&uct_rocm_ipc_export_lock);
}
