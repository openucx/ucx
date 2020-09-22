/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "event.h"

#include <ucm/mmap/mmap.h>
#include <ucm/malloc/malloc_hook.h>
#include <ucm/util/sys.h>
#include <ucs/arch/cpu.h>
#include <ucs/datastruct/khash.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/module.h>
#include <ucs/type/init_once.h>
#include <ucs/type/spinlock.h>

#include <sys/mman.h>
#include <pthread.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <inttypes.h>


UCS_LIST_HEAD(ucm_event_installer_list);

static pthread_spinlock_t ucm_kh_lock;
#define ucm_ptr_hash(_ptr)  kh_int64_hash_func((uintptr_t)(_ptr))
KHASH_INIT(ucm_ptr_size, const void*, size_t, 1, ucm_ptr_hash, kh_int64_hash_equal)

static pthread_rwlock_t ucm_event_lock = PTHREAD_RWLOCK_INITIALIZER;
static ucs_list_link_t ucm_event_handlers;
static int ucm_external_events = 0;
static khash_t(ucm_ptr_size) ucm_shmat_ptrs;

static size_t ucm_shm_size(int shmid)
{
    struct shmid_ds ds;
    int ret;

    ret = shmctl(shmid, IPC_STAT, &ds);
    if (ret < 0) {
        return 0;
    }

    return ds.shm_segsz;
}

static void ucm_event_call_orig(ucm_event_type_t event_type, ucm_event_t *event,
                                void *arg)
{
    switch (event_type) {
    case UCM_EVENT_MMAP:
        if (event->mmap.result == MAP_FAILED) {
            event->mmap.result = ucm_orig_mmap(event->mmap.address,
                                               event->mmap.size,
                                               event->mmap.prot,
                                               event->mmap.flags,
                                               event->mmap.fd,
                                               event->mmap.offset);
        }
        break;
    case UCM_EVENT_MUNMAP:
        if (event->munmap.result == -1) {
            event->munmap.result = ucm_orig_munmap(event->munmap.address,
                                                   event->munmap.size);
        }
        break;
    case UCM_EVENT_MREMAP:
        if (event->mremap.result == MAP_FAILED) {
            event->mremap.result = ucm_orig_mremap(event->mremap.address,
                                                   event->mremap.old_size,
                                                   event->mremap.new_size,
                                                   event->mremap.flags);
        }
        break;
    case UCM_EVENT_SHMAT:
        if (event->shmat.result == MAP_FAILED) {
            event->shmat.result = ucm_orig_shmat(event->shmat.shmid,
                                                 event->shmat.shmaddr,
                                                 event->shmat.shmflg);
        }
        break;
    case UCM_EVENT_SHMDT:
        if (event->shmdt.result == -1) {
            event->shmdt.result = ucm_orig_shmdt(event->shmdt.shmaddr);
        }
        break;
    case UCM_EVENT_SBRK:
        if (event->sbrk.result == MAP_FAILED) {
            event->sbrk.result = ucm_orig_sbrk(event->sbrk.increment);
        }
        break;
    case UCM_EVENT_MADVISE:
        if (event->madvise.result == -1) {
            event->madvise.result = ucm_orig_madvise(event->madvise.addr,
                                                     event->madvise.length,
                                                     event->madvise.advice);
        }
        break;
    default:
        ucm_warn("Got unknown event %d", event_type);
        break;
    }
}

/*
 * Add a handler which calls the original implementation, and declare the callback
 * list so that initially it will be the single element on that list.
 */
static ucm_event_handler_t ucm_event_orig_handler = {
    .list     = UCS_LIST_INITIALIZER(&ucm_event_handlers, &ucm_event_handlers),
    .events   = UCM_EVENT_MMAP | UCM_EVENT_MUNMAP | UCM_EVENT_MREMAP |
                UCM_EVENT_SHMAT | UCM_EVENT_SHMDT | UCM_EVENT_SBRK |
                UCM_EVENT_MADVISE,      /* All events */
    .priority = 0,                      /* Between negative and positive handlers */
    .cb       = ucm_event_call_orig
};
static ucs_list_link_t ucm_event_handlers =
                UCS_LIST_INITIALIZER(&ucm_event_orig_handler.list,
                                     &ucm_event_orig_handler.list);


void ucm_event_dispatch(ucm_event_type_t event_type, ucm_event_t *event)
{
    ucm_event_handler_t *handler;

    ucs_list_for_each(handler, &ucm_event_handlers, list) {
        if (handler->events & event_type) {
            handler->cb(event_type, event, handler->arg);
        }
    }
}

#define ucm_event_lock(_lock_func) \
    { \
        int ret; \
        do { \
            ret = _lock_func(&ucm_event_lock); \
        } while (ret == EAGAIN); \
        if (ret != 0) { \
            ucm_fatal("%s() failed: %s", #_lock_func, strerror(ret)); \
        } \
    }

void ucm_event_enter()
{
    ucm_event_lock(pthread_rwlock_rdlock);
}

void ucm_event_enter_exclusive()
{
    ucm_event_lock(pthread_rwlock_wrlock);
}

void ucm_event_leave()
{
    pthread_rwlock_unlock(&ucm_event_lock);
}

void *ucm_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset)
{
    ucm_event_t event;

    ucm_trace("ucm_mmap(addr=%p length=%lu prot=0x%x flags=0x%x fd=%d offset=%ld)",
              addr, length, prot, flags, fd, (long)offset);

    ucm_event_enter();

    if ((flags & MAP_FIXED) && (addr != NULL)) {
        ucm_dispatch_vm_munmap(addr, length);
    }

    event.mmap.result  = MAP_FAILED;
    event.mmap.address = addr;
    event.mmap.size    = length;
    event.mmap.prot    = prot;
    event.mmap.flags   = flags;
    event.mmap.fd      = fd;
    event.mmap.offset  = offset;
    ucm_event_dispatch(UCM_EVENT_MMAP, &event);

    if (event.mmap.result != MAP_FAILED) {
        /* Use original length */
        ucm_dispatch_vm_mmap(event.mmap.result, length);
    }

    ucm_event_leave();

    return event.mmap.result;
}

int ucm_munmap(void *addr, size_t length)
{
    ucm_event_t event;

    ucm_event_enter();

    ucm_trace("ucm_munmap(addr=%p length=%lu)", addr, length);

    ucm_dispatch_vm_munmap(addr, length);

    event.munmap.result  = -1;
    event.munmap.address = addr;
    event.munmap.size    = length;
    ucm_event_dispatch(UCM_EVENT_MUNMAP, &event);

    ucm_event_leave();

    return event.munmap.result;
}

void ucm_vm_mmap(void *addr, size_t length)
{
    ucm_event_enter();

    ucm_trace("ucm_vm_mmap(addr=%p length=%lu)", addr, length);
    ucm_dispatch_vm_mmap(addr, length);

    ucm_event_leave();
}

void ucm_vm_munmap(void *addr, size_t length)
{
    ucm_event_enter();

    ucm_trace("ucm_vm_munmap(addr=%p length=%lu)", addr, length);
    ucm_dispatch_vm_munmap(addr, length);

    ucm_event_leave();
}

void *ucm_mremap(void *old_address, size_t old_size, size_t new_size, int flags)
{
    ucm_event_t event;

    ucm_event_enter();

    ucm_trace("ucm_mremap(old_address=%p old_size=%lu new_size=%ld flags=0x%x)",
              old_address, old_size, new_size, flags);

    ucm_dispatch_vm_munmap(old_address, old_size);

    event.mremap.result   = MAP_FAILED;
    event.mremap.address  = old_address;
    event.mremap.old_size = old_size;
    event.mremap.new_size = new_size;
    event.mremap.flags    = flags;
    ucm_event_dispatch(UCM_EVENT_MREMAP, &event);

    if (event.mremap.result != MAP_FAILED) {
        /* Use original new_size */
        ucm_dispatch_vm_mmap(event.mremap.result, new_size);
    }

    ucm_event_leave();

    return event.mremap.result;
}

static int ucm_shm_del_entry_from_khash(const void *addr, size_t *size)
{
    khiter_t iter;

    pthread_spin_lock(&ucm_kh_lock);
    iter = kh_get(ucm_ptr_size, &ucm_shmat_ptrs, addr);
    if (iter != kh_end(&ucm_shmat_ptrs)) {
        if (size != NULL) {
            *size = kh_value(&ucm_shmat_ptrs, iter);
        }
        kh_del(ucm_ptr_size, &ucm_shmat_ptrs, iter);
        pthread_spin_unlock(&ucm_kh_lock);
        return 1;
    }

    pthread_spin_unlock(&ucm_kh_lock);
    return 0;
}

void *ucm_shmat(int shmid, const void *shmaddr, int shmflg)
{
#ifdef SHM_REMAP
    uintptr_t attach_addr;
#endif
    ucm_event_t event;
    khiter_t iter;
    size_t size;
    int result;

    ucm_event_enter();

    ucm_trace("ucm_shmat(shmid=%d shmaddr=%p shmflg=0x%x)",
              shmid, shmaddr, shmflg);

    size = ucm_shm_size(shmid);

#if SHM_REMAP
    if ((shmflg & SHM_REMAP) && (shmaddr != NULL)) {
        attach_addr = (uintptr_t)shmaddr;
        if (shmflg & SHM_RND) {
            attach_addr -= attach_addr % SHMLBA;
        }
        ucm_dispatch_vm_munmap((void*)attach_addr, size);
        ucm_shm_del_entry_from_khash((void*)attach_addr, NULL);
    }
#endif

    event.shmat.result  = MAP_FAILED;
    event.shmat.shmid   = shmid;
    event.shmat.shmaddr = shmaddr;
    event.shmat.shmflg  = shmflg;
    ucm_event_dispatch(UCM_EVENT_SHMAT, &event);

    if (event.shmat.result != MAP_FAILED) {
        pthread_spin_lock(&ucm_kh_lock);
        iter = kh_put(ucm_ptr_size, &ucm_shmat_ptrs, event.mmap.result, &result);
        if (result != -1) {
            kh_value(&ucm_shmat_ptrs, iter) = size;
        }
        pthread_spin_unlock(&ucm_kh_lock);
        ucm_dispatch_vm_mmap(event.shmat.result, size);
    }

    ucm_event_leave();

    return event.shmat.result;
}

int ucm_shmdt(const void *shmaddr)
{
    ucm_event_t event;
    size_t size;

    ucm_event_enter();

    ucm_debug("ucm_shmdt(shmaddr=%p)", shmaddr);

    if (!ucm_shm_del_entry_from_khash(shmaddr, &size)) {
        size = ucm_get_shm_seg_size(shmaddr);
    }

    ucm_dispatch_vm_munmap((void*)shmaddr, size);

    event.shmdt.result  = -1;
    event.shmdt.shmaddr = shmaddr;
    ucm_event_dispatch(UCM_EVENT_SHMDT, &event);

    ucm_event_leave();

    return event.shmdt.result;
}

void *ucm_sbrk(intptr_t increment)
{
    ucm_event_t event;

    ucm_event_enter();

    ucm_trace("ucm_sbrk(increment=%+ld)", increment);

    if (increment < 0) {
        ucm_dispatch_vm_munmap(UCS_PTR_BYTE_OFFSET(ucm_orig_sbrk(0), increment),
                               -increment);
    }

    event.sbrk.result    = MAP_FAILED;
    event.sbrk.increment = increment;
    ucm_event_dispatch(UCM_EVENT_SBRK, &event);

    if ((increment > 0) && (event.sbrk.result != MAP_FAILED)) {
        ucm_dispatch_vm_mmap(UCS_PTR_BYTE_OFFSET(ucm_orig_sbrk(0), -increment),
                             increment);
    }

    ucm_event_leave();

    return event.sbrk.result;
}

int ucm_brk(void *addr)
{
#if UCM_BISTRO_HOOKS
    void *old_addr;
    intptr_t increment;
    ucm_event_t event;

    old_addr  = ucm_brk_syscall(0);
    /* in case if addr == NULL - it just returns current pointer */
    increment = addr ? ((intptr_t)addr - (intptr_t)old_addr) : 0;

    ucm_event_enter();

    ucm_trace("ucm_brk(addr=%p)", addr);

    if (increment < 0) {
        ucm_dispatch_vm_munmap(UCS_PTR_BYTE_OFFSET(old_addr, increment),
                               -increment);
    }

    event.sbrk.result    = (void*)-1;
    event.sbrk.increment = increment;
    ucm_event_dispatch(UCM_EVENT_SBRK, &event);

    if ((increment > 0) && (event.sbrk.result != MAP_FAILED)) {
        ucm_dispatch_vm_mmap(old_addr, increment);
    }

    ucm_event_leave();

    return event.sbrk.result == MAP_FAILED ? -1 : 0;
#else
    return -1;
#endif
}

int ucm_madvise(void *addr, size_t length, int advice)
{
    ucm_event_t event;

    ucm_event_enter();

    ucm_trace("ucm_madvise(addr=%p length=%zu advice=%d)", addr, length, advice);

    /* madvise(MADV_DONTNEED) and madvise(MADV_FREE) are releasing pages */
    if ((advice == MADV_DONTNEED)
#if HAVE_DECL_MADV_REMOVE
        || (advice == MADV_REMOVE)
#endif
#if HAVE_DECL_POSIX_MADV_DONTNEED
        || (advice == POSIX_MADV_DONTNEED)
#endif
#if HAVE_DECL_MADV_FREE
        || (advice == MADV_FREE)
#endif
       ) {
        ucm_dispatch_vm_munmap(addr, length);
    }

    event.madvise.result = -1;
    event.madvise.addr   = addr;
    event.madvise.length = length;
    event.madvise.advice = advice;
    ucm_event_dispatch(UCM_EVENT_MADVISE, &event);

    ucm_event_leave();

    return event.madvise.result;
}

void ucm_event_handler_add(ucm_event_handler_t *handler)
{
    ucm_event_handler_t *elem;

    ucm_event_enter_exclusive();
    ucs_list_for_each(elem, &ucm_event_handlers, list) {
        if (handler->priority < elem->priority) {
            ucs_list_insert_before(&elem->list, &handler->list);
            ucm_event_leave();
            return;
        }
    }

    ucs_list_add_tail(&ucm_event_handlers, &handler->list);
    ucm_event_leave();
}

void ucm_event_handler_remove(ucm_event_handler_t *handler)
{
    ucm_event_enter_exclusive();
    ucs_list_del(&handler->list);
    ucm_event_leave();
}

static ucs_status_t ucm_event_install(int events)
{
    static ucs_init_once_t init_once = UCS_INIT_ONCE_INITIALIZER;
    UCS_MODULE_FRAMEWORK_DECLARE(ucm);
    ucm_event_installer_t *event_installer;
    int malloc_events;
    ucs_status_t status;

    UCS_INIT_ONCE(&init_once) {
        ucm_prevent_dl_unload();
    }

    /* TODO lock */
    status = ucm_mmap_install(events);
    if (status != UCS_OK) {
        ucm_debug("failed to install mmap events");
        goto out_unlock;
    }

    ucm_debug("mmap hooks are ready");

    malloc_events = events & ~(UCM_EVENT_MEM_TYPE_ALLOC |
                               UCM_EVENT_MEM_TYPE_FREE);
    status = ucm_malloc_install(malloc_events);
    if (status != UCS_OK) {
        ucm_debug("failed to install malloc events");
        goto out_unlock;
    }

    ucm_debug("malloc hooks are ready");

    /* Call extra event installers */
    UCS_MODULE_FRAMEWORK_LOAD(ucm, UCS_MODULE_LOAD_FLAG_NODELETE);
    ucs_list_for_each(event_installer, &ucm_event_installer_list, list) {
        status = event_installer->install(events);
        if (status != UCS_OK) {
            goto out_unlock;
        }
    }

    status = UCS_OK;

out_unlock:
    return status;

}

ucs_status_t ucm_set_event_handler(int events, int priority,
                                   ucm_event_callback_t cb, void *arg)
{
    ucm_event_installer_t *event_installer;
    ucm_event_handler_t *handler;
    ucs_status_t status;
    int flags;

    if (events & ~(UCM_EVENT_MMAP|UCM_EVENT_MUNMAP|UCM_EVENT_MREMAP|
                   UCM_EVENT_SHMAT|UCM_EVENT_SHMDT|
                   UCM_EVENT_SBRK|
                   UCM_EVENT_MADVISE|
                   UCM_EVENT_VM_MAPPED|UCM_EVENT_VM_UNMAPPED|
                   UCM_EVENT_MEM_TYPE_ALLOC|UCM_EVENT_MEM_TYPE_FREE|
                   UCM_EVENT_FLAG_NO_INSTALL|
                   UCM_EVENT_FLAG_EXISTING_ALLOC)) {
        return UCS_ERR_INVALID_PARAM;
    }

    if (events && !ucm_global_opts.enable_events) {
        return UCS_ERR_UNSUPPORTED;
    }

    /* separate event flags from real events */
    flags   = events & (UCM_EVENT_FLAG_NO_INSTALL |
                        UCM_EVENT_FLAG_EXISTING_ALLOC);
    events &= ~flags;

    if (!(flags & UCM_EVENT_FLAG_NO_INSTALL) && (events & ~ucm_external_events)) {
        status = ucm_event_install(events & ~ucm_external_events);
        if (status != UCS_OK) {
            return status;
        }
    }

    handler = malloc(sizeof(*handler));
    if (handler == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    handler->events   = events;
    handler->priority = priority;
    handler->cb       = cb;
    handler->arg      = arg;

    ucm_event_handler_add(handler);

    if (flags & UCM_EVENT_FLAG_EXISTING_ALLOC) {
        ucs_list_for_each(event_installer, &ucm_event_installer_list, list) {
            event_installer->get_existing_alloc(handler);
        }
    }

    ucm_debug("added user handler (func=%p arg=%p) for events=0x%x prio=%d", cb,
              arg, events, priority);
    return UCS_OK;
}

void ucm_set_external_event(int events)
{
    ucm_event_enter_exclusive();
    ucm_external_events |= events;
    ucm_event_leave();
}

void ucm_unset_external_event(int events)
{
    ucm_event_enter_exclusive();
    ucm_external_events &= ~events;
    ucm_event_leave();
}

void ucm_unset_event_handler(int events, ucm_event_callback_t cb, void *arg)
{
    ucm_event_handler_t *elem, *tmp;
    UCS_LIST_HEAD(gc_list);

    ucm_event_enter_exclusive();
    ucs_list_for_each_safe(elem, tmp, &ucm_event_handlers, list) {
        if ((cb == elem->cb) && (arg == elem->arg)) {
            elem->events &= ~events;
            if (elem->events == 0) {
                ucs_list_del(&elem->list);
                ucs_list_add_tail(&gc_list, &elem->list);
            }
        }
    }
    ucm_event_leave();

    /* Do not release memory while we hold event lock - may deadlock */
    ucs_list_for_each_safe(elem, tmp, &gc_list, list) {
        free(elem);
    }
}

ucs_status_t ucm_test_events(int events)
{
    return ucm_mmap_test_installed_events(events);
}

ucs_status_t ucm_test_external_events(int events)
{
    return ucm_mmap_test_events(events & ucm_external_events, "external");
}

UCS_STATIC_INIT {
    pthread_spin_init(&ucm_kh_lock, PTHREAD_PROCESS_PRIVATE);
    kh_init_inplace(ucm_ptr_size, &ucm_shmat_ptrs);
}

UCS_STATIC_CLEANUP {
    kh_destroy_inplace(ucm_ptr_size, &ucm_shmat_ptrs);
    pthread_spin_destroy(&ucm_kh_lock);
}
