/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "event.h"

#include <ucm/api/ucm.h>
#include <ucm/mmap/mmap.h>
#include <ucm/malloc/malloc_hook.h>
#if HAVE_CUDA
#include <ucm/cuda/cudamem.h>
#endif
#include <ucm/util/ucm_config.h>
#include <ucm/util/log.h>
#include <ucm/util/sys.h>
#include <ucs/arch/cpu.h>

#include <sys/mman.h>
#include <pthread.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>


static pthread_rwlock_t ucm_event_lock = PTHREAD_RWLOCK_INITIALIZER;
static ucs_list_link_t ucm_event_handlers;
static int ucm_external_events = 0;

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
                UCM_EVENT_SHMAT | UCM_EVENT_SHMDT | UCM_EVENT_SBRK, /* All events */
    .priority = 0,                      /* Between negative and positive handlers */
    .cb       = ucm_event_call_orig
};
static ucs_list_link_t ucm_event_handlers =
                UCS_LIST_INITIALIZER(&ucm_event_orig_handler.list,
                                     &ucm_event_orig_handler.list);


static void ucm_event_dispatch(ucm_event_type_t event_type, ucm_event_t *event)
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

static void ucm_event_enter()
{
    ucm_event_lock(pthread_rwlock_rdlock);
}

static void ucm_event_enter_exclusive()
{
    ucm_event_lock(pthread_rwlock_wrlock);
}

static void ucm_event_leave()
{
    pthread_rwlock_unlock(&ucm_event_lock);
}

static UCS_F_ALWAYS_INLINE void
ucm_dispatch_vm_mmap(void *addr, size_t length)
{
    ucm_event_t event;

    event.vm_mapped.address = addr;
    event.vm_mapped.size    = length;
    ucm_event_dispatch(UCM_EVENT_VM_MAPPED, &event);
}

static UCS_F_ALWAYS_INLINE void
ucm_dispatch_vm_munmap(void *addr, size_t length)
{
    ucm_event_t event;

    event.vm_unmapped.address = addr;
    event.vm_unmapped.size    = length;
    ucm_event_dispatch(UCM_EVENT_VM_UNMAPPED, &event);
}

void *ucm_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset)
{
    ucm_event_t event;

    ucm_trace("ucm_mmap(addr=%p length=%lu prot=0x%x flags=0x%x fd=%d offset=%ld)",
              addr, length, prot, flags, fd, offset);

    ucm_event_enter();

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

void *ucm_shmat(int shmid, const void *shmaddr, int shmflg)
{
    ucm_event_t event;
    size_t size;

    ucm_event_enter();

    ucm_trace("ucm_shmat(shmid=%d shmaddr=%p shmflg=0x%x)",
              shmid, shmaddr, shmflg);

    size = ucm_shm_size(shmid);
    event.shmat.result  = MAP_FAILED;
    event.shmat.shmid   = shmid;
    event.shmat.shmaddr = shmaddr;
    event.shmat.shmflg  = shmflg;
    ucm_event_dispatch(UCM_EVENT_SHMAT, &event);

    if (event.shmat.result != MAP_FAILED) {
        ucm_dispatch_vm_mmap(event.shmat.result, size);
    }

    ucm_event_leave();

    return event.shmat.result;
}

int ucm_shmdt(const void *shmaddr)
{
    ucm_event_t event;

    ucm_event_enter();

    ucm_debug("ucm_shmdt(shmaddr=%p)", shmaddr);

    ucm_dispatch_vm_munmap((void*)shmaddr, ucm_get_shm_seg_size(shmaddr));

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
        ucm_dispatch_vm_munmap(ucm_orig_sbrk(0) + increment, -increment);
    }

    event.sbrk.result    = MAP_FAILED;
    event.sbrk.increment = increment;
    ucm_event_dispatch(UCM_EVENT_SBRK, &event);

    if ((increment > 0) && (event.sbrk.result != MAP_FAILED)) {
        ucm_dispatch_vm_mmap(ucm_orig_sbrk(0) - increment, increment);
    }

    ucm_event_leave();

    return event.sbrk.result;
}

#if HAVE_CUDA
cudaError_t ucm_cudaFree(void *addr)
{
    cudaError_t ret;

    ucm_event_enter();

    ucm_trace("ucm_cudaFree(addr=%p )", addr);

    ucm_dispatch_vm_munmap(addr, 0);
    ret = ucm_orig_cudaFree(addr);

    ucm_event_leave();

    return ret;
}
#endif

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
    ucs_status_t status;
    int native_events;

    /* Replace aggregate events with the native events which make them */
    native_events = events & ~(UCM_EVENT_VM_MAPPED | UCM_EVENT_VM_UNMAPPED);
    if (events & UCM_EVENT_VM_MAPPED) {
        native_events |= UCM_EVENT_MMAP | UCM_EVENT_MREMAP |
                         UCM_EVENT_SHMAT | UCM_EVENT_SBRK;
    }
    if (events & UCM_EVENT_VM_UNMAPPED) {
        native_events |= UCM_EVENT_MUNMAP | UCM_EVENT_MREMAP |
                         UCM_EVENT_SHMDT | UCM_EVENT_SBRK;
    }

    /* TODO lock */
    status = ucm_mmap_install(native_events);
    if (status != UCS_OK) {
        ucm_debug("failed to install mmap events");
        goto out_unlock;
    }

    ucm_debug("mmap hooks are ready");

    status = ucm_malloc_install(native_events);
    if (status != UCS_OK) {
        ucm_debug("failed to install malloc events");
        goto out_unlock;
    }

    ucm_debug("malloc hooks are ready");

#if HAVE_CUDA
    status = ucm_cudamem_install();
    if (status != UCS_OK) {
        ucm_debug("failed to install cudamem events");
        goto out_unlock;
    }
    ucm_debug("cudaFree hooks are ready");
#endif

    status = UCS_OK;

out_unlock:
    return status;

}

ucs_status_t ucm_set_event_handler(int events, int priority,
                                   ucm_event_callback_t cb, void *arg)
{
    ucm_event_handler_t *handler;
    ucs_status_t status;

    if (!ucm_global_config.enable_events) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (!(events & (UCM_EVENT_FLAG_NO_INSTALL | ucm_external_events))) {
        status = ucm_event_install(events);
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
    while (!ucs_list_is_empty(&gc_list)) {
        elem = ucs_list_extract_head(&gc_list, ucm_event_handler_t, list);
        free(elem);
    }
}

