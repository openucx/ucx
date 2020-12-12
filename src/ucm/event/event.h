/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCM_EVENT_H_
#define UCM_EVENT_H_

#include <ucm/api/ucm.h>
#include <ucm/util/log.h>
#include <ucs/datastruct/list.h>
#include <ucs/type/status.h>

#define UCM_NATIVE_EVENT_VM_MAPPED (UCM_EVENT_MMAP  | UCM_EVENT_MREMAP | \
                                    UCM_EVENT_SHMAT | UCM_EVENT_SBRK | \
                                    UCM_EVENT_BRK)

#define UCM_NATIVE_EVENT_VM_UNMAPPED (UCM_EVENT_MMAP   | UCM_EVENT_MUNMAP | \
                                      UCM_EVENT_MREMAP | UCM_EVENT_SHMDT  | \
                                      UCM_EVENT_SHMAT  | UCM_EVENT_SBRK   | \
                                      UCM_EVENT_MADVISE | UCM_EVENT_BRK)


typedef struct ucm_event_handler {
    ucs_list_link_t       list;
    int                   events;
    int                   priority;
    ucm_event_callback_t  cb;
    void                  *arg;
} ucm_event_handler_t;


typedef struct ucm_event_installer {
    ucs_status_t          (*install)(int events);
    void                  (*get_existing_alloc)(ucm_event_handler_t *handler);
    ucs_list_link_t       list;
} ucm_event_installer_t;

extern ucs_list_link_t ucm_event_installer_list;

ucs_status_t ucm_set_mmap_hooks();

void ucm_event_handler_add(ucm_event_handler_t *handler);

void ucm_event_handler_remove(ucm_event_handler_t *handler);

void ucm_event_dispatch(ucm_event_type_t event_type, ucm_event_t *event);

void ucm_event_enter();

void ucm_event_enter_exclusive();

void ucm_event_leave();

static UCS_F_ALWAYS_INLINE void
ucm_dispatch_vm_mmap(void *addr, size_t length)
{
    ucm_event_t event;

    ucm_trace("vm_map addr=%p length=%zu", addr, length);

    event.vm_mapped.address = addr;
    event.vm_mapped.size    = length;
    ucm_event_dispatch(UCM_EVENT_VM_MAPPED, &event);
}

static UCS_F_ALWAYS_INLINE void
ucm_dispatch_vm_munmap(void *addr, size_t length)
{
    ucm_event_t event;

    ucm_trace("vm_unmap addr=%p length=%zu", addr, length);

    event.vm_unmapped.address = addr;
    event.vm_unmapped.size    = length;
    ucm_event_dispatch(UCM_EVENT_VM_UNMAPPED, &event);
}

#endif
