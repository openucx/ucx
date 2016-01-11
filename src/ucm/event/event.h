/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCM_EVENT_H_
#define UCM_EVENT_H_

#include <ucm/api/ucm.h>

#include <ucs/datastruct/list.h>
#include <ucs/type/status.h>


typedef struct ucm_event_handler {
    ucs_list_link_t       list;
    int                   events;
    int                   priority;
    ucm_event_callback_t  cb;
    void                  *arg;
} ucm_event_handler_t;


ucs_status_t ucm_set_mmap_hooks();

void ucm_event_handler_add(ucm_event_handler_t *handler);

void ucm_event_handler_remove(ucm_event_handler_t *handler);

void *ucm_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
int ucm_munmap(void *addr, size_t length);
void *ucm_mremap(void *old_address, size_t old_size, size_t new_size, int flags);
void *ucm_shmat(int shmid, const void *shmaddr, int shmflg);
int ucm_shmdt(const void *shmaddr);
void *ucm_sbrk(intptr_t increment);

#endif
