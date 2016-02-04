/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCM_MMAP_H_
#define UCM_MMAP_H_

#include <ucm/api/ucm.h>

ucs_status_t ucm_mmap_install(int events);

void ucm_mmap_event_test_callback(ucm_event_type_t event_type,
                                  ucm_event_t *event, void *arg);


void *ucm_override_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
int ucm_override_munmap(void *addr, size_t length);
void *ucm_override_mremap(void *old_address, size_t old_size, size_t new_size, int flags);
void *ucm_override_shmat(int shmid, const void *shmaddr, int shmflg);
int ucm_override_shmdt(const void *shmaddr);
void *ucm_override_sbrk(intptr_t increment);

#endif
