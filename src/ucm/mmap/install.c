/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "mmap.h"

#include <ucm/api/ucm.h>
#include <ucm/event/event.h>
#include <ucm/util/log.h>
#include <ucm/util/reloc.h>
#include <ucm/util/ucm_config.h>
#include <ucs/sys/math.h>

#include <sys/mman.h>
#include <sys/shm.h>
#include <unistd.h>
#include <pthread.h>


typedef struct ucm_mmap_func {
    ucm_reloc_patch_t patch;
    ucm_event_type_t   event_type;
} ucm_mmap_func_t;

static ucm_mmap_func_t ucm_mmap_funcs[] = {
    { {"mmap",   ucm_override_mmap},   UCM_EVENT_MMAP},
    { {"munmap", ucm_override_munmap}, UCM_EVENT_MUNMAP},
    { {"mremap", ucm_override_mremap}, UCM_EVENT_MREMAP},
    { {"shmat",  ucm_override_shmat},  UCM_EVENT_SHMAT},
    { {"shmdt",  ucm_override_shmdt},  UCM_EVENT_SHMDT},
    { {"sbrk",   ucm_override_sbrk},   UCM_EVENT_SBRK},
    { {NULL, NULL}, 0}
};

void ucm_mmap_event_test_callback(ucm_event_type_t event_type,
                                  ucm_event_t *event, void *arg)
{
    int *out_events = arg;
    *out_events |= event_type;
}

/* Called with lock held */
static ucs_status_t ucm_mmap_test(int events)
{
    static int installed_events = 0;
    ucm_event_handler_t handler;
    int out_events;
    void *p;

    if (ucs_test_all_flags(installed_events, events)) {
        /* All requested events are already installed */
        return UCS_OK;
    }

    /* Install a temporary event handler which will add the supported event
     * type to out_events bitmap.
     */
    handler.events   = events;
    handler.priority = -1;
    handler.cb       = ucm_mmap_event_test_callback;
    handler.arg      = &out_events;
    out_events       = 0;

    ucm_event_handler_add(&handler);

    if (events & (UCM_EVENT_MMAP|UCM_EVENT_MUNMAP|UCM_EVENT_MREMAP)) {
        p = mmap(NULL, 0, 0, 0, -1 ,0);
        p = mremap(p, 0, 0, 0);
        munmap(p, 0);
    }

    if (events & (UCM_EVENT_SHMAT|UCM_EVENT_SHMDT)) {
        p = shmat(0, NULL, 0);
        shmdt(p);
    }

    if (events & UCM_EVENT_SBRK) {
        (void)sbrk(0);
    }

    ucm_event_handler_remove(&handler);

    /* TODO check address / stop all threads */
    installed_events |= out_events;
    ucm_debug("mmap test: got 0x%x out of 0x%x, total: 0x%x", out_events, events,
              installed_events);

    /* Return success iff we caught all wanted events */
    if (!ucs_test_all_flags(out_events, events)) {
        return UCS_ERR_UNSUPPORTED;
    }

    return UCS_OK;
}

/* Called with lock held */
static ucs_status_t ucs_mmap_install_reloc(int events)
{
    static int installed_events = 0;
    ucm_mmap_func_t *entry;
    ucs_status_t status;

    if (!ucm_global_config.enable_mmap_reloc) {
        ucm_debug("installing mmap relocations is disabled by configuration");
        return UCS_ERR_UNSUPPORTED;
    }

    for (entry = ucm_mmap_funcs; entry->patch.symbol != NULL; ++entry) {
        if (!(entry->event_type & events)) {
            /* Not required */
            continue;
        }

        if (entry->event_type & installed_events) {
            /* Already installed */
            continue;
        }

        ucm_debug("mmap: installing relocation table entry for %s = %p for event 0x%x",
                  entry->patch.symbol, entry->patch.value, entry->event_type);

        status = ucm_reloc_modify(&entry->patch);
        if (status != UCS_OK) {
            ucm_warn("failed to install relocation table entry for '%s'",
                     entry->patch.symbol);
            return status;
        }

        installed_events |= entry->event_type;
    }

    return UCS_OK;
}

ucs_status_t ucm_mmap_install(int events)
{
    static pthread_mutex_t install_mutex = PTHREAD_MUTEX_INITIALIZER;
    ucs_status_t status;

    pthread_mutex_lock(&install_mutex);

    status = ucm_mmap_test(events);
    if (status == UCS_OK) {
        goto out_unlock;
    }

    status = ucs_mmap_install_reloc(events);
    if (status != UCS_OK) {
        ucm_debug("failed to install relocations for mmap");
        goto out_unlock;
    }

    status = ucm_mmap_test(events);

out_unlock:
    pthread_mutex_unlock(&install_mutex);
    return status;
}
