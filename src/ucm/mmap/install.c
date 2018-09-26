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
#include <ucm/util/sys.h>
#include <ucm/bistro/bistro.h>
#include <ucs/sys/math.h>
#include <ucs/sys/checker.h>
#include <ucs/debug/assert.h>

#include <sys/mman.h>
#include <sys/shm.h>
#include <unistd.h>
#include <pthread.h>

#define UCM_IS_HOOK_ENABLED(_entry) \
    ((_entry)->hook_type & UCS_BIT(ucm_mmap_hook_mode()))

#define UCM_HOOK_STR \
    ((ucm_mmap_hook_mode() == UCM_MMAP_HOOK_RELOC) ?  "reloc" : "bistro")

extern const char *ucm_mmap_hook_modes[];

typedef enum ucm_mmap_hook_type {
    UCM_HOOK_RELOC  = UCS_BIT(UCM_MMAP_HOOK_RELOC),
    UCM_HOOK_BISTRO = UCS_BIT(UCM_MMAP_HOOK_BISTRO),
    UCM_HOOK_BOTH   = UCM_HOOK_RELOC | UCM_HOOK_BISTRO
} ucm_mmap_hook_type_t;

typedef struct ucm_mmap_func {
    ucm_reloc_patch_t    patch;
    ucm_event_type_t     event_type;
    ucm_event_type_t     deps;
    ucm_mmap_hook_type_t hook_type;
} ucm_mmap_func_t;

static ucm_mmap_func_t ucm_mmap_funcs[] = {
    { {"mmap",    ucm_override_mmap},    UCM_EVENT_MMAP,    0, UCM_HOOK_BOTH},
    { {"munmap",  ucm_override_munmap},  UCM_EVENT_MUNMAP,  0, UCM_HOOK_BOTH},
    { {"mremap",  ucm_override_mremap},  UCM_EVENT_MREMAP,  0, UCM_HOOK_BOTH},
    { {"shmat",   ucm_override_shmat},   UCM_EVENT_SHMAT,   0, UCM_HOOK_BOTH},
    { {"shmdt",   ucm_override_shmdt},   UCM_EVENT_SHMDT,   UCM_EVENT_SHMAT, UCM_HOOK_BOTH},
    { {"sbrk",    ucm_override_sbrk},    UCM_EVENT_SBRK,    0, UCM_HOOK_RELOC},
#if UCM_BISTRO_HOOKS
    { {"brk",     ucm_override_brk},     UCM_EVENT_SBRK,    0, UCM_HOOK_BISTRO},
#endif
    { {"madvise", ucm_override_madvise}, UCM_EVENT_MADVISE, 0, UCM_HOOK_BOTH},
    { {NULL, NULL}, 0}
};

static void ucm_mmap_event_test_callback(ucm_event_type_t event_type,
                                         ucm_event_t *event, void *arg)
{
    int *out_events = arg;

    *out_events |= event_type;
}

void ucm_fire_mmap_events(int events)
{
    void *p;

    if (events & (UCM_EVENT_MMAP|UCM_EVENT_MUNMAP|UCM_EVENT_MREMAP|
                  UCM_EVENT_VM_MAPPED|UCM_EVENT_VM_UNMAPPED)) {
        p = mmap(NULL, 0, 0, 0, -1 ,0);
        p = mremap(p, 0, 0, 0);
        munmap(p, 0);
    }

    if (events & (UCM_EVENT_SHMAT|UCM_EVENT_SHMDT|UCM_EVENT_VM_MAPPED|UCM_EVENT_VM_UNMAPPED)) {
        p = shmat(0, NULL, 0);
        shmdt(p);
    }

    if (events & (UCM_EVENT_SBRK|UCM_EVENT_VM_MAPPED|UCM_EVENT_VM_UNMAPPED)) {
        (void)sbrk(ucm_get_page_size());
        (void)sbrk(-ucm_get_page_size());
    }

    if (events & UCM_EVENT_MADVISE) {
        p = mmap(NULL, ucm_get_page_size(), PROT_READ|PROT_WRITE,
                 MAP_PRIVATE|MAP_ANON, -1, 0);
        if (p != MAP_FAILED) {
            madvise(p, ucm_get_page_size(), MADV_NORMAL);
            munmap(p, ucm_get_page_size());
        } else {
            ucm_debug("mmap failed: %m");
        }
    }
}

/* Called with lock held */
static ucs_status_t ucm_mmap_test(int events)
{
    static int installed_events = 0;
    ucm_event_handler_t handler;
    int out_events;

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

    ucm_fire_mmap_events(events);

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

    if (ucm_mmap_hook_mode() == UCM_MMAP_HOOK_NONE) {
        ucm_debug("installing mmap hooks is disabled by configuration");
        return UCS_ERR_UNSUPPORTED;
    }

    for (entry = ucm_mmap_funcs; entry->patch.symbol != NULL; ++entry) {
        if (!((entry->event_type|entry->deps) & events)) {
            /* Not required */
            continue;
        }

        if (entry->event_type & installed_events) {
            /* Already installed */
            continue;
        }

        if (UCM_IS_HOOK_ENABLED(entry)) {
            ucm_debug("mmap: installing %s hook for %s = %p for event 0x%x", UCM_HOOK_STR,
                      entry->patch.symbol, entry->patch.value, entry->event_type);

            if (ucm_mmap_hook_mode() == UCM_MMAP_HOOK_RELOC) {
                status = ucm_reloc_modify(&entry->patch);
            } else {
                ucs_assert(ucm_mmap_hook_mode() == UCM_MMAP_HOOK_BISTRO);
                status = ucm_bistro_patch(entry->patch.symbol, entry->patch.value, NULL);
            }
            if (status != UCS_OK) {
                ucm_warn("failed to install %s hook for '%s'",
                         UCM_HOOK_STR, entry->patch.symbol);
                return status;
            }

            installed_events |= entry->event_type;
        }
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
