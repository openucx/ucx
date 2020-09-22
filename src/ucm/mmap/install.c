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
#include <ucs/arch/atomic.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/math.h>
#include <ucs/sys/checker.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/arch/bitops.h>
#include <ucs/debug/assert.h>

#include <sys/mman.h>
#include <sys/shm.h>
#include <unistd.h>
#include <pthread.h>

#define UCM_IS_HOOK_ENABLED(_entry) \
    ((_entry)->hook_type & UCS_BIT(ucm_mmap_hook_mode()))

#define UCM_HOOK_STR \
    ((ucm_mmap_hook_mode() == UCM_MMAP_HOOK_RELOC) ?  "reloc" : "bistro")

#define UCM_FIRE_EVENT(_event, _mask, _data, _call)                           \
    do {                                                                      \
        int exp_events = (_event) & (_mask);                                  \
        (_data)->fired_events = 0;                                            \
        _call;                                                                \
        ucm_trace("after %s: got 0x%x/0x%x", UCS_PP_MAKE_STRING(_call),       \
                  (_data)->fired_events, exp_events);                         \
        /* in case if any event is missed - set correcponding bit to 0     */ \
        /* same as equation:                                               */ \
        /* (_data)->out_events &= ~(exp_events ^                           */ \
        /*                          ((_data)->fired_events & exp_events)); */ \
        (_data)->out_events &= ~exp_events | (_data)->fired_events;           \
    } while(0)

#define UCM_MMAP_EVENT_NAME_ENTRY(_event) \
    [ucs_ilog2(UCM_EVENT_##_event)] = #_event

#define UCM_MMAP_MAX_EVENT_NAME_LEN sizeof("VM_UNMAPPED")

#define UCM_MMAP_REPORT_BUF_LEN \
    ((UCM_MMAP_MAX_EVENT_NAME_LEN + 2) * \
    ucs_static_array_size(ucm_mmap_event_name))

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

typedef struct ucm_mmap_test_events_data {
    uint32_t             fired_events;
    int                  out_events;
    pid_t                tid;
} ucm_mmap_test_events_data_t;

static ucm_mmap_func_t ucm_mmap_funcs[] = {
    { {"mmap",    ucm_override_mmap},    UCM_EVENT_MMAP,    UCM_EVENT_NONE,  UCM_HOOK_BOTH},
    { {"munmap",  ucm_override_munmap},  UCM_EVENT_MUNMAP,  UCM_EVENT_NONE,  UCM_HOOK_BOTH},
#if HAVE_MREMAP
    { {"mremap",  ucm_override_mremap},  UCM_EVENT_MREMAP,  UCM_EVENT_NONE,  UCM_HOOK_BOTH},
#endif
    { {"shmat",   ucm_override_shmat},   UCM_EVENT_SHMAT,   UCM_EVENT_NONE,  UCM_HOOK_BOTH},
    { {"shmdt",   ucm_override_shmdt},   UCM_EVENT_SHMDT,   UCM_EVENT_SHMAT, UCM_HOOK_BOTH},
    { {"sbrk",    ucm_override_sbrk},    UCM_EVENT_SBRK,    UCM_EVENT_NONE,  UCM_HOOK_RELOC},
#if UCM_BISTRO_HOOKS
    { {"brk",     ucm_override_brk},     UCM_EVENT_SBRK,    UCM_EVENT_NONE,  UCM_HOOK_BISTRO},
#endif
    { {"madvise", ucm_override_madvise}, UCM_EVENT_MADVISE, UCM_EVENT_NONE,  UCM_HOOK_BOTH},
    { {NULL, NULL}, UCM_EVENT_NONE}
};

static pthread_mutex_t ucm_mmap_install_mutex = PTHREAD_MUTEX_INITIALIZER;
static int ucm_mmap_installed_events = 0; /* events that were reported as installed */

static const char *ucm_mmap_event_name[] = {
    /* Native events */
    UCM_MMAP_EVENT_NAME_ENTRY(MMAP),
    UCM_MMAP_EVENT_NAME_ENTRY(MUNMAP),
    UCM_MMAP_EVENT_NAME_ENTRY(MREMAP),
    UCM_MMAP_EVENT_NAME_ENTRY(SHMAT),
    UCM_MMAP_EVENT_NAME_ENTRY(SHMDT),
    UCM_MMAP_EVENT_NAME_ENTRY(SBRK),
    UCM_MMAP_EVENT_NAME_ENTRY(MADVISE),

    /* Aggregate events */
    UCM_MMAP_EVENT_NAME_ENTRY(VM_MAPPED),
    UCM_MMAP_EVENT_NAME_ENTRY(VM_UNMAPPED),
};

static void ucm_mmap_event_test_callback(ucm_event_type_t event_type,
                                         ucm_event_t *event, void *arg)
{
    ucm_mmap_test_events_data_t *data = arg;

    /* This callback may be called from multiple threads, which are just calling
     * memory allocations/release, and not testing mmap hooks at the moment.
     * So ignore calls from other threads to ensure the only requested events
     * are proceeded.
     */
    if (data->tid == ucs_get_tid()) {
        data->fired_events |= event_type;
    }
}

/* Fire events with pre/post action. The problem is in call sequence: we
 * can't just fire single event - most of the system calls require set of
 * calls to eliminate resource leaks or data corruption, such sequence
 * produces additional events which may affect to event handling. To
 * exclude additional events from processing used pre/post actions where
 * set of handled events is cleared and evaluated for every system call */
static void
ucm_fire_mmap_events_internal(int events, ucm_mmap_test_events_data_t *data)
{
    size_t sbrk_size;
    int sbrk_mask;
    int shmid;
    void *p;

    if (events & (UCM_EVENT_MMAP|UCM_EVENT_MUNMAP|UCM_EVENT_MREMAP|
                  UCM_EVENT_VM_MAPPED|UCM_EVENT_VM_UNMAPPED)) {
        UCM_FIRE_EVENT(events, UCM_EVENT_MMAP|UCM_EVENT_VM_MAPPED,
                       data, p = mmap(NULL, ucm_get_page_size(), PROT_READ | PROT_WRITE,
                                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
#ifdef HAVE_MREMAP
        /* generate MAP event */
        UCM_FIRE_EVENT(events, UCM_EVENT_MREMAP|UCM_EVENT_VM_MAPPED|UCM_EVENT_VM_UNMAPPED,
                       data, p = mremap(p, ucm_get_page_size(),
                                        ucm_get_page_size() * 2, MREMAP_MAYMOVE));
        /* generate UNMAP event */
        UCM_FIRE_EVENT(events, UCM_EVENT_MREMAP|UCM_EVENT_VM_MAPPED|UCM_EVENT_VM_UNMAPPED,
                       data, p = mremap(p, ucm_get_page_size() * 2, ucm_get_page_size(), 0));
#endif
        /* generate UNMAP event */
        UCM_FIRE_EVENT(events, UCM_EVENT_MMAP|UCM_EVENT_VM_MAPPED,
                       data, p = mmap(p, ucm_get_page_size(), PROT_READ | PROT_WRITE,
                                      MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        UCM_FIRE_EVENT(events, UCM_EVENT_MUNMAP|UCM_EVENT_VM_UNMAPPED,
                       data, munmap(p, ucm_get_page_size()));
    }

    if (events & (UCM_EVENT_SHMAT|UCM_EVENT_SHMDT|UCM_EVENT_VM_MAPPED|UCM_EVENT_VM_UNMAPPED)) {
        shmid = shmget(IPC_PRIVATE, ucm_get_page_size(), IPC_CREAT | SHM_R | SHM_W);
        if (shmid == -1) {
            ucm_debug("shmget failed: %m");
            return;
        }

        UCM_FIRE_EVENT(events, UCM_EVENT_SHMAT|UCM_EVENT_VM_MAPPED,
                       data, p = shmat(shmid, NULL, 0));
#ifdef SHM_REMAP
        UCM_FIRE_EVENT(events, UCM_EVENT_SHMAT|UCM_EVENT_VM_MAPPED|UCM_EVENT_VM_UNMAPPED,
                       data, p = shmat(shmid, p, SHM_REMAP));
#endif
        shmctl(shmid, IPC_RMID, NULL);
        UCM_FIRE_EVENT(events, UCM_EVENT_SHMDT|UCM_EVENT_VM_UNMAPPED,
                       data, shmdt(p));
    }

    if (events & (UCM_EVENT_SBRK|UCM_EVENT_VM_MAPPED|UCM_EVENT_VM_UNMAPPED)) {
        if (RUNNING_ON_VALGRIND) {
            /* on valgrind, doing a non-trivial sbrk() causes heap corruption */
            sbrk_size = 0;
            sbrk_mask = UCM_EVENT_SBRK;
        } else {
            sbrk_size = ucm_get_page_size();
            sbrk_mask = UCM_EVENT_SBRK|UCM_EVENT_VM_MAPPED|UCM_EVENT_VM_UNMAPPED;
        }
        UCM_FIRE_EVENT(events, (UCM_EVENT_SBRK|UCM_EVENT_VM_MAPPED) & sbrk_mask,
                       data, (void)sbrk(sbrk_size));
        UCM_FIRE_EVENT(events, (UCM_EVENT_SBRK|UCM_EVENT_VM_UNMAPPED) & sbrk_mask,
                       data, (void)sbrk(-sbrk_size));
    }

    if (events & (UCM_EVENT_MADVISE|UCM_EVENT_VM_UNMAPPED)) {
        UCM_FIRE_EVENT(events, UCM_EVENT_MMAP|UCM_EVENT_VM_MAPPED, data,
                       p = mmap(NULL, ucm_get_page_size(), PROT_READ|PROT_WRITE,
                                MAP_PRIVATE|MAP_ANON, -1, 0));
        if (p != MAP_FAILED) {
            UCM_FIRE_EVENT(events, UCM_EVENT_MADVISE|UCM_EVENT_VM_UNMAPPED, data,
                           madvise(p, ucm_get_page_size(), MADV_DONTNEED));
            UCM_FIRE_EVENT(events, UCM_EVENT_MUNMAP|UCM_EVENT_VM_UNMAPPED, data,
                           munmap(p, ucm_get_page_size()));
        } else {
            ucm_debug("mmap failed: %m");
        }
    }
}

void ucm_fire_mmap_events(int events)
{
    ucm_mmap_test_events_data_t data;

    ucm_fire_mmap_events_internal(events, &data);
}

static void ucm_mmap_event_report_missing(int expected, int actual,
                                          const char *event_type)
{
    int events_count = 0;
    int missing_events;
    int idx;
    char *buf;
    char *buf_p;
    char *end_p;

    UCS_STATIC_ASSERT(UCM_MMAP_REPORT_BUF_LEN <= UCS_ALLOCA_MAX_SIZE)

    buf            = buf_p = ucs_alloca(UCM_MMAP_REPORT_BUF_LEN);
    end_p          = buf_p + UCM_MMAP_REPORT_BUF_LEN;
    missing_events = expected & ~actual &
                     UCS_MASK(ucs_static_array_size(ucm_mmap_event_name));

    ucs_for_each_bit(idx, missing_events) {
        /* coverity[overrun-local] */
        snprintf(buf_p, end_p - buf_p, "%s%s", ((events_count > 0) ? ", " : ""),
                 ucm_mmap_event_name[idx]);
        events_count++;
        buf_p += strlen(buf_p);
    }

    if (events_count) {
        ucm_diag("missing %s memory events: %s", event_type, buf);
    }
}

/* Called with lock held */
static ucs_status_t
ucm_mmap_test_events_nolock(int events, const char *event_type)
{
    ucm_event_handler_t handler;
    ucm_mmap_test_events_data_t data;

    handler.events    = events;
    handler.priority  = -1;
    handler.cb        = ucm_mmap_event_test_callback;
    handler.arg       = &data;
    data.out_events   = events;
    data.tid          = ucs_get_tid();

    ucm_event_handler_add(&handler);
    ucm_fire_mmap_events_internal(events, &data);
    ucm_event_handler_remove(&handler);

    ucm_debug("mmap test: got 0x%x out of 0x%x", data.out_events, events);

    /* Return success if we caught all wanted events */
    if (!ucs_test_all_flags(data.out_events, events)) {
        ucm_mmap_event_report_missing(events, data.out_events, event_type);
        return UCS_ERR_UNSUPPORTED;
    }

    return UCS_OK;
}

ucs_status_t ucm_mmap_test_events(int events, const char *event_type)
{
    ucs_status_t status;

    /*
     * return UCS_OK iff all events are actually working
     */
    pthread_mutex_lock(&ucm_mmap_install_mutex);
    status = ucm_mmap_test_events_nolock(events, event_type);
    pthread_mutex_unlock(&ucm_mmap_install_mutex);

    return status;
}

ucs_status_t ucm_mmap_test_installed_events(int events)
{
    /*
     * return UCS_OK iff all installed events are actually working
     * we don't check the status of events which were not successfully installed
     */
    return ucm_mmap_test_events(events & ucm_mmap_installed_events, "internal");
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

static int ucm_mmap_events_to_native_events(int events)
{
    int native_events;

    native_events = events & ~(UCM_EVENT_MEM_TYPE_ALLOC |
                               UCM_EVENT_MEM_TYPE_FREE);

    if (events & UCM_EVENT_VM_MAPPED) {
        native_events |= UCM_NATIVE_EVENT_VM_MAPPED;
    }
    if (events & UCM_EVENT_VM_UNMAPPED) {
        native_events |= UCM_NATIVE_EVENT_VM_UNMAPPED;
    }

    return native_events;
}

ucs_status_t ucm_mmap_install(int events)
{
    ucs_status_t status;
    int native_events;

    pthread_mutex_lock(&ucm_mmap_install_mutex);

    /* Replace aggregate events with the native events which make them */
    native_events = ucm_mmap_events_to_native_events(events);
    if (ucs_test_all_flags(ucm_mmap_installed_events, native_events)) {
        /* if we already installed these events, check that they are still
         * working, and if not - reinstall them.
         */
        status = ucm_mmap_test_events_nolock(native_events, 0);
        if (status == UCS_OK) {
            goto out_unlock;
        }
    }

    status = ucs_mmap_install_reloc(native_events);
    if (status != UCS_OK) {
        ucm_debug("failed to install relocations for mmap");
        goto out_unlock;
    }

    status = ucm_mmap_test_events_nolock(native_events, 0);
    if (status != UCS_OK) {
        ucm_debug("failed to install mmap events");
        goto out_unlock;
    }

    /* status == UCS_OK */
    ucm_mmap_installed_events |= native_events;
    ucm_debug("mmap installed events = 0x%x", ucm_mmap_installed_events);

out_unlock:
    pthread_mutex_unlock(&ucm_mmap_install_mutex);
    return status;
}
