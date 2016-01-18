/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "malloc_hook.h"

#include <malloc.h>
#undef M_TRIM_THRESHOLD
#undef M_MMAP_THRESHOLD

#include <ucm/api/ucm.h>
#include <ucm/event/event.h>
#include <ucm/mmap/mmap.h>
#include <ucm/ptmalloc3/malloc-2.8.3.h>
#include <ucm/util/log.h>
#include <ucm/util/reloc.h>
#include <ucm/util/ucm_config.h>
#include <ucs/datastruct/queue.h>
#include <ucs/type/component.h>
#include <ucs/type/spinlock.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/sys.h>

#include <string.h>



/* Flags for install_state */
#define UCM_MALLOC_INSTALLED_HOOKS      UCS_BIT(0)  /* Installed malloc hooks */
#define UCM_MALLOC_INSTALLED_SBRK_EVH   UCS_BIT(1)  /* Installed sbrk event handler */
#define UCM_MALLOC_INSTALLED_OPT_SYMS   UCS_BIT(2)  /* Installed optional symbols */
#define UCM_MALLOC_INSTALLED_MALL_SYMS  UCS_BIT(3)  /* Installed malloc symbols */


typedef struct ucm_malloc_hook_state {
    /*
     * State of hook installment
     */
    pthread_mutex_t       install_mutex; /* Protect hooks installation */
    int                   install_state; /* State of hook installation */
    int                   installed_events; /* Which events are working */
    int                   hook_called; /* Our malloc hook was called */
    size_t                (*usable_size)(void*); /* function pointer to get usable size */
#if !NVALGRIND
    void                  (*free)(void*); /*function pointer to release memory */
#endif

    /*
     * Track record of which pointers are ours
     */
    ucs_spinlock_t        lock;       /* Protect heap counters.
                                         Note: Cannot modify events when this lock
                                         is held - may deadlock */
    /* Our heap address range. Used to identify whether a released pointer is ours,
     * or was allocated by the previous heap manager. */
    void                  *heap_start;
    void                  *heap_end;

    /* Save the pointers that we have allocated with mmap, so when they are
     * released we would know they are ours, despite the fact they are not in the
     * heap address range. */
    void                 **ptrs;
    unsigned             num_ptrs;
    unsigned             max_ptrs;

    /**
     * Save the environment strings we've allocated
     */
    char                 **env_strs;
    unsigned             num_env_strs;
} ucm_malloc_hook_state_t;


static ucm_malloc_hook_state_t ucm_malloc_hook_state = {
    .install_mutex    = PTHREAD_MUTEX_INITIALIZER,
    .install_state    = 0,
    .installed_events = 0,
    .hook_called      = 0,
    .usable_size      = malloc_usable_size,
#if !NVALGRIND
    .free             = free,
#endif
    .heap_start       = (void*)-1,
    .heap_end         = (void*)-1,
    .ptrs             = NULL,
    .num_ptrs         = 0,
    .max_ptrs         = 0,
    .env_strs         = NULL,
    .num_env_strs     = 0
};


static void ucm_malloc_mmaped_ptr_add(void *ptr)
{
    unsigned new_max_ptrs;
    void **buf;

    ucs_spin_lock(&ucm_malloc_hook_state.lock);

    if (ucm_malloc_hook_state.num_ptrs == ucm_malloc_hook_state.max_ptrs) {
        /* Enlarge the array if needed */
        if (ucm_malloc_hook_state.max_ptrs == 0) {
            new_max_ptrs = sysconf(_SC_PAGESIZE) / sizeof(void*);
            /* coverity[suspicious_sizeof] */
            buf = ucm_orig_mmap(NULL, new_max_ptrs * sizeof(void*),
                                PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS,
                                -1, 0);
        } else {
            new_max_ptrs = ucm_malloc_hook_state.max_ptrs * 2;
            buf = ucm_orig_mremap(ucm_malloc_hook_state.ptrs,
                                  ucm_malloc_hook_state.max_ptrs * sizeof(void*),
                                  new_max_ptrs * sizeof(void*), MREMAP_MAYMOVE);
        }
        if (buf == MAP_FAILED) {
            ucm_error("failed to allocated memory for mmap pointers: %m")
            goto out_unlock;
        }

        ucm_malloc_hook_state.ptrs     = buf;
        ucm_malloc_hook_state.max_ptrs = new_max_ptrs;
    }

    ucm_malloc_hook_state.ptrs[ucm_malloc_hook_state.num_ptrs] = ptr;
    ++ucm_malloc_hook_state.num_ptrs;
out_unlock:
    ucs_spin_unlock(&ucm_malloc_hook_state.lock);
}

static int ucm_malloc_mmaped_ptr_remove_if_exists(void *ptr)
{
    unsigned i;

    ucs_spin_lock(&ucm_malloc_hook_state.lock);
    for (i = 0; i < ucm_malloc_hook_state.num_ptrs; ++i) {
        if (ucm_malloc_hook_state.ptrs[i] == ptr) {
            --ucm_malloc_hook_state.num_ptrs;
            ucm_malloc_hook_state.ptrs[i] =
                            ucm_malloc_hook_state.ptrs[ucm_malloc_hook_state.num_ptrs];
            ucs_spin_unlock(&ucm_malloc_hook_state.lock);
            return 1;
        }
    }
    ucs_spin_unlock(&ucm_malloc_hook_state.lock);
    return 0;
}

static int ucm_malloc_is_address_in_heap(void *ptr)
{
    int in_heap;

    ucs_spin_lock(&ucm_malloc_hook_state.lock);
    in_heap = (ptr >= ucm_malloc_hook_state.heap_start) &&
              (ptr < ucm_malloc_hook_state.heap_end);
    ucs_spin_unlock(&ucm_malloc_hook_state.lock);
    return in_heap;
}

static int ucm_malloc_address_remove_if_managed(void *ptr, const char *caller)
{
    int is_managed;

    if (ucm_malloc_is_address_in_heap(ptr)) {
        is_managed = 1;
    } else {
        is_managed = ucm_malloc_mmaped_ptr_remove_if_exists(ptr);
    }

    ucm_trace("%s(ptr=%p) - %s (heap [%p..%p])", caller, ptr,
              is_managed ? "ours" : "foreign",
              ucm_malloc_hook_state.heap_start, ucm_malloc_hook_state.heap_end);
    return is_managed;
}

static void ucm_malloc_allocated(void *ptr, size_t size, const char *caller)
{
    VALGRIND_MALLOCLIKE_BLOCK(ptr, size, 0, 0);
    if (ucm_malloc_is_address_in_heap(ptr)) {
        ucm_trace("%s(size=%zu)=%p, in heap [%p..%p]", caller, size, ptr,
                  ucm_malloc_hook_state.heap_start, ucm_malloc_hook_state.heap_end);
    } else {
        ucm_trace("%s(size=%zu)=%p, mmap'ed", caller, size, ptr);
        ucm_malloc_mmaped_ptr_add(ptr);
    }
}

static void ucm_release_foreign_block(void *ptr, const char *caller)
{
#if !NVALGRIND
    if (RUNNING_ON_VALGRIND) {
        /* We want to keep valgrind happy and release foreign memory as well.
         * Otherwise, it's safer to do nothing.
         */
        ucm_trace("%s: release foreign block %p", caller, ptr);
        ucm_malloc_hook_state.free(ptr);
    }
#endif
}

static void *ucm_malloc(size_t size, const void *caller)
{
    void *ptr;

    ucm_malloc_hook_state.hook_called = 1;
    if (ucm_global_config.alloc_alignment > 1) {
        ptr = ucm_dlmemalign(ucm_global_config.alloc_alignment, size);
    } else {
        ptr = ucm_dlmalloc(size);
    }
    ucm_malloc_allocated(ptr, size, "malloc");
    return ptr;
}

static inline void ucm_mem_free(void *ptr, size_t size)
{
    VALGRIND_FREELIKE_BLOCK(ptr, 0);
    VALGRIND_MAKE_MEM_UNDEFINED(ptr, size); /* Make memory accessible to ptmalloc3 */
    ucm_dlfree(ptr);
}

static void *ucm_realloc(void *oldptr, size_t size, const void *caller)
{
    void *newptr;
    size_t oldsz;
    int foreign;

    ucm_malloc_hook_state.hook_called = 1;

    foreign = (oldptr != NULL) &&
              !ucm_malloc_address_remove_if_managed(oldptr, "realloc");
    if ((oldptr != NULL) && (RUNNING_ON_VALGRIND || foreign)) {
        /*  If pointer was created by original malloc(), allocate the new pointer
         * with the new heap, and copy out the data. Then, release the old pointer.
         *  We do the same if we are running with valgrind, so we could use client
         * requests properly.
         */
        newptr = ucm_dlmalloc(size);
        ucm_malloc_allocated(newptr, size, "realloc");

        oldsz = ucm_malloc_hook_state.usable_size(oldptr);
        memcpy(newptr, oldptr, ucs_min(size, oldsz));

        if (foreign) {
            ucm_release_foreign_block(oldptr, "realloc");
        } else{
            ucm_mem_free(oldptr, oldsz);
        }
    } else {
        newptr = ucm_dlrealloc(oldptr, size);
        ucm_malloc_allocated(newptr, size, "realloc");
    }
    return newptr;
}

static void ucm_free(void *ptr, const void *caller)
{
    ucm_malloc_hook_state.hook_called = 1;

    if (ptr == NULL) {
        /* Ignore */
    } else if (ucm_malloc_address_remove_if_managed(ptr, "free")) {
        ucm_mem_free(ptr, ucm_dlmalloc_usable_size(ptr));
    } else {
        ucm_release_foreign_block(ptr, "free");
    }
}

static void *ucm_memalign(size_t alignment, size_t size, const void *caller)
{
    void *ptr;

    ucm_malloc_hook_state.hook_called = 1;
    ptr = ucm_dlmemalign(ucs_max(alignment, ucm_global_config.alloc_alignment), size);
    ucm_malloc_allocated(ptr, size, "memalign");
    return ptr;
}

static void* ucm_calloc(size_t nmemb, size_t size)
{
    void *ptr = ucm_malloc(nmemb * size, NULL);
    if (ptr != NULL) {
        memset(ptr, 0, nmemb * size);
    }
    return ptr;
}

static void* ucm_valloc(size_t size)
{
    return ucm_malloc(size, NULL);
}

static int ucm_posix_memalign(void **memptr, size_t alignment, size_t size)
{
    void *ptr;

    if (!ucs_is_pow2(alignment)) {
        return EINVAL;
    }

    ptr = ucm_memalign(alignment, size, NULL);
    if (ptr == NULL) {
        return ENOMEM;
    }

    *memptr = ptr;
    return 0;
}

/*
 * We remember the string we pass to putenv() so we would be able to release them
 * during library destructor (and thus avoid leaks). Also, if a variable is replaced,
 * we release the old string.
 *
 * This function is **not** thread safe, since setenv() is not thread safe anyway.
 */
static int ucm_add_to_environ(char *env_str)
{
    char *saved_env_str;
    unsigned index;
    size_t len;
    char *p;

    /* Get name length */
    p = strchr(env_str, '=');
    if (p == NULL) {
        len = strlen(env_str); /* Compare whole string */
    } else {
        len = p + 1 - env_str; /* Compare up to and including the '=' character */
    }

    /* Check if we already have variable with same name */
    index = 0;
    while (index < ucm_malloc_hook_state.num_env_strs) {
        saved_env_str = ucm_malloc_hook_state.env_strs[index];
        if ((strlen(saved_env_str) >= len) && !strncmp(env_str, saved_env_str, len)) {
            ucm_trace("replace `%s' with `%s'", saved_env_str, env_str);
            ucm_free(saved_env_str, NULL);
            goto out_insert;
        }
        ++index;
    }

    /* Not found - enlarge array by one */
    index = ucm_malloc_hook_state.num_env_strs;
    ++ucm_malloc_hook_state.num_env_strs;
    ucm_malloc_hook_state.env_strs =
                    ucm_realloc(ucm_malloc_hook_state.env_strs,
                                sizeof(char*) * ucm_malloc_hook_state.num_env_strs,
                                NULL);

out_insert:
    ucm_malloc_hook_state.env_strs[index] = env_str;
    return 0;
}

/*
 * We need to replace setenv() because glibc keeps a search tree of environment
 * strings and releases it with *original* free() (in __tdestroy).
 * If we always use putenv() instead of setenv() this search tree will not be used.
 */
static int ucm_setenv(const char *name, const char *value, int overwrite)
{
    char *curr_value;
    char *env_str;
    int ret;

    curr_value = getenv(name);
    if ((curr_value != NULL) && !overwrite) {
        return 0;
    }

    env_str = ucm_malloc(strlen(name) + 1 + strlen(value) + 1, NULL);
    if (env_str == NULL) {
        errno = ENOMEM;
        ret = -1;
        goto err;
    }

    sprintf(env_str, "%s=%s", name, value);
    ret = putenv(env_str);
    if (ret != 0) {
        goto err_free;
    }

    ucm_add_to_environ(env_str);
    return 0;

err_free:
    ucm_free(env_str, NULL);
err:
    return ret;
}

static void ucm_malloc_sbrk(ucm_event_type_t event_type,
                            ucm_event_t *event, void *arg)
{
    ucs_spin_lock(&ucm_malloc_hook_state.lock);

    /* Copy return value from call. We assume the event handler uses a lock. */
    if (ucm_malloc_hook_state.heap_start == (void*)-1) {
        ucm_malloc_hook_state.heap_start = event->sbrk.result - event->sbrk.increment;
    }
    ucm_malloc_hook_state.heap_end = ucm_orig_sbrk(0);

    ucm_debug("sbrk(%+ld)=%p - adjusting heap to [%p..%p]",
              event->sbrk.increment, event->sbrk.result,
              ucm_malloc_hook_state.heap_start, ucm_malloc_hook_state.heap_end);

    ucs_spin_unlock(&ucm_malloc_hook_state.lock);
}

static int ucs_malloc_is_ready(int events)
{
    /*
     * If malloc hooks are installed - we're good here.
     * Otherwise, we have to make sure all events are indeed working - because
     *  we can't be sure what the existing implementation is doing.
     * The implication of this is that in some cases (e.g infinite mmap threshold)
     *  we will install out memory hooks, even though it may not be required.
     */
    return ucm_malloc_hook_state.hook_called ||
           ucs_test_all_flags(ucm_malloc_hook_state.installed_events, events);
}

static ucm_event_handler_t ucm_malloc_sbrk_handler = {
    .events   = UCM_EVENT_SBRK,
    .priority = 1000,
    .cb       = ucm_malloc_sbrk
};

/* Has to be called with install_mutex held */
static void ucm_malloc_test(int events)
{
    static const size_t small_alloc_count = 128;
    static const size_t small_alloc_size  = 4096;
    static const size_t large_alloc_size  = 4 * 1024 * 1024;
    ucm_event_handler_t handler;
    void *p[small_alloc_count];
    int out_events;
    int i;

    ucm_debug("testing malloc...");

    /* Install a temporary event handler which will add the supported event
     * type to out_events bitmap.
     */
    handler.events   = events;
    handler.priority = -1;
    handler.cb       = ucm_mmap_event_test_callback;
    handler.arg      = &out_events;
    out_events       = 0;

    ucm_event_handler_add(&handler);

    /* Trigger both small and large allocations
     * TODO check address / stop all threads */
    for (i = 0; i < small_alloc_count; ++i) {
        p[i] = malloc(small_alloc_size);
    }
    for (i = 0; i < small_alloc_count; ++i) {
        free(p[i]);
    }
    p[0] = malloc(large_alloc_size);
    p[0] = realloc(p[0], large_alloc_size * 2);
    free(p[0]);

    if (ucm_malloc_hook_state.hook_called) {
        ucm_dlmalloc_trim(0);
    }

    ucm_event_handler_remove(&handler);

    ucm_malloc_hook_state.installed_events |= out_events;

    ucm_debug("malloc test: have 0x%x out of 0x%x, hooks were%s called",
              ucm_malloc_hook_state.installed_events, events,
              ucm_malloc_hook_state.hook_called ? "" : " not");
}

static void ucm_malloc_install_optional_symbols()
{
    #define UCM_MALLOC_HOOK_DL_SYMBOL_PATCH(_name) \
        ucm_reloc_patch_t _name##_patch = { #_name, ucm_dl##_name }
    static UCM_MALLOC_HOOK_DL_SYMBOL_PATCH(mallopt);
    static UCM_MALLOC_HOOK_DL_SYMBOL_PATCH(mallinfo);
    static UCM_MALLOC_HOOK_DL_SYMBOL_PATCH(malloc_stats);
    static UCM_MALLOC_HOOK_DL_SYMBOL_PATCH(malloc_trim);
    static UCM_MALLOC_HOOK_DL_SYMBOL_PATCH(malloc_usable_size);

    if (!ucm_global_config.enable_reloc_hooks) {
        return;
    }

    if (ucm_malloc_hook_state.install_state & UCM_MALLOC_INSTALLED_OPT_SYMS) {
        return;
    }

    ucm_reloc_modify(&mallopt_patch);
    ucm_reloc_modify(&mallinfo_patch);
    ucm_reloc_modify(&malloc_stats_patch);
    ucm_reloc_modify(&malloc_trim_patch);
    ucm_reloc_modify(&malloc_usable_size_patch);

    ucm_malloc_hook_state.install_state |= UCM_MALLOC_INSTALLED_OPT_SYMS;
}

ucs_status_t ucm_malloc_install(int events)
{
    ucs_status_t status;

    pthread_mutex_lock(&ucm_malloc_hook_state.install_mutex);

    events &= UCM_EVENT_MMAP | UCM_EVENT_MUNMAP | UCM_EVENT_MREMAP | UCM_EVENT_SBRK;

    if (ucs_malloc_is_ready(events)) {
        goto out_succ;
    }

    ucm_malloc_test(events);
    if (ucs_malloc_is_ready(events)) {
        goto out_succ;
    }

    if (!ucm_malloc_hook_state.hook_called) {
        /* Try to leak less memory from original malloc */
        malloc_trim(0);
    }

    if (!(ucm_malloc_hook_state.install_state & UCM_MALLOC_INSTALLED_SBRK_EVH)) {
        ucm_debug("installing malloc-sbrk event handler");
        ucm_event_handler_add(&ucm_malloc_sbrk_handler);
        ucm_malloc_hook_state.install_state |= UCM_MALLOC_INSTALLED_SBRK_EVH;
    }

    /* When running on valgrind, don't even try malloc hooks.
     * We want to release original blocks to silence the leak check, so we must
     * have a way to call the original free(), also these hooks don't work with
     * valgrind anyway.
     */
    if (ucm_global_config.enable_malloc_hooks) {
        /* Install using malloc hooks.
         * TODO detect glibc support in configure-time.
         */
        if (!(ucm_malloc_hook_state.install_state & UCM_MALLOC_INSTALLED_HOOKS)) {
            ucm_debug("installing malloc hooks");
            __free_hook     = ucm_free;
            __realloc_hook  = ucm_realloc;
            __malloc_hook   = ucm_malloc;
            __memalign_hook = ucm_memalign;
            ucm_malloc_hook_state.install_state |= UCM_MALLOC_INSTALLED_HOOKS;
        }

        /* Just installed the hooks, test again. */
        ucm_malloc_test(events);
        if (ucm_malloc_hook_state.hook_called) {
            goto out_install_opt_syms;
        }
    }

    /* Install using malloc symbols */
    if (ucm_global_config.enable_reloc_hooks &&
        !(ucm_malloc_hook_state.install_state & UCM_MALLOC_INSTALLED_MALL_SYMS))
    {
        #define UCM_MALLOC_HOOK_UCM_SYMBOL_PATCH(_name) \
            ucm_reloc_patch_t _name##_patch = { #_name, ucm_##_name }
        static UCM_MALLOC_HOOK_UCM_SYMBOL_PATCH(free);
        static UCM_MALLOC_HOOK_UCM_SYMBOL_PATCH(realloc);
        static UCM_MALLOC_HOOK_UCM_SYMBOL_PATCH(malloc);
        static UCM_MALLOC_HOOK_UCM_SYMBOL_PATCH(memalign);
        static UCM_MALLOC_HOOK_UCM_SYMBOL_PATCH(calloc);
        static UCM_MALLOC_HOOK_UCM_SYMBOL_PATCH(valloc);
        static UCM_MALLOC_HOOK_UCM_SYMBOL_PATCH(posix_memalign);
        static UCM_MALLOC_HOOK_UCM_SYMBOL_PATCH(setenv);

        ucm_debug("installing malloc hooks");
        ucm_reloc_modify(&free_patch);
        ucm_reloc_modify(&realloc_patch);
        ucm_reloc_modify(&malloc_patch);
        ucm_reloc_modify(&memalign_patch);
        ucm_reloc_modify(&calloc_patch);
        ucm_reloc_modify(&valloc_patch);
        ucm_reloc_modify(&posix_memalign_patch);
        ucm_reloc_modify(&setenv_patch);
        ucm_malloc_hook_state.install_state |= UCM_MALLOC_INSTALLED_MALL_SYMS;
    }

    /* Just installed the symbols, test again */
    ucm_malloc_test(events);
    if (ucm_malloc_hook_state.hook_called) {
        goto out_install_opt_syms;
    }

    return UCS_ERR_UNSUPPORTED;

out_install_opt_syms:
    ucm_malloc_install_optional_symbols();
out_succ:
    status = UCS_OK;
    pthread_mutex_unlock(&ucm_malloc_hook_state.install_mutex);
    return status;
}

UCS_STATIC_INIT {
    ucs_spinlock_init(&ucm_malloc_hook_state.lock);

    /* Trigger NSS initialization before we install malloc hooks.
     * This is needed because NSS could allocate strings with our malloc(), but
     * release them with the original free(). */
    (void)getlogin();
}

static void UCS_F_DTOR ucm_clear_env()
{
    unsigned i;

    clearenv();
    for (i = 0; i < ucm_malloc_hook_state.num_env_strs; ++i) {
        ucm_free(ucm_malloc_hook_state.env_strs[i], NULL);
    }
    ucm_free(ucm_malloc_hook_state.env_strs, NULL);
}
