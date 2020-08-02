/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "malloc_hook.h"

#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif
#ifdef HAVE_MALLOC_NP_H
#include <malloc_np.h>
#endif
#undef M_TRIM_THRESHOLD
#undef M_MMAP_THRESHOLD
#include "allocator.h" /* have to be included after malloc.h */

#include <ucm/api/ucm.h>
#include <ucm/event/event.h>
#include <ucm/mmap/mmap.h>
#include <ucm/util/log.h>
#include <ucm/util/reloc.h>
#include <ucm/util/khash_safe.h>
#include <ucm/util/sys.h>
#include <ucs/datastruct/queue.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/math.h>
#include <ucs/sys/checker.h>
#include <ucs/sys/sys.h>
#include <ucs/type/spinlock.h>


#include <netdb.h>


/* Flags for install_state */
#define UCM_MALLOC_INSTALLED_HOOKS      UCS_BIT(0)  /* Installed malloc hooks */
#define UCM_MALLOC_INSTALLED_SBRK_EVH   UCS_BIT(1)  /* Installed sbrk event handler */
#define UCM_MALLOC_INSTALLED_OPT_SYMS   UCS_BIT(2)  /* Installed optional symbols */
#define UCM_MALLOC_INSTALLED_MALL_SYMS  UCS_BIT(3)  /* Installed malloc symbols */


/* Mangled symbols of C++ allocators */
#define UCM_OPERATOR_NEW_SYMBOL        "_Znwm"
#define UCM_OPERATOR_DELETE_SYMBOL     "_ZdlPv"
#define UCM_OPERATOR_VEC_NEW_SYMBOL    "_Znam"
#define UCM_OPERATOR_VEC_DELETE_SYMBOL "_ZdaPv"

/* Maximal size for mmap threshold - 32mb */
#define UCM_DEFAULT_MMAP_THRESHOLD_MAX (4ul * 1024 * 1024 * sizeof(long))

/* Take out 12 LSB's, since they are the page-offset on most systems */
#define ucm_mmap_addr_hash(_addr) \
    (khint32_t)((_addr >> 12) ^ (_addr & UCS_MASK(12)))

#define ucm_mmap_ptr_hash(_p)          ucm_mmap_addr_hash((uintptr_t)(_p))
#define ucm_mmap_ptr_equal(_p1, _p2)   ((_p1) == (_p2))

KHASH_INIT(mmap_ptrs, void*, char, 0, ucm_mmap_ptr_hash, ucm_mmap_ptr_equal)


/* Pointer to memory release function */
typedef void (*ucm_release_func_t)(void *ptr);

/* Pointer to get usable size function */
typedef size_t (*ucm_usable_size_func_t)(void *ptr);


typedef struct ucm_malloc_hook_state {
    /*
     * State of hook installment
     */
    pthread_mutex_t          install_mutex; /* Protect hooks installation */
    int                      install_state; /* State of hook installation */
    int                      installed_events; /* Which events are working */
    int                      mmap_thresh_set; /* mmap threshold set by user */
    int                      trim_thresh_set; /* trim threshold set by user */
    int                      hook_called; /* Our malloc hook was called */
    size_t                   max_freed_size; /* Maximal size released so far */

    ucm_usable_size_func_t   usable_size; /* function pointer to get usable size */

    ucm_release_func_t       free; /* function pointer to release memory */

    /*
     * Track record of which pointers are ours
     */
    ucs_recursive_spinlock_t lock; /* Protect heap counters.
                                         Note: Cannot modify events when this lock
                                         is held - may deadlock */
    /* Our heap address range. Used to identify whether a released pointer is ours,
     * or was allocated by the previous heap manager. */
    void                     *heap_start;
    void                     *heap_end;

    /* Save the pointers that we have allocated with mmap, so when they are
     * released we would know they are ours, despite the fact they are not in the
     * heap address range. */
    khash_t(mmap_ptrs)      ptrs;

    /**
     * Save the environment strings we've allocated
     */
    pthread_mutex_t         env_lock;
    char                    **env_strs;
    unsigned                num_env_strs;
} ucm_malloc_hook_state_t;


static ucm_malloc_hook_state_t ucm_malloc_hook_state = {
    .install_mutex    = PTHREAD_MUTEX_INITIALIZER,
    .install_state    = 0,
    .installed_events = 0,
    .mmap_thresh_set  = 0,
    .trim_thresh_set  = 0,
    .hook_called      = 0,
    .max_freed_size   = 0,
    .usable_size      = NULL,
    .free             = NULL,
    .heap_start       = (void*)-1,
    .heap_end         = (void*)-1,
    .ptrs             = {0},
    .env_lock         = PTHREAD_MUTEX_INITIALIZER,
    .env_strs         = NULL,
    .num_env_strs     = 0
};

int ucm_dlmallopt_get(int); /* implemented in ptmalloc */

static void ucm_malloc_mmaped_ptr_add(void *ptr)
{
    int hash_extra_status;
    khiter_t hash_it;

    ucs_recursive_spin_lock(&ucm_malloc_hook_state.lock);

    hash_it = kh_put(mmap_ptrs, &ucm_malloc_hook_state.ptrs, ptr,
                     &hash_extra_status);
    ucs_assert_always(hash_extra_status >= 0);
    ucs_assert_always(hash_it != kh_end(&ucm_malloc_hook_state.ptrs));

    ucs_recursive_spin_unlock(&ucm_malloc_hook_state.lock);
}

static int ucm_malloc_mmaped_ptr_remove_if_exists(void *ptr)
{
    khiter_t hash_it;
    int found;

    ucs_recursive_spin_lock(&ucm_malloc_hook_state.lock);

    hash_it = kh_get(mmap_ptrs, &ucm_malloc_hook_state.ptrs, ptr);
    if (hash_it == kh_end(&ucm_malloc_hook_state.ptrs)) {
        found = 0;
    } else {
        found = 1;
        kh_del(mmap_ptrs, &ucm_malloc_hook_state.ptrs, hash_it);
    }

    ucs_recursive_spin_unlock(&ucm_malloc_hook_state.lock);
    return found;
}

static int ucm_malloc_is_address_in_heap(void *ptr)
{
    int in_heap;

    ucs_recursive_spin_lock(&ucm_malloc_hook_state.lock);
    in_heap = (ptr >= ucm_malloc_hook_state.heap_start) &&
              (ptr < ucm_malloc_hook_state.heap_end);
    ucs_recursive_spin_unlock(&ucm_malloc_hook_state.lock);
    return in_heap;
}

static int ucm_malloc_address_remove_if_managed(void *ptr, const char *debug_name)
{
    int is_managed;

    if (ucm_malloc_is_address_in_heap(ptr)) {
        is_managed = 1;
    } else {
        is_managed = ucm_malloc_mmaped_ptr_remove_if_exists(ptr);
    }

    ucm_trace("%s(ptr=%p) - %s (heap [%p..%p])", debug_name, ptr,
              is_managed ? "ours" : "foreign",
              ucm_malloc_hook_state.heap_start, ucm_malloc_hook_state.heap_end);
    return is_managed;
}

static void ucm_malloc_allocated(void *ptr, size_t size, const char *debug_name)
{
    VALGRIND_MALLOCLIKE_BLOCK(ptr, size, 0, 0);
    if (ucm_malloc_is_address_in_heap(ptr)) {
        ucm_trace("%s(size=%zu)=%p, in heap [%p..%p]", debug_name, size, ptr,
                  ucm_malloc_hook_state.heap_start, ucm_malloc_hook_state.heap_end);
    } else {
        ucm_trace("%s(size=%zu)=%p, mmap'ed", debug_name, size, ptr);
        ucm_malloc_mmaped_ptr_add(ptr);
    }
}

static void ucm_release_foreign_block(void *ptr, ucm_release_func_t orig_free,
                                      const char *debug_name)
{
    if (RUNNING_ON_VALGRIND) {
        /* We want to keep valgrind happy and release foreign memory as well.
         * Otherwise, it's safer to do nothing.
         */
        if (orig_free == NULL) {
            ucm_fatal("%s(): foreign block release function is NULL", debug_name);
        }

        ucm_trace("%s: release foreign block %p", debug_name, ptr);
        orig_free(ptr);
    }
}

static void *ucm_malloc_impl(size_t size, const char *debug_name)
{
    void *ptr;

    ucm_malloc_hook_state.hook_called = 1;
    if (ucm_global_opts.alloc_alignment > 1) {
        ptr = ucm_dlmemalign(ucm_global_opts.alloc_alignment, size);
    } else {
        ptr = ucm_dlmalloc(size);
    }
    ucm_malloc_allocated(ptr, size, debug_name);
    return ptr;
}

static void ucm_malloc_adjust_thresholds(size_t size)
{
    int mmap_thresh;

    if (size > ucm_malloc_hook_state.max_freed_size) {
        /* Valgrind limits the size of brk() segments to 8mb, so must use mmap
         * for large allocations.
         */
        if (!RUNNING_ON_VALGRIND &&
            ucm_global_opts.enable_dynamic_mmap_thresh &&
            !ucm_malloc_hook_state.trim_thresh_set &&
            !ucm_malloc_hook_state.mmap_thresh_set) {
            /* new mmap threshold is increased to the size of released block,
             * new trim threshold is twice that size.
             */
            mmap_thresh = ucs_min(ucs_max(ucm_dlmallopt_get(M_MMAP_THRESHOLD), size),
                                  UCM_DEFAULT_MMAP_THRESHOLD_MAX);
            ucm_trace("adjust mmap threshold to %d", mmap_thresh);
            ucm_dlmallopt(M_MMAP_THRESHOLD, mmap_thresh);
            ucm_dlmallopt(M_TRIM_THRESHOLD, mmap_thresh * 2);
        }

        /* avoid adjusting the threshold for every released block, do it only
         * if the size is larger than ever before.
         */
        ucm_malloc_hook_state.max_freed_size = size;
    }
}

static inline void ucm_mem_free(void *ptr, size_t size)
{
    VALGRIND_FREELIKE_BLOCK(ptr, 0);
    VALGRIND_MAKE_MEM_UNDEFINED(ptr, size); /* Make memory accessible to ptmalloc3 */
    ucm_malloc_adjust_thresholds(size);
    ucm_dlfree(ptr);
}

static void ucm_free_impl(void *ptr, ucm_release_func_t orig_free,
                          const char *debug_name)
{
    ucm_malloc_hook_state.hook_called = 1;

    if (ptr == NULL) {
        /* Ignore */
    } else if (ucm_malloc_address_remove_if_managed(ptr, debug_name)) {
        ucm_mem_free(ptr, ucm_dlmalloc_usable_size(ptr));
    } else {
        ucm_release_foreign_block(ptr, orig_free, debug_name);
    }
}

static void *ucm_memalign_impl(size_t alignment, size_t size, const char *debug_name)
{
    void *ptr;

    ucm_malloc_hook_state.hook_called = 1;
    ptr = ucm_dlmemalign(ucs_max(alignment, ucm_global_opts.alloc_alignment), size);
    ucm_malloc_allocated(ptr, size, debug_name);
    return ptr;
}

static void *ucm_malloc(size_t size, const void *caller)
{
    return ucm_malloc_impl(size, "malloc");
}

static size_t ucm_malloc_usable_size_common(void *mem, int foreign)
{
    return foreign ? ucm_malloc_hook_state.usable_size(mem) :
                     dlmalloc_usable_size(mem);
}

static void *ucm_realloc(void *oldptr, size_t size, const void *caller)
{
    void *newptr;
    size_t oldsz;
    int foreign;

    ucm_malloc_hook_state.hook_called = 1;
    if (oldptr != NULL) {
        foreign = !ucm_malloc_address_remove_if_managed(oldptr, "realloc");
        if (RUNNING_ON_VALGRIND || foreign) {
            /*  If pointer was created by original malloc(), allocate the new pointer
             * with the new heap, and copy out the data. Then, release the old pointer.
             *  We do the same if we are running with valgrind, so we could use client
             * requests properly.
             */
            newptr = ucm_dlmalloc(size);
            ucm_malloc_allocated(newptr, size, "realloc");

            oldsz = ucm_malloc_usable_size_common(oldptr, foreign);
            memcpy(newptr, oldptr, ucs_min(size, oldsz));

            if (foreign) {
                ucm_release_foreign_block(oldptr, ucm_malloc_hook_state.free, "realloc");
            } else{
                ucm_mem_free(oldptr, oldsz);
            }
            return newptr;
        }
    }

    newptr = ucm_dlrealloc(oldptr, size);
    ucm_malloc_allocated(newptr, size, "realloc");
    return newptr;
}

static void ucm_free(void *ptr, const void *caller)
{
    ucm_free_impl(ptr, ucm_malloc_hook_state.free, "free");
}

static void *ucm_memalign(size_t alignment, size_t size, const void *caller)
{
    return ucm_memalign_impl(alignment, size, "memalign");
}

static void* ucm_calloc(size_t nmemb, size_t size)
{
    void *ptr = ucm_malloc_impl(nmemb * size, "calloc");
    if (ptr != NULL) {
        memset(ptr, 0, nmemb * size);
    }
    return ptr;
}

static void* ucm_valloc(size_t size)
{
    return ucm_malloc_impl(size, "valloc");
}

static int ucm_posix_memalign(void **memptr, size_t alignment, size_t size)
{
    void *ptr;

    if (!ucs_is_pow2(alignment)) {
        return EINVAL;
    }

    ptr = ucm_memalign_impl(alignment, size, "posix_memalign");
    if (ptr == NULL) {
        return ENOMEM;
    }

    *memptr = ptr;
    return 0;
}

static void* ucm_operator_new(size_t size)
{
    return ucm_malloc_impl(size, "operator new");
}

static void ucm_operator_delete(void* ptr)
{
    static ucm_release_func_t orig_delete = NULL;
    if (orig_delete == NULL) {
        orig_delete =
            (ucm_release_func_t)ucm_reloc_get_orig(UCM_OPERATOR_DELETE_SYMBOL,
                                                   ucm_operator_delete);
    }
    ucm_free_impl(ptr, orig_delete, "operator delete");
}

static void* ucm_operator_vec_new(size_t size)
{
    return ucm_malloc_impl(size, "operator new[]");
}

static void ucm_operator_vec_delete(void* ptr)
{
    static ucm_release_func_t orig_vec_delete = NULL;
    if (orig_vec_delete == NULL) {
        orig_vec_delete =
            (ucm_release_func_t)ucm_reloc_get_orig(UCM_OPERATOR_VEC_DELETE_SYMBOL,
                                                   ucm_operator_vec_delete);
    }
    ucm_free_impl(ptr, orig_vec_delete, "operator delete[]");
}

static int ucm_vasprintf(char **strp, const char *fmt, va_list ap)
{
    va_list ap_copy;
    size_t length;
    char *str;
    int ret;

    va_copy(ap_copy, ap);
    ret = vsnprintf(NULL, 0, fmt, ap_copy);
    va_end(ap_copy);
    if (ret < 0) {
        return ret;
    }

    length = ret + 1;
    str = ucm_malloc(length, NULL);
    if (str == NULL) {
        return -1;
    }

    ret = vsnprintf(str, length, fmt, ap);
    if (ret < 0) {
        ucm_free(str, NULL);
    } else {
        *strp = str;
    }
    return ret;
}

static int ucm_asprintf(char **strp, const char *fmt, ...)
{
    va_list ap;
    int ret;

    va_start(ap, fmt);
    ret = ucm_vasprintf(strp, fmt, ap);
    va_end(ap);
    return ret;
}

/*
 * We remember the string we pass to putenv() so we would be able to release
 * them if they are overwritten (and thus avoid leaks).
 */
static int ucm_add_to_environ(char *env_str)
{
    char *saved_env_str;
    unsigned idx;
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
    idx = 0;
    while (idx < ucm_malloc_hook_state.num_env_strs) {
        saved_env_str = ucm_malloc_hook_state.env_strs[idx];
        if ((strlen(saved_env_str) >= len) && !strncmp(env_str, saved_env_str, len)) {
            ucm_trace("replace `%s' with `%s'", saved_env_str, env_str);
            ucm_free(saved_env_str, NULL);
            goto out_insert;
        }
        ++idx;
    }

    /* Not found - enlarge array by one */
    idx = ucm_malloc_hook_state.num_env_strs;
    ++ucm_malloc_hook_state.num_env_strs;
    ucm_malloc_hook_state.env_strs =
                    ucm_realloc(ucm_malloc_hook_state.env_strs,
                                sizeof(char*) * ucm_malloc_hook_state.num_env_strs,
                                NULL);

out_insert:
    ucm_malloc_hook_state.env_strs[idx] = env_str;
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

    pthread_mutex_lock(&ucm_malloc_hook_state.env_lock);
    curr_value = getenv(name);
    if ((curr_value != NULL) && !overwrite) {
        ret = 0;
        goto out;
    }

    env_str = ucm_malloc(strlen(name) + 1 + strlen(value) + 1, NULL);
    if (env_str == NULL) {
        errno = ENOMEM;
        ret = -1;
        goto out;
    }

    sprintf(env_str, "%s=%s", name, value);
    ret = putenv(env_str);
    if (ret != 0) {
        goto err_free;
    }

    ucm_add_to_environ(env_str);
    ret = 0;
    goto out;

err_free:
    ucm_free(env_str, NULL);
out:
    pthread_mutex_unlock(&ucm_malloc_hook_state.env_lock);
    return ret;
}

static void ucm_malloc_sbrk(ucm_event_type_t event_type,
                            ucm_event_t *event, void *arg)
{
    ucs_recursive_spin_lock(&ucm_malloc_hook_state.lock);

    /* Copy return value from call. We assume the event handler uses a lock. */
    if (ucm_malloc_hook_state.heap_start == (void*)-1) {
        ucm_malloc_hook_state.heap_start = event->sbrk.result; /* sbrk() returns the previous break */
    }
    ucm_malloc_hook_state.heap_end = ucm_orig_sbrk(0);

    ucm_trace("sbrk(%+ld)=%p - adjusting heap to [%p..%p]",
              event->sbrk.increment, event->sbrk.result,
              ucm_malloc_hook_state.heap_start, ucm_malloc_hook_state.heap_end);

    ucs_recursive_spin_unlock(&ucm_malloc_hook_state.lock);
}

static int ucs_malloc_is_ready(int events, const char *title)
{
    /*
     * In RELOC mode, if malloc hooks are installed - we're good here.
     * Otherwise, we have to make sure all events are indeed working - because
     *  we can't be sure what the existing implementation is doing.
     * The implication of this is that in some cases (e.g infinite mmap threshold)
     *  we will install our memory hooks, even though it may not be required.
     */
    ucm_debug("ucs_malloc_is_ready(%s): have 0x%x/0x%x events;"
              " mmap_mode=%d hook_called=%d",
              title, ucm_malloc_hook_state.installed_events, events,
              ucm_mmap_hook_mode(), ucm_malloc_hook_state.hook_called);

    return ((ucm_mmap_hook_mode() == UCM_MMAP_HOOK_RELOC) &&
            ucm_malloc_hook_state.hook_called) ||
           ucs_test_all_flags(ucm_malloc_hook_state.installed_events, events);
}

static void ucm_malloc_event_test_callback(ucm_event_type_t event_type,
                                           ucm_event_t *event, void *arg)
{
    int *out_events = arg;

    *out_events |= event_type;
}

/* Has to be called with install_mutex held */
static void ucm_malloc_test(int events)
{
    static const size_t small_alloc_count = 128;
    static const size_t small_alloc_size  = 4096;
    static const size_t large_alloc_size  = 4 * UCS_MBYTE;
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
    handler.cb       = ucm_malloc_event_test_callback;
    handler.arg      = &out_events;
    out_events       = 0;

    ucm_event_handler_add(&handler);

    if (ucm_mmap_hook_mode() == UCM_MMAP_HOOK_RELOC) {
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
    } else {
        /* in bistro mode we can't guarantee event fire on malloc calls,
         * let's just try to call sbrk directly & catch it */
        ucm_fire_mmap_events(events);
    }

    ucm_event_handler_remove(&handler);

    ucm_malloc_hook_state.installed_events |= out_events;

    ucm_debug("malloc test: have 0x%x out of 0x%x, malloc/free hooks were%s called",
              ucm_malloc_hook_state.installed_events, events,
              ucm_malloc_hook_state.hook_called ? "" : " not");
}

static void ucm_malloc_populate_glibc_cache()
{
    char hostname[HOST_NAME_MAX];

    /* Trigger NSS initialization before we install malloc hooks.
     * This is needed because NSS could allocate strings with our malloc(), but
     * release them with the original free(). */
    (void)getlogin();
    (void)gethostbyname("localhost");
    (void)gethostname(hostname, sizeof(hostname));
    (void)gethostbyname(hostname);
}

static void ucm_malloc_install_symbols(ucm_reloc_patch_t *patches)
{
    ucm_reloc_patch_t *patch;

    for (patch = patches; patch->symbol != NULL; ++patch) {
        patch->prev_value = NULL;
        ucm_reloc_modify(patch);
    }
}

static void* ucm_malloc_patchlist_prev_value(const ucm_reloc_patch_t *patches,
                                             const char *symbol)
{
    const ucm_reloc_patch_t *patch;
    for (patch = patches; patch->symbol != NULL; ++patch) {
        if (!strcmp(patch->symbol, symbol)) {
            ucm_debug("previous function pointer for '%s' is %p", symbol,
                      patch->prev_value);
            if (patch->prev_value == NULL) {
                goto not_found;
            }

            return patch->prev_value;
        }
    }

not_found:
    ucm_fatal("could not find the previous value of '%s'", symbol);
    return NULL;
}

static int ucm_malloc_mallopt(int param_number, int value)
{
    int success;

    success = ucm_dlmallopt(param_number, value);
    if (success) {
        switch (param_number) {
        case M_TRIM_THRESHOLD:
            ucm_malloc_hook_state.trim_thresh_set = 1;
            break;
        case M_MMAP_THRESHOLD:
            ucm_malloc_hook_state.mmap_thresh_set = 1;
            break;
        }
    }
    return success;
}

static size_t ucm_malloc_usable_size(void *mem)
{
    return ucm_malloc_usable_size_common(mem,
                                         !ucm_malloc_is_address_in_heap(mem));
}

static char *ucm_malloc_blacklist[] = {
    "libnvidia-fatbinaryloader.so",
    NULL
};

static ucm_reloc_patch_t ucm_malloc_symbol_patches[] = {
    { .symbol = "free", .value = ucm_free, .blacklist = ucm_malloc_blacklist },
    { .symbol = "realloc", .value = ucm_realloc, .blacklist = ucm_malloc_blacklist },
    { .symbol = "malloc", .value = ucm_malloc, .blacklist = ucm_malloc_blacklist },
    { .symbol = "memalign", .value = ucm_memalign, .blacklist = ucm_malloc_blacklist },
    { .symbol = "calloc", .value = ucm_calloc, .blacklist = ucm_malloc_blacklist },
    { .symbol = "valloc", .value = ucm_valloc, .blacklist = ucm_malloc_blacklist },
    { .symbol = "posix_memalign", .value = ucm_posix_memalign, .blacklist = ucm_malloc_blacklist },
    { .symbol = "setenv", .value = ucm_setenv, .blacklist = ucm_malloc_blacklist },
    { .symbol = UCM_OPERATOR_NEW_SYMBOL, .value = ucm_operator_new, .blacklist = ucm_malloc_blacklist },
    { .symbol = UCM_OPERATOR_DELETE_SYMBOL, .value = ucm_operator_delete, .blacklist = ucm_malloc_blacklist },
    { .symbol = UCM_OPERATOR_VEC_NEW_SYMBOL, .value = ucm_operator_vec_new, .blacklist = ucm_malloc_blacklist },
    { .symbol = UCM_OPERATOR_VEC_DELETE_SYMBOL, .value = ucm_operator_vec_delete, .blacklist = ucm_malloc_blacklist },
    { .symbol = "asprintf", .value = ucm_asprintf, .blacklist = ucm_malloc_blacklist },
    { .symbol = "__asprintf", .value = ucm_asprintf, .blacklist = ucm_malloc_blacklist },
    { .symbol = "vasprintf", .value = ucm_vasprintf, .blacklist = ucm_malloc_blacklist },
    { .symbol = NULL, .value = NULL }
};

static ucm_reloc_patch_t ucm_malloc_optional_symbol_patches[] = {
    { "mallopt", ucm_malloc_mallopt },
    { "mallinfo", ucm_dlmallinfo },
    { "malloc_stats", ucm_dlmalloc_stats },
    { "malloc_trim", ucm_dlmalloc_trim },
    { "malloc_usable_size", ucm_malloc_usable_size },
    { NULL, NULL }
};

static void ucm_malloc_install_optional_symbols()
{
    if (!(ucm_malloc_hook_state.install_state & UCM_MALLOC_INSTALLED_OPT_SYMS)) {
        ucm_malloc_install_symbols(ucm_malloc_optional_symbol_patches);
        ucm_malloc_hook_state.usable_size    =
            (ucm_usable_size_func_t)ucm_malloc_patchlist_prev_value(
                                        ucm_malloc_optional_symbol_patches,
                                        "malloc_usable_size");
        ucm_malloc_hook_state.install_state |= UCM_MALLOC_INSTALLED_OPT_SYMS;
    }
}

static void ucm_malloc_set_env_mallopt()
{
    /* copy values of M_MMAP_THRESHOLD and M_TRIM_THRESHOLD
     * if they were overriden by the user
     */
    char *p;

    p = getenv("MALLOC_TRIM_THRESHOLD_");
    if (p) {
        ucm_debug("set trim_thresh to %d", atoi(p));
        ucm_malloc_mallopt(M_TRIM_THRESHOLD, atoi(p));
    }

    p = getenv("MALLOC_MMAP_THRESHOLD_");
    if (p) {
        ucm_debug("set mmap_thresh to %d", atoi(p));
        ucm_malloc_mallopt(M_MMAP_THRESHOLD, atoi(p));
    }
}

static void ucm_malloc_init_orig_funcs()
{
    /* We cannot use global initializer for these variables; if we do it,
     * GCC makes them part of .got, and patching .got actually changes the
     * values of these global variables. As a workaround, we initialize
     * them here.
     * NOTE This also makes sure that libucm.so has a reference to these symbols,
     * so patching the relocation tables would find their previous value by libucm
     */
    if (ucm_malloc_hook_state.usable_size == NULL) {
        ucm_malloc_hook_state.usable_size = (size_t (*)(void *))malloc_usable_size;
    }
    if ( ucm_malloc_hook_state.free == NULL) {
        ucm_malloc_hook_state.free = free;
    }
}

ucs_status_t ucm_malloc_install(int events)
{
    static ucm_event_handler_t sbrk_handler = {
        .events   = UCM_EVENT_SBRK,
        .priority = 1000,
        .cb       = ucm_malloc_sbrk
    };
    ucs_status_t status;

    pthread_mutex_lock(&ucm_malloc_hook_state.install_mutex);

    ucm_malloc_init_orig_funcs();

    if (ucs_malloc_is_ready(events, "before test")) {
        goto out_succ;
    }

    ucm_malloc_test(events);
    if (ucs_malloc_is_ready(events, "after test")) {
        goto out_succ;
    }

    if (!ucm_malloc_hook_state.hook_called) {
#ifdef HAVE_MALLOC_TRIM
        /* Try to leak less memory from original malloc */
        malloc_trim(0);
#endif
    }

    if (!(ucm_malloc_hook_state.install_state & UCM_MALLOC_INSTALLED_SBRK_EVH)) {
        ucm_debug("installing malloc-sbrk event handler");
        ucm_event_handler_add(&sbrk_handler);
        ucm_malloc_hook_state.install_state |= UCM_MALLOC_INSTALLED_SBRK_EVH;
    }

    /* When running on valgrind, don't even try malloc hooks.
     * We want to release original blocks to silence the leak check, so we must
     * have a way to call the original free(), also these hooks don't work with
     * valgrind anyway.
     */
#if HAVE_MALLOC_HOOK
    if (ucm_global_opts.enable_malloc_hooks) {
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
    } else
#endif
    {
        ucm_debug("using malloc hooks is disabled by configuration");
    }

    /* Install using malloc symbols */
    if (ucm_global_opts.enable_malloc_reloc) {
        if (!(ucm_malloc_hook_state.install_state & UCM_MALLOC_INSTALLED_MALL_SYMS)) {
            ucm_debug("installing malloc relocations");
            ucm_malloc_populate_glibc_cache();
            ucm_malloc_install_symbols(ucm_malloc_symbol_patches);
            ucm_malloc_hook_state.free           =
                (ucm_release_func_t)ucm_malloc_patchlist_prev_value(
                                        ucm_malloc_symbol_patches, "free");
            ucm_malloc_hook_state.install_state |= UCM_MALLOC_INSTALLED_MALL_SYMS;
        }
    } else {
        ucm_debug("installing malloc relocations is disabled by configuration");
    }

    /* Just installed the symbols, test again */
    ucm_malloc_test(events);
    if (ucs_malloc_is_ready(events, "after install")) {
        goto out_install_opt_syms;
    }

    status = UCS_ERR_UNSUPPORTED;
    goto out_unlock;

out_install_opt_syms:
    ucm_malloc_install_optional_symbols();
    ucm_malloc_set_env_mallopt();
out_succ:
    status = UCS_OK;
out_unlock:
    pthread_mutex_unlock(&ucm_malloc_hook_state.install_mutex);
    return status;
}

void ucm_malloc_state_reset(int default_mmap_thresh, int default_trim_thresh)
{
    ucm_malloc_hook_state.max_freed_size = 0;
    ucm_dlmallopt(M_MMAP_THRESHOLD, default_mmap_thresh);
    ucm_dlmallopt(M_TRIM_THRESHOLD, default_trim_thresh);
    ucm_malloc_set_env_mallopt();
}

UCS_STATIC_INIT {
    ucs_recursive_spinlock_init(&ucm_malloc_hook_state.lock, 0);
    kh_init_inplace(mmap_ptrs, &ucm_malloc_hook_state.ptrs);
}
