/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef LOCKLESS_SYNC_H_
#define LOCKLESS_SYNC_H_

#include "lockless_sync_def.h"

#include <ucs/arch/cpu.h>
#include <ucs/debug/assert.h>


/**
 * Lock-less synchronization for objects stored in a container. The objects
 * may be invalidated and released from an asynchronous garbage collecting
 * thread. The main thread can safely utilize an object acquired with
 * ucs_lockless_sync_get() before it released with ucs_lockless_sync_put().
 * The asynchronous thread may:
 * - Mark regions for invalidation. If ucs_lockless_sync_invalidate() returns true,
 *   the object should be queued for content release.
 * - Release its content if ucs_lockless_sync_release() returns true.
 *
 * Synchronization restrictions:
 * - refcount modified from the main thread only.
 * - flags modified under the container lock only.
 */


enum {
    UCS_LOCKLESS_SYNC_FLAG_STORED  = UCS_BIT(0), /**< Object held by container */
    UCS_LOCKLESS_SYNC_FLAG_READY   = UCS_BIT(2), /**< Object ready for use */
    UCS_LOCKLESS_SYNC_FLAG_INVALID = UCS_BIT(1), /**< Object invalidated */
};


#define UCS_LOCKLESS_SYNC_FMT " %c%c%c ref %u"
#define UCS_LOCKLESS_SYNC_ARG(_obj) \
    ((_obj)->flags & UCS_LOCKLESS_SYNC_FLAG_READY)   ? 'r' : '-', \
    ((_obj)->flags & UCS_LOCKLESS_SYNC_FLAG_INVALID) ? 'i' : '-', \
    ((_obj)->flags & UCS_LOCKLESS_SYNC_FLAG_STORED)  ? 's' : '-', \
    (_obj)->refcount


/**
 * Initialize object.
 * Called from main thread under container lock.
 *
 * @param [in]  obj         Pointer to an object.
 */
static UCS_F_ALWAYS_INLINE void
ucs_lockless_sync_init(ucs_ll_sync_obj_t *obj)
{
    obj->refcount = 1;
    obj->flags    = UCS_LOCKLESS_SYNC_FLAG_STORED;
}


/**
 * Transit object to ready state.
 * Called from main thread under container lock.
 *
 * @param [in]  obj         Pointer to an object.
 */
static UCS_F_ALWAYS_INLINE void
ucs_lockless_sync_set_ready(ucs_ll_sync_obj_t *obj)
{
    obj->refcount++;
    obj->flags |= UCS_LOCKLESS_SYNC_FLAG_READY;
}


/**
 * If object is usable acquire it and return true.
 * Called from main thread.
 *
 * @param [in]  obj         Pointer to an object.
 *
 * @return true if acquired succefully.
 */
static UCS_F_ALWAYS_INLINE int
ucs_lockless_sync_get(ucs_ll_sync_obj_t *obj)
{
    obj->refcount++;
    ucs_memory_cpu_store_fence();

    if (ucs_unlikely((obj->flags & (UCS_LOCKLESS_SYNC_FLAG_READY |
                                    UCS_LOCKLESS_SYNC_FLAG_INVALID)) !=
                     UCS_LOCKLESS_SYNC_FLAG_READY)) {
        obj->refcount--;
        return 0;
    }

    return 1;
}


/**
 * Release object.
 * Called from main thread.
 *
 * @param [in]  obj         Pointer to an object.
 *
 * @return true if object should be destroyed.
 */
static UCS_F_ALWAYS_INLINE int
ucs_lockless_sync_put(ucs_ll_sync_obj_t *obj)
{
    ucs_assert(obj->refcount > 0);
    return --obj->refcount == 0;
}


/**
 * If object is stored in container and has no references return true.
 * Called from main thread.
 *
 * @param [in]  obj         Pointer to an object.
 *
 * @return true if object can be evicted.
 */
static UCS_F_ALWAYS_INLINE int
ucs_lockless_sync_evict(ucs_ll_sync_obj_t *obj)
{
    return (obj->flags & UCS_LOCKLESS_SYNC_FLAG_STORED) && (obj->refcount == 1);
}


/**
 * If object is stored in container mark it as removed and return true.
 * If object has no references and may be destroyed set destroy to true.
 * Called from main thread under container lock.
 *
 * @param [in]  obj         Pointer to an object.
 * @param [out] destroy     true if object should be destroyed.
 *
 * @return true if object should be removed from container.
 */
static UCS_F_ALWAYS_INLINE int
ucs_lockless_sync_remove(ucs_ll_sync_obj_t *obj, int *destroy)
{
    unsigned flags;

    ucs_memory_cpu_load_fence();
    flags = obj->flags;
    if (flags & UCS_LOCKLESS_SYNC_FLAG_STORED) {
        obj->flags &= ~UCS_LOCKLESS_SYNC_FLAG_STORED;
        *destroy = (--obj->refcount == 0);
        return 1;
    }

    *destroy = (obj->refcount == 0);
    return 0;
}

/**
 * Mark object as invalid. Return true is it wasn't marked invalid before and
 * should be invalidated.
 * Called from asynchronous GC thread under container lock.
 *
 * @param [in]  obj         Pointer to an object.
 *
 * @return true if object should be invalidated.
 */
static UCS_F_ALWAYS_INLINE int
ucs_lockless_sync_invalidate(ucs_ll_sync_obj_t *obj)
{
    unsigned flags;

    ucs_memory_cpu_load_fence();
    flags = obj->flags;
    if (flags & UCS_LOCKLESS_SYNC_FLAG_INVALID) {
        return 0;
    }

    ucs_assert(flags & UCS_LOCKLESS_SYNC_FLAG_STORED);
    obj->flags |= UCS_LOCKLESS_SYNC_FLAG_INVALID;
    ucs_memory_cpu_store_fence();
    return 1;
}

/**
 * If objects content may be released return true and mark object as not usable.
 * Called from asynchronous GC thread under container lock.
 *
 * @param [in]  obj         Pointer to an object.
 *
 * @return true if object's content should be released.
 */
static UCS_F_ALWAYS_INLINE int
ucs_lockless_sync_release(ucs_ll_sync_obj_t *obj)
{
    unsigned flags;

    ucs_memory_cpu_load_fence();
    flags = obj->flags;
    ucs_assert(flags & UCS_LOCKLESS_SYNC_FLAG_INVALID);
    if (obj->refcount > !!(flags & UCS_LOCKLESS_SYNC_FLAG_STORED)) {
        return 0;
    }

    if (flags & UCS_LOCKLESS_SYNC_FLAG_READY) {
        obj->flags &= ~UCS_LOCKLESS_SYNC_FLAG_READY;
        return 1;
    }

    return 0;
}

#endif
