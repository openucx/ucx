/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "memtype_cache.h"

#include <ucs/arch/atomic.h>
#include <ucs/type/class.h>
#include <ucs/datastruct/sglib.h>
#include <ucs/datastruct/queue.h>
#include <ucs/debug/log.h>
#include <ucs/profile/profile.h>
#include <ucs/debug/memtrack.h>
#include <ucs/stats/stats.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/math.h>
#include <ucm/api/ucm.h>


#define UCS_MEMTYPE_CACHE_REGION_FMT "{[%p..%p], mem_type: %d} region"
#define UCS_MEMTYPE_CACHE_REGION_ARG(_region) \
    (_region)->address, \
    UCS_PTR_BYTE_OFFSET((_region)->address, \
                        (_region)->size), \
    (_region)->mem_type

/**
 * Comparator returns:
 *
 * - "< 0" if end address of a _elem1 <= starting address of _elem2
 *   +------+         |   +------+
 *   |_elem1|         |   |_elem1|
 *   +------+         |   +------+
 *          +------+  |                 +------+
 *          |_elem2|  |                 |_elem2|
 *          +------+  |                 +------+
 *
 * - "> 0" if starting address of a _elem1 >= end address of _elem1
 *          +------+  |                 +------+
 *          |_elem1|  |                 |_elem1|
 *          +------+  |                 +------+
 *   +------+         |   +------+
 *   |_elem2|         |   |_elem2|
 *   +------+         |   +------+
 *
 * - "0" - otherwise
 *   It searches regions that are intersected (i.e. have at least 1 shared byte)
 */
#define UCS_MEMTYPE_CACHE_RBTREE_CMP(_elem1, _elem2) \
    ((UCS_PTR_BYTE_OFFSET((_elem1)->address, (_elem1)->size) <= (_elem2)->address) ? -1 : \
     (((_elem1)->address >= UCS_PTR_BYTE_OFFSET((_elem2)->address, (_elem2)->size)) ? 1 : 0))

#if ENABLE_DEBUG_DATA
# define ucs_memtype_cache_add(_tree_p, _region) \
    do { \
        ucs_memtype_cache_region_t *found_region; \
        int ret; \
        \
        ret = sglib_ucs_memtype_cache_region_t_add_if_not_member(_tree_p, _region, \
                                                                 &found_region); \
        ucs_assertv(ret, "found - "UCS_MEMTYPE_CACHE_REGION_FMT \
                         "; insert - "UCS_MEMTYPE_CACHE_REGION_FMT, \
                    UCS_MEMTYPE_CACHE_REGION_ARG(found_region), \
                    UCS_MEMTYPE_CACHE_REGION_ARG(_region)); \
    } while (0)

# define ucs_memtype_cache_delete(_tree_p, _region) \
    do { \
        ucs_memtype_cache_region_t *found_region; \
        int ret; \
        \
        ret = sglib_ucs_memtype_cache_region_t_delete_if_member(_tree_p, _region, \
                                                                &found_region); \
        ucs_assertv(ret, "failed to delete "UCS_MEMTYPE_CACHE_REGION_FMT, \
                    UCS_MEMTYPE_CACHE_REGION_ARG(_region)); \
    } while (0)

#else
# define ucs_memtype_cache_add(_tree_p, _region) \
    sglib_ucs_memtype_cache_region_t_add(_tree_p, _region)

# define ucs_memtype_cache_delete(_tree_p, _region) \
    sglib_ucs_memtype_cache_region_t_delete(_tree_p, _region)
#endif

#define ucs_memtype_cache_find(_tree_p, _region) \
    sglib_ucs_memtype_cache_region_t_find_member(_tree_p, _region)
#define ucs_memtype_cache_iter_init(_it, _tree) \
    sglib_ucs_memtype_cache_region_t_it_init(_it, _tree)
#define ucs_memtype_cache_iter_next(_it) \
    sglib_ucs_memtype_cache_region_t_it_next(_it)


SGLIB_DEFINE_RBTREE_PROTOTYPES(ucs_memtype_cache_region_t,
                               left, right, color, UCS_MEMTYPE_CACHE_RBTREE_CMP)
SGLIB_DEFINE_RBTREE_FUNCTIONS(ucs_memtype_cache_region_t,
                              left, right, color, UCS_MEMTYPE_CACHE_RBTREE_CMP)

     
typedef struct sglib_ucs_memtype_cache_region_t_iterator ucs_memtype_cache_iter;

typedef enum {
    UCS_MEMTYPE_CACHE_ACTION_SET_MEMTYPE,
    UCS_MEMTYPE_CACHE_ACTION_REMOVE
} ucs_memtype_cache_action_t;


int ucs_memtype_cache_is_empty(ucs_memtype_cache_t *memtype_cache)
{
    return (memtype_cache->rbtree == NULL);
}

/* Lock must be held in write mode */
static void ucs_memtype_cache_insert(ucs_memtype_cache_t *memtype_cache,
                                     void *address, size_t size,
                                     ucs_memory_type_t mem_type)
{
    ucs_memtype_cache_region_t *region;

    /* Allocate structure for new region */
    region = ucs_malloc(sizeof(ucs_memtype_cache_region_t),
                        "memtype_cache_region");
    if (region == NULL) {
        ucs_warn("failed to allocate memtype_cache region");
        return;
    }

    region->address  = address;
    region->size     = size;
    region->mem_type = mem_type;

    ucs_memtype_cache_add(&memtype_cache->rbtree, region);

    ucs_trace("inserted "UCS_MEMTYPE_CACHE_REGION_FMT" to RB tree",
              UCS_MEMTYPE_CACHE_REGION_ARG(region));
}

static ucs_memtype_cache_region_t*
ucs_memtype_cache_find_region(ucs_memtype_cache_t *memtype_cache,
                              void *address, size_t size)
{
    ucs_memtype_cache_region_t key = {
        .address = address,
        .size    = size
    };

    return ucs_memtype_cache_find(memtype_cache->rbtree, &key);
}

/* find and remove all regions which intersect with specified one */
static void
ucs_memtype_cache_remove_matched_regions(ucs_memtype_cache_t *memtype_cache,
                                         void *address, size_t size,
                                         ucs_queue_head_t *regions)
{
    ucs_memtype_cache_region_t *region;

    while (1) {
        region = ucs_memtype_cache_find_region(memtype_cache, address, size);
        if (region == NULL) {
            break;
        }

        ucs_memtype_cache_delete(&memtype_cache->rbtree, region);

        ucs_trace("removed "UCS_MEMTYPE_CACHE_REGION_FMT" from RB tree",
                  UCS_MEMTYPE_CACHE_REGION_ARG(region));

        ucs_queue_push(regions, &region->elem);
    }
}

UCS_PROFILE_FUNC_VOID(ucs_memtype_cache_update_internal,
                      (memtype_cache, address, size, mem_type, action),
                      ucs_memtype_cache_t *memtype_cache, void *address,
                      size_t size, ucs_memory_type_t mem_type,
                      ucs_memtype_cache_action_t action)
{
    ucs_memtype_cache_region_t *region;
    void *region_end_p, *end_p;
    ucs_queue_head_t regions;

    ucs_queue_head_init(&regions);

    pthread_rwlock_wrlock(&memtype_cache->lock);

    ucs_memtype_cache_remove_matched_regions(memtype_cache, address,
                                             size, &regions);

    /* slice old regions by the new/removed region, to preserve
     * the previous memory type of the non-overlapping parts */
    ucs_queue_for_each_extract(region, &regions, elem, 1) {
        end_p        = UCS_PTR_BYTE_OFFSET(address, size);
        region_end_p = UCS_PTR_BYTE_OFFSET(region->address, region->size);

        ucs_assert((action != UCS_MEMTYPE_CACHE_ACTION_SET_MEMTYPE) ||
                   (region->mem_type == UCS_MEMORY_TYPE_LAST) ||
                   (region->mem_type == mem_type));

        if (address > region->address) {
            /* create previous region */
            ucs_memtype_cache_insert(memtype_cache, region->address,
                                     UCS_PTR_BYTE_DIFF(region->address,
                                                       address),
                                     region->mem_type);
        }

        if (end_p < region_end_p) {
            /* create next region */
            ucs_memtype_cache_insert(memtype_cache, end_p,
                                     UCS_PTR_BYTE_DIFF(end_p, region_end_p),
                                     region->mem_type);
        }

        ucs_free(region);
    }

    if (action == UCS_MEMTYPE_CACHE_ACTION_SET_MEMTYPE) {
        ucs_memtype_cache_insert(memtype_cache, address, size, mem_type);
    }

    pthread_rwlock_unlock(&memtype_cache->lock);
}

void ucs_memtype_cache_update(ucs_memtype_cache_t *memtype_cache, void *address,
                              size_t size, ucs_memory_type_t mem_type)
{
    ucs_memtype_cache_update_internal(memtype_cache, address, size, mem_type,
                                      UCS_MEMTYPE_CACHE_ACTION_SET_MEMTYPE);
}

static void ucs_memtype_cache_event_callback(ucm_event_type_t event_type,
                                              ucm_event_t *event, void *arg)
{
    ucs_memtype_cache_t *memtype_cache = arg;
    ucs_memtype_cache_action_t action;

    if (event_type & UCM_EVENT_MEM_TYPE_ALLOC) {
        action = UCS_MEMTYPE_CACHE_ACTION_SET_MEMTYPE;
    } else if (event_type & UCM_EVENT_MEM_TYPE_FREE) {
        action = UCS_MEMTYPE_CACHE_ACTION_REMOVE;
    } else {
        return;
    }

    ucs_memtype_cache_update_internal(memtype_cache, event->mem_type.address,
                                      event->mem_type.size,
                                      event->mem_type.mem_type, action);
}

UCS_PROFILE_FUNC(ucs_status_t, ucs_memtype_cache_lookup,
                 (memtype_cache, address, size, mem_type_p),
                 ucs_memtype_cache_t *memtype_cache, void *address,
                 size_t size, ucs_memory_type_t *mem_type_p)
{
    ucs_memtype_cache_region_t *region;
    ucs_status_t status;

    pthread_rwlock_rdlock(&memtype_cache->lock);

    region = UCS_PROFILE_CALL(ucs_memtype_cache_find_region,
                              memtype_cache, address, size);
    if ((region == NULL) ||
        (UCS_PTR_BYTE_OFFSET(address, size) >
         UCS_PTR_BYTE_OFFSET(region->address,
                             region->size))) {
        status = UCS_ERR_NO_ELEM;
        goto out_unlock;
    }

    ucs_trace("found "UCS_MEMTYPE_CACHE_REGION_FMT" in RB tree",
              UCS_MEMTYPE_CACHE_REGION_ARG(region));

    *mem_type_p = region->mem_type;
    status      = UCS_OK;

out_unlock:
    pthread_rwlock_unlock(&memtype_cache->lock);
    return status;
}

static UCS_CLASS_INIT_FUNC(ucs_memtype_cache_t)
{
    ucs_status_t status;
    int ret;

    ret = pthread_rwlock_init(&self->lock, NULL);
    if (ret) {
        ucs_error("pthread_rwlock_init() failed: %m");
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    self->rbtree = NULL;

    status = ucm_set_event_handler(UCM_EVENT_MEM_TYPE_ALLOC |
                                   UCM_EVENT_MEM_TYPE_FREE |
                                   UCM_EVENT_FLAG_EXISTING_ALLOC,
                                   1000, ucs_memtype_cache_event_callback,
                                   self);
    if (status != UCS_OK) {
        ucs_error("failed to set UCM memtype event handler: %s",
                  ucs_status_string(status));
        goto err_destroy_rwlock;
    }

    return UCS_OK;

err_destroy_rwlock:
    pthread_rwlock_destroy(&self->lock);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(ucs_memtype_cache_t)
{
    ucs_memtype_cache_region_t *region;
    ucs_memtype_cache_iter iter;

    ucm_unset_event_handler((UCM_EVENT_MEM_TYPE_ALLOC | UCM_EVENT_MEM_TYPE_FREE),
                            ucs_memtype_cache_event_callback, self);

    for (region = ucs_memtype_cache_iter_init(&iter, self->rbtree);
         region != NULL; region = ucs_memtype_cache_iter_next(&iter)) {
        ucs_free(region);
    }

    pthread_rwlock_destroy(&self->lock);
}

UCS_CLASS_DEFINE(ucs_memtype_cache_t, void);
UCS_CLASS_DEFINE_NAMED_NEW_FUNC(ucs_memtype_cache_create, ucs_memtype_cache_t,
                                ucs_memtype_cache_t)
UCS_CLASS_DEFINE_NAMED_DELETE_FUNC(ucs_memtype_cache_destroy, ucs_memtype_cache_t,
                                   ucs_memtype_cache_t)
