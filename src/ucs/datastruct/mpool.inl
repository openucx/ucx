/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_MPOOL_INL_
#define UCS_MPOOL_INL_

#include "mpool.h"

#include <ucs/config/global_opts.h>
#include <ucs/sys/checker.h>
#include <ucs/sys/sys.h>


static inline void *ucs_mpool_get_inline(ucs_mpool_t *mp)
{
    ucs_mpool_elem_t *elem;
    void *obj;

    if (ucs_unlikely(mp->freelist == NULL)) {
        return ucs_mpool_get_grow(mp);
    }

    /* Disconnect an element from the pool */
    elem = mp->freelist;
    VALGRIND_MAKE_MEM_DEFINED(elem, sizeof *elem);
    mp->freelist = elem->next;
    elem->mpool = mp;
    VALGRIND_MAKE_MEM_NOACCESS(elem, sizeof *elem);

    obj = elem + 1;
    VALGRIND_MEMPOOL_ALLOC(mp, obj, mp->data->elem_size - sizeof(ucs_mpool_elem_t));
    return obj;
}

static inline void ucs_mpool_add_to_freelist(ucs_mpool_t *mp, ucs_mpool_elem_t *elem,
                                             int add_to_tail)
{
    ucs_mpool_elem_t *tail;

    if (add_to_tail) {
        elem->next = NULL;
        if (mp->freelist == NULL) {
            mp->freelist = elem;
        } else {
            tail = mp->data->tail;
            VALGRIND_MAKE_MEM_DEFINED(tail, sizeof *tail);
            tail->next = elem;
            VALGRIND_MAKE_MEM_NOACCESS(tail, sizeof *tail);
        }
        mp->data->tail = elem;
    } else {
        elem->next = mp->freelist;
        mp->freelist = elem;
    }
}

static inline ucs_mpool_elem_t *ucs_mpool_obj_to_elem(void *obj)
{
    ucs_mpool_elem_t *elem = (ucs_mpool_elem_t*)obj - 1;
    VALGRIND_MAKE_MEM_DEFINED(elem, sizeof *elem);
    return elem;
}

static inline ucs_mpool_t *ucs_mpool_obj_owner(void *obj)
{
    return ucs_mpool_obj_to_elem(obj)->mpool;
}

static inline void ucs_mpool_put_inline(void *obj)
{
    ucs_mpool_elem_t *elem;
    ucs_mpool_t *mp;

    elem = ucs_mpool_obj_to_elem(obj);
    mp   = elem->mpool;
    ucs_mpool_add_to_freelist(mp, elem,
                              ENABLE_DEBUG_DATA && ucs_global_opts.mpool_fifo);
    VALGRIND_MAKE_MEM_NOACCESS(elem, sizeof *elem);
    VALGRIND_MEMPOOL_FREE(mp, obj);
}

#endif
