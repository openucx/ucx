/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_MPOOL_INL_
#define UCS_MPOOL_INL_

#include "mpool.h"

#include <ucs/sys/sys.h>


static inline void *ucs_mpool_get_inline(ucs_mpool_t *mp)
{
    ucs_mpool_elem_t *elem;
    void *obj;

    if (mp->freelist == NULL) {
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

static inline void ucs_mpool_put_inline(void *obj)
{
    ucs_mpool_elem_t *elem;
    ucs_mpool_t *mp;

    /* Reconnect the element to the pool */
    elem = (ucs_mpool_elem_t*)obj - 1;
    VALGRIND_MAKE_MEM_DEFINED(elem, sizeof *elem);
    mp = elem->mpool;
    elem->next = mp->freelist;
    VALGRIND_MAKE_MEM_NOACCESS(elem, sizeof *elem);
    mp->freelist = elem;
    VALGRIND_MEMPOOL_FREE(mp, obj);
}

#endif
