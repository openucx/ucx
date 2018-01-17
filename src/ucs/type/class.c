/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "class.h"

#include <ucs/debug/assert.h>
#include <ucs/debug/memtrack.h>
#include <ucs/sys/math.h>


UCS_CLASS_INIT_FUNC(void)
{
    return UCS_OK;
}

UCS_CLASS_CLEANUP_FUNC(void)
{
}

ucs_class_t _UCS_CLASS_DECL_NAME(void) = {
    UCS_PP_QUOTE(void),
    0,
    NULL,
    (ucs_class_init_func_t)_UCS_CLASS_INIT_NAME(void),
    (ucs_class_cleanup_func_t)_UCS_CLASS_CLEANUP_NAME(void)
};

void ucs_class_call_cleanup_chain(ucs_class_t *cls, void *obj, int limit)
{
    ucs_class_t *c;
    int depth, skip;

    ucs_assert((limit == -1) || (limit >= 1));

    /* Count how many classes are there */
    for (depth = 0, c = cls; c != NULL; ++depth, c = c->superclass);

    /* Skip some destructors, because we may have a limit here */
    skip = (limit < 0) ? 0 : ucs_max(depth - limit, 0);
    c = cls;
    while (skip-- > 0) {
        c = c->superclass;
    }

    /* Call remaining destructors */
    while (c != NULL) {
        c->cleanup(obj);
        c = c->superclass;
    }
}

void *ucs_class_malloc(ucs_class_t *cls)
{
    return ucs_malloc(cls->size, cls->name);
}

void ucs_class_free(void *obj)
{
    ucs_free(obj);
}
