/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "class.h"

#include <ucs/debug/assert.h>
#include <ucs/debug/memtrack_int.h>
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

    ucs_assert(((limit == -1) || (limit >= 1)) && (cls != NULL));

    /* Count how many classes are there */
    for (depth = 0, c = cls; c != NULL; ++depth, c = c->superclass);

    /* Skip some destructors, because we may have a limit here */
    skip = (limit < 0) ? 0 : ucs_max(depth - limit, 0);
    c = cls;

    /* check for NULL pointer to suppress clang warning */
    while ((skip-- > 0) && (c != NULL)) {
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

void ucs_class_check_new_func_result(ucs_status_t status, void *obj)
{
    ucs_assert((status == UCS_OK) || (obj == NULL));
}
