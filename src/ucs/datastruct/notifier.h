/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2012.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCS_DATASTRUCT_NOTIFIER_H
#define UCS_DATASTRUCT_NOTIFIER_H

#include <ucs/type/callback.h>


/*
 * Forward declarations
 */
typedef struct ucs_notifier_chain      ucs_notifier_chain_t;
typedef struct ucs_notifier_chain_elem ucs_notifier_chain_elem_t;
typedef void                           (*ucs_notifier_chain_func_t)(void *arg);

#define UCS_NOTIFIER_CHAIN_MAX         16


struct ucs_notifier_chain_elem {
    ucs_notifier_chain_func_t func;
    void                      *arg;
    unsigned                  refcount;
};

/**
 * A list of callbacks. It's a circular single-linked list, and cursor points
 * to one of the elements.
 */
struct ucs_notifier_chain {
    ucs_notifier_chain_elem_t elems[UCS_NOTIFIER_CHAIN_MAX];
};


/**
 * Iterate over all elements in the notifier chain
 */
#define ucs_notifier_chain_for_each(_elem, _chain) \
    for (_elem = &(_chain)->elems[0]; _elem->func != NULL; ++_elem)


void ucs_notifier_chain_init(ucs_notifier_chain_t *chain);

static inline void ucs_notifier_chain_call(ucs_notifier_chain_t *chain)
{
    ucs_notifier_chain_elem_t *elem;

    ucs_notifier_chain_for_each(elem, chain) {
        elem->func(elem->arg);
    }
}


/* @return whether added a new entry */
int ucs_notifier_chain_add(ucs_notifier_chain_t *chain,
                           ucs_notifier_chain_func_t func, void *arg);

/* @return whether removed an entry */
int ucs_notifier_chain_remove(ucs_notifier_chain_t *chain,
                              ucs_notifier_chain_func_t func, void *arg);


#endif
