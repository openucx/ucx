/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <ucs/arch/atomic.h>
#include <ucs/debug/log.h>
#include <ucs/debug/debug.h>
#include <ucs/sys/sys.h>

#include "notifier.h"


void ucs_notifier_chain_init(ucs_notifier_chain_t *chain)
{
    ucs_notifier_chain_elem_t *elem;

    for (elem = chain->elems; elem < chain->elems + UCS_NOTIFIER_CHAIN_MAX; ++elem) {
        elem->func     = NULL;
        elem->arg      = NULL;
        elem->refcount = 0;
        VALGRIND_MAKE_MEM_UNDEFINED(&elem->arg, sizeof(elem->arg));
    }
}

int ucs_notifier_chain_add(ucs_notifier_chain_t *chain, ucs_notifier_chain_func_t func, void *arg)
{
    ucs_notifier_chain_elem_t *elem, *free_slot;
    char func_name[200];

    ucs_notifier_chain_for_each(elem, chain) {
        if ((elem->func == func) && (elem->arg == arg)) {
            ucs_atomic_add32(&elem->refcount, 1);
            return 0;
        }
    }

    free_slot = elem;

    if (free_slot - chain->elems >= UCS_NOTIFIER_CHAIN_MAX) {
        ucs_fatal("overflow in progress chain while adding %s",
                  ucs_debug_get_symbol_name(func, func_name, sizeof(func_name)));
    }

    ucs_debug("add %s to progress chain %p",
              ucs_debug_get_symbol_name(func, func_name, sizeof(func_name)),
              chain);
    free_slot->arg      = arg;
    free_slot->refcount = 1;

    ucs_memory_cpu_store_fence();
    free_slot->func     = func;
    return 1;
}

int ucs_notifier_chain_remove(ucs_notifier_chain_t *chain, ucs_notifier_chain_func_t func, void *arg)
{
    ucs_notifier_chain_elem_t *elem, *removed_elem, *last_elem;
    char func_name[200];

    removed_elem = NULL;
    last_elem    = NULL;
    ucs_notifier_chain_for_each(elem, chain) {
        if ((elem->func == func) && (elem->arg == arg)) {
            removed_elem = elem;
        }
        last_elem = elem;
    }

    if (removed_elem == NULL) {
        ucs_debug("callback not found in progress chain");
        return 0;
    }

    if (ucs_atomic_fadd32(&removed_elem->refcount, -1) != 1) {
        return 0;
    }

    ucs_debug("remove %s from progress chain %p",
              ucs_debug_get_symbol_name(func, func_name, sizeof(func_name)),
              chain);
    *removed_elem       = *last_elem;
    last_elem->func     = NULL;
    last_elem->arg      = NULL;
    last_elem->refcount = 0;
    return 1;
}
