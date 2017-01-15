/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_CALLBACKQ_INL
#define UCS_CALLBACKQ_INL

#include <ucs/arch/cpu.h> /* for memory load fence */
#include <ucs/datastruct/callbackq.h>


/**
 * Iterate over all elements in the callback queue.
 * This should be done only from one thread at a time.
 */
#define ucs_callbackq_for_each(_elem, _cbq) \
    for (_elem = (_cbq)->start, \
             ({ ucs_memory_cpu_load_fence(); 1; }); \
         _elem < (_cbq)->end; \
         ++_elem)


/**
 * Call all callbacks on the queue.
 * This should be done only from one thread at a time.
 *
 * Complexity: O(n)
 *
 * @param  [in] cbq      Callback queue whose elements to dispatch.
 */
static inline void ucs_callbackq_dispatch(ucs_callbackq_t *cbq)
{
    ucs_callbackq_elem_t *elem;

    ucs_callbackq_for_each(elem, cbq) {
        elem->cb(elem->arg);
    }
}
#endif
