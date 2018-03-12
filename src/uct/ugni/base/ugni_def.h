/**
 * Copyright (c) UT-Battelle, LLC. 2017. ALL RIGHTS RESERVED.
 * Copyright (c) Los Alamos National Security, LLC. 2018. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_UGNI_DEF_H
#define UCT_UGNI_DEF_H

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <uct/base/uct_worker.h>
#include <ucs/async/async.h>

#define UCT_UGNI_MD_NAME        "ugni"
#define UCT_UGNI_HASH_SIZE      256
#define UCT_UGNI_MAX_DEVICES    2
#define UCT_UGNI_LOCAL_CQ       8192
#define UCT_UGNI_RKEY_MAGIC     0xdeadbeefLL
#define UCT_UGNI_MAX_TYPE_NAME  10
#define LEN_64                  (sizeof(uint64_t))
#define LEN_32                  (sizeof(uint32_t))
#define UGNI_GET_ALIGN          4
#define UCT_UGNI_INIT_FLUSH     1
#define UCT_UGNI_INIT_FLUSH_REQ 2

#define UCT_UGNI_ZERO_LENGTH_POST(len)              \
if (0 == len) {                                     \
    ucs_trace_data("Zero length request: skip it"); \
    return UCS_OK;                                  \
}

#define uct_ugni_enter_async(x) \
do {\
    ucs_trace_async("Taking lock on worker %p", (x)->super.worker); \
    UCS_ASYNC_BLOCK((x)->super.worker->async);                      \
} while(0)

#define uct_ugni_leave_async(x) \
do {\
    ucs_trace_async("Releasing lock on worker %p", (x)->super.worker);  \
    UCS_ASYNC_UNBLOCK((x)->super.worker->async);                        \
} while(0)

#if ENABLE_MT
#define uct_ugni_check_lock_needed(_cdm) UCS_THREAD_MODE_MULTI == (_cdm)->thread_mode
#define uct_ugni_cdm_init_lock(_cdm) ucs_spinlock_init(&(_cdm)->lock)
#define uct_ugni_cdm_destroy_lock(_cdm) ucs_spinlock_destroy(&(_cdm)->lock)
#define uct_ugni_cdm_lock(_cdm) \
if (uct_ugni_check_lock_needed(_cdm)) {  \
    ucs_trace_async("Taking lock");      \
    ucs_spin_lock(&(_cdm)->lock);   \
}
#define uct_ugni_cdm_unlock(_cdm) \
if (uct_ugni_check_lock_needed(_cdm)) {    \
    ucs_trace_async("Releasing lock");        \
    ucs_spin_unlock(&(_cdm)->lock);   \
}
#else
#define uct_ugni_cdm_init_lock(x) UCS_OK
#define uct_ugni_cdm_destroy_lock(x) UCS_OK
#define uct_ugni_cdm_lock(x)
#define uct_ugni_cdm_unlock(x)
#define uct_ugni_check_lock_needed(x) 0
#endif

#endif
