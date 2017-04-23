/**
 * Copyright (c) UT-Battelle, LLC. 2017. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_UGNI_DEF_H
#define UCT_UGNI_DEF_H

#include <ucs/async/async.h>

#define UCT_UGNI_MD_NAME       "ugni"
#define UCT_UGNI_HASH_SIZE     256
#define UCT_UGNI_MAX_DEVICES   2
#define UCT_UGNI_LOCAL_CQ      8192
#define UCT_UGNI_RKEY_MAGIC    0xdeadbeefLL
#define UCT_UGNI_MAX_TYPE_NAME 10
#define LEN_64                 (sizeof(uint64_t))
#define LEN_32                 (sizeof(uint32_t))
#define UGNI_GET_ALIGN         4

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

#endif
