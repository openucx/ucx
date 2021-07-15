/**
* Copyright (C) NVIDIA Corporation. 2021.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <uct/api/uct.h>
#include <assert.h>

int main(int argc, char **argv)
{
    ucs_async_context_t *async;
    uct_worker_h worker;

    /* Initialize context */
    ucs_status_t status = ucs_async_context_create(UCS_ASYNC_MODE_THREAD_SPINLOCK, &async);
    assert(UCS_OK == status);

    /* Create a worker object */
    status = uct_worker_create(async, UCS_THREAD_MODE_SINGLE, &worker);
    assert(UCS_OK == status);

    /* Cleanup */
    uct_worker_destroy(worker);
    ucs_async_context_destroy(async);
    return 0;
}
