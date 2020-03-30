/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/type/init_once.h>
#include <ucs/debug/assert.h>


unsigned ucs_init_once_mutex_unlock(pthread_mutex_t *lock)
{
    int ret = pthread_mutex_unlock(lock);
    ucs_assert_always(ret == 0);
    return 0;
}
