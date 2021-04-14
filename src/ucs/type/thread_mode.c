/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2021.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "thread_mode.h"


const char *ucs_thread_mode_names[] = {
    [UCS_THREAD_MODE_SINGLE]     = "single",
    [UCS_THREAD_MODE_SERIALIZED] = "serialized",
    [UCS_THREAD_MODE_MULTI]      = "multi"
};
