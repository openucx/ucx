/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucs/sys/compiler.h>

extern int test_module_loaded;

UCS_STATIC_INIT {
    ++test_module_loaded;
}
