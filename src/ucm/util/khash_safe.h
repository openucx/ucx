/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCM_KHASH_SAFE_H_
#define UCM_KHASH_SAFE_H_

#include "sys.h"

#define kmalloc  ucm_sys_malloc
#define kcalloc  ucm_sys_calloc
#define kfree    ucm_sys_free
#define krealloc ucm_sys_realloc
#include <ucs/datastruct/khash.h>


#endif
