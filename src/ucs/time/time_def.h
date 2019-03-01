/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_TIME_DEF_H
#define UCS_TIME_DEF_H

#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS

/** @file time_def.h */

/**
 * @ingroup UCS_RESOURCE
 *
 * UCS time units.
 * These are not necessarily aligned with metric time units.
 * MUST compare short time values with UCS_SHORT_TIME_CMP to handle wrap-around.
 */
typedef unsigned long   ucs_time_t;

END_C_DECLS

#endif /* UCS_TIME_DEF_H */
