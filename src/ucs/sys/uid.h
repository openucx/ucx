/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * Copyright (c) UT-Battelle, LLC. 2014-2019. ALL RIGHTS RESERVED.
 * Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_UID_H
#define UCS_UID_H

#include <ucs/sys/compiler_def.h>
#include <stdint.h>

BEGIN_C_DECLS

/** @file uid.h */

/**
 * Read boot ID value or use machine_guid.
 *
 * @return 64-bit value representing system ID.
 */
uint64_t ucs_get_system_id();

END_C_DECLS

#endif
