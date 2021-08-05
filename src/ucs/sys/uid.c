/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2012.  ALL RIGHTS RESERVED.
 * Copyright (c) UT-Battelle, LLC. 2014-2019. ALL RIGHTS RESERVED.
 * Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "uid.h"

#include <ucs/sys/sys.h>


uint64_t ucs_get_system_id()
{
    uint64_t high;
    uint64_t low;
    ucs_status_t status;

    status = ucs_sys_get_boot_id(&high, &low);
    if (status == UCS_OK) {
        return high ^ low;
    }

    return ucs_machine_guid();
}
