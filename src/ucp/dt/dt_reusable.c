/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "dt_reusable.h"

#include <ucs/sys/compiler.h>


void ucp_dt_reusable_completion(uct_completion_t *self, ucs_status_t status)
{
    ucp_dt_reusable_t *reusable = ucs_container_of(self, ucp_dt_reusable_t, nc_comp);
    reusable->nc_status = status;
}

void ucp_dt_reusable_destroy(ucp_dt_reusable_t *reusable) {
    uct_md_mem_dereg(reusable->nc_md, reusable->nc_memh);
}
