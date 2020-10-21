/**
 * Copyright (C) Huawei Technologies Co., Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucx_info.h"

#include <ucg/api/ucg_mpi.h>
#include <ucg/api/ucg_plan_component.h>
#include <ucg/api/ucg_mpi.h>
#include <ucs/debug/memtrack.h>

/* In accordance with @ref enum ucg_predefined */
const char *collective_names[] = {
    "barrier",
    "reduce",
    "gather",
    "bcast",
    "scatter",
    "allreduce",
    NULL
};

#define EMPTY UCG_GROUP_MEMBER_DISTANCE_LAST

ucg_address_t *worker_address = 0;
ucs_status_t dummy_resolve_address(void *cb_group_obj,
                                   ucg_group_member_index_t index,
                                   ucg_address_t **addr, size_t *addr_len)
{
    *addr = worker_address;
    *addr_len = 0; /* special debug flow: replace uct_ep_t with member indexes */
    return UCS_OK;
}

void dummy_release_address(ucg_address_t *addr) { }