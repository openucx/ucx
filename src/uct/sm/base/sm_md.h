/**
 * Copyright (C) 2023 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_SM_MD_H_
#define UCT_SM_MD_H_

#include <uct/base/uct_md.h>
#include <ucs/config/types.h>
#include <ucs/type/status.h>


ucs_status_t uct_sm_rkey_ptr(uct_component_t *component, uct_rkey_t rkey,
                             void *handle, uint64_t raddr, void **laddr_p);

#endif
