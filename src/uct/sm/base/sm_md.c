/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "sm_md.h"


ucs_status_t uct_sm_rkey_ptr(uct_component_t *component, uct_rkey_t rkey,
                             void *handle, uint64_t raddr, void **laddr_p)
{
    /* rkey stores offset from the remote va */
    *laddr_p = UCS_PTR_BYTE_OFFSET(raddr, (ptrdiff_t)rkey);
    return UCS_OK;
}
