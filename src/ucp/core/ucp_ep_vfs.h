/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_EP_VFS_H_
#define UCP_EP_VFS_H_

#include <ucp/api/ucp_def.h>


/**
 * @brief Create objects in VFS to represent endpoint and its features.
 *
 * @param [in] ep Endpoint object to be described.
 */
void ucp_ep_vfs_init(ucp_ep_h ep);

#endif
