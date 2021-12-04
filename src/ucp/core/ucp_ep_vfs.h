/**
 * Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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
