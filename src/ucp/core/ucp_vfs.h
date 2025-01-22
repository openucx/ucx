/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_VFS_H_
#define UCP_VFS_H_

#include <sys/socket.h>
#include <ucs/datastruct/string_buffer.h>

/**
 * Add ip from socket address @a sockaddr to string buffer @a strb.
 *
 * @param [in]  sockaddr  Socket address.
 * @param [in]  strb      String buffer.
 */
void ucp_vfs_read_ip(const struct sockaddr *sockaddr,
                     ucs_string_buffer_t *strb);


/**
 * Add port number from socket address @a sockaddr to string buffer @a strb.
 *
 * @param [in]  sockaddr  Socket address.
 * @param [in]  strb      String buffer.
 */
void ucp_vfs_read_port(const struct sockaddr *sockaddr,
                       ucs_string_buffer_t *strb);

#endif
