/**
 * Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "ucp_vfs.h"

#include <ucs/sys/sock.h>

void ucp_vfs_read_ip(const struct sockaddr *sockaddr, ucs_string_buffer_t *strb)
{
    char ip_str[UCS_SOCKADDR_STRING_LEN];

    if (ucs_sockaddr_get_ipstr(sockaddr, ip_str, UCS_SOCKADDR_STRING_LEN) ==
        UCS_OK) {
        ucs_string_buffer_appendf(strb, "%s\n", ip_str);
    } else {
        ucs_string_buffer_appendf(strb, "<unable to get ip>\n");
    }
}

void ucp_vfs_read_port(const struct sockaddr *sockaddr,
                       ucs_string_buffer_t *strb)
{
    uint16_t port;

    if (ucs_sockaddr_get_port(sockaddr, &port) == UCS_OK) {
        ucs_string_buffer_appendf(strb, "%u\n", port);
    } else {
        ucs_string_buffer_appendf(strb, "<unable to get port>\n");
    }
}
