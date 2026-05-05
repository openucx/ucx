/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "tcp_base.h"

#include <ucs/sys/string.h>

#include <limits.h>

ucs_status_t ucs_tcp_base_set_syn_cnt(int fd, int tcp_syn_cnt)
{
    if (tcp_syn_cnt != UCS_ULUNITS_AUTO) {
        return ucs_socket_setopt(fd, IPPROTO_TCP, TCP_SYNCNT,
                                 (const void*)&tcp_syn_cnt,
                                 sizeof(tcp_syn_cnt));
    }

    return UCS_OK;
}

ucs_status_t ucs_tcp_base_set_user_timeout(int fd, ucs_time_t user_timeout)
{
#ifdef TCP_USER_TIMEOUT
    int user_timeout_ms;
    double user_timeout_ms_d;
#endif

    if (user_timeout == UCS_TIME_AUTO) {
        return UCS_OK;
    }

#ifdef TCP_USER_TIMEOUT
    if (user_timeout == UCS_TIME_INFINITY) {
        user_timeout_ms = 0;
    } else {
        user_timeout_ms_d = ucs_time_to_msec(user_timeout);
        if (user_timeout_ms_d < 1.0) {
            user_timeout_ms = 1;
        } else if (user_timeout_ms_d > INT_MAX) {
            user_timeout_ms = INT_MAX;
        } else {
            user_timeout_ms = user_timeout_ms_d;
        }
    }

    return ucs_socket_setopt(fd, IPPROTO_TCP, TCP_USER_TIMEOUT,
                             (const void*)&user_timeout_ms,
                             sizeof(user_timeout_ms));
#else
    ucs_error("TCP_USER_TIMEOUT is not supported");
    return UCS_ERR_UNSUPPORTED;
#endif
}
