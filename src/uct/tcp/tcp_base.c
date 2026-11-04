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


/* The range of TCP_MAXSEG values accepted by Linux: TCP_MIN_MSS and
 * MAX_TCP_WINDOW */
#define UCT_TCP_MAXSEG_MIN 88
#define UCT_TCP_MAXSEG_MAX 32767


ucs_status_t ucs_tcp_base_set_syn_cnt(int fd, int tcp_syn_cnt)
{
    if (tcp_syn_cnt != UCS_ULUNITS_AUTO) {
        return ucs_socket_setopt(fd, IPPROTO_TCP, TCP_SYNCNT,
                                 (const void*)&tcp_syn_cnt,
                                 sizeof(tcp_syn_cnt));
    }

    return UCS_OK;
}

ucs_status_t ucs_tcp_base_check_maxseg(size_t maxseg)
{
    char buf[64];

    if ((maxseg != UCS_MEMUNITS_AUTO) &&
        ((maxseg < UCT_TCP_MAXSEG_MIN) || (maxseg > UCT_TCP_MAXSEG_MAX))) {
        ucs_error("invalid TCP maximal segment size %s, expected auto or a "
                  "value between %d and %d",
                  ucs_memunits_to_str(maxseg, buf, sizeof(buf)),
                  UCT_TCP_MAXSEG_MIN, UCT_TCP_MAXSEG_MAX);
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

ucs_status_t ucs_tcp_base_set_maxseg(int fd, size_t maxseg)
{
    int maxseg_int;

    if (maxseg == UCS_MEMUNITS_AUTO) {
        return UCS_OK;
    }

    maxseg_int = maxseg;
    return ucs_socket_setopt(fd, IPPROTO_TCP, TCP_MAXSEG,
                             (const void*)&maxseg_int, sizeof(maxseg_int));
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
