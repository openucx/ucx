/**
* Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "tcp_base.h"

ucs_status_t ucs_tcp_base_set_syn_cnt(int fd, int tcp_syn_cnt)
{
    ucs_status_t status;

    status = ucs_socket_setopt(fd, IPPROTO_TCP, TCP_SYNCNT,
                               (const void*)&tcp_syn_cnt, sizeof(int));
    if (status != UCS_OK) {
        ucs_diag("failed to set TCP_SYNCNT to %d for fd %d", tcp_syn_cnt, fd);
    }

    /* return UCS_OK anyway since setting TCP_SYNCNT is done on best effort */
    return UCS_OK;
}
