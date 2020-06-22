/**
* Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_TCP_BASE_H
#define UCT_TCP_BASE_H

#include <stddef.h>
#include <netinet/tcp.h>
#include <ucs/type/status.h>
#include <ucs/sys/sock.h>
#include <ucs/debug/log.h>


/**
 * TCP socket send and receive buffers configuration.
 */
typedef struct uct_tcp_send_recv_buf_config {
    size_t          sndbuf;
    size_t          rcvbuf;
} uct_tcp_send_recv_buf_config_t;


/**
 * Define configuration fields for tcp socket send and receive buffers.
 */
#define UCT_TCP_SEND_RECV_BUF_FIELDS(_offset) \
    {"SNDBUF", "auto", \
     "Socket send buffer size", \
     (_offset) + ucs_offsetof(uct_tcp_send_recv_buf_config_t, sndbuf), UCS_CONFIG_TYPE_MEMUNITS}, \
    \
    {"RCVBUF", "auto", \
     "Socket receive buffer size", \
     (_offset) + ucs_offsetof(uct_tcp_send_recv_buf_config_t, rcvbuf), UCS_CONFIG_TYPE_MEMUNITS}


#define UCT_TCP_SYN_CNT(_offset) \
    {"SYN_CNT", "auto", \
     "Number of SYN retransmits that TCP should send before aborting the attempt\n" \
     "to connect. It cannot exceed 255. auto means to use the system default.", \
     (_offset) , UCS_CONFIG_TYPE_ULUNITS}


ucs_status_t ucs_tcp_base_set_syn_cnt(int fd, int tcp_syn_cnt);

#endif /* UCT_TCP_BASE_H */
