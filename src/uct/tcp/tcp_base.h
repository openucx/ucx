/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
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
#include <ucs/time/time.h>


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


#define UCT_TCP_USER_TIMEOUT(_offset) \
    {"USER_TIMEOUT", "auto", \
     "Maximum time that transmitted data may remain unacknowledged before TCP\n" \
     "forcibly closes the socket. auto means to use the system default.", \
     (_offset), UCS_CONFIG_TYPE_TIME_UNITS}


#define UCT_TCP_MAXSEG(_offset) \
    {"MAXSEG", "auto", \
     "Maximal segment size of TCP connections (TCP_MAXSEG socket option).\n" \
     "Lower it when the network drops larger segments without reporting the\n" \
     "path MTU. Set it to the path MTU minus the fixed IP and TCP headers:\n" \
     "40 bytes for IPv4 or 60 bytes for IPv6 (e.g. 9060 or 9040 for a path\n" \
     "MTU of 9100). TCP and IP options do not have to be subtracted, the\n" \
     "kernel accounts for them. The value must be between 88 and 32767.\n" \
     "auto - use the system default.", \
     (_offset), UCS_CONFIG_TYPE_MEMUNITS}


ucs_status_t ucs_tcp_base_set_syn_cnt(int fd, int tcp_syn_cnt);
ucs_status_t ucs_tcp_base_check_maxseg(size_t maxseg);
ucs_status_t ucs_tcp_base_set_maxseg(int fd, size_t maxseg);
ucs_status_t ucs_tcp_base_server_init(const struct sockaddr *saddr,
                                      socklen_t socklen, int backlog,
                                      int silent_bind, int reuse_addr,
                                      size_t maxseg, int *listen_fd_p);
ucs_status_t ucs_tcp_base_set_user_timeout(int fd, ucs_time_t user_timeout);

#endif /* UCT_TCP_BASE_H */
