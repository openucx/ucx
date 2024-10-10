/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucs_netlink.h"

#include <ucs/sys/sock.h>
#include <ucs/sys/compiler.h>
#include <ucs/type/status.h>
#include <ucs/debug/log.h>

#include <errno.h>
#include <linux/netlink.h>
#include <sys/socket.h>
#include <unistd.h>


static ucs_status_t ucs_netlink_socket_create(int *fd, int protocol)
{
    ucs_status_t ret;
    struct sockaddr_nl sa = {0};

    ret = ucs_socket_create(AF_NETLINK, SOCK_RAW, protocol, fd);
    if (ret != UCS_OK) {
        ucs_diag("failed to create netlink socket %d", ret);
        *fd = -1;
        return ret;
    }

    sa.nl_family = AF_NETLINK;

    if (bind(*fd, (struct sockaddr*)&sa, sizeof(sa)) < 0) {
        ucs_close_fd(fd);
        ucs_diag("failed to bind netlink socket %d", *fd);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

ucs_status_t
ucs_netlink_send_cmd(int protocol, void *nl_protocol_hdr,
                     size_t nl_protocol_hdr_size, char *recv_msg_buf,
                     size_t *recv_msg_buf_len, unsigned short nlmsg_type)
{
    ucs_status_t ret;
    int fd;
    struct nlmsghdr nlh;
    struct iovec iov[2];
    size_t send_msg_len = NLMSG_LENGTH(nl_protocol_hdr_size);

    ret = ucs_netlink_socket_create(&fd, NETLINK_ROUTE);
    if (ret != UCS_OK) {
        ucs_diag("failed to open netlink socket");
        return ret;
    }

    memset(iov, 0, sizeof(iov));
    nlh.nlmsg_len   = send_msg_len;
    nlh.nlmsg_type  = nlmsg_type;
    nlh.nlmsg_flags = NLM_F_REQUEST | NLM_F_DUMP;
    nlh.nlmsg_seq   = 1;
    nlh.nlmsg_pid   = getpid();
    iov[0].iov_base = &nlh;
    iov[0].iov_len  = sizeof(nlh);
    iov[1].iov_base = nl_protocol_hdr;
    iov[1].iov_len  = nl_protocol_hdr_size;

    do {
        ret = ucs_socket_sendv_nb(fd, iov, 2, &send_msg_len);
    } while (ret == UCS_ERR_NO_PROGRESS);

    if (ret != UCS_OK) {
        ucs_diag("failed to send netlink message. returned %d", ret);
        goto out;
    }

    do {
        ret = ucs_socket_recv_nb(fd, recv_msg_buf, recv_msg_buf_len);
    } while (ret == UCS_ERR_NO_PROGRESS);

    if (ret != UCS_OK) {
        ucs_diag("failed to receive route netlink message");
        goto out;
    }

out:
    ucs_close_fd(&fd);
    return ret;
}
