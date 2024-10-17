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
    ucs_status_t status;
    int fd;
    struct nlmsghdr nlh;
    struct iovec iov[2];
    size_t dummy;

    status = ucs_netlink_socket_create(&fd, protocol);
    if (status != UCS_OK) {
        ucs_diag("failed to open netlink socket");
        return status;
    }

    memset(&nlh, 0, sizeof(nlh));
    nlh.nlmsg_len   = NLMSG_LENGTH(nl_protocol_hdr_size);
    nlh.nlmsg_type  = nlmsg_type;
    nlh.nlmsg_flags = NLM_F_REQUEST | NLM_F_DUMP;
    nlh.nlmsg_seq   = 1;
    iov[0].iov_base = &nlh;
    iov[0].iov_len  = sizeof(nlh);
    iov[1].iov_base = nl_protocol_hdr;
    iov[1].iov_len  = nl_protocol_hdr_size;

    do {
        status = ucs_socket_sendv_nb(fd, iov, 2, &dummy);
    } while (status == UCS_ERR_NO_PROGRESS);

    if (status != UCS_OK) {
        ucs_diag("failed to send netlink message. returned %d", status);
        goto out;
    }

    do {
        status = ucs_socket_recv_nb(fd, recv_msg_buf, recv_msg_buf_len);
    } while (status == UCS_ERR_NO_PROGRESS);

    if (status != UCS_OK) {
        ucs_diag("failed to receive route netlink message");
        goto out;
    }

out:
    ucs_close_fd(&fd);
    return status;
}
