/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "netlink.h"

#include <ucs/debug/log.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/sock.h>
#include <ucs/type/status.h>

#include <errno.h>
#include <linux/netlink.h>
#include <sys/socket.h>
#include <unistd.h>


static ucs_status_t ucs_netlink_socket_init(int *fd_p, int protocol)
{
    struct sockaddr_nl sa = {.nl_family = AF_NETLINK};
    ucs_status_t status;

    status = ucs_socket_create(AF_NETLINK, SOCK_RAW, protocol, fd_p);
    if (status != UCS_OK) {
        ucs_error("failed to create netlink socket: %s",
                  ucs_status_string(status));
        goto err;
    }

    if (bind(*fd_p, (struct sockaddr *)&sa, sizeof(sa)) < 0) {
        ucs_error("failed to bind netlink socket: %m (%d)", *fd_p);
        status = UCS_ERR_IO_ERROR;
        goto err_close_socket;
    }

    return UCS_OK;

err_close_socket:
    ucs_close_fd(fd_p);
err:
    return status;
}

ucs_status_t ucs_netlink_send_cmd(int protocol, unsigned short nlmsg_type,
                                  const void *nl_protocol_hdr,
                                  size_t nl_protocol_hdr_size,
                                  char *recv_msg_buf, size_t *recv_msg_buf_len)
{
    struct nlmsghdr nlh = {0};
    ucs_status_t status;
    int fd;
    struct iovec iov[2];
    size_t bytes_sent;

    status = ucs_netlink_socket_init(&fd, protocol);
    if (status != UCS_OK) {
        return status;
    }

    nlh.nlmsg_len   = NLMSG_LENGTH(nl_protocol_hdr_size);
    nlh.nlmsg_type  = nlmsg_type;
    nlh.nlmsg_flags = NLM_F_REQUEST | NLM_F_DUMP;
    iov[0].iov_base = &nlh;
    iov[0].iov_len  = sizeof(nlh);
    iov[1].iov_base = (void *)nl_protocol_hdr;
    iov[1].iov_len  = nl_protocol_hdr_size;

    do {
        status = ucs_socket_sendv_nb(fd, iov, 2, &bytes_sent);
    } while (status == UCS_ERR_NO_PROGRESS);

    if (status != UCS_OK) {
        ucs_error("failed to send netlink message (%s)",
                  ucs_status_string(status));
        goto out;
    }

    do {
        status = ucs_socket_recv_nb(fd, recv_msg_buf, recv_msg_buf_len);
    } while (status == UCS_ERR_NO_PROGRESS);

    if (status != UCS_OK) {
        ucs_error("failed to receive netlink message (%s)",
                  ucs_status_string(status));
        goto out;
    }

out:
    ucs_close_fd(&fd);
    return status;
}

ucs_status_t ucs_netlink_parse_msg(const void *msg, size_t msg_len,
                                   ucs_netlink_parse_cb_t parse_cb, void *arg)
{
    ucs_status_t status        = UCS_INPROGRESS;
    const struct nlmsghdr *nlh = (const struct nlmsghdr *)msg;

    while ((status == UCS_INPROGRESS) && NLMSG_OK(nlh, msg_len) &&
           (nlh->nlmsg_type != NLMSG_DONE)) {
        if (nlh->nlmsg_type == NLMSG_ERROR) {
            struct nlmsgerr *err = (struct nlmsgerr *)NLMSG_DATA(nlh);
            printf("failed to parse netlink message header (%d): %s\n",
                      err->error, strerror(-err->error));
            return UCS_ERR_IO_ERROR;
        }

        status = parse_cb(nlh, NLMSG_DATA(nlh), arg);
        nlh = NLMSG_NEXT(nlh, msg_len);
    }

    return UCS_OK;
}
