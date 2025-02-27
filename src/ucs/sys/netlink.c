/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
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
#include <ucs/debug/memtrack_int.h>

#include <errno.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#include <sys/socket.h>
#include <unistd.h>


typedef struct {
    const struct sockaddr *sa_remote;
    int                    if_index;
    int                    found;
} ucs_netlink_route_info_t;


/**
 * Callback function for parsing individual netlink messages.
 *
 * @param [in] nlh    Pointer to the netlink message header.
 * @param [in] nl_msg Pointer to the netlink message payload.
 * @param [in] arg    User-provided argument passed through from the caller.
 *
 * @return UCS_OK if parsing is complete, UCS_INPROGRESS if there are more
 *         messages to be parsed, or error code otherwise.
 */
typedef ucs_status_t (*ucs_netlink_parse_cb_t)(const struct nlmsghdr *nlh,
                                               const void *nl_msg, void *arg);

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
        ucs_error("failed to bind netlink socket %d: %m", *fd_p);
        status = UCS_ERR_IO_ERROR;
        goto err_close_socket;
    }

    return UCS_OK;

err_close_socket:
    ucs_close_fd(fd_p);
err:
    return status;
}

static ucs_status_t
ucs_netlink_parse_msg(const void *msg, size_t msg_len,
                      ucs_netlink_parse_cb_t parse_cb, void *arg)
{
    ucs_status_t status        = UCS_INPROGRESS;
    const struct nlmsghdr *nlh = (const struct nlmsghdr *)msg;

    while ((status == UCS_INPROGRESS) && NLMSG_OK(nlh, msg_len) &&
           (nlh->nlmsg_type != NLMSG_DONE)) {
        if (nlh->nlmsg_type == NLMSG_ERROR) {
            struct nlmsgerr *err = (struct nlmsgerr *)NLMSG_DATA(nlh);
            ucs_error("received error response from netlink err=%d: %s\n",
                      err->error, strerror(-err->error));
            return UCS_ERR_IO_ERROR;
        }

        status = parse_cb(nlh, NLMSG_DATA(nlh), arg);
        nlh    = NLMSG_NEXT(nlh, msg_len);
    }

    return UCS_OK;
}

static ucs_status_t
ucs_netlink_send_request(int protocol, unsigned short nlmsg_type,
                         const void *protocol_header, size_t header_length,
                         ucs_netlink_parse_cb_t parse_cb, void *arg)
{
    struct nlmsghdr nlh = {0};
    char *recv_msg      = NULL;
    size_t recv_msg_len = 0;
    int netlink_fd      = -1;
    ucs_status_t status;
    struct iovec iov[2];
    size_t bytes_sent;

    status = ucs_netlink_socket_init(&netlink_fd, protocol);
    if (status != UCS_OK) {
        goto out;
    }

    nlh.nlmsg_len   = NLMSG_LENGTH(header_length);
    nlh.nlmsg_type  = nlmsg_type;
    nlh.nlmsg_flags = NLM_F_REQUEST | NLM_F_DUMP;
    iov[0].iov_base = &nlh;
    iov[0].iov_len  = sizeof(nlh);
    iov[1].iov_base = (void *)protocol_header;
    iov[1].iov_len  = header_length;

    do {
        status = ucs_socket_sendv_nb(netlink_fd, iov, 2, &bytes_sent);
    } while (status == UCS_ERR_NO_PROGRESS);

    if (status != UCS_OK) {
        ucs_error("failed to send netlink message on fd=%d: %s",
                  netlink_fd, ucs_status_string(status));
        goto out;
    }

    /* get message size */
    status = ucs_socket_recv_nb(netlink_fd, NULL, MSG_PEEK | MSG_TRUNC,
                                &recv_msg_len);
    if (status != UCS_OK) {
        ucs_error("failed to get netlink message size %d (%s)",
                  status, ucs_status_string(status));
        goto out;
    }

    recv_msg = ucs_malloc(recv_msg_len, "netlink recv message");
    if (recv_msg == NULL) {
        ucs_error("failed to allocate a buffer for netlink receive message of"
                  " size %zu", recv_msg_len);
        goto out;
    }

    status = ucs_socket_recv(netlink_fd, recv_msg, recv_msg_len);
    if (status != UCS_OK) {
        ucs_error("failed to receive netlink message on fd=%d: %s",
                  netlink_fd, ucs_status_string(status));
        goto out;
    }

    status = ucs_netlink_parse_msg(recv_msg, recv_msg_len, parse_cb, arg);

out:
    ucs_close_fd(&netlink_fd);
    ucs_free(recv_msg);
    return status;
}

static ucs_status_t
ucs_netlink_get_route_info(const struct rtattr *rta, int len, int *if_index_p,
                           const void **dst_in_addr)
{
    *if_index_p  = -1;
    *dst_in_addr = NULL;

    for (; RTA_OK(rta, len); rta = RTA_NEXT(rta, len)) {
        if (rta->rta_type == RTA_OIF) {
            *if_index_p = *((const int *)RTA_DATA(rta));
        } else if (rta->rta_type == RTA_DST) {
            *dst_in_addr = RTA_DATA(rta);
        }
    }

    if ((*if_index_p == -1) || (*dst_in_addr == NULL)) {
        ucs_diag("invalid routing table entry");
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

static ucs_status_t
ucs_netlink_parse_rt_entry_cb(const struct nlmsghdr *nlh, const void *nl_msg,
                              void *arg)
{
    ucs_netlink_route_info_t *info = (ucs_netlink_route_info_t *)arg;
    int rule_iface;
    const void *dst_in_addr;

    if (ucs_netlink_get_route_info(RTM_RTA((const struct rtmsg *)nl_msg),
                                   RTM_PAYLOAD(nlh), &rule_iface,
                                   &dst_in_addr) != UCS_OK) {
        return UCS_INPROGRESS;
    }

    if (rule_iface != info->if_index) {
        return UCS_INPROGRESS;
    }

    if (ucs_bitwise_is_equal(ucs_sockaddr_get_inet_addr(info->sa_remote),
                             dst_in_addr,
                             ((const struct rtmsg *)nl_msg)->rtm_dst_len)) {
        info->found = 1;
        return UCS_OK;
    }

    return UCS_INPROGRESS;
}

int ucs_netlink_route_exists(const char *if_name,
                             const struct sockaddr *sa_remote)
{
    ucs_netlink_route_info_t info = {0};
    struct rtmsg rtm              = {0};
    int iface_index;

    iface_index = if_nametoindex(if_name);
    if (iface_index == 0) {
        ucs_error("failed to get interface index (errno %d)", errno);
        goto out;
    }

    rtm.rtm_family = sa_remote->sa_family;
    rtm.rtm_table  = RT_TABLE_MAIN;

    info.if_index  = iface_index;
    info.sa_remote = sa_remote;

    ucs_netlink_send_request(NETLINK_ROUTE, RTM_GETROUTE, &rtm, sizeof(rtm),
                             ucs_netlink_parse_rt_entry_cb, &info);

out:
    return info.found;
}
