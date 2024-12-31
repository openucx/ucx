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
#include <ucs/debug/memtrack_int.h>

#include <errno.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#include <sys/socket.h>
#include <unistd.h>

#define NETLINK_MESSAGE_MAX_SIZE 8195


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

static size_t ucs_netlink_get_msg_size(int sock_fd)
{
    size_t length       = 0;
    ucs_status_t status = ucs_socket_recv_nb(sock_fd, NULL,
                                             MSG_PEEK | MSG_TRUNC, &length);
    if (status != UCS_OK) {
        ucs_error("failed to get netlink message size %d (%s)",
                  sock_fd, ucs_status_string(status));
        return -1;
    }

    return length;
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
            printf("failed to parse netlink message header (%d): %s\n",
                      err->error, strerror(-err->error));
            return UCS_ERR_IO_ERROR;
        }

        status = parse_cb(nlh, NLMSG_DATA(nlh), arg);
        nlh = NLMSG_NEXT(nlh, msg_len);
    }

    return UCS_OK;
}

static ucs_status_t
ucs_netlink_handle_request(int protocol, unsigned short nlmsg_type,
                           const void *nl_protocol_hdr,
                           size_t nl_protocol_hdr_size,
                           ucs_netlink_parse_cb_t parse_cb, void *arg)
{
    struct nlmsghdr nlh = {0};
    char *recv_msg      = NULL;
    ucs_status_t status;
    int fd;
    struct iovec iov[2];
    size_t bytes_sent;
    size_t recv_msg_len;

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

    recv_msg_len = ucs_netlink_get_msg_size(fd);
    if (recv_msg_len == -1) {
        ucs_error("failed to get netlink message size");
        goto out;
    }

    recv_msg = ucs_malloc(recv_msg_len, "netlink recv message");
    if (recv_msg == NULL) {
        ucs_error("failed to allocate a buffer for netlink receive message");
        goto out;
    }

    do {
        status = ucs_socket_recv(fd, recv_msg, recv_msg_len);
    } while (status == UCS_ERR_NO_PROGRESS);

    if (status != UCS_OK) {
        ucs_error("failed to receive netlink message (%s)",
                  ucs_status_string(status));
        goto out;
    }

    status = ucs_netlink_parse_msg(recv_msg, recv_msg_len, parse_cb, arg);
    if (status != UCS_OK) {
        ucs_error("failed to parse netlink message (%s)",
                  ucs_status_string(status));
        goto out;
    }

out:
    ucs_close_fd(&fd);
    free(recv_msg);
    return status;
}

static ucs_status_t
ucs_netlink_get_route_info(const struct rtattr *rta, int len, int *if_idx,
                           void **dst_in_addr)
{
    *if_idx      = -1;
    *dst_in_addr = NULL;

    for (; RTA_OK(rta, len); rta = RTA_NEXT(rta, len)) {
        if (rta->rta_type == RTA_OIF) {
            *if_idx = *((int *)RTA_DATA(rta));
        } else if (rta->rta_type == RTA_DST) {
            *dst_in_addr = RTA_DATA(rta);
        }
    }

    if ((*if_idx == -1) || (*dst_in_addr == NULL)) {
        ucs_diag("invalid routing table entry");
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

static int ucs_netlink_is_rule_matching(const struct rtmsg *rtm, size_t rtm_len,
                                        const struct sockaddr *sa_remote,
                                        int iface_index)
{
    int   rule_iface;
    void *dst_in_addr;

    if (ucs_netlink_get_route_info(RTM_RTA(rtm), rtm_len, &rule_iface,
                                   &dst_in_addr) != UCS_OK) {
        return 0;
    }

    if (rule_iface != iface_index) {
        return 0;
    }

    return ucs_bitwise_is_equal(ucs_sockaddr_get_inet_addr(sa_remote),
                                dst_in_addr, rtm->rtm_dst_len);
}

static ucs_status_t
ucs_netlink_parse_rt_entry_cb(const struct nlmsghdr *nlh, const void *nl_msg,
                              void *arg)
{
    ucs_netlink_route_info_t *info = (ucs_netlink_route_info_t *)arg;

    if (ucs_netlink_is_rule_matching((const struct rtmsg *)nl_msg,
                                     RTM_PAYLOAD(nlh), info->sa_remote,
                                     info->if_index)) {
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
    ucs_status_t status;
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

    status = ucs_netlink_handle_request(NETLINK_ROUTE, RTM_GETROUTE, &rtm,
                                        sizeof(rtm),
                                        ucs_netlink_parse_rt_entry_cb, &info);
    if (status != UCS_OK) {
        ucs_error("failed to handle netlink route request (%s)",
                  ucs_status_string(status));
        goto out;
    }

out:
    return info.found;
}