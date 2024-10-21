/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "ucs_netlink.h"

#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/sock.h>
#include <ucs/type/status.h>

#include <errno.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#include <sys/socket.h>
#include <unistd.h>


/* *******************************************************
 * General Netlink utilities                             *
 * ***************************************************** */

static ucs_status_t ucs_netlink_socket_init(int *fd, int protocol)
{
    struct sockaddr_nl sa = {0};
    ucs_status_t status;

    status = ucs_socket_create(AF_NETLINK, SOCK_RAW, protocol, fd);
    if (status != UCS_OK) {
        ucs_error("failed to create netlink socket %d (%s)", status,
                  ucs_status_string(status));
        goto err;
    }

    sa.nl_family = AF_NETLINK;

    if (bind(*fd, (struct sockaddr*)&sa, sizeof(sa)) < 0) {
        ucs_error("failed to bind netlink socket %d", *fd);
        status = UCS_ERR_IO_ERROR;
        goto err_close_socket;
    }

    return UCS_OK;

err_close_socket:
    ucs_close_fd(fd);
err:
    *fd = -1;
    return status;
}

ucs_status_t ucs_netlink_send_cmd(int protocol, unsigned short nlmsg_type,
                                  void *nl_protocol_hdr,
                                  size_t nl_protocol_hdr_size,
                                  char *recv_msg_buf, size_t *recv_msg_buf_len)
{
    struct nlmsghdr nlh = {0};
    ucs_status_t status;
    int fd;
    struct iovec iov[2];
    size_t dummy;

    status = ucs_netlink_socket_init(&fd, protocol);
    if (status != UCS_OK) {
        ucs_error("failed to open netlink socket");
        return status;
    }

    memset(&nlh, 0, sizeof(nlh));
    nlh.nlmsg_len   = NLMSG_LENGTH(nl_protocol_hdr_size);
    nlh.nlmsg_type  = nlmsg_type;
    nlh.nlmsg_flags = NLM_F_REQUEST | NLM_F_DUMP;
    iov[0].iov_base = &nlh;
    iov[0].iov_len  = sizeof(nlh);
    iov[1].iov_base = nl_protocol_hdr;
    iov[1].iov_len  = nl_protocol_hdr_size;

    do {
        status = ucs_socket_sendv_nb(fd, iov, 2, &dummy);
    } while (status == UCS_ERR_NO_PROGRESS);

    if (status != UCS_OK) {
        ucs_error("failed to send netlink message %d (%s)", status,
                  ucs_status_string(status));
        goto out;
    }

    do {
        status = ucs_socket_recv_nb(fd, recv_msg_buf, recv_msg_buf_len);
    } while (status == UCS_ERR_NO_PROGRESS);

    if (status != UCS_OK) {
        ucs_error("failed to receive netlink message");
        goto out;
    }

out:
    ucs_close_fd(&fd);
    return status;
}

ucs_status_t ucs_netlink_parse_msg(void *msg, size_t msg_len,
                                   ucs_netlink_parse_cb_t parse_cb, void *arg)
{
    struct nlmsghdr *nlh;
    ucs_status_t status = UCS_INPROGRESS;

    for (nlh = (struct nlmsghdr*)msg;
         (status == UCS_INPROGRESS) && NLMSG_OK(nlh, msg_len) &&
         (nlh->nlmsg_type != NLMSG_DONE) && (nlh->nlmsg_type != NLMSG_ERROR);
         nlh = NLMSG_NEXT(nlh, msg_len)) {
        status = parse_cb(nlh, NLMSG_DATA(nlh), arg);
    }

    if (nlh->nlmsg_type == NLMSG_ERROR) {
        struct nlmsgerr *err = (struct nlmsgerr*)NLMSG_DATA(nlh);
        ucs_error("failed to parse netlink message header (%d)", err->error);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}


/* *******************************************************
 * Route Netlink utilities                               *
 * ***************************************************** */

#define NETLINK_MESSAGE_MAX_SIZE 8195

struct route_info {
    struct sockaddr_storage *sa_remote;
    int                     if_index;
    int                     matching;
};


static void ucs_rtnetlink_get_route_info(int **if_idx, void **dst_in_addr,
                                         struct rtattr *rta, int len)
{
    *if_idx      = NULL;
    *dst_in_addr = NULL;

    for (; RTA_OK(rta, len); rta = RTA_NEXT(rta, len)) {
        if (rta->rta_type == RTA_OIF) {
            *if_idx = RTA_DATA(rta);
        } else if (rta->rta_type == RTA_DST) {
            *dst_in_addr = RTA_DATA(rta);
        }
    }
}

static int ucs_rtnetlink_is_rule_matching(struct rtmsg *rtm, size_t rtm_len,
                                          struct sockaddr_storage *sa_remote,
                                          int oif)
{
    int *rule_iface;
    void *dst_in_addr;
    void *remote_addr;

    if (rtm->rtm_family != sa_remote->ss_family) {
        return 0;
    }

    ucs_rtnetlink_get_route_info(&rule_iface, &dst_in_addr, RTM_RTA(rtm),
                                 rtm_len);
    if (rule_iface == NULL || dst_in_addr == NULL) {
        return 0;
    }

    if (*rule_iface == oif) {
        if (sa_remote->ss_family == AF_INET) {
            remote_addr = &((struct sockaddr_in*)sa_remote)->sin_addr;
        } else { /* AF_INET6 */
            remote_addr = &((struct sockaddr_in6*)sa_remote)->sin6_addr;
        }

        if (ucs_bitwise_is_equal(remote_addr, dst_in_addr, rtm->rtm_dst_len)) {
            return 1;
        }
    }

    return 0;
}

ucs_status_t
ucs_rtnetlink_parse_entry(struct nlmsghdr *nlh, void *nl_msg, void *arg)
{
    struct route_info *info = (struct route_info*)arg;
    if (ucs_rtnetlink_is_rule_matching((struct rtmsg*)nl_msg, RTM_PAYLOAD(nlh),
                                       info->sa_remote, info->if_index)) {
        info->matching = 1;
        return UCS_OK;
    }

    return UCS_INPROGRESS;
}

int ucs_netlink_rule_exists(const char *iface,
                            struct sockaddr_storage *sa_remote)
{
    char *recv_msg         = NULL;
    struct route_info info = {0};
    struct rtmsg rtm       = {0};
    ucs_status_t status;
    size_t recv_msg_len;
    int oif;

    rtm.rtm_family = sa_remote->ss_family;
    rtm.rtm_table  = RT_TABLE_MAIN;

    recv_msg_len = NETLINK_MESSAGE_MAX_SIZE;
    recv_msg = ucs_malloc(NETLINK_MESSAGE_MAX_SIZE, "netlink recv message");
    if (recv_msg == NULL) {
        ucs_error("failed to allocate a buffer for netlink receive message");
        goto out;
    }

    status = ucs_netlink_send_cmd(NETLINK_ROUTE, RTM_GETROUTE, &rtm,
                                  sizeof(rtm), recv_msg, &recv_msg_len);
    if (status != UCS_OK) {
        ucs_error("failed to send netlink route message (%d)", status);
        goto out;
    }

    oif = if_nametoindex(iface);
    if (oif == 0) {
        ucs_error("failed to get interface index");
        goto out;
    }

    info.if_index  = oif;
    info.sa_remote = sa_remote;

    status = ucs_netlink_parse_msg(recv_msg, recv_msg_len,
                                   ucs_rtnetlink_parse_entry, &info);
    if (status != UCS_OK) {
        ucs_error("failed to parse netlink route message (%d)", status);
        goto out;
    }

out:
    if (recv_msg != NULL) {
        free(recv_msg);
    }

    return info.matching;
}
