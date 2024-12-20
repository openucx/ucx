/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/rtnetlink.h>
#include <ucs/sys/netlink.h>
#include <ucs/sys/sock.h>
#include <ucs/type/status.h>

#include <errno.h>
#include <linux/rtnetlink.h>

#define NETLINK_MESSAGE_MAX_SIZE 8195


typedef struct {
    const struct sockaddr *sa_remote;
    int                    if_index;
    int                    found;
} ucs_netlink_route_info_t;

static ucs_status_t
ucs_rtnetlink_get_route_info(const struct rtattr *rta, int len, int *if_idx,
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
        ucs_diag("either iface index or dest address are missing "
                  "in the routing table entry");
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

static int ucs_rtnetlink_is_rule_matching(const struct rtmsg *rtm, size_t rtm_len,
                                          const struct sockaddr *sa_remote,
                                          int iface_index)
{
    int   rule_iface;
    void *dst_in_addr;

    if (ucs_rtnetlink_get_route_info(RTM_RTA(rtm), rtm_len, &rule_iface,
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

    if (ucs_rtnetlink_is_rule_matching((const struct rtmsg *)nl_msg,
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
    char *recv_msg                = NULL;
    ucs_netlink_route_info_t info = {0};
    struct rtmsg rtm              = {0};
    ucs_status_t status;
    size_t recv_msg_len;
    int iface_index;

    iface_index = if_nametoindex(if_name);
    if (iface_index == 0) {
        ucs_error("failed to get interface index (errno %d)", errno);
        goto out;
    }

    rtm.rtm_family = sa_remote->sa_family;
    rtm.rtm_table  = RT_TABLE_MAIN;

    recv_msg_len = NETLINK_MESSAGE_MAX_SIZE;
    recv_msg     = ucs_malloc(NETLINK_MESSAGE_MAX_SIZE, "netlink recv message");
    if (recv_msg == NULL) {
        ucs_error("failed to allocate a buffer for netlink receive message");
        goto out;
    }

    status = ucs_netlink_send_cmd(NETLINK_ROUTE, RTM_GETROUTE, &rtm,
                                  sizeof(rtm), recv_msg, &recv_msg_len);
    if (status != UCS_OK) {
        ucs_error("failed to send netlink route message (%s)",
                  ucs_status_string(status));
        goto out;
    }

    info.if_index  = iface_index;
    info.sa_remote = sa_remote;

    status = ucs_netlink_parse_msg(recv_msg, recv_msg_len,
                                   ucs_netlink_parse_rt_entry_cb, &info);
    if (status != UCS_OK) {
        ucs_error("failed to parse netlink route message (%s)",
                  ucs_status_string(status));
        goto out;
    }

out:
    ucs_free(recv_msg);
    return info.found;
}
