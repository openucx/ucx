/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "netlink.h"

#include <ucs/datastruct/khash.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/sock.h>
#include <ucs/type/status.h>
#include <ucs/type/init_once.h>

#include <errno.h>
#include <linux/rtnetlink.h>
#include <pthread.h>
#include <sys/socket.h>
#include <unistd.h>


typedef struct {
    const struct sockaddr *sa_remote;
    int                    if_index;
    int                    found;
} ucs_netlink_route_info_t;


typedef struct {
    struct sockaddr_storage dest;
    uint8_t                 subnet_prefix_len;
} ucs_netlink_route_entry_t;

UCS_ARRAY_DECLARE_TYPE(ucs_netlink_rt_rules_t, unsigned,
                       ucs_netlink_route_entry_t);

KHASH_INIT(ucs_netlink_rt_cache, khint32_t, ucs_netlink_rt_rules_t, 1,
           kh_int_hash_func, kh_int_hash_equal);
static khash_t(ucs_netlink_rt_cache) ucs_netlink_routing_table_cache;

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

        status = parse_cb(nlh, arg);
        nlh    = NLMSG_NEXT(nlh, msg_len);
    }

    return UCS_OK;
}

ucs_status_t
ucs_netlink_send_request(int protocol, unsigned short nlmsg_type,
                         unsigned short nlmsg_flags,
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
    nlh.nlmsg_flags = NLM_F_REQUEST | nlmsg_flags;
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
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

static ucs_status_t
ucs_netlink_parse_rt_entry_cb(const struct nlmsghdr *nlh, void *arg)
{
    const struct rtmsg *rt_msg = NLMSG_DATA(nlh);
    const void *dst_in_addr;
    ucs_netlink_route_entry_t *new_rule;
    ucs_netlink_rt_rules_t *iface_rules;
    int iface_index;
    khiter_t iter;
    int khret;

    if (ucs_netlink_get_route_info(RTM_RTA(rt_msg), RTM_PAYLOAD(nlh),
                                   &iface_index, &dst_in_addr) != UCS_OK) {
        return UCS_INPROGRESS;
    }

    iter = kh_put(ucs_netlink_rt_cache, &ucs_netlink_routing_table_cache,
                  iface_index, &khret);
    if (khret == UCS_KH_PUT_FAILED) {
        ucs_error("failed to put net iface index (%d) in the cache", iface_index);
        return UCS_ERR_IO_ERROR;
    }

    /* if the iface was not present in the hash table before, initialize the
       array of rules */
    iface_rules = &kh_val(&ucs_netlink_routing_table_cache, iter);
    if (khret != UCS_KH_PUT_KEY_PRESENT) {
        ucs_array_init_dynamic(iface_rules);
    }

    new_rule = ucs_array_append(iface_rules,
                                ucs_error("could not allocate route entry");
                                return UCS_ERR_NO_MEMORY);

    memset(&new_rule->dest, 0, sizeof(sizeof(new_rule->dest)));
    new_rule->dest.ss_family = rt_msg->rtm_family;
    if (UCS_OK != ucs_sockaddr_set_inet_addr((struct sockaddr *)&new_rule->dest,
                                             dst_in_addr)) {
        ucs_array_pop_back(iface_rules);
        return UCS_ERR_IO_ERROR;
    }

    new_rule->subnet_prefix_len = rt_msg->rtm_dst_len;

    return UCS_INPROGRESS;
}

static void ucs_netlink_lookup_route(ucs_netlink_route_info_t *info)
{
    ucs_netlink_rt_rules_t *iface_rules;
    ucs_netlink_route_entry_t *curr_entry;
    khiter_t iter;

    iter = kh_get(ucs_netlink_rt_cache, &ucs_netlink_routing_table_cache,
                  info->if_index);
    if (iter == kh_end(&ucs_netlink_routing_table_cache)) {
        info->found = 0;
        return;
    }

    iface_rules = &kh_val(&ucs_netlink_routing_table_cache, iter);
    ucs_array_for_each(curr_entry, iface_rules) {
        if (ucs_sockaddr_is_same_subnet(
                                info->sa_remote,
                                (const struct sockaddr *)&curr_entry->dest,
                                curr_entry->subnet_prefix_len)) {
            info->found = 1;
            return;
        }
    }
}

int ucs_netlink_route_exists(int if_index, const struct sockaddr *sa_remote)
{
    static ucs_init_once_t init_once = UCS_INIT_ONCE_INITIALIZER;
    struct rtmsg rtm                 = {0};
    ucs_netlink_route_info_t info;

    UCS_INIT_ONCE(&init_once) {
        rtm.rtm_family = AF_INET;
        rtm.rtm_table  = RT_TABLE_MAIN;
        ucs_netlink_send_request(NETLINK_ROUTE, RTM_GETROUTE, NLM_F_DUMP, &rtm,
                                 sizeof(rtm), ucs_netlink_parse_rt_entry_cb,
                                 NULL);

        rtm.rtm_family = AF_INET6;
        ucs_netlink_send_request(NETLINK_ROUTE, RTM_GETROUTE, NLM_F_DUMP, &rtm,
                                 sizeof(rtm), ucs_netlink_parse_rt_entry_cb,
                                 NULL);
    }

    info.if_index  = if_index;
    info.sa_remote = sa_remote;
    info.found     = 0;
    ucs_netlink_lookup_route(&info);

    return info.found;
}
