/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "netlink.h"
#include "netlink_int.h"

#include <ucs/datastruct/array.h>
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


static ucs_netlink_route_table_t ucs_netlink_routing_table_cache;

void ucs_netlink_route_table_init(ucs_netlink_route_table_t *route_table)
{
    kh_init_inplace(ucs_netlink_rt_cache, route_table);
}

void ucs_netlink_route_table_cleanup(ucs_netlink_route_table_t *route_table)
{
    ucs_netlink_rt_rules_t iface_rules;

    kh_foreach_value(route_table, iface_rules, {
        ucs_array_cleanup_dynamic(&iface_rules);
    })
    kh_destroy_inplace(ucs_netlink_rt_cache, route_table);
}

ucs_status_t
ucs_netlink_route_table_add(ucs_netlink_route_table_t *route_table,
                            int if_index, const struct sockaddr *dest,
                            uint8_t subnet_prefix_len, uint8_t route_type)
{
    ucs_netlink_route_entry_t *new_rule;
    ucs_netlink_rt_rules_t *iface_rules;
    ucs_status_t status;
    khiter_t iter;
    int khret;

    iter = kh_put(ucs_netlink_rt_cache, route_table, if_index, &khret);
    if (khret == UCS_KH_PUT_FAILED) {
        ucs_error("failed to put net iface index (%d) in the route table",
                  if_index);
        return UCS_ERR_NO_MEMORY;
    }

    /* if the iface was not present in the hash table before, initialize the
       array of rules */
    iface_rules = &kh_val(route_table, iter);
    if (khret != UCS_KH_PUT_KEY_PRESENT) {
        ucs_array_init_dynamic(iface_rules);
    }

    new_rule = ucs_array_append(iface_rules,
                                ucs_error("could not allocate route entry");
                                return UCS_ERR_NO_MEMORY);

    memset(&new_rule->dest, 0, sizeof(new_rule->dest));
    status = ucs_sockaddr_copy((struct sockaddr *)&new_rule->dest, dest);
    if (status != UCS_OK) {
        ucs_array_pop_back(iface_rules);
        return status;
    }

    new_rule->subnet_prefix_len = subnet_prefix_len;
    new_rule->route_type        = route_type;
    return UCS_OK;
}

static inline int ucs_netlink_is_msg_done(const struct nlmsghdr *nlh)
{
    return (nlh->nlmsg_type == NLMSG_DONE);
}

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
           !ucs_netlink_is_msg_done(nlh)) {
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
    int netlink_fd      = -1;
    size_t recv_msg_len;
    char *recv_msg;
    int msg_done;
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
    do {
        recv_msg_len = 0;
        status = ucs_socket_recv_nb(netlink_fd, NULL, MSG_PEEK | MSG_TRUNC,
                                    &recv_msg_len);
        if (status != UCS_OK) {
            ucs_error("failed to get netlink message size %d (%s)",
                    status, ucs_status_string(status));
            goto out;
        }

        recv_msg = ucs_malloc(recv_msg_len, "netlink recv message");
        if (recv_msg == NULL) {
            ucs_error("failed to allocate a buffer for netlink receive message"
                      " of size %zu", recv_msg_len);
            goto out;
        }

        status = ucs_socket_recv(netlink_fd, recv_msg, recv_msg_len);
        if (status != UCS_OK) {
            ucs_error("failed to receive netlink message on fd=%d: %s",
                    netlink_fd, ucs_status_string(status));
            ucs_free(recv_msg);
            goto out;
        }

        status   = ucs_netlink_parse_msg(recv_msg, recv_msg_len, parse_cb, arg);
        msg_done = ucs_netlink_is_msg_done((const struct nlmsghdr *)recv_msg);
        ucs_free(recv_msg);
    } while ((nlmsg_flags & NLM_F_DUMP) && !msg_done);

out:
    ucs_close_fd(&netlink_fd);
    return status;
}

static ucs_status_t
ucs_netlink_get_route_info(const struct rtattr *rta, int len, int *if_index_p,
                           const void **dst_in_addr, size_t rtm_dst_len)
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

    if (/* Network interface index is not valid */
        (*if_index_p == -1) ||
        /* dst_in_addr required but not present */
        ((rtm_dst_len != 0) && (*dst_in_addr == NULL))) {
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

static ucs_status_t
ucs_netlink_parse_rt_entry_cb(const struct nlmsghdr *nlh, void *arg)
{
    ucs_netlink_route_table_t *route_table = arg;
    const struct rtmsg *rt_msg             = NLMSG_DATA(nlh);
    struct sockaddr_storage dest;
    const void *dst_in_addr;
    ucs_status_t status;
    int iface_index;

    if (ucs_netlink_get_route_info(RTM_RTA(rt_msg), RTM_PAYLOAD(nlh),
                                   &iface_index, &dst_in_addr,
                                   rt_msg->rtm_dst_len) != UCS_OK) {
        return UCS_INPROGRESS;
    }

    memset(&dest, 0, sizeof(dest));
    dest.ss_family = rt_msg->rtm_family;
    if (dst_in_addr != NULL) {
        status = ucs_sockaddr_set_inet_addr((struct sockaddr *)&dest,
                                            dst_in_addr);
        if (status != UCS_OK) {
            return status;
        }
    }

    status = ucs_netlink_route_table_add(route_table, iface_index,
                                         (const struct sockaddr *)&dest,
                                         rt_msg->rtm_dst_len,
                                         rt_msg->rtm_type);
    return (status == UCS_OK) ? UCS_INPROGRESS : status;
}

static int
ucs_netlink_lookup_in_iface_rules_by_type(const struct sockaddr *sa_remote,
                                          ucs_netlink_rt_rules_t *iface_rules,
                                          uint8_t route_type)
{
    int found_netmask_len = -1;
    ucs_netlink_route_entry_t *curr_entry;

    ucs_array_for_each(curr_entry, iface_rules) {
        if ((route_type != RTN_UNSPEC) &&
            (curr_entry->route_type != route_type)) {
            continue;
        }

        if ((curr_entry->subnet_prefix_len > found_netmask_len) &&
            ucs_sockaddr_is_same_subnet(
                    sa_remote, (const struct sockaddr*)&curr_entry->dest,
                    curr_entry->subnet_prefix_len)) {
            found_netmask_len = curr_entry->subnet_prefix_len;
        }
    }

    return found_netmask_len;
}

static int
ucs_netlink_lookup_in_iface_rules(const struct sockaddr *sa_remote,
                                  ucs_netlink_rt_rules_t *iface_rules)
{
    return ucs_netlink_lookup_in_iface_rules_by_type(sa_remote, iface_rules,
                                                     RTN_UNSPEC);
}

static void ucs_netlink_init_routing_table_cache(void)
{
    static ucs_init_once_t init_once = UCS_INIT_ONCE_INITIALIZER;
    struct rtmsg rtm                 = {0};

    UCS_INIT_ONCE(&init_once) {
        ucs_netlink_route_table_init(&ucs_netlink_routing_table_cache);

        rtm.rtm_table  = RT_TABLE_UNSPEC; /* fetch all the tables */
        rtm.rtm_family = AF_INET;
        ucs_netlink_send_request(NETLINK_ROUTE, RTM_GETROUTE, NLM_F_DUMP, &rtm,
                                 sizeof(rtm), ucs_netlink_parse_rt_entry_cb,
                                 &ucs_netlink_routing_table_cache);

        rtm.rtm_family = AF_INET6;
        ucs_netlink_send_request(NETLINK_ROUTE, RTM_GETROUTE, NLM_F_DUMP, &rtm,
                                 sizeof(rtm), ucs_netlink_parse_rt_entry_cb,
                                 &ucs_netlink_routing_table_cache);
    }
}

/* Return the netmask length of the best route to the destination through the
   given interface, or -1 if no such route exists */
static int
ucs_netlink_lookup_route(ucs_netlink_route_table_t *route_table, int if_index,
                         const struct sockaddr *sa_remote)
{
    khiter_t iter;

    iter = kh_get(ucs_netlink_rt_cache, route_table, if_index);
    if (iter == kh_end(route_table)) {
        return -1;
    }

    return ucs_netlink_lookup_in_iface_rules(sa_remote,
                                             &kh_val(route_table, iter));
}

static int
ucs_netlink_max_netmask_len(ucs_netlink_route_table_t *route_table,
                            const struct sockaddr *sa_remote)
{
    int max_netmask_len = -1;
    ucs_netlink_rt_rules_t iface_rules;

    kh_foreach_value(route_table, iface_rules, {
        int curr_netmask_len = ucs_netlink_lookup_in_iface_rules(sa_remote,
                                                                 &iface_rules);
        if (curr_netmask_len > max_netmask_len) {
            max_netmask_len = curr_netmask_len;
        }
    })

    return max_netmask_len;
}

int ucs_netlink_route_exists(int if_index, const struct sockaddr *sa_remote,
                             int *netmask_len_p)
{
    int netmask_len;

    ucs_netlink_init_routing_table_cache();
    netmask_len = ucs_netlink_lookup_route(&ucs_netlink_routing_table_cache,
                                           if_index, sa_remote);

    if (netmask_len_p != NULL) {
        *netmask_len_p = netmask_len;
    }

    return (netmask_len > -1);
}

int ucs_netlink_local_route_ndev_index_in_table(
                                    ucs_netlink_route_table_t *route_table,
                                    const struct sockaddr *sa_remote)
{
    int best_netmask_len = -1;
    int best_if_index    = -1;
    ucs_netlink_rt_rules_t iface_rules;
    khint32_t if_index;

    kh_foreach(route_table, if_index, iface_rules, {
        int curr_netmask_len = ucs_netlink_lookup_in_iface_rules_by_type(
                sa_remote, &iface_rules, RTN_LOCAL);
        if (curr_netmask_len > best_netmask_len) {
            best_netmask_len = curr_netmask_len;
            best_if_index    = if_index;
        }
    })

    return best_if_index;
}

int ucs_netlink_get_local_route_ndev_index(const struct sockaddr *sa_remote)
{
    ucs_netlink_init_routing_table_cache();
    return ucs_netlink_local_route_ndev_index_in_table(
            &ucs_netlink_routing_table_cache, sa_remote);
}

int ucs_netlink_route_matches_in_table(ucs_netlink_route_table_t *route_table,
                                       int if_index,
                                       const struct sockaddr *sa_remote,
                                       ucs_netlink_route_check_t route_check)
{
    int netmask_len;

    netmask_len = ucs_netlink_lookup_route(route_table, if_index, sa_remote);

    /* If there is no route to the destination through this interface, not even
     * a default route, return 0. */
    if (netmask_len < 0) {
        return 0;
    }

    /* Relaxed mode accepts any matching non-default route. */
    if ((route_check == UCS_NETLINK_ROUTE_CHECK_RELAXED) &&
        (netmask_len > 0)) {
        return 1;
    }

    /* Accept the route if it is the best match. In relaxed mode, this check is
     * reached only for a default route, so the default is accepted only when
     * no interface has a more specific route. */
    return (ucs_netlink_max_netmask_len(route_table, sa_remote) == netmask_len);
}

int ucs_netlink_route_matches(int if_index, const struct sockaddr *sa_remote,
                              ucs_netlink_route_check_t route_check)
{
    ucs_netlink_init_routing_table_cache();
    return ucs_netlink_route_matches_in_table(
            &ucs_netlink_routing_table_cache, if_index, sa_remote, route_check);
}
