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
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>


typedef struct {
    const struct sockaddr *sa_remote;
    int                    if_index;
    int                    netmask_len;
} ucs_netlink_route_info_t;


typedef struct {
    struct sockaddr_storage dest;
    uint8_t                 subnet_prefix_len;
    uint8_t                 route_type;
} ucs_netlink_route_entry_t;

UCS_ARRAY_DECLARE_TYPE(ucs_netlink_rt_rules_t, unsigned,
                       ucs_netlink_route_entry_t);

KHASH_INIT(ucs_netlink_rt_cache, khint32_t, ucs_netlink_rt_rules_t, 1,
           kh_int_hash_func, kh_int_hash_equal);
static khash_t(ucs_netlink_rt_cache) ucs_netlink_routing_table_cache;

/* Network interface index -> index of the VRF device it is enslaved to */
KHASH_INIT(ucs_netlink_vrf_cache, khint32_t, int, 1, kh_int_hash_func,
           kh_int_hash_equal);
static khash_t(ucs_netlink_vrf_cache) ucs_netlink_vrf_master_cache;

/* Look up an initialized VRF cache */
static int ucs_netlink_vrf_master_lookup(int if_index)
{
    khiter_t iter = kh_get(ucs_netlink_vrf_cache, &ucs_netlink_vrf_master_cache,
                           if_index);

    if (iter == kh_end(&ucs_netlink_vrf_master_cache)) {
        return -1;
    }

    return kh_val(&ucs_netlink_vrf_master_cache, iter);
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

ucs_status_t ucs_netlink_parse_msg(const void *msg, size_t msg_len,
                                   ucs_netlink_parse_cb_t parse_cb, void *arg,
                                   ucs_log_level_t err_log_level,
                                   int *msg_done_p)
{
    int dump_intr = 0;
    const struct nlmsghdr *nlh;
    ucs_status_t status;
    int err;

    *msg_done_p = 0;

    for (nlh = msg; NLMSG_OK(nlh, msg_len); nlh = NLMSG_NEXT(nlh, msg_len)) {
        if (!dump_intr && (nlh->nlmsg_flags & NLM_F_DUMP_INTR)) {
            ucs_debug("netlink dump was interrupted");
            dump_intr = 1;
        }

        if (ucs_netlink_is_msg_done(nlh) || (nlh->nlmsg_type == NLMSG_ERROR)) {
            err = (nlh->nlmsg_len >= NLMSG_LENGTH(sizeof(int))) ?
                          *(const int*)NLMSG_DATA(nlh) :
                          0;
            if (ucs_netlink_is_msg_done(nlh)) {
                *msg_done_p = 1;
                if (err >= 0) {
                    break;
                }
            }

            goto err;
        }

        status = parse_cb(nlh, arg);
        if (status != UCS_INPROGRESS) {
            return status;
        }
    }

    return dump_intr ? UCS_ERR_BUSY : UCS_OK;

err:
    ucs_log(err_log_level, "received error response from netlink err %d: %s",
            err, strerror(-err));
    return UCS_ERR_IO_ERROR;
}

ucs_status_t ucs_netlink_send_request(int protocol, unsigned short nlmsg_type,
                                      unsigned short nlmsg_flags,
                                      const void *protocol_header,
                                      size_t header_length,
                                      ucs_netlink_parse_cb_t parse_cb,
                                      void *arg, ucs_log_level_t err_log_level)
{
    struct nlmsghdr nlh = {0};
    int netlink_fd      = -1;
    int dump_intr       = 0;
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
            status = UCS_ERR_NO_MEMORY;
            goto out;
        }

        status = ucs_socket_recv(netlink_fd, recv_msg, recv_msg_len);
        if (status != UCS_OK) {
            ucs_error("failed to receive netlink message on fd=%d: %s",
                    netlink_fd, ucs_status_string(status));
            ucs_free(recv_msg);
            goto out;
        }

        status = ucs_netlink_parse_msg(recv_msg, recv_msg_len, parse_cb, arg,
                                       err_log_level, &msg_done);
        ucs_free(recv_msg);
        if (status == UCS_ERR_BUSY) {
            dump_intr = 1;
            status    = UCS_OK;
        }
    } while ((nlmsg_flags & NLM_F_DUMP) && (status == UCS_OK) && !msg_done);

    status = ((status == UCS_OK) && dump_intr) ? UCS_ERR_BUSY : status;

out:
    ucs_close_fd(&netlink_fd);
    return status;
}

static ucs_status_t
ucs_netlink_add_route_entry(const struct rtmsg *rt_msg, int if_index,
                            const void *dst_in_addr)
{
    ucs_netlink_route_entry_t *new_rule;
    ucs_netlink_rt_rules_t *iface_rules;
    khiter_t iter;
    int khret;

    iter = kh_put(ucs_netlink_rt_cache, &ucs_netlink_routing_table_cache,
                  if_index, &khret);
    if (khret == UCS_KH_PUT_FAILED) {
        ucs_error("failed to put net iface index (%d) in the cache", if_index);
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

    memset(&new_rule->dest, 0, sizeof(new_rule->dest));
    new_rule->dest.ss_family = rt_msg->rtm_family;
    if (dst_in_addr != NULL) {
        if (ucs_sockaddr_set_inet_addr((struct sockaddr *)&new_rule->dest,
                                       dst_in_addr) != UCS_OK) {
            ucs_array_pop_back(iface_rules);
            return UCS_ERR_IO_ERROR;
        }
    }

    new_rule->subnet_prefix_len = rt_msg->rtm_dst_len;
    new_rule->route_type        = rt_msg->rtm_type;

    return UCS_OK;
}

static ucs_status_t
ucs_netlink_parse_rt_entry_cb(const struct nlmsghdr *nlh, void *arg)
{
    const struct rtmsg *rt_msg = NLMSG_DATA(nlh);
    const struct rtattr *rta   = RTM_RTA(rt_msg);
    int len                    = RTM_PAYLOAD(nlh);
    const struct rtattr *multipath = NULL;
    const void *dst_in_addr        = NULL;
    int if_index                   = -1;
    const struct rtnexthop *nh;
    int nh_len;
    ucs_status_t status;

    for (; RTA_OK(rta, len); rta = RTA_NEXT(rta, len)) {
        if (rta->rta_type == RTA_OIF) {
            if_index = *((const int *)RTA_DATA(rta));
        } else if (rta->rta_type == RTA_DST) {
            dst_in_addr = RTA_DATA(rta);
        } else if (rta->rta_type == RTA_MULTIPATH) {
            multipath = rta;
        }
    }

    /* dst_in_addr required but not present */
    if ((rt_msg->rtm_dst_len != 0) && (dst_in_addr == NULL)) {
        return UCS_INPROGRESS;
    }

    if (multipath != NULL) {
        /* ECMP route: the output interfaces are in the nested nexthops */
        nh     = RTA_DATA(multipath);
        nh_len = RTA_PAYLOAD(multipath);
        for (; RTNH_OK(nh, nh_len); nh = RTNH_NEXT(nh)) {
            nh_len -= RTNH_ALIGN(nh->rtnh_len);
            status  = ucs_netlink_add_route_entry(rt_msg, nh->rtnh_ifindex,
                                                  dst_in_addr);
            if (status != UCS_OK) {
                return status;
            }
        }
    } else if (if_index != -1) {
        status = ucs_netlink_add_route_entry(rt_msg, if_index, dst_in_addr);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_INPROGRESS;
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
        rtm.rtm_table  = RT_TABLE_UNSPEC; /* fetch all the tables */
        rtm.rtm_family = AF_INET;
        ucs_netlink_send_request(NETLINK_ROUTE, RTM_GETROUTE, NLM_F_DUMP, &rtm,
                                 sizeof(rtm), ucs_netlink_parse_rt_entry_cb,
                                 NULL, UCS_LOG_LEVEL_ERROR);

        rtm.rtm_family = AF_INET6;
        ucs_netlink_send_request(NETLINK_ROUTE, RTM_GETROUTE, NLM_F_DUMP, &rtm,
                                 sizeof(rtm), ucs_netlink_parse_rt_entry_cb,
                                 NULL, UCS_LOG_LEVEL_ERROR);
    }
}

static void ucs_netlink_lookup_route(ucs_netlink_route_info_t *info)
{
    ucs_netlink_rt_rules_t *iface_rules;
    khiter_t iter;

    ucs_netlink_init_routing_table_cache();

    iter = kh_get(ucs_netlink_rt_cache, &ucs_netlink_routing_table_cache,
                  info->if_index);
    if (iter == kh_end(&ucs_netlink_routing_table_cache)) {
        return;
    }

    iface_rules       = &kh_val(&ucs_netlink_routing_table_cache, iter);
    info->netmask_len = ucs_netlink_lookup_in_iface_rules(info->sa_remote,
                                                          iface_rules);
}

static int
ucs_netlink_max_netmask_len(int if_index, const struct sockaddr *sa_remote)
{
    int vrf_index       = ucs_netlink_get_vrf_master_index(if_index);
    int max_netmask_len = -1;
    ucs_netlink_rt_rules_t iface_rules;
    khint32_t curr_if_index;

    kh_foreach(&ucs_netlink_routing_table_cache, curr_if_index, iface_rules, {
        uint8_t route_type;
        int curr_netmask_len;

        /* Longest prefix match is done only within one routing domain, but
         * local addresses of other domains are matched too: other domains do
         * not deliver them locally, but route them out of the host */
        route_type       = (ucs_netlink_vrf_master_lookup(curr_if_index) ==
                            vrf_index) ? RTN_UNSPEC : RTN_LOCAL;
        curr_netmask_len = ucs_netlink_lookup_in_iface_rules_by_type(
                sa_remote, &iface_rules, route_type);
        if (curr_netmask_len > max_netmask_len) {
            max_netmask_len = curr_netmask_len;
        }
    })

    return max_netmask_len;
}

int ucs_netlink_route_exists(int if_index, const struct sockaddr *sa_remote,
                             int *netmask_len_p)
{
    ucs_netlink_route_info_t info = {
        .if_index    = if_index,
        .sa_remote   = sa_remote,
        .netmask_len = -1
    };

    ucs_netlink_lookup_route(&info);

    if (netmask_len_p != NULL) {
        *netmask_len_p = info.netmask_len;
    }

    return (info.netmask_len > -1);
}

int ucs_netlink_get_local_route_ndev_index(const struct sockaddr *sa_remote)
{
    int best_netmask_len = -1;
    int best_if_index    = -1;
    ucs_netlink_rt_rules_t iface_rules;
    khint32_t if_index;

    ucs_netlink_init_routing_table_cache();

    kh_foreach(&ucs_netlink_routing_table_cache, if_index, iface_rules, {
        int curr_netmask_len = ucs_netlink_lookup_in_iface_rules_by_type(
                sa_remote, &iface_rules, RTN_LOCAL);
        if (curr_netmask_len > best_netmask_len) {
            best_netmask_len = curr_netmask_len;
            best_if_index    = if_index;
        }
    })

    return best_if_index;
}

int ucs_netlink_is_local_addr(int if_index, const struct sockaddr *sa_remote)
{
    int vrf_index = ucs_netlink_get_vrf_master_index(if_index);
    ucs_netlink_rt_rules_t iface_rules;
    khint32_t curr_if_index;

    ucs_netlink_init_routing_table_cache();

    kh_foreach(&ucs_netlink_routing_table_cache, curr_if_index, iface_rules, {
        /* Local routes of other routing domains are not used */
        if (ucs_netlink_vrf_master_lookup(curr_if_index) != vrf_index) {
            continue;
        }

        if (ucs_netlink_lookup_in_iface_rules_by_type(sa_remote, &iface_rules,
                                                      RTN_LOCAL) >= 0) {
            return 1;
        }
    })

    return 0;
}

int ucs_netlink_route_matches(int if_index, const struct sockaddr *sa_remote,
                              ucs_netlink_route_check_t route_check)
{
    int netmask_len;

    /* If there is no route to the destination through this interface, not even
     * a default route, return 0. */
    if (!ucs_netlink_route_exists(if_index, sa_remote, &netmask_len)) {
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
    return (ucs_netlink_max_netmask_len(if_index, sa_remote) == netmask_len);
}

static int ucs_netlink_is_vrf_kind(const struct rtattr *rta)
{
    return (RTA_PAYLOAD(rta) == sizeof("vrf")) &&
           !memcmp(RTA_DATA(rta), "vrf", sizeof("vrf"));
}

static ucs_status_t
ucs_netlink_parse_vrf_master_cb(const struct nlmsghdr *nlh, void *arg)
{
    const struct ifinfomsg *ifi_msg = NLMSG_DATA(nlh);
    int master_index                = -1;
    int is_vrf_slave                = 0;
    int is_vrf                      = 0;
    const struct rtattr *rta, *info_rta;
    int len, info_len, khret;
    khiter_t iter;

    if (nlh->nlmsg_type != RTM_NEWLINK) {
        return UCS_INPROGRESS;
    }

    len = IFLA_PAYLOAD(nlh);
    for (rta = IFLA_RTA(ifi_msg); RTA_OK(rta, len); rta = RTA_NEXT(rta, len)) {
        if (rta->rta_type == IFLA_MASTER) {
            master_index = *(const int*)RTA_DATA(rta);
        } else if (rta->rta_type == IFLA_LINKINFO) {
            info_len = RTA_PAYLOAD(rta);
            for (info_rta = RTA_DATA(rta); RTA_OK(info_rta, info_len);
                 info_rta = RTA_NEXT(info_rta, info_len)) {
                if (info_rta->rta_type == IFLA_INFO_KIND) {
                    is_vrf = ucs_netlink_is_vrf_kind(info_rta);
                } else if (info_rta->rta_type == IFLA_INFO_SLAVE_KIND) {
                    is_vrf_slave = ucs_netlink_is_vrf_kind(info_rta);
                }
            }
        }
    }

    /* A VRF device is the master of its own routing domain */
    if (is_vrf) {
        master_index = ifi_msg->ifi_index;
    } else if (!is_vrf_slave || (master_index == -1)) {
        return UCS_INPROGRESS;
    }

    iter = kh_put(ucs_netlink_vrf_cache, &ucs_netlink_vrf_master_cache,
                  ifi_msg->ifi_index, &khret);
    if (khret == UCS_KH_PUT_FAILED) {
        ucs_error("failed to put net iface index (%d) in the VRF cache",
                  ifi_msg->ifi_index);
        return UCS_ERR_NO_MEMORY;
    }

    kh_val(&ucs_netlink_vrf_master_cache, iter) = master_index;
    return UCS_INPROGRESS;
}

static void ucs_netlink_init_vrf_master_cache(void)
{
    static ucs_init_once_t init_once = UCS_INIT_ONCE_INITIALIZER;
    struct ifinfomsg ifi_msg         = {.ifi_family = AF_UNSPEC};

    UCS_INIT_ONCE(&init_once) {
        ucs_netlink_send_request(NETLINK_ROUTE, RTM_GETLINK, NLM_F_DUMP,
                                 &ifi_msg, sizeof(ifi_msg),
                                 ucs_netlink_parse_vrf_master_cb, NULL,
                                 UCS_LOG_LEVEL_ERROR);
    }
}

int ucs_netlink_get_vrf_master_index(int if_index)
{
    ucs_netlink_init_vrf_master_cache();
    return ucs_netlink_vrf_master_lookup(if_index);
}

void ucs_netlink_foreach_route(int if_index, ucs_netlink_route_cb_t cb,
                               void *arg)
{
    ucs_netlink_rt_rules_t *iface_rules;
    ucs_netlink_route_entry_t *entry;
    khiter_t iter;

    ucs_netlink_init_routing_table_cache();

    iter = kh_get(ucs_netlink_rt_cache, &ucs_netlink_routing_table_cache,
                  if_index);
    if (iter == kh_end(&ucs_netlink_routing_table_cache)) {
        return;
    }

    iface_rules = &kh_val(&ucs_netlink_routing_table_cache, iter);
    ucs_array_for_each(entry, iface_rules) {
        cb((const struct sockaddr*)&entry->dest, entry->subnet_prefix_len,
           entry->route_type, arg);
    }
}
