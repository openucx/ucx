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
#include <ucs/sys/sys.h>
#include <ucs/type/status.h>
#include <ucs/type/init_once.h>

#include <errno.h>
#include <limits.h>
#include <linux/rtnetlink.h>
#include <pthread.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>


/* Supported by Linux 4.13 and later */
#ifndef RTM_F_FIB_MATCH
#define RTM_F_FIB_MATCH 0x2000
#endif

/* 198.51.100.1, a TEST-NET-2 address which is not assigned to any host */
#define UCS_NETLINK_TEST_NET_2_ADDR 0xc6336401


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


typedef struct {
    struct rtmsg rtm;
    /* Space for RTA_DST, RTA_SRC and RTA_OIF attributes */
    char         attrs[(2 * RTA_SPACE(sizeof(struct in6_addr))) +
                       RTA_SPACE(sizeof(int))];
} ucs_netlink_route_req_t;


typedef struct {
    uint64_t dst[2]; /* IPv4 or IPv6 address */
    uint64_t src[2];
    int      if_index;
    int      family;
} ucs_netlink_route_key_t;


typedef struct {
    ucs_status_t        status;
    ucs_netlink_route_t route;
} ucs_netlink_route_result_t;


static UCS_F_ALWAYS_INLINE khint32_t
ucs_netlink_route_key_hash(ucs_netlink_route_key_t key)
{
    return kh_int64_hash_func(key.dst[0] ^ key.dst[1] ^ key.if_index);
}

static UCS_F_ALWAYS_INLINE int
ucs_netlink_route_key_equal(ucs_netlink_route_key_t key1,
                            ucs_netlink_route_key_t key2)
{
    return !memcmp(&key1, &key2, sizeof(key1));
}

/* Results of kernel route lookups */
KHASH_INIT(ucs_netlink_route_cache, ucs_netlink_route_key_t,
           ucs_netlink_route_result_t, 1, ucs_netlink_route_key_hash,
           ucs_netlink_route_key_equal);
static khash_t(ucs_netlink_route_cache) ucs_netlink_route_cache;
static pthread_mutex_t ucs_netlink_route_cache_lock = PTHREAD_MUTEX_INITIALIZER;

UCS_ARRAY_DECLARE_TYPE(ucs_netlink_rt_rules_t, unsigned,
                       ucs_netlink_route_entry_t);

KHASH_INIT(ucs_netlink_rt_cache, khint32_t, ucs_netlink_rt_rules_t, 1,
           kh_int_hash_func, kh_int_hash_equal);
static khash_t(ucs_netlink_rt_cache) ucs_netlink_routing_table_cache;

/* Network interface index -> index of the VRF device it is enslaved to */
KHASH_INIT(ucs_netlink_vrf_cache, khint32_t, int, 1, kh_int_hash_func,
           kh_int_hash_equal);
static khash_t(ucs_netlink_vrf_cache) ucs_netlink_vrf_master_cache;

UCS_ARRAY_DECLARE_TYPE(ucs_netlink_host_addrs_t, unsigned,
                       struct sockaddr_storage);

/* Addresses of all network interfaces, in all routing domains */
static ucs_netlink_host_addrs_t ucs_netlink_host_addrs =
        UCS_ARRAY_DYNAMIC_INITIALIZER;

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
    /* Route lookups fail with these errors when the kernel has no route */
    return ((err == -ENETUNREACH) || (err == -EHOSTUNREACH)) ?
                   UCS_ERR_UNREACHABLE :
                   UCS_ERR_IO_ERROR;
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
    const struct rtmsg *rt_msg = NLMSG_DATA(nlh);
    const void *dst_in_addr;
    ucs_netlink_route_entry_t *new_rule;
    ucs_netlink_rt_rules_t *iface_rules;
    int iface_index;
    khiter_t iter;
    int khret;

    if (ucs_netlink_get_route_info(RTM_RTA(rt_msg), RTM_PAYLOAD(nlh),
                                   &iface_index, &dst_in_addr,
                                   rt_msg->rtm_dst_len) != UCS_OK) {
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

static void ucs_netlink_route_req_add_attr(ucs_netlink_route_req_t *req,
                                           size_t *len_p, unsigned short type,
                                           const void *data, size_t data_len)
{
    struct rtattr *rta = UCS_PTR_BYTE_OFFSET(req, *len_p);

    ucs_assert((*len_p + RTA_SPACE(data_len)) <= sizeof(*req));

    rta->rta_type = type;
    rta->rta_len  = RTA_LENGTH(data_len);
    memcpy(RTA_DATA(rta), data, data_len);
    *len_p += RTA_SPACE(data_len);
}

static ucs_status_t
ucs_netlink_route_req_add_addr(ucs_netlink_route_req_t *req, size_t *len_p,
                               unsigned short type, const struct sockaddr *sa,
                               unsigned char *prefix_len_p)
{
    size_t addr_size;
    ucs_status_t status;

    status = ucs_sockaddr_inet_addr_sizeof(sa, &addr_size);
    if (status != UCS_OK) {
        return status;
    }

    ucs_netlink_route_req_add_attr(req, len_p, type,
                                   ucs_sockaddr_get_inet_addr(sa), addr_size);
    *prefix_len_p = addr_size * CHAR_BIT;
    return UCS_OK;
}

static void ucs_netlink_route_add_oif(ucs_netlink_route_t *route, int if_index)
{
    if (route->num_oifs >= UCS_NETLINK_ROUTE_MAX_OIFS) {
        ucs_debug("route has more than %d output interfaces, ignoring %d",
                  UCS_NETLINK_ROUTE_MAX_OIFS, if_index);
        return;
    }

    route->oifs[route->num_oifs++] = if_index;
}

/* Check whether a route is not through an interface, but through another
 * interface of the routing domain of a VRF device, or out of any VRF if -1 */
static int ucs_netlink_route_is_via_other_oif(const ucs_netlink_route_t *route,
                                              int if_index, int vrf_index)
{
    int is_via_domain_oif = 0;
    unsigned i;

    for (i = 0; i < route->num_oifs; ++i) {
        if (route->oifs[i] == if_index) {
            return 0;
        }

        if (ucs_netlink_get_vrf_master_index(route->oifs[i]) == vrf_index) {
            is_via_domain_oif = 1;
        }
    }

    return is_via_domain_oif;
}

static ucs_status_t
ucs_netlink_parse_route_cb(const struct nlmsghdr *nlh, void *arg)
{
    ucs_netlink_route_t *route = arg;
    const struct rtmsg *rt_msg = NLMSG_DATA(nlh);
    int len                    = RTM_PAYLOAD(nlh);
    const struct rtnexthop *nh;
    const struct rtattr *rta;
    int nh_len;

    if (nlh->nlmsg_type != RTM_NEWROUTE) {
        return UCS_INPROGRESS;
    }

    route->type       = rt_msg->rtm_type;
    route->prefix_len = rt_msg->rtm_dst_len;
    route->num_oifs   = 0;

    for (rta = RTM_RTA(rt_msg); RTA_OK(rta, len); rta = RTA_NEXT(rta, len)) {
        if (rta->rta_type == RTA_OIF) {
            ucs_netlink_route_add_oif(route, *(const int*)RTA_DATA(rta));
        } else if (rta->rta_type == RTA_MULTIPATH) {
            /* A fib match reply lists all the nexthops of the route, including
             * the ones the kernel does not use. RTNH_OK() reads the nexthop
             * length before checking the remaining length. */
            nh_len = RTA_PAYLOAD(rta);
            for (nh = RTA_DATA(rta);
                 (nh_len >= (int)sizeof(*nh)) && RTNH_OK(nh, nh_len);
                 nh_len -= RTNH_ALIGN(nh->rtnh_len), nh = RTNH_NEXT(nh)) {
                if (!(nh->rtnh_flags & (RTNH_F_DEAD | RTNH_F_LINKDOWN))) {
                    ucs_netlink_route_add_oif(route, nh->rtnh_ifindex);
                }
            }
        }
    }

    return UCS_OK;
}

static ucs_status_t ucs_netlink_route_lookup(const struct sockaddr *sa_remote,
                                             const struct sockaddr *sa_local,
                                             int if_index,
                                             ucs_netlink_route_t *route_p)
{
    ucs_netlink_route_req_t req = {{0}};
    ucs_netlink_route_t route   = {.type = RTN_UNSPEC};
    size_t req_len              = NLMSG_ALIGN(sizeof(req.rtm));
    ucs_status_t status;

    req.rtm.rtm_family = sa_remote->sa_family;
    req.rtm.rtm_flags  = RTM_F_FIB_MATCH;

    status = ucs_netlink_route_req_add_addr(&req, &req_len, RTA_DST, sa_remote,
                                            &req.rtm.rtm_dst_len);
    if (status != UCS_OK) {
        return status;
    }

    if (sa_local != NULL) {
        status = ucs_netlink_route_req_add_addr(&req, &req_len, RTA_SRC,
                                                sa_local, &req.rtm.rtm_src_len);
        if (status != UCS_OK) {
            return status;
        }
    }

    if (if_index != 0) {
        ucs_netlink_route_req_add_attr(&req, &req_len, RTA_OIF, &if_index,
                                       sizeof(if_index));
    }

    status = ucs_netlink_send_request(NETLINK_ROUTE, RTM_GETROUTE, 0, &req,
                                      req_len, ucs_netlink_parse_route_cb,
                                      &route, UCS_LOG_LEVEL_TRACE);
    if (status == UCS_ERR_UNREACHABLE) {
        /* No route is a valid reply, which the callers report if needed */
        return status;
    } else if (status != UCS_OK) {
        ucs_debug("netlink route request failed: %s",
                  ucs_status_string(status));
        return status;
    }

    if (route.type == RTN_UNSPEC) {
        ucs_debug("netlink reply to a route request has no route");
        return UCS_ERR_IO_ERROR;
    }

    *route_p = route;
    return UCS_OK;
}

static int ucs_netlink_probe_fib_match(void)
{
    struct sockaddr_in sin = {.sin_family = AF_INET};
    ucs_netlink_route_t route;
    unsigned lo_index;

    /* The loopback address has a local route, unless the kernel does not
     * understand the request */
    sin.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if ((ucs_get_loopback_ndev_index(&lo_index) != UCS_OK) ||
        (ucs_netlink_route_lookup((const struct sockaddr*)&sin, NULL, 0,
                                  &route) != UCS_OK) ||
        (route.type != RTN_LOCAL)) {
        return 0;
    }

    /* A kernel which ignores the fib match flag replies with an on-link route
     * through the requested interface instead of an error */
    sin.sin_addr.s_addr = htonl(UCS_NETLINK_TEST_NET_2_ADDR);
    return ucs_netlink_route_lookup((const struct sockaddr*)&sin, NULL,
                                    lo_index, &route) == UCS_ERR_UNREACHABLE;
}

static int ucs_netlink_is_fib_match_supported(void)
{
    static ucs_init_once_t init_once = UCS_INIT_ONCE_INITIALIZER;
    static int supported;

    UCS_INIT_ONCE(&init_once) {
        supported = ucs_netlink_probe_fib_match();
        ucs_debug("netlink route lookup by fib match is %ssupported",
                  supported ? "" : "not ");
    }

    return supported;
}

ucs_status_t ucs_netlink_route_get(const struct sockaddr *sa_remote,
                                   const struct sockaddr *sa_local,
                                   int if_index, ucs_netlink_route_t *route_p)
{
    ucs_netlink_route_result_t *result;
    ucs_netlink_route_key_t key;
    ucs_status_t status;
    size_t addr_size;
    khiter_t iter;
    int khret;

    ucs_assert((sa_local == NULL) ||
               (sa_local->sa_family == sa_remote->sa_family));

    if (!ucs_netlink_is_fib_match_supported()) {
        return UCS_ERR_UNSUPPORTED;
    }

    status = ucs_sockaddr_inet_addr_sizeof(sa_remote, &addr_size);
    if (status != UCS_OK) {
        return status;
    }

    memset(&key, 0, sizeof(key));
    memcpy(key.dst, ucs_sockaddr_get_inet_addr(sa_remote), addr_size);
    if (sa_local != NULL) {
        memcpy(key.src, ucs_sockaddr_get_inet_addr(sa_local), addr_size);
    }
    key.if_index = if_index;
    key.family   = sa_remote->sa_family;

    pthread_mutex_lock(&ucs_netlink_route_cache_lock);
    iter = kh_put(ucs_netlink_route_cache, &ucs_netlink_route_cache, key,
                  &khret);
    if (khret == UCS_KH_PUT_FAILED) {
        status = UCS_ERR_NO_MEMORY;
        goto out_unlock;
    }

    result = &kh_val(&ucs_netlink_route_cache, iter);
    if (khret != UCS_KH_PUT_KEY_PRESENT) {
        result->status = ucs_netlink_route_lookup(sa_remote, sa_local, if_index,
                                                  &result->route);
    }

    status = result->status;
    if (status == UCS_OK) {
        *route_p = result->route;
    } else if (status != UCS_ERR_UNREACHABLE) {
        /* Other errors can be transient */
        kh_del(ucs_netlink_route_cache, &ucs_netlink_route_cache, iter);
    }

out_unlock:
    pthread_mutex_unlock(&ucs_netlink_route_cache_lock);
    return status;
}

ucs_status_t ucs_netlink_kernel_route_matches(int if_index,
                                              const struct sockaddr *sa_remote,
                                              const struct sockaddr *sa_local,
                                              uint8_t *best_prefix_len_p)
{
    int vrf_index = ucs_netlink_get_vrf_master_index(if_index);
    ucs_netlink_route_t route, best_route;
    ucs_status_t status;

    *best_prefix_len_p = 0;

    status = ucs_netlink_route_get(sa_remote, sa_local, if_index, &route);
    if ((status != UCS_OK) || (route.prefix_len > 0)) {
        return status;
    }

    /* The lookup is done without the source address, because per-source
     * routing tables can have a default route to any destination. A VRF
     * lookup falls through to the main table when the VRF table has no
     * route, so routes through other domains are ignored. */
    status = ucs_netlink_route_get(sa_remote, NULL, ucs_max(vrf_index, 0),
                                   &best_route);
    if ((status == UCS_OK) && (best_route.prefix_len > 0) &&
        ucs_netlink_route_is_via_other_oif(&best_route, if_index, vrf_index)) {
        *best_prefix_len_p = best_route.prefix_len;
        return UCS_ERR_UNREACHABLE;
    }

    return UCS_OK;
}

static ucs_status_t
ucs_netlink_parse_host_addr_cb(const struct nlmsghdr *nlh, void *arg)
{
    const struct ifaddrmsg *ifa_msg = NLMSG_DATA(nlh);
    const void *in_addr             = NULL;
    struct sockaddr_storage *addr;
    const struct rtattr *rta;
    int len;

    if ((nlh->nlmsg_type != RTM_NEWADDR) ||
        ((ifa_msg->ifa_family != AF_INET) &&
         (ifa_msg->ifa_family != AF_INET6))) {
        return UCS_INPROGRESS;
    }

    /* IFA_ADDRESS is the peer address of a point-to-point interface, and
     * IFA_LOCAL is its local address */
    len = IFA_PAYLOAD(nlh);
    for (rta = IFA_RTA(ifa_msg); RTA_OK(rta, len); rta = RTA_NEXT(rta, len)) {
        if ((rta->rta_type == IFA_LOCAL) ||
            ((rta->rta_type == IFA_ADDRESS) && (in_addr == NULL))) {
            in_addr = RTA_DATA(rta);
        }
    }

    if (in_addr == NULL) {
        return UCS_INPROGRESS;
    }

    addr = ucs_array_append(&ucs_netlink_host_addrs,
                            ucs_error("could not allocate host address");
                            return UCS_ERR_NO_MEMORY);

    memset(addr, 0, sizeof(*addr));
    addr->ss_family = ifa_msg->ifa_family;
    if (ucs_sockaddr_set_inet_addr((struct sockaddr*)addr, in_addr) != UCS_OK) {
        ucs_array_pop_back(&ucs_netlink_host_addrs);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_INPROGRESS;
}

static void ucs_netlink_init_host_addrs(void)
{
    static ucs_init_once_t init_once = UCS_INIT_ONCE_INITIALIZER;
    struct ifaddrmsg ifa_msg         = {.ifa_family = AF_UNSPEC};

    UCS_INIT_ONCE(&init_once) {
        ucs_netlink_send_request(NETLINK_ROUTE, RTM_GETADDR, NLM_F_DUMP,
                                 &ifa_msg, sizeof(ifa_msg),
                                 ucs_netlink_parse_host_addr_cb, NULL,
                                 UCS_LOG_LEVEL_ERROR);
    }
}

int ucs_netlink_is_host_addr(const struct sockaddr *sa)
{
    const struct sockaddr_storage *addr;

    ucs_netlink_init_host_addrs();

    ucs_array_for_each(addr, &ucs_netlink_host_addrs) {
        if ((addr->ss_family == sa->sa_family) &&
            !ucs_sockaddr_ip_cmp((const struct sockaddr*)addr, sa)) {
            return 1;
        }
    }

    return 0;
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
