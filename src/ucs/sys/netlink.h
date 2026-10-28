/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_NETLINK_H
#define UCS_NETLINK_H

#include <ucs/type/status.h>

#include <linux/netlink.h>
#include <netinet/in.h>

struct rtmsg;

BEGIN_C_DECLS

typedef enum {
    UCS_NETLINK_ROUTE_CHECK_BEST,
    UCS_NETLINK_ROUTE_CHECK_RELAXED
} ucs_netlink_route_check_t;

/**
 * Callback function for parsing individual netlink messages.
 *
 * @param [in] nlh    Pointer to the netlink message header.
 * @param [in] arg    User-provided argument passed through from the caller.
 *
 * @return UCS_OK if parsing is complete, UCS_INPROGRESS if there are more
 *         messages to be parsed, or error code otherwise.
 */
typedef ucs_status_t (*ucs_netlink_parse_cb_t)(const struct nlmsghdr *nlh,
                                               void *arg);

/**
 * Callback function for parsing a route from a netlink message.
 *
 * @param [in] rt_msg       Pointer to the route message header.
 * @param [in] if_index     Network interface index for the route.
 * @param [in] dst_in_addr  Pointer to the destination address, or NULL when
 *                          the message has no RTA_DST attribute.
 * @param [in] arg          User-provided argument passed through from the
 *                          caller.
 *
 * @return UCS_OK on success, or an error code otherwise.
 */
typedef ucs_status_t (*ucs_netlink_route_parse_func_t)(
        const struct rtmsg *rt_msg, int if_index, const void *dst_in_addr,
        void *arg);

/*
 * Send a netlink request and parse the response.
 *
 * @param [in]  protocol         Netlink protocol (e.g. NETLINK_ROUTE).
 * @param [in]  nlmsg_type       Protocol message type (e.g. NETLINK_GETROUTE).
 * @param [in]  nlmsg_flags      Flags for message header (e.g. NLM_F_ROOT).
 * @param [in]  protocol_header  Netlink protocol header.
 * @param [in]  header_length    Netlink protocol header length.
 * @param [in]  parse_cb         Callback function to parse the response.
 * @param [in]  arg              User-provided argument for the parse callback.
 */
ucs_status_t
ucs_netlink_send_request(int protocol, unsigned short nlmsg_type,
                         unsigned short nlmsg_flags,
                         const void *protocol_header, size_t header_length,
                         ucs_netlink_parse_cb_t parse_cb, void *arg);

/**
 * Parse a route netlink message.
 *
 * The callback is invoked once for a top-level RTA_OIF, or once for every live
 * nexthop in an RTA_MULTIPATH attribute when RTA_OIF is absent.
 *
 * @param [in] nlh         Pointer to the netlink message header.
 * @param [in] parse_func  Callback function to process the route.
 * @param [in] arg         User-provided argument for the callback.
 *
 * @return UCS_OK if the route is skipped or all callback invocations succeed,
 *         or the error code returned by the callback otherwise.
 */
ucs_status_t
ucs_netlink_parse_route(const struct nlmsghdr *nlh,
                        ucs_netlink_route_parse_func_t parse_func, void *arg);


/**
 * Check whether a route exists for a given network interface and
 * destination address.
 *
 * @param [in]  if_index           A global index representing the network
 *                                 interface, as assigned by the system
 *                                 (e.g., obtained via if_nametoindex()).
 * @param [in]  sa_remote          Pointer to the destination address.
 * @param [out] netmask_len_p      Optional pointer to store the netmask length
 *                                 of the route found. Pass NULL if this
 *                                 information is not needed.
 *
 * @return 1 if a route exists, or 0 otherwise.
 */
int ucs_netlink_route_exists(int if_index, const struct sockaddr *sa_remote,
                             int *netmask_len_p);

/**
 * Get the network interface index of a local route to a given destination
 * address.
 *
 * @param [in]  sa_remote        Pointer to the destination address.
 *
 * @return Network interface index of a local route, or -1 if not found.
 */
int ucs_netlink_get_local_route_ndev_index(const struct sockaddr *sa_remote);

/**
 * Check whether a route to a given destination address through a network
 * interface matches the requested policy.
 *
 * @param [in]  if_index         Network interface index.
 * @param [in]  sa_remote        Pointer to the destination address.
 * @param [in]  route_check      When set to UCS_NETLINK_ROUTE_CHECK_RELAXED,
 *                               accept any non-default route, or a default
 *                               route if nothing better exists. Otherwise,
 *                               accept only the best route.
 *
 * @return 1 if the route is accepted by the requested policy, or 0 otherwise.
 */
int ucs_netlink_route_matches(int if_index, const struct sockaddr *sa_remote,
                              ucs_netlink_route_check_t route_check);

END_C_DECLS

#endif /* UCS_NETLINK_H */
