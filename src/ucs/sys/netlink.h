/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_NETLINK_H
#define UCS_NETLINK_H

#include <ucs/config/types.h>
#include <ucs/datastruct/khash.h>
#include <ucs/type/status.h>

#include <linux/netlink.h>
#include <netinet/in.h>

BEGIN_C_DECLS

typedef enum {
    UCS_NETLINK_ROUTE_CHECK_BEST,
    UCS_NETLINK_ROUTE_CHECK_RELAXED
} ucs_netlink_route_check_t;

KHASH_MAP_INIT_INT(ucs_netlink_table_flags, char);
typedef khash_t(ucs_netlink_table_flags) ucs_netlink_rt_parse_ctx_t;

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
 * Parse a received netlink datagram buffer containing one or more messages.
 *
 * @param [in]  msg            Pointer to the received netlink message buffer.
 * @param [in]  msg_len        Length of the message buffer in bytes.
 * @param [in]  parse_cb       Callback function to parse each message.
 * @param [in]  arg            User-provided argument for the parse callback.
 * @param [in]  err_log_level  Log level used when an NLMSG_ERROR is received.
 * @param [out] msg_done_p     Set to 1 if NLMSG_DONE was reached, 0 otherwise.
 *
 * @return UCS_OK if the buffer was parsed, UCS_ERR_BUSY if a dump was
 *         interrupted (NLM_F_DUMP_INTR), or error code otherwise.
 */
ucs_status_t ucs_netlink_parse_msg(const void *msg, size_t msg_len,
                                   ucs_netlink_parse_cb_t parse_cb, void *arg,
                                   ucs_log_level_t err_log_level,
                                   int *msg_done_p);

/** Reset the cached netlink routing table (not thread-safe, for testing). */
void ucs_netlink_reset_routing_table_cache(void);

/** Parse an RTM_NEWRULE entry (@a arg: @ref ucs_netlink_rt_parse_ctx_t*). */
ucs_status_t
ucs_netlink_parse_rule_entry_cb(const struct nlmsghdr *nlh, void *arg);

/** Parse an RTM_NEWROUTE entry (@a arg: @ref ucs_netlink_rt_parse_ctx_t*). */
ucs_status_t
ucs_netlink_parse_rt_entry_cb(const struct nlmsghdr *nlh, void *arg);

/**
 * Send a netlink request and parse the response.
 *
 * @param [in]  protocol         Netlink protocol (e.g. NETLINK_ROUTE).
 * @param [in]  nlmsg_type       Protocol message type (e.g. RTM_GETROUTE).
 * @param [in]  nlmsg_flags      Flags for message header (e.g. NLM_F_DUMP).
 * @param [in]  protocol_header  Netlink protocol header.
 * @param [in]  header_length    Netlink protocol header length.
 * @param [in]  parse_cb         Callback function to parse the response.
 * @param [in]  arg              User-provided argument for the parse callback.
 * @param [in]  err_log_level    Log level used when an NLMSG_ERROR is received.
 *
 * @return UCS_OK if the request succeeded, UCS_ERR_BUSY if a dump was
 *         interrupted (NLM_F_DUMP_INTR) and should be retried, or error code
 *         otherwise.
 */
ucs_status_t ucs_netlink_send_request(int protocol, unsigned short nlmsg_type,
                                      unsigned short nlmsg_flags,
                                      const void *protocol_header,
                                      size_t header_length,
                                      ucs_netlink_parse_cb_t parse_cb,
                                      void *arg, ucs_log_level_t err_log_level);

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
