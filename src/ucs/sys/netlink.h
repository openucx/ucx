/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_NETLINK_H
#define UCS_NETLINK_H

#include <ucs/config/types.h>
#include <ucs/type/status.h>

#include <linux/netlink.h>
#include <netinet/in.h>

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

/**
 * Parse an RTM_NEWROUTE entry into the routing table cache.
 *
 * @param [in] nlh    Pointer to the netlink message header.
 * @param [in] arg    Unused.
 *
 * @return UCS_INPROGRESS if parsing should continue, or error code otherwise.
 */
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
 * Check whether a destination address is assigned to the local host in the
 * routing domain of a network interface, i.e. an interface of the same VRF (or
 * an interface out of any VRF) has a local route to it.
 *
 * @param [in]  if_index         Network interface index.
 * @param [in]  sa_remote        Pointer to the destination address.
 *
 * @return 1 if the destination address is local, or 0 otherwise.
 */
int ucs_netlink_is_local_addr(int if_index, const struct sockaddr *sa_remote);

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

/**
 * Callback function for iterating over the routes of a network interface.
 *
 * @param [in] dest        Destination network address of the route.
 * @param [in] prefix_len  Destination prefix length in bits (0 for a default
 *                         route).
 * @param [in] route_type  Route type (RTN_xxx).
 * @param [in] arg         User-provided argument passed through from the
 *                         caller.
 */
typedef void (*ucs_netlink_route_cb_t)(const struct sockaddr *dest,
                                       unsigned prefix_len, uint8_t route_type,
                                       void *arg);

/**
 * Invoke a callback for each route of a network interface, as recorded in the
 * routing table snapshot taken on the first netlink route query.
 *
 * @param [in]  if_index  Network interface index.
 * @param [in]  cb        Callback to invoke for each route.
 * @param [in]  arg       User-provided argument for the callback.
 */
void ucs_netlink_foreach_route(int if_index, ucs_netlink_route_cb_t cb,
                               void *arg);

/**
 * Get the VRF (L3 master) device of a network interface: the VRF device it is
 * enslaved to, or the interface itself if it is a VRF device. Addresses of such
 * interface can be used only by sockets bound to the VRF.
 * VRF membership of all interfaces is read once and cached at the first call.
 *
 * @param [in]  if_index         Network interface index.
 *
 * @return Network interface index of the VRF device, or -1 if the interface
 *         is neither a VRF device nor enslaved to one.
 */
int ucs_netlink_get_vrf_master_index(int if_index);

END_C_DECLS

#endif /* UCS_NETLINK_H */
