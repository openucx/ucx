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

BEGIN_C_DECLS

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
 * Check whether a routing table rule exists for a given network
 * interface name and a destination address.
 *
 * @param [in]  if_name    Pointer to the name of the interface.
 * @param [in]  sa_remote  Pointer to the destination address.
 *
 * @return 1 if rule exists, or 0 otherwise.
 */
int ucs_netlink_route_exists(const char *if_name,
                             const struct sockaddr *sa_remote);

END_C_DECLS

#endif /* UCS_NETLINK_H */
