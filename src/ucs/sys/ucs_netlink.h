/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_NETLINK_H
#define UCS_NETLINK_H

#include <ucs/type/status.h>

#include <linux/netlink.h>
#include <stddef.h>

BEGIN_C_DECLS


/**
 * Callback function for parsing individual netlink messages.
 *
 * @param [in] nlh    Pointer to the netlink message header.
 * @param [in] nl_msg Pointer to the netlink message payload.
 * @param [in] arg    User-provided argument passed through from the caller.
 *
 * @return UCS_OK if parsing is complete, UCS_INPROGRESS if there are more
 *         messages to be parsed, or error code otherwise.
 */
typedef ucs_status_t (*ucs_netlink_parse_cb_t)(struct nlmsghdr *nlh,
                                               void *nl_msg, void *arg);


/**
 * Sends and receives a netlink message using a user allocated buffer.
 *
 * @param [in]    protocol             The communication protocol to be used
 *                                     (NETLINK_ROUTE, NETLINK_NETFILTER, etc.).
 * @param [in]    nlmsg_type           Netlink message type (RTM_GETROUTE,
 *                                     RTM_GETNEIGH, etc.).
 * @param [in]    nl_protocol_hdr      A struct that holds nl protocol specific
 *                                     details and is placed in nlmsghdr.
 * @param [in]    nl_protocol_hdr_size Protocol struct size.
 * @param [out]   recv_msg_buf         The buffer that will hold the received
 *                                     message.
 * @param [inout] recv_msg_buf_len     Pointer to the size of the buffer and to
 *                                     store the length of the received message.
 *
 * @return UCS_OK if received successfully, or error code otherwise.
 */
ucs_status_t ucs_netlink_send_cmd(int protocol, unsigned short nlmsg_type,
                                  void *nl_protocol_hdr,
                                  size_t nl_protocol_hdr_size,
                                  char *recv_msg_buf, size_t *recv_msg_buf_len);


/**
 * Iterates over the netlink headers and parses each one of them
 * using a callback function provided by the caller.
 *
 * @param [in]  msg       Pointer to the full netlink message.
 * @param [in]  msg_len   Length of the netlink message in bytes.
 * @param [in]  parse_cb  The callback function to parse each sub-message (entry).
 * @param [in]  arg       Pointer to the callback function arguments.
 *
 * @return UCS_OK if parsed successfully, or error code otherwise.
 */
ucs_status_t ucs_netlink_parse_msg(void *msg, size_t msg_len,
                                   ucs_netlink_parse_cb_t parse_cb, void *arg);


END_C_DECLS

#endif /* UCS_NETLINK_H */
