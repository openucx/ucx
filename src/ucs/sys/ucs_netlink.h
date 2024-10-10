/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_NETLINK_H
#define UCS_NETLINK_H

#include <ucs/type/status.h>

#include <stddef.h>
#include <linux/netlink.h>


#define ucs_netlink_foreach(elem, msg, len) \
    for (elem = (struct nlmsghdr *)msg; \
         NLMSG_OK(elem, len) && (elem->nlmsg_type != NLMSG_DONE) && \
         (elem->nlmsg_type != NLMSG_ERROR); \
         elem = NLMSG_NEXT(elem, len))

#define ucs_netlink_handle_parse_error(nlh, action) \
    do { \
        if (nlh->nlmsg_type == NLMSG_ERROR) { \
            ucs_diag("failed to parse netlink message header (%d)", \
                    ((struct nlmsgerr*)NLMSG_DATA(nlh))->error); \
            action; \
        } \
    } while (0)


/**
 * Sends and receives a netlink message using a user allocated buffer.
 *
 * @param [in]    protocol         The communication protocol to be used
 *                                 (NETLINK_ROUTE, NETLINK_NETFILTER, etc.).
 * @param [in]    nl_protocol_hdr  A struct that holds nl protocol specific
 *                                 details and is placed in nlmsghdr.
 * @param [in]    nl_protocol_hdr_size Protocol struct size.
 * @param [out]   recv_msg_buf     The buffer that will hold the received message.
 * @param [inout] recv_msg_buf_len Pointer to the size of the buffer and to
 *                                 store the length of the received message.
 * @param [in]    nlmsg_type       Netlink message type (RTM_GETROUTE,
 *                                 RTM_GETNEIGH, etc.).
 *
 * @return UCS_OK if received successfully, or error code otherwise.
 */
ucs_status_t
ucs_netlink_send_cmd(int protocol, void *nl_protocol_hdr,
                     size_t nl_protocol_hdr_size, char *recv_msg_buf,
                     size_t *recv_msg_buf_len, unsigned short nlmsg_type);


#endif // UCS_NETLINK_H
