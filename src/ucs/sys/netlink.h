/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_NETLINK_H
#define UCS_NETLINK_H

#include <ucs/type/status.h>

#include <netinet/in.h>

BEGIN_C_DECLS


/**
 * Check whether a routing table rule exists for a given network
 * interface name and a destination address.
 *
 * @param [in]  if_index   An index representing the network interface.
 * @param [in]  sa_remote  Pointer to the destination address.
 *
 * @return 1 if rule exists, or 0 otherwise.
 */
int ucs_netlink_route_exists(int if_index, const struct sockaddr *sa_remote);

END_C_DECLS

#endif /* UCS_NETLINK_H */
