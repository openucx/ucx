/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_NETLINK_INT_H
#define UCS_NETLINK_INT_H

#include "netlink.h"

#include <ucs/datastruct/array.h>
#include <ucs/datastruct/khash.h>
#include <ucs/type/status.h>

#include <stdint.h>
#include <sys/socket.h>

BEGIN_C_DECLS

/* A single routing rule of a network interface */
typedef struct {
    struct sockaddr_storage dest;
    uint8_t                 subnet_prefix_len;
    uint8_t                 route_type;
} ucs_netlink_route_entry_t;


UCS_ARRAY_DECLARE_TYPE(ucs_netlink_rt_rules_t, unsigned,
                       ucs_netlink_route_entry_t);

KHASH_INIT(ucs_netlink_rt_cache, khint32_t, ucs_netlink_rt_rules_t, 1,
           kh_int_hash_func, kh_int_hash_equal);


/* Routing rules of all network interfaces, keyed by interface index */
typedef khash_t(ucs_netlink_rt_cache) ucs_netlink_route_table_t;


/**
 * Initialize an empty route table.
 *
 * @param [out] route_table    Route table to initialize.
 */
void ucs_netlink_route_table_init(ucs_netlink_route_table_t *route_table);


/**
 * Release the resources of a route table.
 *
 * @param [in]  route_table    Route table to clean up.
 */
void ucs_netlink_route_table_cleanup(ucs_netlink_route_table_t *route_table);


/**
 * Add a routing rule to a route table.
 *
 * @param [in]  route_table        Route table to add the rule to.
 * @param [in]  if_index           Network interface index of the rule.
 * @param [in]  dest               Destination address of the rule.
 * @param [in]  subnet_prefix_len  Subnet prefix length of the rule.
 * @param [in]  route_type         Route type of the rule (e.g. RTN_UNICAST).
 *
 * @return UCS_OK if the rule was added, or error code otherwise.
 */
ucs_status_t
ucs_netlink_route_table_add(ucs_netlink_route_table_t *route_table,
                            int if_index, const struct sockaddr *dest,
                            uint8_t subnet_prefix_len, uint8_t route_type);


/**
 * Check whether a route to a given destination address through a network
 * interface matches the requested policy, according to the given route table.
 *
 * @param [in]  route_table      Route table to search in.
 * @param [in]  if_index         Network interface index.
 * @param [in]  sa_remote        Pointer to the destination address.
 * @param [in]  route_check      Route matching policy, see
 *                               @ref ucs_netlink_route_matches.
 *
 * @return 1 if the route is accepted by the requested policy, or 0 otherwise.
 */
int ucs_netlink_route_matches_in_table(ucs_netlink_route_table_t *route_table,
                                       int if_index,
                                       const struct sockaddr *sa_remote,
                                       ucs_netlink_route_check_t route_check);


/**
 * Get the network interface index of a local route to a given destination
 * address, according to the given route table.
 *
 * @param [in]  route_table      Route table to search in.
 * @param [in]  sa_remote        Pointer to the destination address.
 *
 * @return Network interface index of a local route, or -1 if not found.
 */
int ucs_netlink_local_route_ndev_index_in_table(
                                    ucs_netlink_route_table_t *route_table,
                                    const struct sockaddr *sa_remote);

END_C_DECLS

#endif /* UCS_NETLINK_INT_H */
