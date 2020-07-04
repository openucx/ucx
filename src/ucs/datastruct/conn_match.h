/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_CONN_MATCH_H_
#define UCS_CONN_MATCH_H_

#include <ucs/datastruct/khash.h>
#include <ucs/datastruct/hlist.h>

#include <inttypes.h>


BEGIN_C_DECLS


/**
 * Connection sequence number
 */
typedef uint64_t ucs_conn_sn_t;


/**
 * Connection queue type
 */
typedef enum ucs_conn_match_queue_type {
    /* Queue type for connections created by API and not
     * connected to remote peer */
    UCS_CONN_MATCH_QUEUE_EXP,
    /* Queue type for connections created internally as
     * connected to remote peer, but not provided to user yet */
    UCS_CONN_MATCH_QUEUE_UNEXP,
    /* Number of queues that are used by connection matching */
    UCS_CONN_MATCH_QUEUE_LAST
} ucs_conn_match_queue_type_t;


/**
 * Structure to embed in a connection entry to support matching with remote
 * peer's connections.
 */
typedef struct ucs_conn_match_elem {
    ucs_hlist_link_t          list;          /* List entry into endpoint
                                                matching structure */
} ucs_conn_match_elem_t;


/**
 * Function to get the address of the connection between the peers.
 *
 * @param [in]  elem          Pointer to the connection matching element.
 *
 * @return Pointer to the address of the connection between the peers.
 */
typedef const void*
(*ucs_conn_match_get_address_t)(const ucs_conn_match_elem_t *elem);


/**
 * Function to get the sequence number of the connection between the peers.
 *
 * @param [in] elem  Pointer to the connection matching element.
 *
 * @return Sequnce number of the given connection between the peers.
 */
typedef ucs_conn_sn_t
(*ucs_conn_match_get_conn_sn_t)(const ucs_conn_match_elem_t *elem);


/**
 * Function to get string representation of the connection address.
 *
 * @param [in]  address     Pointer to the connection address.
 * @param [out] str         A string filled with the address.
 * @param [in]  max_size    Size of a string (considering '\0'-terminated symbol).
 *
 * @return A resulted string filled with the address.
 */
typedef const char*
(*ucs_conn_match_address_str_t)(const void *address,
                                char *str, size_t max_size);


/**
 * Connection matching operations
 */
typedef struct ucs_conn_match_ops {
    ucs_conn_match_get_address_t get_address;
    ucs_conn_match_get_conn_sn_t get_conn_sn;
    ucs_conn_match_address_str_t address_str;
} ucs_conn_match_ops_t;


/**
 * Connection peer entry - allows *ordered* matching of connections between
 * a pair of connected peers. The expected/unexpected lists are *not* circular
 */
typedef struct ucs_conn_match_peer ucs_conn_match_peer_t;


KHASH_TYPE(ucs_conn_match, ucs_conn_match_peer_t*, char)


/**
 * Context for matching connections
 */
typedef struct ucs_conn_match_ctx {
    khash_t(ucs_conn_match)      hash;           /* Hash of matched connections */
    size_t                       address_length; /* Length of the addresses used for the
                                                    connection between peers */
    ucs_conn_match_ops_t         ops;            /* User's connection matching operations */
} ucs_conn_match_ctx_t;


/**
 * Initialize the connection matching context.
 *
 * @param [in] conn_match_ctx    Pointer to the connection matching context.
 * @param [in] address_length    Length of the addresses used for the connection
 *                               between peers.
 * @param [in] ops               Pointer to the user-defined connection matching
 *                               operations.
 */
void ucs_conn_match_init(ucs_conn_match_ctx_t *conn_match_ctx,
                         size_t address_length,
                         const ucs_conn_match_ops_t *ops);


/**
 * Cleanup the connection matching context.
 *
 * @param [in] conn_match_ctx    Pointer to the connection matching context.
 */
void ucs_conn_match_cleanup(ucs_conn_match_ctx_t *conn_match_ctx);


/**
 * Get the next value of the connection sequence number between two peers.
 *
 * @param [in] conn_match_ctx    Pointer to the connection matching context.
 * @param [in] address           Pointer to the address of the connection
 *                               between the peers.
 *
 * @return The next value of the connection sequence number, this value is unique
 *         for the given connection.
 */
ucs_conn_sn_t ucs_conn_match_get_next_sn(ucs_conn_match_ctx_t *conn_match_ctx,
                                         const void *address);


/**
 * Insert the connection matching entry to the context.
 *
 * @param [in] conn_match_ctx    Pointer to the connection matching context.
 * @param [in] address           Pointer to the address of the connection
 *                               between the peers.
 * @param [in] conn_sn           Connection sequence number of the connection.
 * @param [in] elem              Pointer to the connection matching structure.
 * @param [in] conn_queue_type   Connection queue which should be used to insert
 *                               the connection matching element to.
 */
void ucs_conn_match_insert(ucs_conn_match_ctx_t *conn_match_ctx,
                           const void *address, ucs_conn_sn_t conn_sn,
                           ucs_conn_match_elem_t *elem,
                           ucs_conn_match_queue_type_t conn_queue_type);


/**
 * Retrieve the connection matching entry from the context.
 *
 * @param [in] conn_match_ctx    Pointer to the connection matching context.
 * @param [in] address           Pointer to the address of the connection
 *                               between the peers.
 * @param [in] conn_sn           Connection sequence number of the connection.
 * @param [in] conn_queue_type   Connection queue which should be used to retrieve
 *                               the connection matching element from.
 *
 * @return Pointer to the found connection matching entry.
 */
ucs_conn_match_elem_t *
ucs_conn_match_retrieve(ucs_conn_match_ctx_t *conn_match_ctx,
                        const void *address, ucs_conn_sn_t conn_sn,
                        ucs_conn_match_queue_type_t conn_queue_type);


/**
 * Remove the connection matching entry from the context.
 *
 * @param [in] conn_match_ctx    Pointer to the connection matching context.
 * @param [in] elem              Pointer to the connection matching element.
 * @param [in] conn_queue_type   Connection queue which should be used to remove
 *                               the connection matching element from.
 *
 * @note Connection @conn_match matching entry must be present in the queue
 *       pointed by @conn_queue_type.
 */
void ucs_conn_match_remove_elem(ucs_conn_match_ctx_t *conn_match_ctx,
                                ucs_conn_match_elem_t *elem,
                                ucs_conn_match_queue_type_t conn_queue_type);


END_C_DECLS


#endif
