/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_INTERVAL_MAP_H_
#define UCS_INTERVAL_MAP_H_

#include <ucs/datastruct/rbtree.h>
#include <ucs/sys/compiler_def.h>
#include <stdint.h>
#include <stddef.h>

BEGIN_C_DECLS

/**
 * Node of a @ref ucs_interval_map_t, embedded in the caller's object.
 */
typedef struct {
    ucs_rbtree_node_t super;   /**< Balancing links, must be first */
    uint64_t          start;   /**< Interval start, inclusive */
    uint64_t          end;     /**< Interval end, exclusive */
    uint64_t          max_end; /**< Maximum 'end' in this subtree */
} ucs_interval_map_node_t;


/**
 * A map of half-open intervals [start, end) to caller-owned objects.
 *
 * Does not merge overlapping and touching intervals into one node.
 */
typedef struct {
    ucs_rbtree_t rb;        /**< Balanced tree ordered by 'start' */
    size_t       num_nodes; /**< Number of intervals */
} ucs_interval_map_t;


/**
 * Callback for @ref ucs_interval_map_foreach_overlapping. It must not modify
 * the map.
 */
typedef void (*ucs_interval_map_cb_t)(ucs_interval_map_node_t *node, void *arg);


/**
 * @brief Initialize an empty interval map
 *
 * @param [in]  map  Interval map to initialize.
 */
void ucs_interval_map_init(ucs_interval_map_t *map);


/**
 * @brief Number of intervals in the map
 *
 * @param [in]  map  Interval map.
 *
 * @return Number of intervals currently in @a map.
 */
static UCS_F_ALWAYS_INLINE size_t
ucs_interval_map_count(const ucs_interval_map_t *map)
{
    return map->num_nodes;
}


/**
 * @brief Insert an interval. Overlaps and duplicates remain distinct.
 *
 * @param [in]  map    Interval map.
 * @param [in]  node   Caller-allocated node. Its contents are overwritten.
 * @param [in]  start  Interval start, inclusive.
 * @param [in]  end    Interval end, exclusive. Must be greater than @a start.
 */
void ucs_interval_map_insert(ucs_interval_map_t *map,
                             ucs_interval_map_node_t *node, uint64_t start,
                             uint64_t end);


/**
 * @brief Remove an interval. Other nodes are neither moved nor invalidated.
 *
 * @param [in]  map   Interval map.
 * @param [in]  node  Node to remove, which must be in @a map.
 */
void ucs_interval_map_remove(ucs_interval_map_t *map,
                             ucs_interval_map_node_t *node);


/**
 * @brief Find an interval containing [start, end)
 *
 * @param [in]  map    Interval map.
 * @param [in]  start  Start of the range to contain, inclusive.
 * @param [in]  end    End of the range to contain, exclusive. Must be greater
 *                     than @a start.
 *
 * @return The first node found with (start <= @a start) and (end >= @a end),
 *         or NULL if none contains the whole range. Node ordering is not
 *         guaranteed to be stable across map modifications.
 */
ucs_interval_map_node_t *
ucs_interval_map_find_containing(const ucs_interval_map_t *map, uint64_t start,
                                 uint64_t end);


/**
 * @brief Invoke @a cb for every interval overlapping [start, end)
 *
 * @param [in]  map    Interval map.
 * @param [in]  start  Start of the range to overlap, inclusive.
 * @param [in]  end    End of the range to overlap, exclusive. Must be greater
 *                     than @a start.
 * @param [in]  cb     Called once per overlapping interval. Must not modify
 *                     @a map.
 * @param [in]  arg    Passed to @a cb.
 */
void ucs_interval_map_foreach_overlapping(const ucs_interval_map_t *map,
                                          uint64_t start, uint64_t end,
                                          ucs_interval_map_cb_t cb, void *arg);

END_C_DECLS

#endif
