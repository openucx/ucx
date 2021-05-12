/**
* Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_PTR_MAP_H_
#define UCS_PTR_MAP_H_

#include "khash.h"

#include <ucs/sys/compiler.h>
#include <ucs/type/status.h>

#include <stdint.h>

BEGIN_C_DECLS


/**
 * The minimal required pointer alignment.
 */
#define UCS_PTR_MAP_KEY_MIN_ALIGN       UCS_BIT(1)


/**
 * Invalid key.
 */
#define UCS_PTR_MAP_KEY_INVALID ((ucs_ptr_map_key_t)0)


/**
 * Key to find pointer in @ref ucs_ptr_map_t.
 */
typedef uintptr_t ucs_ptr_map_key_t;


KHASH_TYPE(ucs_ptr_map_impl, ucs_ptr_map_key_t, void*);
typedef khash_t(ucs_ptr_map_impl) ucs_ptr_hash_t;


/**
 * Associative container key -> object pointer.
 */
typedef struct ucs_ptr_map {
    uint64_t        next_id; /**< Generator of unique keys per map. */
    ucs_ptr_hash_t  hash;    /**< Hash table to store indirect key to pointer
                                  associations. */
} ucs_ptr_map_t;

END_C_DECLS

#endif
