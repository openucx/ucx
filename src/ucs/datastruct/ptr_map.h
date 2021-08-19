/**
* Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_PTR_MAP_H_
#define UCS_PTR_MAP_H_

#include "khash.h"

#include <ucs/sys/compiler.h>
#include <ucs/type/spinlock.h>
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


typedef struct ucs_ptr_map {
    uint64_t       next_id; /**< Generator of unique keys per map. */
    ucs_ptr_hash_t hash; /**< Hash table to store indirect key to pointer
                              associations. */
} ucs_ptr_map_t;


typedef struct ucs_ptr_safe_hash {
    ucs_ptr_hash_t hash; /**< Hash table to store indirect key to pointer
                              associations. */
    ucs_spinlock_t lock; /**< Spin lock to synchronize access to hash table. */
} ucs_ptr_safe_hash_t;


/**
 * Associative container key -> object pointer.
 *
 * @param [in]  _is_put_thread_safe  Defines whether UCS_PTR_MAP_PUT is thread
 *                                   safe or not. If the operation is safe, then
 *                                   it is safe to use the operation
 *                                   concurrently with itself, UCS_PTR_MAP_GET
 *                                   or UCS_PTR_MAP_DEL operations. 0 or 1 are
 *                                   only acceptable values for the parameter.
 *
 * @note Using UCS_PTR_MAP_GET and UCS_PTR_MAP_DEL with themselves and among
 *       themselves is not thread safe regardless of @a _is_put_thread_safe.
 */
#define UCS_PTR_MAP_TYPE(_name, _is_put_thread_safe) \
    typedef struct { \
        ucs_ptr_map_t       ptr_map; \
        ucs_ptr_safe_hash_t safe[_is_put_thread_safe]; \
    } UCS_PTR_MAP_T(_name);


#define UCS_PTR_MAP_T(_name) UCS_PP_TOKENPASTE3(ucs_ptr_map_, _name, _t)


/* Internal helper function */
ucs_status_t ucs_ptr_map_init(ucs_ptr_map_t *map, int is_put_thread_safe,
                              ucs_ptr_safe_hash_t *safe_hash);


void ucs_ptr_map_destroy(ucs_ptr_map_t *map, int is_put_thread_safe,
                         ucs_ptr_safe_hash_t *safe_hash);


ucs_status_t
ucs_ptr_safe_hash_put(ucs_ptr_map_t *map, void *ptr, ucs_ptr_map_key_t *key,
                      ucs_ptr_safe_hash_t *safe_hash);


ucs_status_t
ucs_ptr_safe_hash_get(ucs_ptr_map_t *map, ucs_ptr_map_key_t key, int extract,
                      void **ptr_p, ucs_ptr_safe_hash_t *safe_hash);

END_C_DECLS

#endif
