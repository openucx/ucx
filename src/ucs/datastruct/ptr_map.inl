/**
* Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_PTR_MAP_INL_
#define UCS_PTR_MAP_INL_

#include "ptr_map.h"

#include <ucs/debug/log.h>

BEGIN_C_DECLS


/**
 * The flag is present in key if it is indirect key.
 */
#define UCS_PTR_MAP_KEY_INDIRECT_FLAG   UCS_BIT(0)


KHASH_IMPL(ucs_ptr_map_impl, ucs_ptr_map_key_t, void*, 1,
           kh_int64_hash_func, kh_int64_hash_equal);

/**
 * Initialize a pointer map.
 *
 * @param [in]  map     Map to initialize.
 * @return      UCS_OK on success otherwise error code as defined by
 *              @ref ucs_status_t.
 */
static inline ucs_status_t ucs_ptr_map_init(ucs_ptr_map_t *map)
{
    map->next_id = 0;
    kh_init_inplace(ucs_ptr_map_impl, &map->hash);
    return UCS_OK;
}

/**
 * Destroy a pointer map.
 *
 * @param [in]  map     Map to destroy.
 */
static inline void ucs_ptr_map_destroy(ucs_ptr_map_t *map)
{
    size_t size = kh_size(&map->hash);

    if (size != 0) {
        ucs_warn("ptr map %p contains %zd elements on destroy", map, size);
    }

    kh_destroy_inplace(ucs_ptr_map_impl, &map->hash);
}

/**
 * Put a pointer into the map.
 *
 * @param [in]  map       Container.
 * @param [in]  ptr       Object pointer.
 * @param [in]  indirect  If nonzero, the pointer @a ptr is stored in the
 *                        internal hash table, associated with a unique indirect
 *                        key which is returned from this function. Otherwise,
 *                        the returned value is the integer representation of
 *                        the pointer @ptr.
 * @param [out] key       Key to object pointer @ptr if operation completed
 *                        successfully otherwise value is undefined.
 * @return      UCS_OK on success otherwise error code as defined by
 *              @ref ucs_status_t.
 * @note @ptr must be aligned on @ref UCS_PTR_MAP_KEY_MIN_ALIGN.
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
ucs_ptr_map_put(ucs_ptr_map_t *map, void *ptr, int indirect,
                ucs_ptr_map_key_t *key)
{
    khiter_t iter;
    int ret;

    if (ucs_likely(!indirect)) {
        *key = (uintptr_t)ptr;
        ucs_assert(!(*key & UCS_PTR_MAP_KEY_MIN_ALIGN));
        return UCS_OK;
    }

    *key = (map->next_id += UCS_PTR_MAP_KEY_MIN_ALIGN) |
           UCS_PTR_MAP_KEY_INDIRECT_FLAG;

    iter = kh_put(ucs_ptr_map_impl, &map->hash, *key, &ret);
    if (ucs_unlikely(ret == UCS_KH_PUT_FAILED)) {
        return UCS_ERR_NO_MEMORY;
    } else if (ucs_unlikely(ret == UCS_KH_PUT_KEY_PRESENT)) {
        return UCS_ERR_ALREADY_EXISTS;
    }

    kh_value(&map->hash, iter) = ptr;
    return UCS_OK;
}

/**
 * Get a pointer value from the map by its key.
 *
 * @param [in]  map     Container to get the pointer value from.
 * @param [in]  key     Key to look up in the container.
 * @return object pointer on success, otherwise NULL.
 */
static UCS_F_ALWAYS_INLINE void*
ucs_ptr_map_get(const ucs_ptr_map_t *map, ucs_ptr_map_key_t key)
{
    khiter_t iter;

    if (ucs_likely(!(key & UCS_PTR_MAP_KEY_INDIRECT_FLAG))) {
        return (void*)key;
    }

    iter = kh_get(ucs_ptr_map_impl, &map->hash, key);
    return ucs_unlikely(iter == kh_end(&map->hash)) ? NULL :
           kh_value(&map->hash, iter);
}

/**
 * Extract a pointer value from the map by its key.
 *
 * @param [in]  map     Container to get the pointer value from.
 * @param [in]  key     Key to look up in the container.
 * @return object pointer on success, otherwise NULL.
 */
static UCS_F_ALWAYS_INLINE void*
ucs_ptr_map_extract(ucs_ptr_map_t *map, ucs_ptr_map_key_t key)
{
    khiter_t iter;
    void *value;

    if (ucs_likely(!(key & UCS_PTR_MAP_KEY_INDIRECT_FLAG))) {
        return (void*)key;
    }

    iter = kh_get(ucs_ptr_map_impl, &map->hash, key);
    if (ucs_unlikely(iter == kh_end(&map->hash))) {
        return NULL;
    }

    value = kh_value(&map->hash, iter);
    kh_del(ucs_ptr_map_impl, &map->hash, iter);
    return value;
}

/**
 * Remove object pointer from the map by key.
 *
 * @param [in]  map     Container.
 * @param [in]  key     Key to object pointer.
 * @return       - UCS_OK on success
 *               - UCS_ERR_NO_ELEM if the key is not found in the internal hash
 *                 table.
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
ucs_ptr_map_del(ucs_ptr_map_t *map, ucs_ptr_map_key_t key)
{
    return ucs_likely(ucs_ptr_map_extract(map, key) != NULL) ?
           UCS_OK : UCS_ERR_NO_ELEM;
}

END_C_DECLS

#endif
