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


/**
 * Returns whether the key is indirect or not.
 *
 * @param [in]  key     Key to object pointer.
 *
 * @return 0 - direct, otherside - indirect.
 */
#define ucs_ptr_map_key_indirect(_key) ((_key) & UCS_PTR_MAP_KEY_INDIRECT_FLAG)


KHASH_IMPL(ucs_ptr_map_impl, ucs_ptr_map_key_t, void*, 1,
           kh_int64_hash_func, kh_int64_hash_equal);

static UCS_F_ALWAYS_INLINE ucs_ptr_map_key_t
ucs_ptr_map_create_key(ucs_ptr_map_t *map)
{
    return (map->next_id += UCS_PTR_MAP_KEY_MIN_ALIGN) |
            UCS_PTR_MAP_KEY_INDIRECT_FLAG;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucs_ptr_hash_put(ucs_ptr_hash_t *hash, ucs_ptr_map_key_t key, void *ptr)
{
    khiter_t iter;
    int ret;

    iter = kh_put(ucs_ptr_map_impl, hash, key, &ret);
    if (ucs_unlikely(ret == UCS_KH_PUT_FAILED)) {
        return UCS_ERR_NO_MEMORY;
    } else if (ucs_unlikely(ret == UCS_KH_PUT_KEY_PRESENT)) {
        return UCS_ERR_ALREADY_EXISTS;
    }

    kh_value(hash, iter) = ptr;
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucs_ptr_hash_get(ucs_ptr_hash_t *hash, ucs_ptr_map_key_t key, int extract,
                 void **ptr_p)
{
    khiter_t iter;

    iter = kh_get(ucs_ptr_map_impl, hash, key);
    if (ucs_unlikely(iter == kh_end(hash))) {
        *ptr_p = NULL; /* To suppress compiler warning */
        return UCS_ERR_NO_ELEM;
    }

    *ptr_p = kh_value(hash, iter);
    if (extract) {
        kh_del(ucs_ptr_map_impl, hash, iter);
    }
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucs_ptr_map_put(ucs_ptr_map_t *map, void *ptr, int indirect,
                ucs_ptr_map_key_t *key, int is_put_thread_safe,
                ucs_ptr_safe_hash_t *safe_hash)
{
    if (ucs_likely(!indirect)) {
        *key = (uintptr_t)ptr;
        ucs_assert(!(*key & UCS_PTR_MAP_KEY_MIN_ALIGN));
        ucs_assert(*key != 0);
        return UCS_ERR_NO_PROGRESS;
    }

    if (is_put_thread_safe) {
        return ucs_ptr_safe_hash_put(map, ptr, key, safe_hash);
    }

    *key = ucs_ptr_map_create_key(map);
    return ucs_ptr_hash_put(&map->hash, *key, ptr);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucs_ptr_map_get(ucs_ptr_map_t *map, ucs_ptr_map_key_t key, int extract,
                void **ptr_p, int is_put_thread_safe,
                ucs_ptr_safe_hash_t *safe_hash)
{
    ucs_status_t status;

    if (ucs_likely(!ucs_ptr_map_key_indirect(key))) {
        *ptr_p = (void*)key;
        return UCS_ERR_NO_PROGRESS;
    }

    status = ucs_ptr_hash_get(&map->hash, key, extract, ptr_p);
    if (ucs_likely(status != UCS_ERR_NO_ELEM)) {
        return status;
    }

    if (is_put_thread_safe) {
        status = ucs_ptr_safe_hash_get(map, key, extract, ptr_p, safe_hash);
    }

    return status;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucs_ptr_map_del(ucs_ptr_map_t *map, ucs_ptr_map_key_t key,
                int is_put_thread_safe, ucs_ptr_safe_hash_t *safe_hash)
{
    void UCS_V_UNUSED *dummy;
    return ucs_ptr_map_get(map, key, 1, &dummy, is_put_thread_safe, safe_hash);
}

#define UCS_PTR_MAP_IMPL(_name, _is_put_thread_safe) \
static UCS_F_ALWAYS_INLINE ucs_status_t \
ucs_ptr_map_init_##_name(UCS_PTR_MAP_T(_name) *map) \
{ \
    UCS_STATIC_ASSERT(((_is_put_thread_safe) == 0) || \
                      ((_is_put_thread_safe) == 1)); \
    return ucs_ptr_map_init(&map->ptr_map, _is_put_thread_safe, map->safe); \
} \
\
static UCS_F_ALWAYS_INLINE void \
ucs_ptr_map_destroy_##_name(UCS_PTR_MAP_T(_name) *map) \
{ \
    ucs_ptr_map_destroy(&map->ptr_map, _is_put_thread_safe, map->safe); \
} \
\
static UCS_F_ALWAYS_INLINE ucs_status_t \
ucs_ptr_map_put_##_name(UCS_PTR_MAP_T(_name) *map, void *ptr, int indirect, \
                       ucs_ptr_map_key_t *key) \
{ \
    return ucs_ptr_map_put(&map->ptr_map, ptr, indirect, key, \
                           _is_put_thread_safe, map->safe); \
} \
\
static UCS_F_ALWAYS_INLINE ucs_status_t \
ucs_ptr_map_get_##_name(UCS_PTR_MAP_T(_name) *map, ucs_ptr_map_key_t key, \
                       int extract, void **ptr_p) \
{ \
    return ucs_ptr_map_get(&map->ptr_map, key, extract, ptr_p, \
                           _is_put_thread_safe, map->safe); \
} \
\
static UCS_F_ALWAYS_INLINE ucs_status_t \
ucs_ptr_map_del_##_name(UCS_PTR_MAP_T(_name) *map, ucs_ptr_map_key_t key) \
{ \
    return ucs_ptr_map_del(&map->ptr_map, key, _is_put_thread_safe, \
                           map->safe); \
}


#define UCS_PTR_MAP_DEFINE(_name, _is_put_thread_safe) \
    UCS_PTR_MAP_TYPE(_name, _is_put_thread_safe) \
    UCS_PTR_MAP_IMPL(_name, _is_put_thread_safe)

/**
 * Initialize a pointer map.
 *
 * @param [in]  map  Map to initialize.
 * @return      UCS_OK on success otherwise error code as defined by
 *              @ref ucs_status_t.
 */
#define UCS_PTR_MAP_INIT(_name, _map) ucs_ptr_map_init_##_name(_map)

/**
 * Destroy a pointer map.
 *
 * @param [in]  map  Map to destroy.
 */
#define UCS_PTR_MAP_DESTROY(_name, _map) ucs_ptr_map_destroy_##_name(_map)

/**
 * Put a pointer into the map.
 *
 * @param [in]  map       Container.
 * @param [in]  ptr       Object pointer.
 * @param [in]  indirect  If nonzero, the pointer @a ptr is stored in the
 *                        internal hash table, associated with a unique indirect
 *                        key which is returned from this function. Otherwise,
 *                        the returned value is the integer representation of
 *                        the pointer @a ptr.
 * @param [out] key       Key to object pointer @ptr if operation completed
 *                        successfully otherwise value is undefined.
 * @return UCS_OK on success, UCS_ERR_NO_PROGRESS if this key is direct and
 *         therefore no action was performed, otherwise error code as defined
 *         by @ref ucs_status_t.
 * @note @a ptr must be aligned on @ref UCS_PTR_MAP_KEY_MIN_ALIGN.
 */
#define UCS_PTR_MAP_PUT(_name, _map, _ptr, _indirect, _key) \
    ucs_ptr_map_put_##_name(_map, _ptr, _indirect, _key)

/**
 * Get a pointer value from the map by its key.
 *
 * @param [in]  map     Container to get the pointer value from.
 * @param [in]  key     Key to look up in the container.
 * @param [in]  extract Whether to remove the key from the map.
 * @param [out] ptr_p   If successful, set to the pointer found in the map.
 *
 * @return UCS_OK if found, UCS_ERR_NO_PROGRESS if this key is direct and
 *         therefore no action was performed, UCS_ERR_NO_ELEM if not found.
 */
#define UCS_PTR_MAP_GET(_name, _map, _key, _extract, _ptr_p) \
    ucs_ptr_map_get_##_name(_map, _key, _extract, _ptr_p)

/**
 * Remove object pointer from the map by key.
 *
 * @param [in]  map Container.
 * @param [in]  key Key to object pointer.
 *
 * @return - UCS_OK on success
 *         - UCS_ERR_NO_PROGRESS if this key is direct and therefore no action
 *           was performed
 *         - UCS_ERR_NO_ELEM if the key is not found in the internal hash
 *           table.
 */
#define UCS_PTR_MAP_DEL(_name, _map, _key) ucs_ptr_map_del_##_name(_map, _key)

END_C_DECLS

#endif
