/**
 * Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "ptr_map.inl"


static void ucs_ptr_hash_destroy(ucs_ptr_hash_t *ptr_hash)
{
    size_t size = kh_size(ptr_hash);
    if (size != 0) {
        ucs_warn("ptr hash %p contains %zd elements on destroy", ptr_hash,
                 size);
    }
    kh_destroy_inplace(ucs_ptr_map_impl, ptr_hash);
}

static ucs_status_t ucs_ptr_safe_hash_init(ucs_ptr_safe_hash_t *safe_hash)
{
    kh_init_inplace(ucs_ptr_map_impl, &safe_hash->hash);
    return ucs_spinlock_init(&safe_hash->lock, 0);
}

static void ucs_ptr_safe_hash_destroy(ucs_ptr_safe_hash_t *safe_hash)
{
    ucs_ptr_hash_destroy(&safe_hash->hash);
    ucs_spinlock_destroy(&safe_hash->lock);
}

ucs_status_t ucs_ptr_safe_hash_get(ucs_ptr_map_t *map, ucs_ptr_map_key_t key,
                                   int extract, void **ptr_p,
                                   ucs_ptr_safe_hash_t *safe_hash)
{
    ucs_status_t status;

    ucs_spin_lock(&safe_hash->lock);
    status = ucs_ptr_hash_get(&safe_hash->hash, key, 1, ptr_p);
    ucs_spin_unlock(&safe_hash->lock);

    if ((status == UCS_OK) && !extract) {
        ucs_ptr_hash_put(&map->hash, key, *ptr_p);
    }

    return status;
}

ucs_status_t ucs_ptr_safe_hash_put(ucs_ptr_map_t *map, void *ptr,
                                   ucs_ptr_map_key_t *key,
                                   ucs_ptr_safe_hash_t *safe_hash)
{
    ucs_status_t status;

    ucs_spin_lock(&safe_hash->lock);
    *key   = ucs_ptr_map_create_key(map);
    status = ucs_ptr_hash_put(&safe_hash->hash, *key, ptr);
    ucs_spin_unlock(&safe_hash->lock);

    return status;
}

ucs_status_t ucs_ptr_map_init(ucs_ptr_map_t *map, int is_put_thread_safe,
                              ucs_ptr_safe_hash_t *safe_hash)
{
    ucs_status_t status = UCS_OK;

    UCS_STATIC_ASSERT(!ucs_ptr_map_key_indirect(UCS_PTR_MAP_KEY_INVALID));
    map->next_id = 0;
    kh_init_inplace(ucs_ptr_map_impl, &map->hash);

    if (is_put_thread_safe) {
        status = ucs_ptr_safe_hash_init(safe_hash);
    }

    return status;
}

void ucs_ptr_map_destroy(ucs_ptr_map_t *map, int is_put_thread_safe,
                         ucs_ptr_safe_hash_t *safe_hash)
{
    ucs_ptr_hash_destroy(&map->hash);

    if (is_put_thread_safe) {
        ucs_ptr_safe_hash_destroy(safe_hash);
    }
}
