/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2023. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_LRU_H_
#define UCS_LRU_H_

#include <stddef.h>
#include <stdint.h>


#include <ucs/datastruct/khash.h>
#include <ucs/datastruct/list.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/type/status.h>

/* LRU element data structure */
typedef struct {
    /* Key to use as hash table input */
    void           *key;
    /* Linked list item */
    ucs_list_link_t list;
} ucs_lru_element_t;


KHASH_INIT(ucs_lru_hash, uint64_t, ucs_lru_element_t*, 1, kh_int64_hash_func,
           kh_int64_hash_equal)


/* Hash table type for LRU cache */
typedef khash_t(ucs_lru_hash) ucs_lru_hash_t;


/* LRU cache data structure */
typedef struct ucs_lru {
    /* Hash table of addresses as keys */
    ucs_lru_hash_t  hash;
    /* Linked list ordered by most recently accessed */
    ucs_list_link_t list;
    /* Number of elements currently in cache */
    size_t          capacity;
} ucs_lru_t;


typedef struct ucs_lru *ucs_lru_h;


/**
 * @brief Create a new LRU cache object.
 *
 * @param [in]    capacity  Cache capacity.
 * @param [inout] lru_p     Pointer to the allocated LRU struct. Filled with the
 *                          LRU handle.
 *
 * @return UCS_OK if successful, or an error code as defined by
 * @ref ucs_status_t otherwise.
 */
ucs_status_t ucs_lru_create(size_t capacity, ucs_lru_h *lru_p);


/**
 * @brief Destroys an LRU cache object.
 *
 * @param [in] lru  Handle to the LRU cache.
 */
void ucs_lru_destroy(ucs_lru_h lru);


static UCS_F_ALWAYS_INLINE ucs_lru_element_t *ucs_lru_pop(ucs_lru_h lru)
{
    ucs_lru_element_t *tail;
    khint_t iter;

    tail = ucs_list_tail(&lru->list, ucs_lru_element_t, list);
    iter = kh_get(ucs_lru_hash, &lru->hash, (uint64_t)tail->key);

    ucs_list_del(&tail->list);
    kh_del(ucs_lru_hash, &lru->hash, iter);
    return tail;
}


/**
 * @brief Checks if a given key exists in the LRU cache.
 *
 * @param [in] lru  Handle to the LRU cache.
 * @param [in] key  Element's key.
 *
 * @return 1 if entry was found, 0 otherwise.
 */
static UCS_F_ALWAYS_INLINE int ucs_lru_is_present(ucs_lru_h lru, void *key)
{
    return kh_get(ucs_lru_hash, &lru->hash, (uint64_t)key) !=
           kh_end(&lru->hash);
}


/**
 * @brief Insert or update an element in the cache.
 *
 * @param [in] lru  Handle to the LRU cache.
 * @param [in] key  Element's key.
 *
 */
static UCS_F_ALWAYS_INLINE void ucs_lru_push(ucs_lru_h lru, void *key)
{
    khint_t iter;
    int ret;
    ucs_lru_element_t **elem_p;

    iter = kh_put(ucs_lru_hash, &lru->hash, (uint64_t)key, &ret);
    ucs_assert(ret != UCS_KH_PUT_FAILED);

    elem_p = &kh_val(&lru->hash, iter);

    if (ucs_likely(ret == UCS_KH_PUT_KEY_PRESENT)) {
        ucs_list_del(&(*elem_p)->list);
    } else if (kh_size(&lru->hash) > lru->capacity) {
        *elem_p = ucs_lru_pop(lru);
    } else {
        *elem_p = (ucs_lru_element_t*)ucs_malloc(sizeof(**elem_p),
                                                 "ucs_lru_element");
    }

    (*elem_p)->key = key;
    ucs_list_add_head(&lru->list, &(*elem_p)->list);
}


/**
 * @brief Resets an LRU object.
 *
 * @param [in] lru  Handle to the LRU cache.
 *
 */
void ucs_lru_reset(ucs_lru_h lru);


static UCS_F_ALWAYS_INLINE void **ucs_lru_next_key(ucs_list_link_t *elem)
{
    return &ucs_container_of(elem->next, ucs_lru_element_t, list)->key;
}


/**
 * Iterate over elements of the LRU.
 *
 * @param [in] _elem  Pointer to the current key (void**).
 * @param [in] _lru   Handle to the LRU cache.
 */
#define ucs_lru_for_each(_elem, _lru) \
    for (_elem = ucs_lru_next_key(&(_lru)->list); \
         &ucs_container_of((_elem), ucs_lru_element_t, key)->list != \
         &(_lru)->list; \
         _elem = ucs_lru_next_key( \
                 &ucs_container_of((_elem), ucs_lru_element_t, key)->list))

#endif
