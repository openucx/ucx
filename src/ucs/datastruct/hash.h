/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCS_HASH_H_
#define UCS_HASH_H_

#include "sglib_wrapper.h"


/**
 * Defines types and functions for a thread-safe hash table.
 * (see sglib docs for further explanations)
 *
 * @param _type     Type of elements in the hash table.
 * @param _next     "next" filed inside "_type".
 * @param _dim      Hash table size.
 * @param _compare  Function/macro to compare 2 elements
 * @param _hash     Function/macro to produce hash value of an element.
 *
 * types:
 * - ucs_hashed_##_type
 * - ucs_hashed_##_type##_alloc_cb_t
 * - ucs_hashed_##_type##_elem_cb_t
 *
 * functions:
 * - ucs_hashed_##_type##_init
 * - ucs_hashed_##_type##_add
 * - ucs_hashed_##_type##_add_if
 * - ucs_hashed_##_type##_remove
 * - ucs_hashed_##_type##_remove_if
 * - ucs_hashed_##_type##_find
 *
 */
#define UCS_DEFINE_THREAD_SAFE_HASH(_type, _next, _dim, _compare, _hash) \
    \
    /** \
     * Hash table type \
     */ \
    typedef struct ucs_hashed_##_type { \
        pthread_rwlock_t lock; \
        _type*           hash[_dim]; \
    } ucs_hashed_##_type; \
    \
    \
    /** \
     * Callback function for new element allocation. \
     * Called with hash table lock held. \
     * \
     * @param search   Element which was searched for and not found. \
     * @paran arg      User-defined argument. \
     * @paran elem     Should be filled with new allocated element. \
     */ \
    typedef ucs_status_t (*ucs_hashed_##_type##_alloc_cb_t)(_type *search, void *arg, \
                                                            _type **elem); \
    \
    /** \
     * Callback function for processing element. \
     * Called with hash table lock held. \
     * \
     * @paran elem     Element to process. \
     * @paran arg      User-defined argument. \
     */ \
    typedef ucs_status_t (*ucs_hashed_##_type##_elem_cb_t) (_type* elem, void *arg); \
    \
    \
    SGLIB_DEFINE_LIST_PROTOTYPES(_type, _compare, _next) \
    SGLIB_DEFINE_HASHED_CONTAINER_PROTOTYPES(_type, _dim, _hash) \
    \
    SGLIB_DEFINE_LIST_FUNCTIONS(_type, _compare, _next) \
    SGLIB_DEFINE_HASHED_CONTAINER_FUNCTIONS(_type, _dim, _hash) \
    \
    \
    /** \
     * Initializes the hash table. \
     */ \
    void ucs_hashed_##_type##_init(ucs_hashed_##_type *table) \
    { \
        pthread_rwlock_init(&table->lock, NULL); \
        sglib_hashed_##_type##_init(table->hash); \
    } \
    \
    \
    /** \
     * Add new element to hash table, fail if already exists. \
     * \
     * @param table    Hash table. \
     * @paran elem     Element to add. \
     */ \
    ucs_status_t ucs_hashed_##_type##_add(ucs_hashed_##_type *table, _type *elem) \
    { \
        ucs_status_t status; \
        _type *member; \
        \
        pthread_rwlock_wrlock(&table->lock); \
        if (!sglib_hashed_##_type##_add_if_not_member(table->hash, elem, &member)) { \
            status = UCS_ERR_ALREADY_EXISTS; \
        } else { \
            status = UCS_OK; \
        } \
        pthread_rwlock_unlock(&table->lock); \
        return status; \
    } \
    \
    \
    /** \
     * Allocate and add new element to the hash table. \
     * If not exists - allocate it using the allocation callback. \
     * If exists - pass it to user-define callback.
     * \
     * @param table     Hash table. \
     * @param search    Element to search for. \
     * @param alloc_cb  Called to allocate new element, if not exists. \
     * @param exists_cb Called to process the element, if already exists. May be NULL. \
     * @param arg       User-defined argument passed to callbacks. \
     * \
     * @return The return value of alloc_cb or exists_cb, whichever was used.
     */ \
    ucs_status_t ucs_hashed_##_type##_add_if(ucs_hashed_##_type *table, _type *search, \
                                             ucs_hashed_##_type##_alloc_cb_t alloc_cb, \
                                             ucs_hashed_##_type##_elem_cb_t  exists_cb, \
                                             void *arg) \
    { \
        ucs_status_t status; \
        _type *elem, *member; \
        int UCS_V_UNUSED ret; \
        \
        pthread_rwlock_wrlock(&table->lock); \
        elem = sglib_hashed_##_type##_find_member(table->hash, search); \
        if (elem != NULL) { \
            if (exists_cb == NULL) { \
                status = UCS_ERR_ALREADY_EXISTS; \
            } else { \
                status = exists_cb(elem, arg); \
            } \
        } else { \
            status = alloc_cb(search, arg, &elem); \
            if (status == UCS_OK) { \
                ret = sglib_hashed_##_type##_add_if_not_member(table->hash, elem, \
                                                               &member); \
                ucs_assert(ret && (member == NULL)); \
            } \
        } \
        pthread_rwlock_unlock(&table->lock); \
        return status; \
    } \
    \
    \
    /** \
     * Remove an element from the hash table. \
     * \
     * @param table     Hash table. \
     * @param search    Element to remove. \
     * @param elem      Filled with the removed element.
     * \
     * @return UCS_ERR_NO_ELEM if not exists, or UCS_OK if removed.
     */ \
    ucs_status_t ucs_hashed_##_type##_remove(ucs_hashed_##_type *table, \
                                             _type *search, _type **elem) \
    { \
        ucs_status_t status; \
        \
        pthread_rwlock_wrlock(&table->lock); \
        if (!sglib_hashed_##_type##_delete_if_member(table->hash, search, elem)) { \
            status = UCS_ERR_NO_ELEM; \
            ucs_assert(*elem == NULL); \
        } else { \
            status = UCS_OK; \
        } \
        pthread_rwlock_unlock(&table->lock); \
        return status; \
    } \
    \
    \
    /** \
     * Remove an element from the hash table if it passes user-defined test. \
     * \
     * @param table     Hash table. \
     * @param search    Element to remove. \
     * @param remove_cb Called to test whether an element should be removed It should. \
     *                  return UCS_OK in order to remove the element. \
     * @param arg       User-defined argument passed to callback. \
     * @param elem      Filled with the removed element.
     * \
     * @return UCS_ERR_NO_ELEM if not exists, or the return value from remove_cb.
     */ \
    ucs_status_t ucs_hashed_##_type##_remove_if(ucs_hashed_##_type *table, _type *search, \
                                                ucs_hashed_##_type##_elem_cb_t remove_cb, \
                                                void *arg, _type **elem) \
    { \
        ucs_status_t status; \
        \
        pthread_rwlock_wrlock(&table->lock); \
        *elem = sglib_hashed_##_type##_find_member(table->hash, search); \
        if (*elem == NULL) { \
            status = UCS_ERR_NO_ELEM; \
        } else { \
            status = remove_cb(*elem, arg); \
            if (status == UCS_OK) { \
                sglib_hashed_##_type##_delete(table->hash, *elem); \
            } else { \
                *elem = NULL; \
            } \
        } \
        pthread_rwlock_unlock(&table->lock); \
        return status; \
    } \
    \
    \
    /** \
     * Find an element in the hash table and call user-defined callback on it. \
     * \
     * @param table     Hash table. \
     * @param search    Element to find. \
     * @param find_cb   Called if the element is found.
     * @param arg       User-defined argument passed to callback. \
     * \
     * @return UCS_ERR_NO_ELEM if not exists, or the return value from find_cb.
     */ \
    ucs_status_t ucs_hashed_##_type##_find(ucs_hashed_##_type *table, _type *search, \
                                           ucs_hashed_##_type##_elem_cb_t find_cb, \
                                           void *arg) \
    { \
        ucs_status_t status; \
        _type *elem; \
        \
        pthread_rwlock_rdlock(&table->lock); \
        elem = sglib_hashed_##_type##_find_member(table->hash, search); \
        if (elem == NULL) { \
            status = UCS_ERR_NO_ELEM; \
        } else { \
            status = find_cb(elem, arg); \
        } \
        pthread_rwlock_unlock(&table->lock); \
        return status; \
    } \
    \
    \
    /** \
     * Check if a hash table is empty. \
     * \
     * @param table     Hash table. \
     * \
     * @return Whether the hash table is empty. \
     */ \
    int ucs_hashed_##_type##_is_empty(ucs_hashed_##_type *table) \
    { \
        pthread_rwlock_rdlock(&table->lock); \
        struct sglib_hashed_##_type##_iterator it; \
        return sglib_hashed_##_type##_it_init(&it, table->hash) == NULL; \
        pthread_rwlock_unlock(&table->lock); \
    } \
    \
    \
    /** \
     * Iterate over the hash table and invoke the callback for every element. \
     * \
     * @param table     Hash table. \
     * @param iter_cb   Function to call for every element. \
     * @param arg       User-defined argument passed to callback. \
     * \
     * @return UCS_ERR_NO_ELEM if not exists, or the return value from find_cb.
     */ \
    void ucs_hashed_##_type##_iter(ucs_hashed_##_type *table, \
                                   ucs_hashed_##_type##_elem_cb_t iter_cb, \
                                   void *arg) \
    { \
        struct sglib_hashed_##_type##_iterator it; \
        _type *elem; \
        \
        pthread_rwlock_rdlock(&table->lock); \
        for (elem = sglib_hashed_##_type##_it_init(&it, table->hash); \
             elem != NULL; \
             elem = sglib_hashed_##_type##_it_next(&it)) \
        { \
            iter_cb(elem, arg); \
        } \
        pthread_rwlock_unlock(&table->lock); \
    }

#endif
