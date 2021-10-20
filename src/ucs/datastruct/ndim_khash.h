/**
 * See file LICENSE for terms.
 */

#ifndef UCS_NDIM_HASH_H_
#define UCS_NDIM_HASH_H_

#include <ucs/datastruct/khash.h>
#include <ucs/datastruct/mpool.h>

#include <ucs/sys/math.h>

#define NDIM_HASH_MPOOL_GROWTH_FACTOR (2.0)

#define __NDIM_HASH_TYPE(name, ndim, khkey_t, khval_t) \
    typedef struct ndh_##name##_node_s { \
        khkey_t keys[ndim]; \
        khval_t value; \
    } ndnode_t(name); \
    \
    __KHASH_TYPE(name, khkey_t, ndnode_t(name)*) \
    __KHASH_TYPE(name##_reverse, khval_t, ndnode_t(name)*) \
    \
    typedef struct ndh_##name##_s { \
        khash_t(name)           by_dim[ndim]; \
        ucs_mpool_t             node_mp; \
        khash_t(name##_reverse) reverse; \
    } ndhash_t(name);

#define __NDIM_HASH_PROTOTYPES(name, ndim, khkey_t, khval_t) \
    __KHASH_PROTOTYPES(name, khkey_t, ndnode_t(name)*) \
    __KHASH_PROTOTYPES(name##_reverse, khval_t, ndnode_t(name)*) \
    extern ndhash_t(name) * ndh_init_##name(void); \
    extern ndhash_t(name) * ndh_init_##name##_inplace(ndhash_t(name) * h); \
    extern void ndh_destroy_##name(ndhash_t(name) * h); \
    extern void ndh_destroy_##name##_inplace(ndhash_t(name) * h); \
    extern void ndh_clear_##name(ndhash_t(name) * h); \
    extern int ndh_resize_##name(ndhash_t(name) * h, khint_t new_size); \
    extern int ndh_insert_##name(ndhash_t(name) * h, const khkey_t key[ndim], \
                                 khval_t value); \
    extern int ndh_anyv_##name(ndhash_t(name) * h, unsigned cnt, \
                               ndnode_t(name) * *res); \
    extern int ndh_delv_##name(ndhash_t(name) * h, khval_t value);

#define __NDIM_HASH_IMPL(name, ndim, SCOPE, khkey_t, khval_t, __khash_func, \
                         __khash_equal, __vhash_func, __vhash_equal) \
    __KHASH_IMPL(name, SCOPE, khkey_t, ndnode_t(name)*, 1, 1, __khash_func, \
                 __khash_equal) \
    __KHASH_IMPL(name##_reverse, SCOPE, khval_t, ndnode_t(name)*, 1, 1, \
                 __vhash_func, __vhash_equal) \
    SCOPE ndhash_t(name) * ndh_init_##name(void) \
    { \
        ndhash_t(name) *ndh = (ndhash_t( \
                name)*)kcalloc(1, sizeof(ndhash_t(name))); \
        if ((ndh != NULL) && \
            (ndh_init_mpool(&ndh->node_mp, #name, sizeof(ndnode_t(name))))) { \
            kfree(ndh); \
            ndh = NULL; \
        } \
        return ndh; \
    } \
    SCOPE ndhash_t(name) * ndh_init_##name##_inplace(ndhash_t(name) * h) \
    { \
        return ((!ndh_init_mpool(&h->node_mp, #name, \
                                 sizeof(ndnode_t(name))) && \
                 (kmemset(h->by_dim, 0, ndim * sizeof(khash_t(name)))))) ? \
                       h : \
                       NULL; \
    } \
    SCOPE void ndh_destroy_##name##_inplace(ndhash_t(name) * h) \
    { \
        unsigned idx; \
        for (idx = 0; idx < ndim; idx++) { \
            kh_destroy_inplace(name, &h->by_dim[idx]); \
        } \
        ucs_mpool_cleanup(&h->node_mp, 0); \
    } \
    SCOPE void ndh_destroy_##name(ndhash_t(name) * h) \
    { \
        ndh_destroy_inplace(name, h); \
        kfree(h); \
    } \
    SCOPE void ndh_clear_##name(ndhash_t(name) * h) \
    { \
        int i; \
        kh_clear(name##_reverse, &h->reverse); \
        for (i = 0; i < ndim; i++) { \
            kh_clear(name, &h->by_dim[i]); \
        } \
    } \
    SCOPE int ndh_resize_##name(ndhash_t(name) * h, khint_t new_size) \
    { \
        int i, ret = kh_resize(name##_reverse, &h->reverse, new_size); \
        for (i = 0; ((ret == 0) && (i < ndim)); i++) { \
            ret = kh_resize(name, &h->by_dim[i], new_size); \
        } \
        return ret; \
    } \
    SCOPE int ndh_insert_##name##_node(ndhash_t(name) * h, \
                                       ndnode_t(name) * node) \
    { \
        int ret; \
        unsigned i; \
        khiter_t iter; \
        khkey_t key_iter; \
        iter = kh_put(name##_reverse, &h->reverse, node->value, &ret); \
        if (ret == UCS_KH_PUT_FAILED) \
            return -1; \
        kh_value(&h->reverse, iter) = node; \
        for (i = 0; i < ndim; i++) { \
            key_iter = node->keys[i]; \
            iter     = kh_put(name, &h->by_dim[i], key_iter, &ret); \
            if (ret == UCS_KH_PUT_FAILED) \
                return -1; \
            kh_value(&h->by_dim[i], iter) = node; \
        } \
        return 0; \
    } \
    SCOPE int ndh_insert_##name(ndhash_t(name) * h, khkey_t key[ndim], \
                                khval_t value) \
    { \
        ndnode_t(name) *node = (ndnode_t(name)*)ucs_mpool_get( \
               &h->node_mp); \
        if (node == NULL) { \
            return -1; \
        } \
        node->value = value; \
        memcpy(node->keys, key, sizeof(node->keys)); \
        return ndh_insert_##name##_node(h, node); \
    } \
    SCOPE int ndh_anyv_##name(ndhash_t(name) * h, unsigned cnt, \
                              ndnode_t(name) * *res) \
    { \
        return kh_anyv(name##_reverse, &h->reverse, cnt, res); \
    } \
    SCOPE int ndh_delv_##name(ndhash_t(name) * h, khval_t value) \
    { \
        unsigned i; \
        ndnode_t(name) * node; \
        khiter_t iter = kh_get(name##_reverse, &h->reverse, value); \
        if (iter == kh_end(&h->reverse)) { \
            return -1; \
        } \
        node = kh_value(&h->reverse, iter); \
        if (!node) { \
            return -1; \
        } \
        for (i = 0; i < ndim; i++) { \
            iter = kh_getv(name, &h->by_dim[i], node->keys[i], &node); \
            if (iter == kh_end(&h->reverse)) { \
                return -1; \
            } \
            kh_del(name, &h->by_dim[i], iter); \
        } \
        ucs_mpool_put(node); \
        return 0; \
    }

static ucs_mpool_ops_t ndh_mpool_ops = {ucs_mpool_chunk_malloc,
                                        ucs_mpool_chunk_free, NULL, NULL, NULL};

static inline int
ndh_init_mpool(ucs_mpool_t *mp, const char *name, size_t elem_size)
{
    ucs_mpool_params_t mp_params;

    ucs_mpool_params_reset(&mp_params);

    mp_params.ops         = &ndh_mpool_ops;
    mp_params.name        = name;
    mp_params.elem_size   = elem_size;
    mp_params.grow_factor = NDIM_HASH_MPOOL_GROWTH_FACTOR;

    return UCS_OK != ucs_mpool_init(&mp_params, mp);
}

#define NDIM_HASH_DECLARE(name, ndim, khkey_t, khval_t) \
    __NDIM_HASH_TYPE(name, ndim, khkey_t, khval_t) \
    __NDIM_HASH_PROTOTYPES(name, ndim, khkey_t, khval_t)

#define NDIM_HASH_INIT2(name, ndim, SCOPE, khkey_t, khval_t, __khash_func, \
                        __khash_equal, __vhash_func, __vhash_equal) \
    __NDIM_HASH_TYPE(name, ndim, khkey_t, khval_t) \
    __NDIM_HASH_IMPL(name, ndim, SCOPE, khkey_t, khval_t, __khash_func, \
                     __khash_equal, __vhash_func, __vhash_equal)

#define NDIM_HASH_INIT(name, ndim, khkey_t, khval_t, __khash_func, \
                       __khash_equal, __vhash_func, __vhash_equal) \
    NDIM_HASH_INIT2(name, ndim, static kh_inline klib_unused, khkey_t, \
                    khval_t, __khash_func, __khash_equal, __vhash_func, \
                    __vhash_equal)

#define NDIM_HASH_TYPE(name, ndim, khkey_t, khval_t) \
    __NDIM_HASH_TYPE(name, ndim, khkey_t, khval_t)

#define NDIM_HASH_IMPL(name, ndim, khkey_t, khval_t, __khash_func, \
                       __khash_equal, __vhash_func, __vhash_equal) \
    __NDIM_HASH_IMPL(name, ndim, static ndh_inline klib_unused, khkey_t, \
                     khval_t, __khash_func, __khash_equal, __vhash_func, \
                     __vhash_equal)

/* Other convenient macros... */

/*!
  @abstract Type of the n-dimensional hash table.
  @param  name  Name of the n-dimensional hash table [symbol]
 */
#define ndhash_t(name) ndh_##name##_t

/*!
  @abstract Type of the n-dimensional hash table node.
  @param  name  Name of the n-dimensional hash table [symbol]
 */
#define ndnode_t(name) ndh_##name##_node_t

/*! @function
  @abstract     Initiate an n-dimensional hash table.
  @param  name  Name of the n-dimensional hash table [symbol]
  @return       Pointer to the n-dimensional hash table [ndhash_t(name)*]
 */
#define ndh_init(name) ndh_init_##name()

/*! @function
  @abstract     Initiate an n-dimensional hash table in the in-place case.
  @param  name  Name of the n-dimensional hash table [symbol]
  @param  h     Pointer to the n-dimensional hash table [ndhash_t(name)*]
 */
#define ndh_init_inplace(name, h) ndh_init_##name##_inplace(h)

/*! @function
  @abstract     Destroy an n-dimensional hash table.
  @param  name  Name of the n-dimensional hash table [symbol]
  @param  h     Pointer to the n-dimensional hash table [ndhash_t(name)*]
 */
#define ndh_destroy(name, h) ndh_destroy_##name(h)

/*! @function
  @abstract     Destroy an n-dimensional hash table in the in-place case.
  @param  name  Name of the n-dimensional hash table [symbol]
  @param  h     Pointer to the n-dimensional hash table [ndhash_t(name)*]
 */
#define ndh_destroy_inplace(name, h) ndh_destroy_##name##_inplace(h)

/*! @function
  @abstract     Reset the n-dimensional hash table without deallocating memory.
  @param  name  Name of the n-dimensional hash table [symbol]
  @param  h     Pointer to the hash table [ndhash_t(name)*]
 */
#define ndh_clear(name, h) ndh_clear_##name(h)

/*! @function
  @abstract     Resize the n-dimensional hash table.
  @param  name  Name of the n-dimensional hash table [symbol]
  @param  h     Pointer to the hash table [ndhash_t(name)*]
  @param  s     New size [khint_t]
 */
#define ndh_resize(name, h, s) ndh_resize_##name(h, s)

/*! @function
  @abstract     Insert a key-value pair to the n-dimensional hash table.
  @param  name  Name of the n-dimensional hash table [symbol]
  @param  h     Pointer to the n-dimensional hash table [ndhash_t(name)*]
  @param  k     Key vector [vector of type of keys]
  @param  v     Value [type of values]
  @return       Status, non-zero for error [integer]
 */
#define ndh_insert(name, h, k, v) ndh_insert_##name(h, k, v)

/*! @function
  @abstract     Remove a key from the n-dimensional hash table.
  @param  name  Name of the n-dimensional hash table [symbol]
  @param  h     Pointer to the n-dimensional hash table [ndhash_t(name)*]
  @param  k     Number of nodes requested [number]
  @param  res   An array of k hash table nodes to write to [ndnode_t(name)**]
  @return       Status, non-zero for error [integer]
 */
#define ndh_anyv(name, h, k, res) ndh_anyv_##name(h, k, res)

/*! @function
  @abstract     Remove a key from the n-dimensional hash table.
  @param  name  Name of the n-dimensional hash table [symbol]
  @param  h     Pointer to the n-dimensional hash table [ndhash_t(name)*]
  @param  v     Value to be located and deleted [type of values]
  @return       Status, non-zero for error [integer]
 */
#define ndh_delv(name, h, v) ndh_delv_##name(h, v)

/* More conenient interfaces */

/*! @function
  @abstract     Instantiate an n-dimensional hash map containing integer keys
  @param  name  Name of the n-dimensional hash table [symbol]
  @param  ndim  Number of dimensions [number]
 */
#define NDIM_HASH_INIT_INT(name, ndim) \
    NDIM_HASH_INIT(name, ndim, khint32_t, uintptr_t, kh_int_hash_func, \
                   kh_int_hash_equal, kh_int64_hash_func, kh_int64_hash_equal)

/*! @function
  @abstract     Instantiate an n-dimensional map with 64-bit integer keys and coefficients
  @param  name  Name of the n-dimensional hash table [symbol]
  @param  ndim  Number of dimensions [number]
 */
#define NDIM_HASH_INIT_INT64(name, ndim) \
    NDIM_HASH_INIT(name, ndim, khint64_t, uintptr_t, kh_int_hash_func, \
                   kh_int_hash_equal, kh_int64_hash_func, kh_int64_hash_equal)

/*! @function
  @abstract     Instantiate an n-dimensional hash map containing string keys
  @param  name  Name of the n-dimensional hash table [symbol]
  @param  ndim  Number of dimensions [number]
 */
#define NDIM_HASH_INIT_STR(name, ndim) \
    NDIM_HASH_INIT(name, ndim, kh_cstr_t, uintptr_t, kh_str_hash_func, \
                   kh_str_hash_equal, kh_int64_hash_func, kh_int64_hash_equal)

#endif
