/**
 * See file LICENSE for terms.
 */

#ifndef UCS_NDIM_KNN_H_
#define UCS_NDIM_KNN_H_

#include "ndim_khash.h"

/**
 * This file contains 3 variants for the N-dimensional K-nearest neighbor search
 * based on the Hamming distance metric over non-ordered discrete data spaces.
 * This metric is uniquely suitable for caching mechanisms which can re-use
 * objects with partial match to the lookup vector - and prefer the best
 * available match.
 *
 * 1. Exact K nearest neighbors (prefix: KNN)
 * 2. C-approximate K-nearest neighbors (prefix: ANN)
 * 3. C-approximate K-nearest neighbors, Weighted Hamming distance (prefix: WNN)
 */

#define __NDIM_KNN_TYPE(name, ndim, khkey_t, khval_t, sum_t) \
    __NDIM_HASH_TYPE(name, ndim, khkey_t, khval_t) \
    \
    typedef struct nn_##name##_node_s { \
        ndnode_t(name) super; \
        unsigned query_idx; \
        sum_t    query_sum; \
        unsigned query_pos; \
    } nn_node_t(name); \
    \
    typedef struct knn_##name##_s { \
        ndhash_t(name) super; \
        unsigned query_idx; \
    } knn_t(name);

#define __NDIM_ANN_TYPE(name, ndim, khkey_t, khval_t, coeff_t) \
    __NDIM_KNN_TYPE(name, ndim, khkey_t, khval_t, coeff_t) \
    \
    typedef struct ann_##name##_s { \
        knn_t(name) super; \
        coeff_t approximation; \
    } ann_t(name);

#define __NDIM_WNN_TYPE(name, ndim, khkey_t, khval_t, coeff_t) \
    __NDIM_ANN_TYPE(name, ndim, khkey_t, khval_t, coeff_t) \
    \
    typedef struct wnn_##name##_s { \
        ann_t(name) super; \
        unsigned order[ndim]; \
        coeff_t  coefficients[ndim]; \
        coeff_t  coefficient_sums[ndim]; \
    } wnn_t(name);

#define __NDIM_KNN_PROTOTYPES(name, K, ndim, khkey_t, khval_t) \
    __NDIM_HASH_PROTOTYPES(name, ndim, khkey_t, khval_t) \
    extern knn_t(name) * knn_init_##name##_inplace(knn_t(name) * knn); \
    extern knn_t(name) * knn_init_##name(void); \
    extern void knn_destroy_##name##_inplace(knn_t(name) * knn); \
    extern void knn_destroy_##name(knn_t(name) * knn); \
    extern void knn_clear_##name(knn_t(name) * knn); \
    extern int knn_resize_##name(knn_t(name) * knn, khint_t new_size); \
    extern int knn_insert_##name(knn_t(name) * knn, const khkey_t key[ndim], \
                                 khval_t value); \
    extern int knn_delv_##name(knn_t(name) * knn, khval_t value); \
    extern int knn_search_##name(knn_t(name) * knn, const khkey_t key[ndim], \
                                 khval_t result[K]);

#define __NDIM_ANN_PROTOTYPES(name, K, ndim, khkey_t, khval_t, coeff_t) \
    __NDIM_KNN_PROTOTYPES(name, K, ndim, khkey_t, khval_t) \
    extern ann_t(name) * ann_init_##name##_inplace(ann_t(name) * ann, \
                                                   coeff_t approximation); \
    extern ann_t(name) * ann_init_##name(coeff_t approximation); \
    extern void ann_destroy_##name##_inplace(ann_t(name) * ann); \
    extern void ann_destroy_##name(ann_t(name) * ann); \
    extern void ann_clear_##name(ann_t(name) * ann); \
    extern int ann_resize_##name(ann_t(name) * ann, khint_t new_size); \
    extern int ann_insert_##name(ann_t(name) * ann, const khkey_t key[ndim], \
                                 khval_t value); \
    extern int ann_delv_##name(ann_t(name) * ann, khval_t value); \
    extern int ann_search_##name(ann_t(name) * ann, const khkey_t key[ndim], \
                                 khval_t result[K]);

#define __NDIM_WNN_PROTOTYPES(name, K, ndim, khkey_t, khval_t, coeff_t) \
    __NDIM_ANN_PROTOTYPES(name, K, ndim, khkey_t, khval_t, coeff_t) \
    extern wnn_t(name) * \
            wnn_init_##name##_inplace(wnn_t(name) * wnn, \
                                      coeff_t approximation, \
                                      const coeff_t coefficients[ndim]); \
    extern wnn_t(name) * wnn_init_##name(coeff_t approximation, \
                                         const coeff_t coefficients[ndim]); \
    extern void wnn_destroy_##name##_inplace(wnn_t(name) * wnn); \
    extern void wnn_destroy_##name(wnn_t(name) * wnn); \
    extern void wnn_clear_##name(wnn_t(name) * wnn); \
    extern int wnn_resize_##name(wnn_t(name) * wnn, khint_t new_size); \
    extern int wnn_insert_##name(wnn_t(name) * wnn, const khkey_t key[ndim], \
                                 khval_t value); \
    extern int wnn_delv_##name(wnn_t(name) * wnn, khval_t value); \
    extern int wnn_search_##name(wnn_t(name) * wnn, const khkey_t key[ndim], \
                                 khval_t result[K]);

#define __NDIM_KNN_IMPL(name, K, ndim, SCOPE, khkey_t, khval_t, sum_t, \
                        __khash_func, __khash_equal, __vhash_func, \
                        __vhash_equal) \
    __NDIM_HASH_IMPL(name, ndim, SCOPE, khkey_t, khval_t, __khash_func, \
                     __khash_equal, __vhash_func, __vhash_equal) \
    typedef int (*knn_update_##name)(knn_t(name) * knn, unsigned dim, \
                                     nn_node_t(name) * node, \
                                     nn_node_t(name) * nearest[K + 1]); \
    SCOPE int knn_init_common_##name(knn_t(name) * knn) \
    { \
        size_t i; \
        if (ndh_init_mpool(&knn->super.node_mp, #name, \
                           sizeof(nn_node_t(name)))) { \
            return -1; \
        } \
        \
        for (i = 0; i < ndim; i++) { \
            kh_init_inplace(name, &knn->super.by_dim[i]); \
        } \
        \
        kh_init_inplace(name##_reverse, &knn->super.reverse); \
        knn->query_idx = 0; \
        return 0; \
    } \
    SCOPE knn_t(name) * knn_init_##name##_inplace(knn_t(name) * knn) \
    { \
        return knn_init_common_##name(knn) ? knn : NULL; \
    } \
    SCOPE knn_t(name) * knn_init_##name(void) \
    { \
        knn_t(name) *knn = (knn_t(name)*)kcalloc(1, sizeof(knn_t(name))); \
        if ((knn != NULL) && (knn_init_common_##name(knn))) { \
            kfree(knn); \
            knn = NULL; \
        } \
        return knn; \
    } \
    SCOPE void knn_destroy_##name##_inplace(knn_t(name) * knn) \
    { \
        ndh_destroy_inplace(name, &knn->super); \
    } \
    SCOPE void knn_destroy_##name(knn_t(name) * knn) \
    { \
        ndh_destroy(name, &knn->super); \
    } \
    SCOPE void knn_clear_##name(knn_t(name) * knn) \
    { \
        ndh_clear(name, &knn->super); \
    } \
    SCOPE int knn_resize_##name(knn_t(name) * knn, khint_t new_size) \
    { \
        return ndh_resize(name, &knn->super, new_size); \
    } \
    SCOPE int knn_insert##name##_internal(knn_t(name) * knn, \
                                          const unsigned order[ndim], \
                                          const khkey_t key[ndim], \
                                          khval_t value) \
    { \
        unsigned i; \
        nn_node_t(name) *node = (nn_node_t(name)*)ucs_mpool_get( \
                &knn->super.node_mp); \
        for (i = 0; i < ndim; i++) { \
            node->super.keys[i] = key[order ? order[i] : i]; \
        } \
        node->super.value = value; \
        node->query_idx   = 0; \
        node->query_sum   = 0; \
        node->query_pos   = 0; \
        return ndh_insert_##name##_node(&knn->super, &node->super); \
    } \
    SCOPE int knn_insert_##name(knn_t(name) * knn, const khkey_t key[ndim], \
                                khval_t value) \
    { \
        return knn_insert##name##_internal(knn, NULL, key, value); \
    } \
    SCOPE int knn_delv_##name(knn_t(name) * knn, khval_t value) \
    { \
        return ndh_delv(name, &knn->super, value); \
    } \
    SCOPE int knn_update_node_##name##_internal( \
            knn_t(name) * knn, const sum_t inc[ndim], unsigned dim, \
            nn_node_t(name) * node, nn_node_t(name) * nearest[K + 1]) \
    { \
        unsigned i; \
        sum_t sum = node->query_sum = node->query_sum + (inc ? inc[dim] : 1); \
        \
        if (K == 1) { \
            if (ucs_unlikely(nearest[1]->query_sum < sum)) { \
                nearest[1] = node; \
            } \
            return 0; \
        } \
        \
        if (ucs_unlikely(sum > nearest[0]->query_sum)) { \
            for (i = node->query_pos; (K > 1) && (i < (K - 1)); i++) { \
                if (nearest[i + 1]->query_sum > sum) { \
                    nearest[i]      = node; \
                    node->query_pos = i; \
                    return 0; \
                } \
                nearest[i] = nearest[i + 1]; \
            } \
            nearest[i] = node; \
        } \
        return 0; \
    } \
    static int knn_update_node_##name(knn_t(name) * knn, unsigned dim, \
                                      nn_node_t(name) * node, \
                                      nn_node_t(name) * nearest[K + 1]) \
    { \
        return knn_update_node_##name##_internal(knn, NULL, dim, node, \
                                                 nearest); \
    } \
    SCOPE int knn_search_##name##_internal(knn_t(name) * knn, \
                                           const unsigned order[ndim], \
                                           knn_update_##name update_f, \
                                           const khkey_t key[ndim], \
                                           khval_t result[K]) \
    { \
        khkey_t dk; \
        khash_t(name) *h; \
        khint_t first, iter, step, mask; \
        nn_node_t(name) * node, *volatile nearest[K + 1] = {0}; \
        unsigned i, query_idx                            = knn->query_idx++; \
        \
        if (ndh_anyv(name, &knn->super, K + 1, (ndnode_t(name)**)nearest)) { \
            return -1; \
        } \
        for (i = 0; i < (K + 1); i++) { \
            if ((node = nearest[i]) == NULL) { \
                return -1; \
            } \
            node->query_idx = query_idx; \
            node->query_sum = 0; \
            node->query_pos = i + 1; \
        } \
        for (i = 0; i < ndim; i++) { \
            step  = 0; \
            dk    = key[order ? order[i] : i]; \
            h     = &knn->super.by_dim[i]; \
            mask  = h->n_buckets - 1; \
            first = iter = kh_get_first_##name(h, dk, &step); \
            while (ucs_likely(iter != kh_end(h))) { \
                node = ucs_derived_of(kh_value(h, iter), nn_node_t(name)); \
                if (node->query_idx != query_idx) { \
                    node->query_idx = query_idx; \
                    node->query_sum = 0; \
                    node->query_pos = 0; \
                } \
                if (update_f(knn, i, (nn_node_t(name)*)node, \
                             (nn_node_t(name)**)nearest)) { \
                    i = ndim; \
                    break; \
                } \
                if ((iter = ((iter + ++step) & mask)) == first) \
                    break; \
                iter = kh_get_next_##name(h, dk, first, iter, &step, 0, NULL); \
            } \
        } \
        \
        for (i = 0; i < K; i++) { \
            result[i] = nearest[K - i]->super.value; \
        } \
        return 0; \
    } \
    SCOPE int knn_search_##name(knn_t(name) * knn, const khkey_t key[ndim], \
                                khval_t result[K]) \
    { \
        return knn_search_##name##_internal(knn, NULL, knn_update_node_##name, \
                                            key, result); \
    }

#define __NDIM_ANN_IMPL(name, K, ndim, SCOPE, khkey_t, khval_t, coeff_t, \
                        __khash_func, __khash_equal, __vhash_func, \
                        __vhash_equal) \
    __NDIM_KNN_IMPL(name, K, ndim, SCOPE, khkey_t, khval_t, coeff_t, \
                    __khash_func, __khash_equal, __vhash_func, __vhash_equal) \
    SCOPE int ann_init_common_##name(ann_t(name) * ann, coeff_t approximation) \
    { \
        if (knn_init_common_##name(&ann->super)) { \
            return -1; \
        } \
        \
        ann->approximation = approximation; \
        return 0; \
    } \
    SCOPE ann_t(name) * ann_init_##name##_inplace(ann_t(name) * ann, \
                                                  coeff_t approximation) \
    { \
        return ann_init_common_##name(ann, approximation) ? ann : NULL; \
    } \
    SCOPE ann_t(name) * ann_init_##name(coeff_t approximation) \
    { \
        ann_t(name) *ann = (ann_t(name)*)kcalloc(1, sizeof(ann_t(name))); \
        if ((ann != NULL) && (ann_init_common_##name(ann, approximation))) { \
            kfree(ann); \
            ann = NULL; \
        } \
        return ann; \
    } \
    SCOPE void ann_destroy_##name##_inplace(ann_t(name) * ann) \
    { \
        nn_destroy_inplace(k, name, &ann->super); \
    } \
    SCOPE void ann_destroy_##name(ann_t(name) * ann) \
    { \
        nn_destroy(k, name, &ann->super); \
    } \
    SCOPE void ann_clear_##name(ann_t(name) * ann) \
    { \
        nn_clear(k, name, &ann->super); \
    } \
    SCOPE int ann_resize_##name(ann_t(name) * ann, khint_t new_size) \
    { \
        return nn_resize(k, name, &ann->super, new_size); \
    } \
    SCOPE int ann_insert_##name##_internal(ann_t(name) * ann, \
                                           const unsigned order[ndim], \
                                           const khkey_t key[ndim], \
                                           khval_t value) \
    { \
        return knn_insert##name##_internal(&ann->super, order, key, value); \
    } \
    SCOPE int ann_insert_##name(ann_t(name) * ann, const khkey_t key[ndim], \
                                khval_t value) \
    { \
        return nn_insert(k, name, &ann->super, key, value); \
    } \
    SCOPE int ann_delv_##name(ann_t(name) * ann, khval_t value) \
    { \
        return nn_delv(k, name, &ann->super, value); \
    } \
    SCOPE int ann_update_node_##name##_internal( \
            knn_t(name) * knn, const coeff_t inc[ndim], unsigned dim, \
            nn_node_t(name) * node, nn_node_t(name) * nearest[K + 1]) \
    { \
        ann_t(name) *ann = ucs_derived_of(knn, ann_t(name)); \
        knn_update_node_##name##_internal(knn, inc, dim, node, nearest); \
        if (ucs_unlikely(node->query_pos > 0) && \
            ((nearest[0]->query_sum + K - dim) < \
             (nearest[1]->query_sum * ann->approximation))) { \
            return 1; \
        } \
        return 0; \
    } \
    static int ann_update_node_##name(knn_t(name) * knn, unsigned dim, \
                                      nn_node_t(name) * node, \
                                      nn_node_t(name) * nearest[K + 1]) \
    { \
        return ann_update_node_##name##_internal(knn, NULL, dim, node, \
                                                 nearest); \
    } \
    SCOPE int ann_search_##name##_internal(ann_t(name) * ann, \
                                           const unsigned order[ndim], \
                                           knn_update_##name update_f, \
                                           const khkey_t key[ndim], \
                                           khval_t result[K]) \
    { \
        return knn_search_##name##_internal(&ann->super, order, update_f, key, \
                                            result); \
    } \
    SCOPE int ann_search_##name(ann_t(name) * ann, const khkey_t key[ndim], \
                                khval_t result[K]) \
    { \
        return ann_search_##name##_internal(ann, NULL, ann_update_node_##name, \
                                            key, result); \
    }

#define __NDIM_WNN_IMPL(name, K, ndim, SCOPE, khkey_t, khval_t, coeff_t, \
                        __khash_func, __khash_equal, __vhash_func, \
                        __vhash_equal) \
    __NDIM_ANN_IMPL(name, K, ndim, SCOPE, khkey_t, khval_t, coeff_t, \
                    __khash_func, __khash_equal, __vhash_func, __vhash_equal) \
    SCOPE int wnn_init_common_##name(wnn_t(name) * wnn, coeff_t approximation, \
                                     const coeff_t coefficients[ndim]) \
    { \
        coeff_t next, sum = 0; \
        unsigned i, j, k, min; \
        if (ann_init_common_##name(&wnn->super, approximation)) { \
            return -1; \
        } \
        \
        for (i = 0; i < ndim; i++) { \
            for (min = ndim, j = 0; j < ndim; j++) { \
                for (k = 0; ((k < i) && (wnn->order[ndim - k - 1] != j)); k++) \
                    ; \
                if ((k == i) /* j isn't among last i enties of order[] */ && \
                    ((min == ndim) || \
                     (coefficients[j] < coefficients[min]))) { \
                    min = j; \
                } \
            } \
            if (min == ndim) { \
                return -1; \
            } \
            next                                = coefficients[min]; \
            wnn->coefficients[ndim - i - 1]     = next; \
            sum                                += next; \
            wnn->coefficient_sums[ndim - i - 1] = sum; \
            wnn->order[ndim - i - 1]            = min; \
        } \
        return 0; \
    } \
    SCOPE wnn_t(name) * wnn_init_##name##_inplace(wnn_t(name) * wnn, \
                                                  coeff_t approximation, \
                                                  const coeff_t coeff[ndim]) \
    { \
        return wnn_init_common_##name(wnn, approximation, coeff) ? wnn : NULL; \
    } \
    SCOPE wnn_t(name) * \
            wnn_init_##name(coeff_t approximation, const coeff_t coeff[ndim]) \
    { \
        wnn_t(name) *wnn = (wnn_t(name)*)kcalloc(1, sizeof(wnn_t(name))); \
        if ((wnn != NULL) && \
            (wnn_init_common_##name(wnn, approximation, coeff))) { \
            kfree(wnn); \
            wnn = NULL; \
        } \
        return wnn; \
    } \
    SCOPE void wnn_destroy_##name##_inplace(wnn_t(name) * wnn) \
    { \
        nn_destroy_inplace(a, name, &wnn->super); \
    } \
    SCOPE void wnn_destroy_##name(wnn_t(name) * wnn) \
    { \
        nn_destroy(a, name, &wnn->super); \
    } \
    SCOPE void wnn_clear_##name(wnn_t(name) * wnn) \
    { \
        nn_clear(a, name, &wnn->super); \
    } \
    SCOPE int wnn_resize_##name(wnn_t(name) * wnn, khint_t new_size) \
    { \
        return nn_resize(a, name, &wnn->super, new_size); \
    } \
    SCOPE int wnn_insert_##name(wnn_t(name) * wnn, const khkey_t key[ndim], \
                                khval_t value) \
    { \
        return ann_insert_##name##_internal(&wnn->super, wnn->order, key, \
                                            value); \
    } \
    SCOPE int wnn_delv_##name(wnn_t(name) * wnn, khval_t value) \
    { \
        return nn_delv(a, name, &wnn->super, value); \
    } \
    static int wnn_update_node_##name(knn_t(name) * knn, unsigned dim, \
                                      nn_node_t(name) * node, \
                                      nn_node_t(name) * nearest[K + 1]) \
    { \
        wnn_t(name) *wnn = ucs_derived_of(knn, wnn_t(name)); \
        ann_update_node_##name##_internal(knn, wnn->coefficients, dim, node, \
                                          nearest); \
        if (ucs_unlikely(node->query_pos > 0) && \
            ((nearest[0]->query_sum + wnn->coefficient_sums[dim]) < \
             (nearest[1]->query_sum * wnn->super.approximation))) { \
            return 1; \
        } \
        return 0; \
    } \
    SCOPE int wnn_search_##name(wnn_t(name) * wnn, const khkey_t key[ndim], \
                                khval_t result[K]) \
    { \
        return ann_search_##name##_internal(&wnn->super, wnn->order, \
                                            wnn_update_node_##name, key, \
                                            result); \
    }

#define NDIM_KNN_DECLARE(name, K, ndim, khkey_t, khval_t) \
    __NDIM_KNN_TYPE(name, ndim, khkey_t, khval_t, unsigned) \
    __NDIM_KNN_PROTOTYPES(name, K, ndim, khkey_t, khval_t)

#define NDIM_ANN_DECLARE(name, K, ndim, khkey_t, khval_t, coeff_t) \
    __NDIM_ANN_TYPE(name, ndim, khkey_t, khval_t, coeff_t) \
    __NDIM_ANN_PROTOTYPES(name, K, ndim, khkey_t, khval_t, coeff_t)

#define NDIM_WNN_DECLARE(name, K, ndim, khkey_t, khval_t, coeff_t) \
    __NDIM_WNN_TYPE(name, ndim, khkey_t, khval_t, coeff_t) \
    __NDIM_WNN_PROTOTYPES(name, K, ndim, khkey_t, khval_t, coeff_t)

#define NDIM_KNN_INIT2(name, K, ndim, SCOPE, khkey_t, khval_t, _ign, \
                       __khash_func, __khash_equal, __vhash_func, \
                       __vhash_equal) \
    __NDIM_KNN_TYPE(name, ndim, khkey_t, khval_t, unsigned) \
    __NDIM_KNN_IMPL(name, K, ndim, SCOPE, khkey_t, khval_t, unsigned, \
                    __khash_func, __khash_equal, __vhash_func, __vhash_equal)

#define NDIM_ANN_INIT2(name, K, ndim, SCOPE, khkey_t, khval_t, coeff_t, \
                       __khash_func, __khash_equal, __vhash_func, \
                       __vhash_equal) \
    __NDIM_ANN_TYPE(name, ndim, khkey_t, khval_t, coeff_t) \
    __NDIM_ANN_IMPL(name, K, ndim, SCOPE, khkey_t, khval_t, coeff_t, \
                    __khash_func, __khash_equal, __vhash_func, __vhash_equal)

#define NDIM_WNN_INIT2(name, K, ndim, SCOPE, khkey_t, khval_t, coeff_t, \
                       __khash_func, __khash_equal, __vhash_func, \
                       __vhash_equal) \
    __NDIM_WNN_TYPE(name, ndim, khkey_t, khval_t, coeff_t) \
    __NDIM_WNN_IMPL(name, K, ndim, SCOPE, khkey_t, khval_t, coeff_t, \
                    __khash_func, __khash_equal, __vhash_func, __vhash_equal)


#define NDIM_KNN_INIT(name, K, ndim, khkey_t, khval_t, __khash_func, \
                      __khash_equal, __vhash_func, __vhash_equal) \
    NDIM_KNN_INIT2(name, K, ndim, static kh_inline klib_unused, khkey_t, \
                   khval_t, k, __khash_func, __khash_equal, __vhash_func, \
                   __vhash_equal)

#define NDIM_ANN_INIT(name, K, ndim, khkey_t, khval_t, coeff_t, __khash_func, \
                      __khash_equal, __vhash_func, __vhash_equal) \
    NDIM_ANN_INIT2(name, K, ndim, static kh_inline klib_unused, khkey_t, \
                   khval_t, coeff_t, __khash_func, __khash_equal, \
                   __vhash_func, __vhash_equal)

#define NDIM_WNN_INIT(name, K, ndim, khkey_t, khval_t, coeff_t, __khash_func, \
                      __khash_equal, __vhash_func, __vhash_equal) \
    NDIM_WNN_INIT2(name, K, ndim, static kh_inline klib_unused, khkey_t, \
                   khval_t, coeff_t, __khash_func, __khash_equal, \
                   __vhash_func, __vhash_equal)


#define NDIM_KNN_TYPE(name, ndim, khkey_t, khval_t) \
    __NDIM_KNN_TYPE(name, ndim, khkey_t, khval_t, unsigned)

#define NDIM_ANN_TYPE(name, ndim, khkey_t, khval_t, coeff_t) \
    __NDIM_ANN_TYPE(name, ndim, khkey_t, khval_t, coeff_t)

#define NDIM_WNN_TYPE(name, ndim, khkey_t, khval_t, coeff_t) \
    __NDIM_WNN_TYPE(name, ndim, khkey_t, khval_t, coeff_t)


#define NDIM_KNN_IMPL(name, K, ndim, khkey_t, khval_t, __khash_func, \
                      __khash_equal, __vhash_func, __vhash_equal) \
    __NDIM_KNN_IMPL(name, K, ndim, static kh_inline klib_unused, khkey_t, \
                    khval_t, unsigned, __khash_func, __khash_equal, \
                    __vhash_func, __vhash_equal)

#define NDIM_ANN_IMPL(name, K, ndim, khkey_t, khval_t, coeff_t, __khash_func, \
                      __khash_equal, __vhash_func, __vhash_equal) \
    __NDIM_ANN_IMPL(name, K, ndim, static kh_inline klib_unused, khkey_t, \
                    khval_t, coeff_t, __khash_func, __khash_equal, \
                    __vhash_func, __vhash_equal)

#define NDIM_WNN_IMPL(name, K, ndim, khkey_t, khval_t, coeff_t, __khash_func, \
                      __khash_equal, __vhash_func, __vhash_equal) \
    __NDIM_WNN_IMPL(name, K, ndim, static kh_inline klib_unused, khkey_t, \
                    khval_t, coeff_t, __khash_func, __khash_equal, \
                    __vhash_func, __vhash_equal)

/* Other convenient macros... */

/*!
  @abstract Type of the n-dimensional ANN lookup.
  @param  name  Name of the n-dimensional ANN lookup [symbol]
 */
#define nn_t(prefix, name) prefix##nn_##name##_t
#define knn_t(name)        nn_t(k, name)
#define ann_t(name)        nn_t(a, name)
#define wnn_t(name)        nn_t(w, name)

/*!
  @abstract Type of the n-dimensional ANN lookup node.
  @param  name  Name of the n-dimensional ANN lookup [symbol]
 */
#define nn_node_t(name) nn_##name##_node_t

/*! @function
  @abstract      Initiate an n-dimensional ANN lookup.
  @param  name   Name of the n-dimensional ANN lookup [symbol]
  @return        Pointer to the n-dimensional ANN lookup [nn_t(prefix, name)*]
 */
#define knn_init(name) knn_init_##name()

/*! @function
  @abstract      Initiate an n-dimensional ANN lookup in the in-place case.
  @param  name   Name of the n-dimensional ANN lookup [symbol]
  @param  h      Pointer to the n-dimensional ANN lookup [nn_t(prefix, name)*]
 */
#define knn_init_inplace(name, h) knn_init_##name##_inplace(h)

/*! @function
  @abstract      Initiate an n-dimensional ANN lookup.
  @param  name   Name of the n-dimensional ANN lookup [symbol]
  @param  a      Approximation coefficient, for ANN lookups [number]
  @return        Pointer to the n-dimensional ANN lookup [nn_t(prefix, name)*]
 */
#define ann_init(name, a) ann_init_##name(a)

/*! @function
  @abstract      Initiate an n-dimensional ANN lookup in the in-place case.
  @param  name   Name of the n-dimensional ANN lookup [symbol]
  @param  h      Pointer to the n-dimensional ANN lookup [nn_t(prefix, name)*]
  @param  a      Approximation coefficient, for ANN lookups [number]
 */
#define ann_init_inplace(name, h, a) ann_init_##name##_inplace(h, a

/*! @function
  @abstract      Initiate an n-dimensional ANN lookup.
  @param  name   Name of the n-dimensional ANN lookup [symbol]
  @param  a      Approximation coefficient, for ANN lookups [number]
  @param  c      Coefficients for the wighted distance function [array of numbers]
  @return        Pointer to the n-dimensional ANN lookup [nn_t(prefix, name)*]
 */
#define wnn_init(name, a, c) wnn_init_##name(a, c)

/*! @function
  @abstract      Initiate an n-dimensional ANN lookup in the in-place case.
  @param  name   Name of the n-dimensional ANN lookup [symbol]
  @param  h      Pointer to the n-dimensional ANN lookup [nn_t(prefix, name)*]
  @param  a      Approximation coefficient, for ANN lookups [number]
  @param  c      Coefficients for the wighted distance function [array of numbers]
 */
#define wnn_init_inplace(prefix, name, h, a, c) \
    wnn_init_##name##_inplace(h, a, c)

/*! @function
  @abstract      Destroy an n-dimensional ANN lookup.
  @param  prefix One-letter prefix indicating the lookup type [k, a, or w]
  @param  name   Name of the n-dimensional ANN lookup [symbol]
  @param  h      Pointer to the n-dimensional ANN lookup [nn_t(prefix, name)*]
 */
#define nn_destroy(prefix, name, h) prefix##nn_destroy_##name(h)

/*! @function
  @abstract      Destroy an n-dimensional ANN lookup in the in-place case.
  @param  prefix One-letter prefix indicating the lookup type [k, a, or w]
  @param  name   Name of the n-dimensional ANN lookup [symbol]
  @param  h      Pointer to the n-dimensional ANN lookup [nn_t(prefix, name)*]
 */
#define nn_destroy_inplace(prefix, name, h) \
    prefix##nn_destroy_##name##_inplace(h)

/*! @function
  @abstract      Reset the n-dimensional ANN lookup without deallocating memory.
  @param  prefix One-letter prefix indicating the lookup type [k, a, or w]
  @param  name   Name of the n-dimensional ANN lookup [symbol]
  @param  h      Pointer to the ANN lookup [nn_t(prefix, name)*]
 */
#define nn_clear(prefix, name, h) prefix##nn_clear_##name(h)

/*! @function
  @abstract      Resize the n-dimensional ANN lookup
  @param  prefix One-letter prefix indicating the lookup type [k, a, or w]
  @param  name   Name of the n-dimensional ANN lookup [symbol]
  @param  h      Pointer to the ANN lookup [nn_t(prefix, name)*]
  @param  s      New size [khint_t]
 */
#define nn_resize(prefix, name, h, s) prefix##nn_resize_##name(h, s)

/*! @function
  @abstract      Insert a key-value pair to the n-dimensional ANN lookup.
  @param  prefix One-letter prefix indicating the lookup type [k, a, or w]
  @param  name   Name of the n-dimensional ANN lookup [symbol]
  @param  h      Pointer to the n-dimensional ANN lookup [nn_t(prefix, name)*]
  @param  k      Key vector [vector of type of keys]
  @param  v      Value [type of values]
  @return        Status, non-zero for error [integer]
 */
#define nn_insert(prefix, name, h, k, v) prefix##nn_insert_##name(h, k, v)

/*! @function
  @abstract      Remove a key from the n-dimensional ANN lookup.
  @param  prefix One-letter prefix indicating the lookup type [k, a, or w]
  @param  name   Name of the n-dimensional ANN lookup [symbol]
  @param  h      Pointer to the n-dimensional ANN lookup [nn_t(prefix, name)*]
  @param  k      Iterator to the element to be deleted [Khint_t]
  @return        Status, non-zero for error [integer]
 */
#define nn_delv(prefix, name, h, k) prefix##nn_delv_##name(h, k)

/*! @function
  @abstract      Lookup the C-approximate nearest neighbor to a given key.
  @param  prefix One-letter prefix indicating the lookup type [k, a, or w]
  @param  name   Name of the n-dimensional ANN lookup [symbol]
  @param  h      Pointer to the n-dimensional ANN lookup [nn_t(prefix, name)*]
  @param  k      Key vector [vector of type of keys]
  @param  res    Resulting values vector [vector of type of values]
  @return        Status, non-zero for error [integer]
 */
#define nn_search(prefix, name, h, k, res) prefix##nn_search_##name(h, k, res)

/* More conenient interfaces */

/*! @function
  @abstract     Prepare for n-dimensional NN search by integer keys
  @param  name  Name of the n-dimensional NN lookup [symbol]
  @param  k     Amount of neighbors to retrieve per search [number]
  @param  ndim  Number of dimensions [number]
  @param  khval_t  Type of values [type]
 */
#define NDIM_KNN_INIT_INT(name, K, ndim) \
    NDIM_KNN_INIT(name, K, ndim, khint32_t, uintptr_t, kh_int_hash_func, \
                  kh_int_hash_equal, kh_int64_hash_func, kh_int64_hash_equal)

/*! @function
  @abstract     Prepare for n-dimensional NN search by 64-bit integer keys
  @param  name  Name of the n-dimensional NN lookup [symbol]
  @param  k     Amount of neighbors to retrieve per search [number]
  @param  ndim  Number of dimensions [number]
 */
#define NDIM_ANN_INIT_INT64(name, K, ndim) \
    NDIM_ANN_INIT(name, K, ndim, khint64_t, uintptr_t, khint64_t, \
                  kh_int_hash_func, kh_int_hash_equal, kh_int64_hash_func, \
                  kh_int64_hash_equal)

/*! @function
  @abstract     Prepare for n-dimensional NNS by strings with float coefficients
  @param  name  Name of the n-dimensional NN lookup [symbol]
  @param  k     Amount of neighbors to retrieve per search [number]
  @param  ndim  Number of dimensions [number]
 */
#define NDIM_WNN_INIT_STR(name, K, ndim) \
    NDIM_WNN_INIT(name, K, ndim, kh_cstr_t, uintptr_t, float, \
                  kh_str_hash_func, kh_str_hash_equal, kh_int64_hash_func, \
                  kh_int64_hash_equal)

#endif
