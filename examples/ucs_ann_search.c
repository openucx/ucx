/**
 * See file LICENSE for terms.
 */

#include <ucs/datastruct/ndim_knn.h>

#define EXPORTED __attribute__((visibility("default")))
#define DIM_LIST 1, 2, 3, 16, 20, 25, 50, 96, 100, 128, 200, 256, 784, 800, 960

#define WNN_BY_DIM(_K, _ndim) \
    NDIM_WNN_INIT2(char_to_unsigned_##_ndim##_dim_k##_K, _K, _ndim, EXPORTED, \
                   char, unsigned, khint32_t, kh_int_hash_func, \
                   kh_int_hash_equal, kh_int_hash_func, kh_int_hash_equal) \
    NDIM_WNN_INIT2(int32_to_int32_##_ndim##_dim_k##_K, _K, _ndim, EXPORTED, \
                   khint32_t, unsigned, khint32_t, kh_int_hash_func, \
                   kh_int_hash_equal, kh_int_hash_func, kh_int_hash_equal) \
    NDIM_WNN_INIT2(int64_to_int64_##_ndim##_dim_k##_K, _K, _ndim, EXPORTED, \
                   khint64_t, khint64_t, khint64_t, kh_int_hash_func, \
                   kh_int_hash_equal, kh_int64_hash_func, kh_int64_hash_equal) \
    NDIM_WNN_INIT2(float_to_unsigned_##_ndim##_dim_k##_K, _K, _ndim, EXPORTED, \
                   float, unsigned, int, kh_int_hash_func, kh_int_hash_equal, \
                   kh_int_hash_func, kh_int_hash_equal) \
    NDIM_WNN_INIT2(float_to_ptr_##_ndim##_dim_k##_K, _K, _ndim, EXPORTED, \
                   float, uintptr_t, float, kh_int_hash_func, \
                   kh_int_hash_equal, kh_int64_hash_func, kh_int64_hash_equal) \
    NDIM_WNN_INIT2(cstr_to_ptr_##_ndim##_dim_k##_K, _K, _ndim, EXPORTED, \
                   kh_cstr_t, unsigned, float, kh_str_hash_func, \
                   kh_str_hash_equal, kh_int_hash_func, kh_int_hash_equal)

UCS_PP_FOREACH(WNN_BY_DIM, 1, DIM_LIST)
UCS_PP_FOREACH(WNN_BY_DIM, 10, DIM_LIST)
