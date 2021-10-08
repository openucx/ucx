/**
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "mpool_set.h"
#include "mpool.h"
#include "mpool_set.inl"

#include <ucs/sys/math.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>



static void ucs_mpool_set_cleanup_common(ucs_mpool_set_t *mp_set, int max_idx,
                                         int leak_check)
{
    ucs_mpool_t *mpools = mp_set->data;
    int i;

    for (i = 0; i < max_idx; ++i) {
        ucs_mpool_cleanup(&mpools[i], leak_check);
    }

    ucs_free(mp_set->data);
}

ucs_status_t
ucs_mpool_set_init(ucs_mpool_set_t *mp_set, size_t *sizes, unsigned sizes_count,
                   size_t max_mp_entry_size, size_t priv_size,
                   size_t priv_elem_size, size_t align_offset, size_t alignment,
                   unsigned elems_per_chunk, unsigned max_elems,
                   ucs_mpool_ops_t *ops, const char *name)
{
    int i, size_log2, mpools_num;
    int prev_idx, mps_idx, map_idx, max_idx;
    size_t size;
    ucs_mpool_t *mpools;
    ucs_status_t status;

    if (sizes_count == 0) {
        ucs_error("creation of empty mpool_set is not allowed");
        return UCS_ERR_INVALID_PARAM;
    }

    if ((max_mp_entry_size > UCS_MPOOL_SET_MAX_SIZE) ||
        (max_mp_entry_size == 0)) {
        ucs_error("invalid maximal mpool element size %zu", max_mp_entry_size);
        return UCS_ERR_INVALID_PARAM;
    }

    mp_set->bitmap = 0;
    max_idx        = UCS_MPOOL_SET_SIZE - 1;
    for (i = 0; i < sizes_count; ++i) {
        if (!ucs_is_pow2(sizes[i])) {
            ucs_error("wrong mpool size %zu, it must be power of 2", sizes[i]);
            return UCS_ERR_INVALID_PARAM;
        }

        if (sizes[i] > max_mp_entry_size) {
            /* Ignore sizes bigger than maximal allowed value */
            continue;
        }

        mp_set->bitmap |= sizes[i];
    }

    /* max_mp_entry_size is allowed to be non pow of 2. If this is the case, set
     * the last bit in the map indicating that max_mp_entry_size is not equal to
     * any element in sizes array.
     */
    if (!ucs_is_pow2(max_mp_entry_size) ||
        !(max_mp_entry_size & mp_set->bitmap)) {
        mp_set->bitmap |= UCS_BIT(max_idx);
    }

    mpools_num   = ucs_popcount(mp_set->bitmap);
    mp_set->data = ucs_malloc((mpools_num * sizeof(*mpools)) + priv_size,
                              "mpools_set");
    if (mp_set->data == NULL) {
        ucs_error("failed to allocate mpool set private data");
        return UCS_ERR_NO_MEMORY;
    }

    mpools   = mp_set->data;
    mps_idx  = 0;
    prev_idx = max_idx;
    ucs_for_each_bit(size_log2, mp_set->bitmap) {
        map_idx = max_idx - size_log2;
        size    = (map_idx == 0) ? max_mp_entry_size : UCS_BIT(size_log2);
        status  = ucs_mpool_init(&mpools[mps_idx], priv_size,
                                 size + priv_elem_size, align_offset, alignment,
                                 elems_per_chunk, max_elems, ops, name);
        if (status != UCS_OK) {
            goto err;
        }

        /* mp_set->map is an array of pointers to memory pools. Array index is
         * log2 of the corresponding memory pool element size. Indexation is
         * done in the reverse order to speedup array index calculation in
         * ucs_mpool_set_get(). Since there is no separate memory pool for
         * every pow of 2 value, initialize all array elements in the range of
         * [current_index, last_initalized_index).
         * So, eventually every mp_set->map array element will point to a
         * certain mpool with minimal element size capable of storing elements
         * up to 2^(max_idx - index).
         */
        for (i = prev_idx; i >= map_idx; --i) {
            mp_set->map[i] = &mpools[mps_idx];
        }
        prev_idx = map_idx - 1;
        ++mps_idx;
    }

    ucs_debug("mpool_set:%s, sizes map 0x%x, largest size %zu, mpools num %d",
              name, mp_set->bitmap, max_mp_entry_size, mpools_num);

    return UCS_OK;

err:
    ucs_mpool_set_cleanup_common(mp_set, mps_idx, 0);

    return status;
}

void ucs_mpool_set_cleanup(ucs_mpool_set_t *mp_set, int leak_check)
{
    ucs_mpool_set_cleanup_common(mp_set, ucs_popcount(mp_set->bitmap),
                                 leak_check);
}

void *ucs_mpool_set_priv(ucs_mpool_set_t *mp_set)
{
    return (ucs_mpool_t*)mp_set->data + ucs_popcount(mp_set->bitmap);
}

const char *ucs_mpool_set_name(ucs_mpool_set_t *mp_set)
{
    /* All mpools in the set contain the same name, can just return the name
     * of the first one.
     */
    return ((ucs_mpool_t*)mp_set->data)->data->name;
}

