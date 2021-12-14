/**
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */


#ifndef UCS_MPOOL_SET_INL_
#define UCS_MPOOL_SET_INL_

#include "mpool_set.h"

#include <ucs/datastruct/mpool.inl>
#include <ucs/debug/assert.h>
#include <ucs/arch/bitops.h>
#include <ucs/sys/math.h>


/**
 * Get an element from the memory pool set.
 *
 * @param mp_set           Memory pool set structure.
 * @param size             Needed size of the element. Private area size (if
 *                         needed) is not included here, it has to be specified
 *                         during mpool set initialization instead.
 *
 * @return New allocated object, or NULL if cannot allocate.
 */
static UCS_F_ALWAYS_INLINE void*
ucs_mpool_set_get_inline(ucs_mpool_set_t *mp_set, size_t size)
{
    uint32_t elem_size = (uint32_t)(size + 1);
    unsigned idx;

    /* - use elem_size equal to size + 1 to avoid passing 0 to
     *   ucs_count_leading_zero_bits()
     * - do not roundup the elem_size if it is pow of 2 already, to not use
     *   bigger mpool than needed
     */
    idx = ucs_count_leading_zero_bits(elem_size) -
          !ucs_is_pow2_or_zero(elem_size);

    ucs_assertv(size < UINT32_MAX, "size %zu", size);
    ucs_assertv(idx < UCS_MPOOL_SET_SIZE, "idx %u", idx);

    return ucs_mpool_get_inline(mp_set->map[idx]);
}

/**
 * Return an object to the memory pool set.
 *
 * @param obj              Object to return.
 */
static UCS_F_ALWAYS_INLINE void ucs_mpool_set_put_inline(void *obj)
{
    ucs_mpool_put_inline(obj);
}

#endif /* UCS_MPOOL_SET_INL_ */
