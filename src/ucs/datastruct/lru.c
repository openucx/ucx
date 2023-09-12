/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2023. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "lru.h"

#include <ucs/datastruct/list.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>


ucs_status_t ucs_lru_create(size_t capacity, ucs_lru_h *lru_p)
{
    ucs_lru_h lru;
    ucs_status_t status;

    if (capacity == 0) {
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    lru = ucs_calloc(1, sizeof(*lru), "ucs_lru");
    if (lru == NULL) {
        ucs_error("failed to allocate LRU (capacity: %lu)", capacity);
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    kh_init_inplace(ucs_lru_hash, &lru->hash);

    /* Resize the cache to the required capacity. Need to allocate extra space
     * in order to avoid collisions in the hash table. */
    if (kh_resize(ucs_lru_hash, &lru->hash, capacity * 2) < 0) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free;
    }

    ucs_list_head_init(&lru->list);

    lru->capacity = capacity;
    *lru_p        = lru;
    return UCS_OK;

err_free:
    ucs_free(lru);
err:
    return status;
}

void ucs_lru_destroy(ucs_lru_h lru)
{
    kh_destroy_inplace(ucs_lru_hash, &lru->hash);
    ucs_free(lru);
}

