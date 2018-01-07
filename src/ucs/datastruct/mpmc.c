/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "mpmc.h"

#include <ucs/arch/atomic.h>
#include <ucs/arch/bitops.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/memtrack.h>


ucs_status_t ucs_mpmc_queue_init(ucs_mpmc_queue_t *mpmc, uint32_t length)
{
    uint32_t i;

    mpmc->length   = ucs_roundup_pow2(length);
    mpmc->shift    = ucs_ilog2(mpmc->length);
    if (mpmc->length >= UCS_BIT(UCS_MPMC_VALID_SHIFT)) {
        return UCS_ERR_INVALID_PARAM;
    }

    mpmc->consumer = 0;
    mpmc->producer = 0;
    mpmc->queue = ucs_malloc(sizeof(int) * mpmc->length, "mpmc");
    if (mpmc->queue == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    for (i = 0; i < mpmc->length; ++i) {
        mpmc->queue[i] = UCS_BIT(UCS_MPMC_VALID_SHIFT);
    }

    return UCS_OK;
}

void ucs_mpmc_queue_cleanup(ucs_mpmc_queue_t *mpmc)
{
    ucs_free(mpmc->queue);
}

static inline uint32_t __ucs_mpmc_queue_valid_bit(ucs_mpmc_queue_t *mpmc, uint32_t location)
{
    return (location >> mpmc->shift) & 1;
}

ucs_status_t ucs_mpmc_queue_push(ucs_mpmc_queue_t *mpmc, uint32_t value)
{
    uint32_t location;

    ucs_assert((value >> UCS_MPMC_VALID_SHIFT) == 0);

    do {
        location = mpmc->producer;
        if (UCS_CIRCULAR_COMPARE32(location, >=, mpmc->consumer + mpmc->length)) {
            /* Queue is full */
            return UCS_ERR_EXCEEDS_LIMIT;
        }
    } while (ucs_atomic_cswap32(&mpmc->producer, location, location + 1) != location);

    mpmc->queue[location & (mpmc->length - 1)] = value |
                    (__ucs_mpmc_queue_valid_bit(mpmc, location) << UCS_MPMC_VALID_SHIFT);
    return UCS_OK;
}


ucs_status_t ucs_mpmc_queue_pull(ucs_mpmc_queue_t *mpmc, uint32_t *value_p)
{
    uint32_t location, value;

    location = mpmc->consumer;
    if (location == mpmc->producer) {
        /* Producer not started yet */
        return UCS_ERR_NO_PROGRESS;
    }

    value = mpmc->queue[location & (mpmc->length - 1)];
    if ((value >> UCS_MPMC_VALID_SHIFT) != __ucs_mpmc_queue_valid_bit(mpmc, location)) {
        /* Producer not finished yet */
        return UCS_ERR_NO_PROGRESS;
    }

    if (ucs_atomic_cswap32(&mpmc->consumer, location, location + 1) != location) {
        /* Someone else consumed */
        return UCS_ERR_NO_PROGRESS;
    }

    *value_p = value & UCS_MASK(UCS_MPMC_VALID_SHIFT);
    return UCS_OK;
}
