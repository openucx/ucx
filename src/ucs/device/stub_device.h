/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_STUB_DEVICE_CUH
#define UCS_STUB_DEVICE_CUH

#include <stdint.h>

/* Device function */
#define UCS_F_DEVICE static inline


/* Device library function */
#define UCS_F_DEVICE_LIB


/*
  * Read a 64-bit atomic value from a global memory address.
  */
UCS_F_DEVICE uint64_t ucs_device_atomic64_read(const uint64_t *ptr)
{
    return *ptr;
}


/*
  * Write a 64-bit value to counter global memory address.
  */
UCS_F_DEVICE void ucs_device_atomic64_write(uint64_t *ptr, uint64_t value)
{
    *ptr = value;
}


/*
  * Read the 64-bit GPU global nanosecond timer
  */
UCS_F_DEVICE uint64_t ucs_device_get_time_ns(void)
{
    return 0;
}

/*
  * Load a constant from global memory.
  */
#define ucs_device_load_const(_ptr) (*(_ptr))

#endif /* UCS_STUB_DEVICE_CUH */
