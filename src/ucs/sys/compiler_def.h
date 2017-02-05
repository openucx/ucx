/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#ifndef UCS_COMPILER_DEF_H
#define UCS_COMPILER_DEF_H

/* Packed structure */
#define UCS_S_PACKED             __attribute__((packed))

/* Always inline the function */
#ifdef __GNUC__
#define UCS_F_ALWAYS_INLINE      inline __attribute__ ((always_inline))
#else
#define UCS_F_ALWAYS_INLINE      inline
#endif

/* The i-th bit */
#define UCS_BIT(i)               (1ul << (i))

/* Mask of bits 0..i-1 */
#define UCS_MASK(i)              (UCS_BIT(i) - 1)


#endif /* UCS_COMPILER_DEF_H */
