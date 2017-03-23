/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#ifndef UCS_COMPILER_DEF_H
#define UCS_COMPILER_DEF_H


#ifdef __cplusplus
# define BEGIN_C_DECLS  extern "C" {
# define END_C_DECLS    }
#else
# define BEGIN_C_DECLS
# define END_C_DECLS
#endif

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

/*
 * Enable compiler checks for printf-like formatting.
 *
 * @param fmtargN number of formatting argument
 * @param vargN   number of variadic argument
 */
#define UCS_F_PRINTF(fmtargN, vargN) __attribute__((format(printf, fmtargN, vargN)))

#endif /* UCS_COMPILER_DEF_H */
