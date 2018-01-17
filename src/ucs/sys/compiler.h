/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_COMPILER_H_
#define UCS_COMPILER_H_

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "preprocessor.h"
#include "compiler_def.h"

#include <ucs/debug/assert.h>
#include <stddef.h>
#include <stdarg.h>
#include <alloca.h>

#ifndef ULLONG_MAX
#define ULLONG_MAX (__LONG_LONG_MAX__ * 2ULL + 1)
#endif


#ifdef __ICC
#  pragma warning(disable: 268)
#endif

/* A function which should not be optimized */
#if defined(HAVE_ATTRIBUTE_NOOPTIMIZE) && (HAVE_ATTRIBUTE_NOOPTIMIZE == 1)
#define UCS_F_NOOPTIMIZE __attribute__((optimize("O0")))
#else
#define UCS_F_NOOPTIMIZE
#endif


/* Helper macro for address arithmetic in bytes */
#define UCS_PTR_BYTE_OFFSET(_ptr, _offset) \
    ((void *)((uintptr_t)(_ptr) + (_offset)))

/**
 * Copy words from _src to _dst.
 *
 * @param _dst         Destination buffer.
 * @param _src         Source buffer.
 * @param _word_type   Type to use for copying.
 * @param _size        Number of bytes to copy.
 */
#define UCS_WORD_COPY(_dst, _src, _word_type, _size) \
    { \
        unsigned i; \
        for (i = 0; i < (_size) / sizeof(_word_type); ++i) { \
            *((_word_type*)(_dst) + i) = *((_word_type*)(_src) + i); \
        } \
    }

#define UCS_ALLOCA_MAX_SIZE  1200

/**
 * alloca which makes sure the size is small enough.
 */
#define ucs_alloca(_size) \
    ({ \
        ucs_assertv((_size) <= UCS_ALLOCA_MAX_SIZE, "alloca(%zu)", (size_t)(_size)); \
        alloca(_size); \
    })


/**
 * Define cache-line padding variable inside a structure
 *
 * @param ...    List of types, of the variables which should be padded to cache line.
 */
#define UCS_CACHELINE_PADDING(...) \
    char UCS_PP_APPEND_UNIQUE_ID(pad)[UCS_SYS_CACHE_LINE_SIZE - \
                                      UCS_CACHELINE_PADDING_MISALIGN(__VA_ARGS__)];
#define UCS_CACHELINE_PADDING_SIZEOF(_, _x) \
    + sizeof(_x)
#define UCS_CACHELINE_PADDING_MISALIGN(...) \
    ((UCS_PP_FOREACH(UCS_CACHELINE_PADDING_SIZEOF, _, __VA_ARGS__)) % UCS_SYS_CACHE_LINE_SIZE)

#endif
