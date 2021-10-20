/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) Arm, Ltd. 2021. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_COMPILER_DEF_H
#define UCS_COMPILER_DEF_H

/* Note: Place "@file <file name>.h" after BEGIN_C_DECS
 * to avoid bugs in a documentation */
#ifdef __cplusplus
# define BEGIN_C_DECLS  extern "C" {
# define END_C_DECLS    }
#else
# define BEGIN_C_DECLS
# define END_C_DECLS
#endif

/*
 * Assertions which are checked in compile-time
 *
 * Usage: UCS_STATIC_ASSERT(condition)
 */
#define UCS_STATIC_ASSERT(_cond) \
     switch(0) {case 0:case (_cond):;}

/* Maximal allocation size for on-stack buffers */
#define UCS_ALLOCA_MAX_SIZE  1200

/* Aliasing structure */
#define UCS_S_MAY_ALIAS __attribute__((may_alias))

/* A function without side effects */
#define UCS_F_PURE   __attribute__((pure))

/* A function which does not return */
#define UCS_F_NORETURN __attribute__((noreturn))

/* Packed structure */
#define UCS_S_PACKED             __attribute__((packed))

/* Avoid inlining the function */
#define UCS_F_NOINLINE __attribute__ ((noinline))

/* Shared library constructor and destructor */
#define UCS_F_CTOR __attribute__((constructor))
#define UCS_F_DTOR __attribute__((destructor))

/* Silence "defined but not used" error for static function */
#define UCS_F_MAYBE_UNUSED __attribute__((used))

/* Non-null return */
#define UCS_F_NON_NULL __attribute__((nonnull))

/* Always inline the function */
#ifdef __GNUC__
#define UCS_F_ALWAYS_INLINE      inline __attribute__ ((always_inline))
#else
#define UCS_F_ALWAYS_INLINE      inline
#endif

/* Silence "uninitialized variable" for stupid compilers (gcc 4.1)
 * which can't optimize properly.
 */
#if (((__GNUC__ == 4) && (__GNUC_MINOR__ == 1)) || !defined(__OPTIMIZE__))
#  define UCS_V_INITIALIZED(_v)  (_v = (ucs_typeof(_v))0)
#else
#  define UCS_V_INITIALIZED(_v)  ((void)0)
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

/* Unused variable */
#define UCS_V_UNUSED __attribute__((unused))

/* Aligned variable */
#define UCS_V_ALIGNED(_align) __attribute__((aligned(_align)))

/* Used for labels */
#define UCS_EMPTY_STATEMENT {}

/* Helper macro for address arithmetic in bytes */
#define UCS_PTR_BYTE_OFFSET(_ptr, _offset) \
    ((void *)((intptr_t)(_ptr) + (intptr_t)(_offset)))

/* Helper macro to calculate an address with offset equal to size of _type */
#define UCS_PTR_TYPE_OFFSET(_ptr, _type) \
    ((void *)((ucs_typeof(_type) *)(_ptr) + 1))

/* Helper macro to calculate ptr difference (_end - _start) */
#define UCS_PTR_BYTE_DIFF(_start, _end) \
    ((ptrdiff_t)((uintptr_t)(_end) - (uintptr_t)(_start)))


/**
 * Size of statically-declared array
 */
#define ucs_static_array_size(_array) \
    (sizeof(_array) / sizeof((_array)[0]))


/**
 * @return Offset of _member in _type. _type is a structure type.
 */
#define ucs_offsetof(_type, _member) \
    ((unsigned long)&( ((_type*)0)->_member ))


/**
 * Get a pointer to a struct containing a member.
 *
 * @param _ptr     Pointer to the member.
 * @param _type    Container type.
 * @param _member  Element member inside the container.

 * @return Address of the container structure.
 */
#define ucs_container_of(_ptr, _type, _member) \
    ( (_type*)( (char*)(void*)(_ptr) - ucs_offsetof(_type, _member) )  )


/**
 * Get the type of a structure or variable.
 * 
 * @param _type  Return the type of this argument.
 * 
 * @return The type of the given argument.
 */
#define ucs_typeof(_type) \
    __typeof__(_type)


/**
 * @return Address of a derived structure. It must have a "super" member at offset 0.
 * NOTE: we use the built-in offsetof here because we can't use ucs_offsetof() in
 *       a constant expression.
 */
#define ucs_derived_of(_ptr, _type) \
    ({\
        UCS_STATIC_ASSERT(offsetof(_type, super) == 0) \
        ucs_container_of(_ptr, _type, super); \
    })

/**
 * @param _type   Structure type.
 * @param _field  Field of structure.
 *
 * @return Size of _field in _type.
 */
#define ucs_field_sizeof(_type, _field) \
    sizeof(((_type*)0)->_field)

/**
 * @param _type   Structure type.
 * @param _field  Field of structure.
 *
 * @return Type of _field in _type.
 */
#define ucs_field_type(_type, _field) \
    ucs_typeof(((_type*)0)->_field)

/**
 * Prevent compiler from reordering instructions
 */
#define ucs_compiler_fence()       asm volatile(""::: "memory")

/**
 * Prefetch cache line
 */
#define ucs_prefetch(p)            __builtin_prefetch(p)

/* Branch prediction */
#define ucs_likely(x)              __builtin_expect(x, 1)
#define ucs_unlikely(x)            __builtin_expect(x, 0)

/* Check if an expression is a compile-time constant */
#define ucs_is_constant(expr)      __builtin_constant_p(expr)

/*
 * Define code which runs at global constructor phase
 */
#define UCS_STATIC_INIT \
    static void UCS_F_CTOR UCS_PP_APPEND_UNIQUE_ID(ucs_initializer_ctor)()

/*
 * Define code which runs at global destructor phase
 */
#define UCS_STATIC_CLEANUP \
    static void UCS_F_DTOR UCS_PP_APPEND_UNIQUE_ID(ucs_initializer_dtor)()

/*
 * Check if the two types are the same
 */
#define ucs_same_type(_type1, _type2) \
    __builtin_types_compatible_p(_type1, _type2)

#endif /* UCS_COMPILER_DEF_H */
