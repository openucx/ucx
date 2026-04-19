/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2021. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#ifndef UCS_TYPE_FLOAT_H
#define UCS_TYPE_FLOAT_H

#include <stdint.h>
#include <string.h>

/* IEEE754 double layout helpers without bitfields.
 */
static inline void
ucs_double_to_words(double d, uint32_t *hi, uint32_t *lo)
{
    uint32_t w[2];
    memcpy(w, &d, sizeof(w));
#if defined(FLOAT_WORDS_BIGENDIAN)
    *hi = w[0];
    *lo = w[1];
#else
    *hi = w[1];
    *lo = w[0];
#endif
}

static inline double
ucs_words_to_double(uint32_t hi, uint32_t lo)
{
    uint32_t w[2];
    double d;
#if defined(FLOAT_WORDS_BIGENDIAN)
    w[0] = hi;
    w[1] = lo;
#else
    w[0] = lo;
    w[1] = hi;
#endif
    memcpy(&d, w, sizeof(d));
    return d;
}

static inline uint32_t ucs_ieee_double_sign(uint32_t hi)
{
    return hi >> 31;
}

static inline uint32_t ucs_ieee_double_exp(uint32_t hi)
{
    return (hi >> 20) & 0x7ffu;
}

static inline uint32_t ucs_ieee_double_mant_hi20(uint32_t hi)
{
    return hi & 0x000fffffu;
}

#ifndef IEEE754_DOUBLE_BIAS
#  define IEEE754_DOUBLE_BIAS 1023
#endif

#include <ucs/sys/preprocessor.h>
#include <ucs/debug/assert.h>
#include <ucs/arch/bitops.h>

BEGIN_C_DECLS


typedef uint8_t ucs_fp8_t;

/**
 * Bits number in the exponent part of a packed floating-point number
 */
#define _UCS_FP8_EXPONENT_BITS 4


/**
 * Bits number in the mantissa part of a packed floating-point number
 */
#define _UCS_FP8_MANTISSA_BITS 4


/**
 * The ratio of the value obtained after packing and unpacking to
 * the original number
 */
#define UCS_FP8_PRECISION \
    ((double)UCS_MASK(_UCS_FP8_MANTISSA_BITS) / UCS_BIT(_UCS_FP8_MANTISSA_BITS))


/**
 * Bits number in the exponent part of an IEEE754 double
 */
#define _UCS_FP8_IEEE_EXPONENT_BITS 11


/**
 * Bits number in the significant mantissa part of an IEEE754 double
 */
#define _UCS_FP8_IEEE_MANTISSA_BITS 20


/**
 * Shift of the packed mantissa representation, relative to the IEEE representation
 */
#define _UCS_FP_MANTISSA_OFFSET \
    (_UCS_FP8_IEEE_MANTISSA_BITS - _UCS_FP8_MANTISSA_BITS)


/**
 * A special value of exponent which represents NaN in an IEEE754 double
 */
#define _UCS_FP8_IEEE_NAN_EXPONENT UCS_MASK(_UCS_FP8_IEEE_EXPONENT_BITS)


/**
 * A special value of exponent which represents NaN in a packed floating-point number
 */
#define _UCS_FP8_NAN UCS_MASK(_UCS_FP8_EXPONENT_BITS)


/**
 * The offset of an IEEE754 exponent representation from a packed exponent representation
 */
#define _UCS_FP8_EXPONENT_OFFSET (IEEE754_DOUBLE_BIAS - 1)


/**
 * Internal macro to construct floating-point type identifier from a name and a suffix
 */
#define _UCS_FP8_IDENTIFIER(_name, _suffix) \
    UCS_PP_TOKENPASTE3(ucs_fp8_, _name, _suffix)


/**
 * Mask the exponent part of a packed floating-point number
 */
#define _UCS_FP8_EXPONENT_MASK (UCS_MASK(_UCS_FP8_EXPONENT_BITS))


/**
 * pack a double-precision floating-point number in a given range to a single byte.
 * The packing is lossy and the unpacked number is assumed to be
 * non-negative.
 *
 * @param value Pack this number
 * @param min   Min supported value (assumed to be a power of 2)
 * @param max   Max supported value (assumed to be a power of 2)
 *
 * @return A single byte which represents the given number
 */
static UCS_F_ALWAYS_INLINE ucs_fp8_t ucs_fp8_pack(double value, uint64_t min,
                                                  uint64_t max)
{
    uint32_t hi, lo;
    uint32_t exp;
    uint32_t mant0;
    uint8_t exponent;
    int8_t min_exponent, max_exponent;

    min_exponent = ucs_ilog2(min);
    max_exponent = ucs_ilog2(max);

    ucs_double_to_words(value, &hi, &lo);
    exp   = ucs_ieee_double_exp(hi);
    mant0 = ucs_ieee_double_mant_hi20(hi);

    if (ucs_unlikely(exp == _UCS_FP8_IEEE_NAN_EXPONENT)) {
        /* NaN maps to a special value for NaN */
        exponent = _UCS_FP8_NAN;
    } else if (ucs_unlikely(exp >
                            (max_exponent + _UCS_FP8_EXPONENT_OFFSET))) {
        /* A number beyond the max supported is capped */
        exponent                  = max_exponent - min_exponent;
        /* zero mantissa in-place */
        mant0 = 0;
        lo    = 0;
        hi    = (hi & 0xfff00000u); /* keep sign+exp, clear mantissa hi20 */
    } else if (ucs_unlikely(exp <=
                            min_exponent + _UCS_FP8_EXPONENT_OFFSET)) {
        if (ucs_unlikely(value == 0)) {
            /* 0 maps to a special value for 0 */
            exponent = 0;
        } else {
            /* A number below the min supported is rounded up */
            exponent                  = 1;
            mant0 = 0;
            lo    = 0;
            hi    = (hi & 0xfff00000u);
        }
    } else {
        exponent = exp - _UCS_FP8_EXPONENT_OFFSET -
                   min_exponent;
    }

    return exponent | ((mant0 >> _UCS_FP_MANTISSA_OFFSET)
                       << _UCS_FP8_EXPONENT_BITS);
}


/**
 * Unpack a byte to a double-precision floating-point number in a given range.
 *
 * @param value Unpack this number
 * @param min   Min supported value (assumed to be a power of 2)
 * @param max   Max supported value (assumed to be a power of 2)
 *
 * @return A double-precision floating-point number which approximates the
 *         original unpacked value
 */
static UCS_F_ALWAYS_INLINE double
ucs_fp8_unpack(ucs_fp8_t value, uint64_t min, uint64_t max)
{
    uint32_t hi = 0, lo = 0;
    uint32_t exp;
    uint32_t mant0;
    uint8_t exponent                = value & _UCS_FP8_EXPONENT_MASK;

    (void)max;

    if (ucs_unlikely(exponent == 0)) {
        exp = 0;
    } else if (ucs_unlikely(exponent == _UCS_FP8_NAN)) {
        exp = _UCS_FP8_IEEE_NAN_EXPONENT;
    } else {
        exp = exponent + _UCS_FP8_EXPONENT_OFFSET +
                                   ucs_ilog2(min);
    }
    mant0 = (uint32_t)(value >> _UCS_FP8_EXPONENT_BITS);
    mant0 = (mant0 << _UCS_FP_MANTISSA_OFFSET) & 0x000fffffu;

    hi = ((exp & 0x7ffu) << 20) | (mant0 & 0x000fffffu);
    lo = 0;

    return ucs_words_to_double(hi, lo);
}


/**
 * Declare a packed floating-point type.
 *
 * The packed type uses a portable and platform-independent underlying
 * representation (an 8-bit char), able to perform a (lossy) packing and
 * unpacking from a double (8-byte) type.
 *
 * The packed type is defined by the required min and max values -
 * the exponent is scaled accordingly, to accommodate the needed range.
 *
 * Special values (0 and NaN) are packed and unpacked in a loseless way.
 *
 * max/min <= 2^14 must hold, as only 4 bits are used for exponent representation.
 *
 * @param _name Packed type name
 * @param _min  Min supported number (assumed to be a power of 2)
 * @param _max  Max supported number (assumed to be a power of 2)
 */
#define UCS_FP8_DECLARE_TYPE(_name, _min, _max) \
    \
    static UCS_F_ALWAYS_INLINE ucs_fp8_t _UCS_FP8_IDENTIFIER(_name, _pack)( \
            double value) \
    { \
        /* 2 is subtracted because of special values for 0 and NaN */ \
        ucs_assert(ucs_ilog2((_max) / (_min)) < \
                   UCS_BIT(_UCS_FP8_EXPONENT_BITS) - 2); \
        return ucs_fp8_pack(value, _min, _max); \
    } \
    \
    static UCS_F_ALWAYS_INLINE double _UCS_FP8_IDENTIFIER(_name, _unpack)( \
            ucs_fp8_t value) \
    { \
        return ucs_fp8_unpack(value, _min, _max); \
    }


/**
 * Pack a double-precision floating-point number of a given type to a single byte.
 * The packing is lossy and the unpacked number is assumed to be
 * non-negative.
 *
 * @param _name  Packed type name
 * @param _value Pack this number
 *
 * @return A single byte which represents the given number
 */
#define UCS_FP8_PACK(_name, _value) _UCS_FP8_IDENTIFIER(_name, _pack)(_value)


/**
 * Unpack a byte to a double-precision floating-point number of a given type.
 *
 * @param _name  Packed type name
 * @param _value Unpack this number
 *
 * @return A double-precision floating-point number which approximates the
 *         original unpacked value
 */
#define UCS_FP8_UNPACK(_name, _value) \
    _UCS_FP8_IDENTIFIER(_name, _unpack)(_value)


/**
 * Pack a double-precision floating-point number of a given type to a single
 * byte and then unpack it. Can be useful for simulating the loss of precision
 * when comparing with an unpacked value.
 *
 * @param _name  Packed type name
 * @param _value Convert this number
 *
 * @return A double-precision floating-point number which approximates the
 *         original value
 */
#define UCS_FP8_PACK_UNPACK(_name, _value) \
    UCS_FP8_UNPACK(_name, UCS_FP8_PACK(_name, _value))


END_C_DECLS

#endif
