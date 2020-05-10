/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/algorithm/crc.h>

#include <string.h>


/* CRC-16-CCITT */
#define UCS_CRC16_POLY    0x8408u

/* CRC-32 (ISO 3309) */
#define UCS_CRC32_POLY    0xedb88320l

#define UCS_CRC_CALC(_width, _buffer, _size, _crc) \
    do { \
        const uint8_t *end = (const uint8_t*)(UCS_PTR_BYTE_OFFSET(_buffer, _size)); \
        const uint8_t *p; \
        uint8_t bit; \
        \
        if ((_size) != 0) { \
            for (p = (_buffer); p < end; ++p) { \
                (_crc) ^= *p; \
                for (bit = 0; bit < 8; ++bit) { \
                    (_crc) = ((_crc) >> 1) ^ (-(int)((_crc) & 1) & \
                                              UCS_CRC ## _width ## _POLY); \
                } \
            } \
        } \
        (_crc) = ~(_crc); \
    } while (0)


uint16_t ucs_crc16(const void *buffer, size_t size)
{
    uint16_t crc = UINT16_MAX;
    UCS_CRC_CALC(16, buffer, size, crc);
    return crc;
}

uint16_t ucs_crc16_string(const char *s)
{
    return ucs_crc16((const char*)s, strlen(s));
}

uint32_t ucs_crc32(uint32_t prev_crc, const void *buffer, size_t size)
{
    uint32_t crc = ~prev_crc;
    UCS_CRC_CALC(32, buffer, size, crc);
    return crc;
}
