/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "crc.h"

#include <string.h>


uint16_t ucs_crc16(const void *buffer, size_t size)
{
    const uint8_t *p;
    uint16_t result;
    uint8_t data;
    int bit;

    if (size == 0) {
        return 0;
    }

    result = -1;
    for (p = buffer; p < (const uint8_t*)(buffer + size); ++p) {
        data = *p;
        for (bit = 0; bit < 8; ++bit) {
            result >>= 1;
            if ((result ^ data) & 1) {
                result = result ^ 0x8048;
            }
            data >>= 1;
        }
    };

    result = ((result & 0xff) << 8) | ((result >> 8) & 0xff);
    return ~result;
}

uint16_t ucs_crc16_string(const char *s)
{
    return ucs_crc16((char*)s, strlen(s));
}
