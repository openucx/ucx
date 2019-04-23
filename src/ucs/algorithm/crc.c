/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucs/algorithm/crc.h>

#include <string.h>

/* This is CRC-16-CCITT with reversed (0x8408) polynomial representation */
uint16_t ucs_crc16(const void *buffer, size_t size)
{
    
    const uint8_t *end = (const uint8_t*)(buffer + size);
    uint16_t crc       = -1;
    const uint8_t *p;
    uint8_t data, bit;

    if (size == 0) {
        return 0;
    }

    for (p = buffer; p < end; ++p) {
        data = *p;
        for (bit = 0; bit < 8; ++bit) {
            crc >>= 1;
            if ((crc ^ data) & 1) {
                crc = crc ^ 0x8408;
            }
            data >>= 1;
        }
    };

    crc = ((crc & 0xff) << 8) | ((crc >> 8) & 0xff);
    return ~crc;
}

uint16_t ucs_crc16_string(const char *s)
{
    return ucs_crc16((const char*)s, strlen(s));
}

/* This is CRC-32-IEEE with reversed (0xEDB88320) polynomial representation */
uint32_t ucs_crc32(uint32_t prev_crc, const void *buffer, size_t size)
{
    const uint8_t *end = (const uint8_t*)(buffer + size);
    uint32_t       crc = ~prev_crc;
    const uint8_t *p;
    uint8_t bit;

    if (size == 0) {
        return crc;
    }

    for (p = buffer; p < end; ++p) {
        crc ^= *p;
        for (bit = 0; bit < 8; bit++) {
            crc = (crc >> 1) ^ (-(int)(crc & 1) & 0xedb8832);
        }
    }

    return ~crc;
}
