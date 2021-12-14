/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_ALGORITHM_CRC_H_
#define UCS_ALGORITHM_CRC_H_

#include <ucs/sys/compiler_def.h>

#include <stddef.h>
#include <stdint.h>

BEGIN_C_DECLS

/** @file crc.h */

/**
 * Calculate CRC16 of an arbitrary buffer.
 *
 * @param [in]  buffer     Buffer to compute crc for.
 * @param [in]  size       Buffer size.
 *
 * @return crc16() function of the buffer.
 */
uint16_t ucs_crc16(const void *buffer, size_t size);


/**
 * Calculate CRC16 of a NULL-terminated string.
 *
 * @param [in]  s          NULL-terminated string to compute crc for.
 *
 * @return crc16() function of the string.
 */
uint16_t ucs_crc16_string(const char *s);


/**
 * Calculate CRC32 of an arbitrary buffer.
 *
 * @param [in]  prev_crc   Initial CRC value.
 * @param [in]  buffer     Buffer to compute crc for.
 * @param [in]  size       Buffer size.
 *
 * @return crc32() function of the buffer.
 */
uint32_t ucs_crc32(uint32_t prev_crc, const void *buffer, size_t size);

END_C_DECLS

#endif
