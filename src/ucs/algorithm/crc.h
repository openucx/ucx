/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_ALGORITHM_CRC_H_
#define UCS_ALGORITHM_CRC_H_

#include <stddef.h>
#include <stdint.h>

#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS

/**
 * Calculate CRC16 of an arbitrary buffer.
 *
 * @param [in]  buffer  Buffer to compute crc for.
 * @param [in]  size    Buffer size.
 *
 * @return crc16() function of the buffer.
 */
uint16_t ucs_crc16(const void *buffer, size_t size);


/**
 * Calculate CRC16 of a NULL-terminated string.
 */
uint16_t ucs_crc16_string(const char *s);

END_C_DECLS

#endif
