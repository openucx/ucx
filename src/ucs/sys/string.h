/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_STRING_H_
#define UCS_STRING_H_

#include "compiler_def.h"

#include <stddef.h>
#include <stdint.h>
#include <sys/socket.h>

BEGIN_C_DECLS

/** @file string.h */

/**
 * Expand a partial path to full path.
 *
 * @param path       Path to expand.
 * @param fullpath   Filled with full path.
 * @param max        Room in "fullpath"
 */
void ucs_expand_path(const char *path, char *fullpath, size_t max);


/**
 * Fill a filename template. The following values in the string are replaced:
 *  %p - replaced by process id
 *  %h - replaced by host name
 *
 * @param tmpl   File name template (possibly containing formatting sequences)
 * @param buf    Filled with resulting file name
 * @param max    Maximal size of destination buffer.
 */
void ucs_fill_filename_template(const char *tmpl, char *buf, size_t max);


/**
 * Format a string to a buffer of given size, and fill the rest of the buffer
 * with '\0'. Also, guarantee that the last char in the buffer is '\0'.
 *
 * @param buf  Buffer to format the string to.
 * @param size Buffer size.
 * @param fmt  Format string.
 */
void ucs_snprintf_zero(char *buf, size_t size, const char *fmt, ...)
    UCS_F_PRINTF(3, 4);


/**
 * Same as strncpy(), but guarantee that the last char in the buffer is '\0'.
 */
void ucs_strncpy_zero(char *dest, const char *src, size_t max);


/**
 * Return a number filled with the first characters of the string.
 */
uint64_t ucs_string_to_id(const char *str);


/**
 * Convert a memory units value to a string which is abbreviated if possible.
 * For example:
 *  1024 -> 1kb
 *
 * @param value  Value to convert.
 * @param buf    Buffer to place the string.
 * @param max    Maximal length of the buffer.
 */
void ucs_memunits_to_str(size_t value, char *buf, size_t max);


/**
 * Copy string limited by len bytes. Destination string is always ended by '\0'
 *
 * @param dst Destination buffer
 * @param src Source string
 * @param len Maximum string length to copy
 *
 * @return address of destination buffer
 */
char* ucs_strncpy_safe(char *dst, const char *src, size_t len);


/**
 * Remove whitespace characters in the beginning and end of the string, as
 * detected by isspace(3). Returns a pointer to the new string (which may be a
 * substring of 'str'). The original string 'str' may be modified in-place.
 *
 * @param str  String to remove whitespaces from.
 * @return Pointer to the new string, with leading/trailing whitespaces removed.
 */
char *ucs_strtrim(char *str);


END_C_DECLS

#endif
