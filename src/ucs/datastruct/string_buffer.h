/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_STRING_BUFFER_H_
#define UCS_STRING_BUFFER_H_

#include <ucs/sys/compiler_def.h>
#include <ucs/type/status.h>
#include <stddef.h>


BEGIN_C_DECLS

/**
 * String buffer - a dynamic NULL-terminated character buffer which can grow
 * on demand.
 */
typedef struct ucs_string_buffer {
    char        *buffer;  /* Buffer pointer */
    size_t      length;   /* Actual string length */
    size_t      capacity; /* Allocated memory size */
} ucs_string_buffer_t;


/**
 * Initialize a string buffer
 *
 * @param [out] strb   String buffer to initialize.
 */
void ucs_string_buffer_init(ucs_string_buffer_t *strb);


/**
 * Cleanup a string buffer and release any memory associated with it.
 *
 * @param [out] strb   String buffer to clean up.
 */
void ucs_string_buffer_cleanup(ucs_string_buffer_t *strb);


/**
 * Append a formatted string to the string buffer.
 *
 * @param [inout] strb   String buffer to append to.
 * @param [in]    fmt    Format string.
 *
 * @return UCS_OK on success or UCS_ERR_NO_MEOMRY if could not allocate memory
 * to grow the string.
 */
ucs_status_t ucs_string_buffer_appendf(ucs_string_buffer_t *strb,
                                       const char *fmt, ...)
    UCS_F_PRINTF(2, 3);


/**
 * Remove specific characters from the end of the string.
 *
 * @param [inout] strb     String buffer remote characters from.
 * @param [in]    charset  C-string with the set of characters to remove.
 *                         If NULL, this function removes whitespace characters,
 *                         as defined by isspace (3).
 *
 * This function removes the largest contiguous suffix from the input string
 * 'strb', which consists entirely of characters in 'charset'.
 */
void ucs_string_buffer_rtrim(ucs_string_buffer_t *strb, const char *charset);


/**
 * Return a temporary pointer to a C-style string which represents the string
 * buffer. The returned string is valid only as long as no other operation is
 * done on the string buffer (including append).
 *
 * @param [in]   strb   String buffer to convert to a C-style string
 *
 * @return C-style string representing the data in the buffer.
 */
const char *ucs_string_buffer_cstr(const ucs_string_buffer_t *strb);


END_C_DECLS

#endif
