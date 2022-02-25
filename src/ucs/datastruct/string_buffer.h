/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_STRING_BUFFER_H_
#define UCS_STRING_BUFFER_H_

#include <ucs/sys/compiler_def.h>
#include <ucs/type/status.h>
#include <ucs/datastruct/array.h>
#include <sys/uio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>


BEGIN_C_DECLS

UCS_ARRAY_DECLARE_TYPE(string_buffer, size_t, char)


/**
 * Dynamic string buffer initializer. The backing storage should be released
 * explicitly by calling @ref ucs_string_buffer_cleanup()
 */
#define UCS_STRING_BUFFER_INITIALIZER \
    { \
        UCS_ARRAY_DYNAMIC_INITIALIZER \
    }


/**
 * Declare a string buffer which is using an existing string as backing store.
 * Such string buffer does not allocate additional memory and does not have to
 * be cleaned-up, and it can also be used to build a string onto existing
 * C-string buffer passed as a function argument.
 *
 * @param _var       String buffer variable name
 * @param _buffer    Buffer to use as backing store.
 * @param _capacity  Buffer capacity.
 *
 * Example:
 *
 * @code{.c}
 * char * build_my_string(char *buffer, size_t max_length)
 * {
 *    UCS_STRING_BUFFER_FIXED(strb, buffer, max_length);
 *    ucs_string_buffer_appendf(&strb, "%x%x", 57005, 48879);
 *    return buffer;
 * }
 * @endcode
 */
#define UCS_STRING_BUFFER_FIXED(_var, _buffer, _capacity) \
    ucs_string_buffer_t _var = { \
        UCS_ARRAY_FIXED_INITIALIZER(_buffer, _capacity) \
    }


/**
 * Declare a string buffer which is using a static array as backing store.
 * Such string buffer does not allocate additional memory and does not have to
 * be cleaned-up.
 *
 * @param _var     String buffer variable name
 * @param _buffer  Buffer to use as backing store.
 *
 * Example:
 *
 * @code{.c}
 * char buffer[100];
 * UCS_STRING_BUFFER_FIXED(strb, buffer);
 *
 * ucs_string_buffer_appendf(&strb, "%x%x", 57005, 48879);
 * @endcode
 */
#define UCS_STRING_BUFFER_STATIC(_var, _buffer) \
    UCS_STRING_BUFFER_FIXED(_var, _buffer, ucs_static_array_size(_buffer))


#define UCS_STRING_BUFFER_ONSTACK(_var, _capacity) \
    UCS_STRING_BUFFER_FIXED(_var, \
                            UCS_ARRAY_ALLOC_ONSTACK(string_buffer, _capacity), \
                            _capacity)


/**
 * String buffer - a dynamic NULL-terminated character buffer which can grow
 * on demand.
 */
typedef struct ucs_string_buffer {
    ucs_array_t(string_buffer) str;
} ucs_string_buffer_t;


/**
 * Initialize a string buffer
 *
 * @param [out] strb   String buffer to initialize.
 */
void ucs_string_buffer_init(ucs_string_buffer_t *strb);


/**
 * Initialize a string buffer with fixed-size buffer as backing storage.
 *
 * @param [out] strb      String buffer to initialize.
 * @param [in]  buffer    Buffer to use as backing storage.
 * @param [in]  capacity  Buffer size.
 */
void ucs_string_buffer_init_fixed(ucs_string_buffer_t *strb, char *buffer,
                                  size_t capacity);


/**
 * Cleanup a string buffer and release any memory associated with it.
 *
 * @param [out] strb   String buffer to clean up.
 */
void ucs_string_buffer_cleanup(ucs_string_buffer_t *strb);


/**
 * Reset a string buffer to initial empty state.
 *
 * @param [out] strb   String buffer reset.
 */
void ucs_string_buffer_reset(ucs_string_buffer_t *strb);


/**
 * Get the number of characters in a string buffer
 *
 * @param [out] strb   Return the length of this string buffer.
 *
 * @return String buffer length.
 */
size_t ucs_string_buffer_length(ucs_string_buffer_t *strb);


/**
 * Append a formatted string to the string buffer.
 *
 * @param [inout] strb   String buffer to append to.
 * @param [in]    fmt    Format string.
 *
 * @note If the string cannot grow to the required length, only some of the
 *       characters would be appended.
 */
void ucs_string_buffer_appendf(ucs_string_buffer_t *strb, const char *fmt, ...)
    UCS_F_PRINTF(2, 3);


/**
 * Append a hex dump to the string buffer.
 *
 * @param [inout] strb       String buffer to append to.
 * @param [in]    data       Raw data to hex-dump.
 * @param [in]    size       Raw data size.
 * @param [in]    per_line   Add a newline character after this number of bytes.
 *
 * @note If the string cannot grow to the required length, only some of the
 *       characters would be appended.
 */
void ucs_string_buffer_append_hex(ucs_string_buffer_t *strb, const void *data,
                                  size_t size, size_t per_line);


/**
 * Append a flag bitmask representation to the string buffer.
 *
 * @param [inout] strb        String buffer to append to.
 * @param [in]    mask        Append the representation of this mask value.
 * @param [in]    flag_names  If non-NULL, use this array as flag names.
 */
void ucs_string_buffer_append_flags(ucs_string_buffer_t *strb, uint64_t mask,
                                    const char **flag_names);


/**
 * Append an IO vector representation to the string buffer.
 *
 * @param [inout] strb        String buffer to append to.
 * @param [in]    iov         Pointer to an IO vector.
 * @param [in]    iovcnt      Number of entries in the IO vector.
 */
void ucs_string_buffer_append_iovec(ucs_string_buffer_t *strb,
                                    const struct iovec *iov, size_t iovcnt);


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
 * @param [in]   strb   String buffer to convert to a C-style string.
 *
 * @return C-style string representing the data in the buffer.
 */
const char *ucs_string_buffer_cstr(const ucs_string_buffer_t *strb);


/**
 * Print the string buffer to a stream as multi-line text.
 *
 * @param [in]  strb          String buffer to print.
 * @param [in]  line_prefix   Prefix to prepend to each output line.
 * @param [in]  stream        Stream to print to.
 */
void ucs_string_buffer_dump(const ucs_string_buffer_t *strb,
                            const char *line_prefix, FILE *stream);


/**
 * Return a pointer to a C-style string which represents the string buffer. The
 * returned pointer should be freed with method which deallocates memory, e.g.
 * ucs_free. There is no need to call ucs_string_buffer_cleanup in case of
 * extracting memory using this method.
 *
 * @param [inout] strb String buffer to convert to a C-style string.
 *
 * @return C-style string representing the data in the buffer.
 */
char *ucs_string_buffer_extract_mem(ucs_string_buffer_t *strb);


/**
 * Get the next token from the string. This operation can overwrite some of the
 * string with '\0' characters, to separate the tokens.
 *
 * @param [in]  strb        String buffer to get next token from.
 * @param [in]  token       Pointer to the current token, or NULL to start
 *                          from the beginning.
 * @param [in]  delimiters  Set of characters that separate between tokens.
 *
 * @return Pointer to the next token, after the given @a token, or NULL if no
 *         more tokens are found.
 */
char *ucs_string_buffer_next_token(ucs_string_buffer_t *strb, char *token,
                                   const char *delimiters);


/**
 * Append repeat character to a string buffer.
 *
 * @param [inout] strb     String buffer to append characters to.
 * @param [in]    c        Character to append.
 * @param [in]    count    Number of times to append @a c.
 */
void ucs_string_buffer_appendc(ucs_string_buffer_t *strb, int c, size_t count);


/**
 * Split the string to tokens and iterate over them. This operation can
 * overwrite some of the string with '\0' characters.
 *
 * @param _tok    A variable of type 'char *' which will be assigned to the
 *                current token.
 * @param _strb   String to iterate over.
 * @param _delim  Set of characters that separate between tokens.
 */
#define ucs_string_buffer_for_each_token(_tok, _strb, _delim) \
    for (_tok = ucs_string_buffer_next_token(_strb, NULL, _delim); \
         _tok != NULL; \
         _tok = ucs_string_buffer_next_token(_strb, _tok, _delim))

END_C_DECLS

#endif
