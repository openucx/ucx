/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_STRING_SET_H_
#define UCS_STRING_SET_H_

#include <ucs/datastruct/khash.h>
#include <ucs/datastruct/string_buffer.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/type/status.h>
#include <stddef.h>

BEGIN_C_DECLS


/*
 * Define ucs_string_set_t as a khash/set type
 */
KHASH_INIT(ucs_string_set, char*, char, 0, kh_str_hash_func, kh_str_hash_equal)
typedef khash_t(ucs_string_set) ucs_string_set_t;


/**
 * Initialize a string set
 *
 * @param [out] sset   String set to initialize.
 */
void ucs_string_set_init(ucs_string_set_t *sset);


/**
 * Cleanup a string set and release any memory associated with it.
 *
 * @param [out] sset   String set to clean up.
 */
void ucs_string_set_cleanup(ucs_string_set_t *sset);


/**
 * Add a copy of a string to the string set
 *
 * @param [inout] sset  String set to add to.
 * @param [in]    str   String to add. The passed string can be released
 *                      immediately after this call, since the contents of the
 *                      string are copied to an internal buffer.
 *
 * @param UCS_OK if successful, or UCS_ERR_NO_MEMORY if could not allocate
 *         enough memory to add the string.
 */
ucs_status_t ucs_string_set_add(ucs_string_set_t *sset, const char *str);


/**
 * Add a formatted string to the string set
 *
 * @param [inout] sset  String set to add to.
 * @param [in]    fmt   Format string to add.
 *
 * @param UCS_OK if successful, or UCS_ERR_NO_MEMORY if could not allocate
 *         enough memory to add the string.
 */
ucs_status_t ucs_string_set_addf(ucs_string_set_t *sset, const char *fmt, ...)
    UCS_F_PRINTF(2, 3);


/**
 * Check whether a string set contains a given string
 *
 * @param [in]   sset   String set to check.
 * @param [in]   str    String to check if contained in the set.
 *
 * @return Nonzero if the string is contained in the set, 0 otherwise.
 */
int ucs_string_set_contains(const ucs_string_set_t *sset, const char *str);


/**
 * Print set contents to a string buffer in a lexicographical order
 *
 * @param [in]    sset   String set whose contents to print.
 * @param [inout] strb   Append the strings in the set to this string buffer.
 * @param [in]    sep    Separator string to insert between every two printed
 *                       strings, for example: ","
 *
 * @param UCS_OK if successful, or UCS_ERR_NO_MEMORY if could not allocate
 *         enough memory to sort the set or to grow the string buffer.
 */
ucs_status_t ucs_string_set_print_sorted(const ucs_string_set_t *sset,
                                         ucs_string_buffer_t *strb,
                                         const char *sep);


END_C_DECLS

#endif
