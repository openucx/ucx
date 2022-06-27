/**
 * Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_STRING_DISTANCE_H_
#define UCS_STRING_DISTANCE_H_

#include <ucs/sys/compiler_def.h>
#include <stddef.h>

BEGIN_C_DECLS

/** @file string_distance.h */

/**
 * Calculate Levenshtein distance between two strings.
 *
 * @param [in]  str1  First NULL-terminated string.
 * @param [in]  str2  Second NULL-terminated string.
 *
 * @return Distance between the strings.
 */
size_t ucs_string_distance(const char *str1, const char *str2);

END_C_DECLS

#endif
