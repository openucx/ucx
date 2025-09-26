/**
 * Copyright (C) 2022 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "string_distance.h"
#include <ucs/sys/math.h>
#include <ucs/sys/sys.h>
#include <string.h>


size_t ucs_string_distance(const char *str1, const char *str2)
{
    const size_t len1 = strlen(str1);
    const size_t len2 = strlen(str2);
    size_t *distances = ucs_alloca((len1 + 1) * sizeof(*distances));
    size_t distance_backup, distance_prev, min_prev, addition;
    size_t i, j;

    /* We explicitly init distances[len1] to prevent static
     * analysis false positive (uninitialized return value) */
    distances[len1] = len1;
    for (j = 1; j <= len1; ++j) {
        distances[j] = j;
    }

    for (i = 1; i <= len2; ++i) {
        distances[0]  = i;
        distance_prev = i - 1;

        for (j = 1; j <= len1; ++j) {
            distance_backup = distances[j];
            min_prev        = ucs_min(distances[j] + 1, distances[j - 1] + 1);
            addition        = str1[j - 1] != str2[i - 1];
            distances[j]    = ucs_min(min_prev, distance_prev + addition);
            distance_prev   = distance_backup;
        }
    }

    return distances[len1];
}
