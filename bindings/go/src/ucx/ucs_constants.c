/*
 * Copyright (C) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "ucs_constants.h"

#include <ucs/sys/string.h>
#include <stdlib.h>

const char* ucxgo_get_ucs_mem_type_name(ucs_memory_type_t idx) {
    return ucs_memory_type_names[idx];
}

ssize_t ucxgo_parse_ucs_mem_type_name(void* value) {
    ssize_t idx = ucs_string_find_in_list((const char *)value, ucs_memory_type_names, 0);
    free(value);
    return idx;
}
