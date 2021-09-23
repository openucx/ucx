/**
 * Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "uct_vfs_attr.h"
#include <ucs/vfs/base/vfs_obj.h>


static void uct_md_vfs_read_flag(void *obj, ucs_string_buffer_t *strb,
                                 void *arg_ptr, uint64_t arg_u64)
{
    ucs_string_buffer_appendf(strb, "1\n");
}

void uct_vfs_init_flags(void *obj, uint64_t obj_flags,
                        const uct_vfs_flag_info_t *flags, unsigned long n)
{
    unsigned long i;

    for (i = 0; i < n; ++i) {
        if (obj_flags & flags[i].flag) {
            ucs_vfs_obj_add_ro_file(obj, uct_md_vfs_read_flag, NULL, 0,
                                    "capability/flag/%s", flags[i].name);
        }
    }
}
