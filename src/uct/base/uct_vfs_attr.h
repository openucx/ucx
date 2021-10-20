/**
 * Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_VFS_ATTR_H_
#define UCT_VFS_ATTR_H_

#include <stdint.h>


typedef struct {
    uint64_t   flag;
    const char *name;
} uct_vfs_flag_info_t;


/**
 * Add read-only files representing flags to objects @a obj directory in VFS.
 *
 * @param [in] obj       Pointer to the object.
 * @param [in] obj_flags Capability flags of @a obj object.
 * @param [in] flags     Array of all object's capability flags.
 * @param [in] n         Size of @a flags array.
 */
void uct_vfs_init_flags(void *obj, uint64_t obj_flags,
                        const uct_vfs_flag_info_t *flags, unsigned long n);

#endif
