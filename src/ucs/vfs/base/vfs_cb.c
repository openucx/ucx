/**
 * Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "vfs_cb.h"
#include <ucs/debug/log_def.h>
#include <ucs/sys/string.h>

void ucs_vfs_show_memory_address(void *obj, ucs_string_buffer_t *strb,
                                 void *arg_ptr, uint64_t arg_u64)
{
    ucs_string_buffer_appendf(strb, "%p\n", obj);
}

void ucs_vfs_show_primitive(void *obj, ucs_string_buffer_t *strb, void *arg_ptr,
                            uint64_t arg_u64)
{
    ucs_vfs_primitive_type_t type = arg_u64;
    unsigned long ulvalue;
    long lvalue;

    UCS_STATIC_ASSERT(UCS_VFS_TYPE_FLAG_UNSIGNED >= UCS_VFS_TYPE_LAST);
    UCS_STATIC_ASSERT(UCS_VFS_TYPE_FLAG_HEX >= UCS_VFS_TYPE_LAST);

    if (type == UCS_VFS_TYPE_POINTER) {
        ucs_string_buffer_appendf(strb, "%p\n", *(void**)arg_ptr);
    } else if (type == UCS_VFS_TYPE_STRING) {
        ucs_string_buffer_appendf(strb, "%s\n", (char*)arg_ptr);
    } else {
        switch (type & ~(UCS_VFS_TYPE_FLAG_UNSIGNED | UCS_VFS_TYPE_FLAG_HEX)) {
        case UCS_VFS_TYPE_CHAR:
            lvalue  = *(char*)arg_ptr;
            ulvalue = *(unsigned char*)arg_ptr;
            break;
        case UCS_VFS_TYPE_SHORT:
            lvalue  = *(short*)arg_ptr;
            ulvalue = *(unsigned short*)arg_ptr;
            break;
        case UCS_VFS_TYPE_INT:
            lvalue  = *(int*)arg_ptr;
            ulvalue = *(unsigned int*)arg_ptr;
            break;
        case UCS_VFS_TYPE_LONG:
            lvalue  = *(long*)arg_ptr;
            ulvalue = *(unsigned long*)arg_ptr;
            break;
        default:
            ucs_warn("vfs object %p attribute %p: incorrect type 0x%lx", obj,
                     arg_ptr, arg_u64);
            ucs_string_buffer_appendf(strb, "<unable to get the value>\n");
            return;
        }

        if (type & UCS_VFS_TYPE_FLAG_HEX) {
            ucs_string_buffer_appendf(strb, "%lx\n", ulvalue);
        } else if (type & UCS_VFS_TYPE_FLAG_UNSIGNED) {
            ucs_string_buffer_appendf(strb, "%lu\n", ulvalue);
        } else {
            ucs_string_buffer_appendf(strb, "%ld\n", lvalue);
        }
    }
}

void ucs_vfs_show_ulunits(void *obj, ucs_string_buffer_t *strb, void *arg_ptr,
                          uint64_t arg_u64)
{
    char buf[64];

    ucs_config_sprintf_ulunits(buf, sizeof(buf), arg_ptr, NULL);
    ucs_string_buffer_appendf(strb, "%s\n", buf);
}

void ucs_vfs_show_memunits(void *obj, ucs_string_buffer_t *strb, void *arg_ptr,
                           uint64_t arg_u64)
{
    char buf[64];

    ucs_memunits_to_str(*(size_t*)arg_ptr, buf, sizeof(buf));
    ucs_string_buffer_appendf(strb, "%s\n", buf);
}
