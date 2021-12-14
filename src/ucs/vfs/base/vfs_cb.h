/**
 * Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_VFS_CB_H_
#define UCS_VFS_CB_H_

#include <ucs/datastruct/string_buffer.h>
#include <ucs/sys/compiler_def.h>
#include <stdint.h>

BEGIN_C_DECLS

/* Defines type of primitive variables */
typedef enum {
    /* Basic type definitions */
    UCS_VFS_TYPE_POINTER,
    UCS_VFS_TYPE_STRING,
    UCS_VFS_TYPE_CHAR,
    UCS_VFS_TYPE_SHORT,
    UCS_VFS_TYPE_INT,
    UCS_VFS_TYPE_LONG,
    UCS_VFS_TYPE_LAST,

    /* Type modifiers */
    UCS_VFS_TYPE_FLAG_UNSIGNED = UCS_BIT(14),
    UCS_VFS_TYPE_FLAG_HEX      = UCS_BIT(15),

    /* Convenience flags */
    UCS_VFS_TYPE_I8      = UCS_VFS_TYPE_CHAR,
    UCS_VFS_TYPE_U8      = UCS_VFS_TYPE_FLAG_UNSIGNED | UCS_VFS_TYPE_CHAR,
    UCS_VFS_TYPE_I16     = UCS_VFS_TYPE_SHORT,
    UCS_VFS_TYPE_U16     = UCS_VFS_TYPE_FLAG_UNSIGNED | UCS_VFS_TYPE_SHORT,
    UCS_VFS_TYPE_I32     = UCS_VFS_TYPE_INT,
    UCS_VFS_TYPE_U32     = UCS_VFS_TYPE_FLAG_UNSIGNED | UCS_VFS_TYPE_INT,
    UCS_VFS_TYPE_U32_HEX = UCS_VFS_TYPE_U32 | UCS_VFS_TYPE_FLAG_HEX,
    UCS_VFS_TYPE_ULONG   = UCS_VFS_TYPE_FLAG_UNSIGNED | UCS_VFS_TYPE_LONG,
    UCS_VFS_TYPE_SSIZET  = UCS_VFS_TYPE_LONG,
    UCS_VFS_TYPE_SIZET   = UCS_VFS_TYPE_ULONG
} ucs_vfs_primitive_type_t;


/**
 * Callback function to fill the memory address of an object to the string
 * buffer.
 *
 * @param [in]    obj      Pointer to the object.
 * @param [inout] strb     String buffer filled with the object's information.
 * @param [in]    arg_ptr  Unused.
 * @param [in]    arg_u64  Unused.
 */
void ucs_vfs_show_memory_address(void *obj, ucs_string_buffer_t *strb,
                                 void *arg_ptr, uint64_t arg_u64);


/**
 * Callback function to show a variable of a primitive C type.
 *
 * @param [in]    obj      Pointer to the object.
 * @param [inout] strb     String buffer filled with the object's information.
 * @param [in]    arg_ptr  Points to the variable to show.
 * @param [in]    arg_u64  Specifies type flags for the variable, as defined in
 *                         @ref ucs_vfs_primitive_type_t.
 */
void ucs_vfs_show_primitive(void *obj, ucs_string_buffer_t *strb, void *arg_ptr,
                            uint64_t arg_u64);


/**
 * Callback function to fill a value of an unsigned long type to the string
 * buffer. The function handles 'auto' and 'infinity' values.
 *
 * @param [in]    obj      Pointer to the object.
 * @param [inout] strb     String buffer filled with the object's information.
 * @param [in]    arg_ptr  Pointer to the value of an unsigned long type.
 * @param [in]    arg_u64  Unused.
 */
void ucs_vfs_show_ulunits(void *obj, ucs_string_buffer_t *strb, void *arg_ptr,
                          uint64_t arg_u64);


/**
 * Callback function to fill memory units to the string buffer. The function
 * handles 'auto' and 'infinity' values.
 *
 * @param [in]    obj      Pointer to the object.
 * @param [inout] strb     String buffer filled with the object's information.
 * @param [in]    arg_ptr  Pointer to the memory unit value.
 * @param [in]    arg_u64  Unused.
 */
void ucs_vfs_show_memunits(void *obj, ucs_string_buffer_t *strb, void *arg_ptr,
                           uint64_t arg_u64);

END_C_DECLS

#endif
