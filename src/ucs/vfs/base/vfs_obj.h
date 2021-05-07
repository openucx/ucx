/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_VFS_H_
#define UCS_VFS_H_

#include <ucs/datastruct/string_buffer.h>
#include <ucs/sys/compiler_def.h>
#include <stdint.h>

BEGIN_C_DECLS

/* This header file defines API for manipulating VFS object tree structure */


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
    UCS_VFS_TYPE_UNSIGNED = UCS_BIT(14),
    UCS_VFS_TYPE_HEX      = UCS_BIT(15),

    /* Convenience flags */
    UCS_VFS_TYPE_I8       = UCS_VFS_TYPE_CHAR,
    UCS_VFS_TYPE_U8       = UCS_VFS_TYPE_UNSIGNED | UCS_VFS_TYPE_CHAR,
    UCS_VFS_TYPE_I16      = UCS_VFS_TYPE_SHORT,
    UCS_VFS_TYPE_U16      = UCS_VFS_TYPE_UNSIGNED | UCS_VFS_TYPE_SHORT,
    UCS_VFS_TYPE_I32      = UCS_VFS_TYPE_INT,
    UCS_VFS_TYPE_U32      = UCS_VFS_TYPE_UNSIGNED | UCS_VFS_TYPE_INT,
    UCS_VFS_TYPE_U32_HEX  = UCS_VFS_TYPE_U32 | UCS_VFS_TYPE_HEX,
    UCS_VFS_TYPE_ULONG    = UCS_VFS_TYPE_UNSIGNED | UCS_VFS_TYPE_LONG,
    UCS_VFS_TYPE_SSIZET   = UCS_VFS_TYPE_LONG,
    UCS_VFS_TYPE_SIZET    = UCS_VFS_TYPE_ULONG
} ucs_vfs_primitive_type_t;


/**
 * Structure to describe the vfs node.
 */
typedef struct {
    /**
     * Size of the content in case of read-only file, and number of child
     * directories if node is directory.
     */
    size_t size;
    /**
     * File mode can be either regular file (S_IFREG) or directory (S_IFDIR)
     * depending of the type of the vfs node.
     */
    int    mode;
} ucs_vfs_path_info_t;


/**
 * Function type to fill information about an object to the string buffer.
 *
 * @param [in]    obj      Pointer to the object.
 * @param [inout] strb     String buffer filled with the object's information.
 * @param [in]    arg_ptr  Optional pointer argument passed to the function.
 * @param [in]    arg_u64  Optional numeric argument passed to the function.
 */
typedef void (*ucs_vfs_file_show_cb_t)(void *obj, ucs_string_buffer_t *strb,
                                       void *arg_ptr, uint64_t arg_u64);


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
 * buffer. The function handles 'auto' and 'infinty' values.
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
 * handles 'auto' and 'infinty' values.
 *
 * @param [in]    obj      Pointer to the object.
 * @param [inout] strb     String buffer filled with the object's information.
 * @param [in]    arg_ptr  Pointer to the memory unit value.
 * @param [in]    arg_u64  Unused.
 */
void ucs_vfs_show_memunits(void *obj, ucs_string_buffer_t *strb, void *arg_ptr,
                           uint64_t arg_u64);


/**
 * Function to update representation of object in VFS.
 *
 * @param [in] obj Pointer to the object to be updated.
 */
typedef void (*ucs_vfs_refresh_cb_t)(void *obj);


/**
 * Function to process VFS nodes during reading of the parent directory.
 *
 * @param [in] name Path to directory.
 * @param [in] arg  Pointer to the arguments.
 */
typedef void (*ucs_vfs_list_dir_cb_t)(const char *name, void *arg);


/**
 * Add directory representing object in VFS. If @a parent_obj is NULL, the mount
 * directory will be used as the base for @a rel_path.
 *
 * @param [in] parent_obj Pointer to the parent object. @a rel_path is relative
 *                        to @a parent_obj directory.
 * @param [in] obj        Pointer to the object to be represented in VFS.
 * @param [in] rel_path   Format string which specifies relative path
 *                        @a obj directory.
 */
void ucs_vfs_obj_add_dir(void *parent_obj, void *obj, const char *rel_path, ...)
        UCS_F_PRINTF(3, 4);


/**
 * Add read-only file describing object features in VFS. If @a obj is NULL, the
 * mount directory will be used as the base for @a rel_path.
 *
 * @param [in] obj      Pointer to the object. @a rel_path is relative to @a obj
 *                      directory.
 * @param [in] text_cb  Callback method that generates the content of the file.
 * @param [in] arg_ptr  Optional pointer argument that is passed to the callback
 *                      method.
 * @param [in] arg_u64  Optional numeric argument that is passed to the callback
 *                      method.
 * @param [in] rel_path Format string which specifies relative path to the file.
 */
void ucs_vfs_obj_add_ro_file(void *obj, ucs_vfs_file_show_cb_t text_cb,
                             void *arg_ptr, uint64_t arg_u64,
                             const char *rel_path, ...) UCS_F_PRINTF(5, 6);


/**
 * Recursively remove directories and files associated with the object and its
 * children from VFS. The method removes all empty parent sub-directories.
 *
 * @param [in] obj Pointer to the object to be deleted with its children from
 *                 VFS.
 */
void ucs_vfs_obj_remove(void *obj);


/**
 * Invalidate VFS node and set method to update the node.
 *
 * @param [in] obj        Pointer to the object to be invalidate.
 * @param [in] refresh_cb Method to update the node associated with the object.
 */
void ucs_vfs_obj_set_dirty(void *obj, ucs_vfs_refresh_cb_t refresh_cb);


/**
 * Fill information about VFS node corresponding to the specified path.
 *
 * @param [in]  path       String which specifies path to find the node in VFS.
 * @param [out] info       VFS object information.
 *
 * @return UCS_OK          VFS node corresponding to specified path exists.
 *         UCS_ERR_NO_ELEM Otherwise.
 *
 * @note The content of the file defined by ucs_vfs_file_show_cb_t of the node.
 *       The method initiates refresh of the node defined by
 *       ucs_vfs_refresh_cb_t of the node.
 */
ucs_status_t ucs_vfs_path_get_info(const char *path, ucs_vfs_path_info_t *info);


/**
 * Read the content of VFS node corresponding to the specified path. The content
 * of the file defined by ucs_vfs_file_show_cb_t of the node.
 *
 * @param [in]    path     String which specifies path to find the node in VFS.
 * @param [inout] strb     String buffer to be filled by the content of the
 *                         file.
 *
 * @return UCS_OK          VFS node corresponding to specified path exists and
 *                         the node is a file.
 * @return UCS_ERR_NO_ELEM Otherwise.
 */
ucs_status_t
ucs_vfs_path_read_file(const char *path, ucs_string_buffer_t *strb);


/**
 * Invoke callback @a dir_cb for children of VFS node corresponding to the
 * specified path.
 *
 * @param [in] path        String which specifies path to find the node in VFS.
 * @param [in] dir_cb      Callback method to be invoked for each child of the
 *                         VFS node.
 * @param [in] arg         Arguments to be passed to the callback method.
 *
 * @return UCS_OK          VFS node corresponding to specified path exists and
 *                         the node is a directory.
 *         UCS_ERR_NO_ELEM Otherwise.
 *
 * @note The method initiates refresh of the node defined by
 *       ucs_vfs_refresh_cb_t of the node.
 */
ucs_status_t ucs_vfs_path_list_dir(const char *path,
                                   ucs_vfs_list_dir_cb_t dir_cb, void *arg);

END_C_DECLS

#endif
