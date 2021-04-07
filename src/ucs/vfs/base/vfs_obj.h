/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_VFS_H_
#define UCS_VFS_H_

#include <ucs/datastruct/string_buffer.h>
#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS

/* This header file defines API for manipulating VFS object tree structure */

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
 * Function to fill buffer @a strb with the information about @a obj.
 * 
 * @param [in]    obj  Pointer to the object to be described.
 * @param [inout] strb String buffer to be filled by the description of the
 *                     object @a obj.
 */
typedef void (*ucs_vfs_file_show_cb_t)(void *obj, ucs_string_buffer_t *strb);


/**
 * Callback function to show memory address of object.
 *
 * @param [in]    obj  Pointer to the object.
 * @param [inout] strb String buffer to be filled by memory address of @a obj.
 */
void ucs_vfs_memory_address_show_cb(void *obj, ucs_string_buffer_t *strb);


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
 * @param [in] rel_path Format string which specifies relative path to the file.
 */
void ucs_vfs_obj_add_ro_file(void *obj, ucs_vfs_file_show_cb_t text_cb,
                             const char *rel_path, ...) UCS_F_PRINTF(3, 4);


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
 * @param [in]  path       String wich specifies path to find the node in VFS.
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
 * @param [in]    path     String wich specifies path to find the node in VFS.
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
 * @param [in] path        String wich specifies path to find the node in VFS.
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
