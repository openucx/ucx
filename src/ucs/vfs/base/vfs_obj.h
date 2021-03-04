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

typedef void (*ucs_vfs_file_show_cb_t)(void *obj, ucs_string_buffer_t *strb);


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

END_C_DECLS

#endif
