/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "vfs_obj.h"

#include <ucs/datastruct/khash.h>
#include <ucs/datastruct/list.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/log_def.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/type/init_once.h>
#include <ucs/type/spinlock.h>
#include <ucs/sys/string.h>
#include <stdarg.h>
#include <stdint.h>
#include <sys/stat.h>


typedef enum {
    UCS_VFS_NODE_TYPE_DIR,
    UCS_VFS_NODE_TYPE_RO_FILE,
    UCS_VFS_NODE_TYPE_RW_FILE,
    UCS_VFS_NODE_TYPE_SUBDIR,
    UCS_VFS_NODE_TYPE_SYM_LINK,
    UCS_VFS_NODE_TYPE_LAST
} ucs_vfs_node_type_t;


#define UCS_VFS_FLAGS_DIRTY UCS_BIT(0)


typedef struct ucs_vfs_node ucs_vfs_node_t;
struct ucs_vfs_node {
    /* Node type. ucs_vfs_node_type_t contains possible node types. */
    ucs_vfs_node_type_t type;
    /* Reference count. Increase the value to avoid deletion while executing
       updating callbacks. */
    int                 refcount;
    /* Node flags. E.g. to indicate need for an update. */
    uint8_t             flags;
    /* Pointer to an object describing by the node. */
    void                *obj;
    /* Pointer to parent node in VFS hierarchy. */
    ucs_vfs_node_t      *parent;
    /* List of children in VFS hierarchy. */
    ucs_list_link_t     children;
    /* List item to represent the node in parent's list of children. */
    ucs_list_link_t     list;
    union {
        /* Callback method to read content of the file. */
        ucs_vfs_file_read_cb_t read_cb;
        /* Callback method to update content of the directory. */
        ucs_vfs_refresh_cb_t   refresh_cb;
        /* Pointer to the target node of symbolic link. */
        ucs_vfs_node_t         *target;
    };
    /* Callback method to write data to the file. */
    ucs_vfs_file_write_cb_t write_cb;
    /* Pointer to an optional argument passed to the text_cb. */
    void                    *arg_ptr;
    /* Type of value passed to ucs_vfs_show_primitive referenced by arg_ptr. */
    uint64_t                arg_u64;
    /* List of symbolic links targeting to the node. */
    ucs_list_link_t         links;
    /* List item to represent the node in target node's list of links. */
    ucs_list_link_t         link_list;
    /* Path to the node in VFS. */
    char                    path[0];
};

KHASH_MAP_INIT_STR(vfs_path, ucs_vfs_node_t*);
KHASH_MAP_INIT_INT64(vfs_obj, ucs_vfs_node_t*);

static ucs_init_once_t ucs_vfs_init_once = UCS_INIT_ONCE_INITIALIZER;

struct {
    ucs_spinlock_t    lock;
    ucs_vfs_node_t    root;
    khash_t(vfs_path) path_hash;
    khash_t(vfs_obj)  obj_hash;
} ucs_vfs_obj_context = {};

#define ucs_vfs_kh_put(_name, _h, _k, _node) \
    { \
        int khret; \
        khiter_t khiter = kh_put(_name, _h, _k, &khret); \
        ucs_assert((khret == UCS_KH_PUT_BUCKET_EMPTY) || \
                   (khret == UCS_KH_PUT_BUCKET_CLEAR)); \
        kh_val(_h, khiter) = _node; \
    }

#define ucs_vfs_kh_del_key(_name, _h, _k) \
    { \
        khiter_t khiter = kh_get(_name, _h, _k); \
        ucs_assert(khiter != kh_end(_h)); \
        kh_del(_name, _h, khiter); \
    }

#define ucs_vfs_kh_find(_name, _h, _k, _node) \
    { \
        khiter_t khiter = kh_get(_name, _h, _k); \
        _node           = (khiter != kh_end(_h)) ? kh_val(_h, khiter) : NULL; \
    }


/* must be called with lock held */
static ucs_vfs_node_t *ucs_vfs_node_find_by_path(const char *path)
{
    ucs_vfs_node_t *node;

    ucs_vfs_kh_find(vfs_path, &ucs_vfs_obj_context.path_hash, path, node);
    ucs_assert((node == NULL) || !strcmp(node->path, path));

    return node;
}

/* must be called with lock held */
static ucs_vfs_node_t *ucs_vfs_node_find_by_obj(void *obj)
{
    ucs_vfs_node_t *node;

    ucs_vfs_kh_find(vfs_obj, &ucs_vfs_obj_context.obj_hash, (uintptr_t)obj,
                    node);
    ucs_assert((node == NULL) || (node->obj == obj));

    return node;
}

/* must be called with lock held */
static void ucs_vfs_node_init(ucs_vfs_node_t *node, ucs_vfs_node_type_t type,
                              void *obj, ucs_vfs_node_t *parent_node)
{
    node->type       = type;
    node->refcount   = 1;
    /* coverity[missing_lock] */
    node->flags      = 0;
    node->obj        = obj;
    node->parent     = parent_node;
    node->read_cb    = NULL;
    node->write_cb   = NULL;
    node->refresh_cb = NULL;
    node->arg_ptr    = NULL;
    node->arg_u64    = 0;
    node->target     = NULL;
    ucs_list_head_init(&node->children);
    ucs_list_head_init(&node->links);
}

static void ucs_vfs_global_init()
{
    UCS_INIT_ONCE(&ucs_vfs_init_once) {
        ucs_spinlock_init(&ucs_vfs_obj_context.lock, 0);
        ucs_vfs_node_init(&ucs_vfs_obj_context.root, UCS_VFS_NODE_TYPE_DIR,
                          NULL, NULL);
        kh_init_inplace(vfs_obj, &ucs_vfs_obj_context.obj_hash);
        kh_init_inplace(vfs_path, &ucs_vfs_obj_context.path_hash);
    }
}

/* must be called with lock held */
static ucs_vfs_node_t *ucs_vfs_node_create(ucs_vfs_node_t *parent_node,
                                           const char *path,
                                           ucs_vfs_node_type_t type, void *obj)
{
    ucs_vfs_node_t *node;

    node = ucs_malloc(sizeof(*node) + strlen(path) + 1, "vfs_node");
    if (node == NULL) {
        ucs_error("Failed to allocate vfs_node");
        return NULL;
    }

    /* initialize node */
    ucs_vfs_node_init(node, type, obj, parent_node);
    strcpy(node->path, path);

    /* add to parent */
    ucs_list_add_head(&parent_node->children, &node->list);

    /* add to obj hash */
    if (node->obj != NULL) {
        ucs_vfs_kh_put(vfs_obj, &ucs_vfs_obj_context.obj_hash,
                       (uintptr_t)node->obj, node);
    }

    /* add to path hash */
    ucs_vfs_kh_put(vfs_path, &ucs_vfs_obj_context.path_hash, node->path, node);

    return node;
}

/* must be called with lock held */
static ucs_vfs_node_t *ucs_vfs_node_get_by_obj(void *obj)
{
    if (obj == NULL) {
        return &ucs_vfs_obj_context.root;
    }

    return ucs_vfs_node_find_by_obj(obj);
}

/* must be called with lock held */
static void ucs_vfs_node_build_path(ucs_vfs_node_t *parent_node,
                                    const char *name, char *path_buf,
                                    size_t path_buf_size)
{
    if (parent_node == &ucs_vfs_obj_context.root) {
        ucs_snprintf_safe(path_buf, path_buf_size, "/%s", name);
    } else {
        ucs_snprintf_safe(path_buf, path_buf_size, "%s/%s", parent_node->path,
                          name);
    }
}

/* must be called with lock held */
static ucs_vfs_node_t *
ucs_vfs_node_add_subdir(ucs_vfs_node_t *parent_node, const char *name)
{
    char path_buf[PATH_MAX];
    ucs_vfs_node_t *node;

    ucs_vfs_node_build_path(parent_node, name, path_buf, sizeof(path_buf));
    node = ucs_vfs_node_find_by_path(path_buf);
    if (node != NULL) {
        return node;
    }

    return ucs_vfs_node_create(parent_node, path_buf, UCS_VFS_NODE_TYPE_SUBDIR,
                               NULL);
}

static int ucs_vfs_node_need_update_path(ucs_vfs_node_type_t type,
                                         const char *path, void *obj)
{
    return (type == UCS_VFS_NODE_TYPE_DIR) &&
           (ucs_vfs_node_find_by_path(path) != NULL) &&
           (ucs_vfs_node_find_by_obj(obj) == NULL);
}

/* must be called with lock held */
static void
ucs_vfs_node_update_path(void *obj, char *path_buf, size_t path_buf_size)
{
    size_t pos = strlen(path_buf);

    ucs_snprintf_safe(path_buf + pos, path_buf_size - pos, "_%p", obj);
}

/* must be called with lock held */
static ucs_status_t ucs_vfs_node_add(void *parent_obj, ucs_vfs_node_type_t type,
                                     void *obj, const char *rel_path,
                                     va_list ap, ucs_vfs_node_t **new_node)
{
    ucs_vfs_node_t *current_node;
    char rel_path_buf[PATH_MAX];
    char abs_path_buf[PATH_MAX];
    char *token, *next_token;

    current_node = ucs_vfs_node_get_by_obj(parent_obj);
    if (current_node == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }

    /* generate the relative path */
    ucs_vsnprintf_safe(rel_path_buf, sizeof(rel_path_buf), rel_path, ap);

    /* Build parent nodes along the rel_path, without associated object */
    next_token = rel_path_buf;
    token      = strsep(&next_token, "/");
    while (next_token != NULL) {
        current_node = ucs_vfs_node_add_subdir(current_node, token);
        if (current_node == NULL) {
            return UCS_ERR_NO_MEMORY;
        }

        token = strsep(&next_token, "/");
    }

    ucs_vfs_node_build_path(current_node, token, abs_path_buf,
                            sizeof(abs_path_buf));

    if (ucs_vfs_node_need_update_path(type, abs_path_buf, obj)) {
        ucs_vfs_node_update_path(obj, abs_path_buf, sizeof(abs_path_buf));
    }

    if (ucs_vfs_node_find_by_path(abs_path_buf) != NULL) {
        return UCS_ERR_ALREADY_EXISTS;
    }

    current_node = ucs_vfs_node_create(current_node, abs_path_buf, type, obj);
    if (current_node == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    *new_node = current_node;

    return UCS_OK;
}

/* must be called with lock held */
static int ucs_vfs_check_node(ucs_vfs_node_t *node, ucs_vfs_node_type_t type)
{
    return (node != NULL) && (node->type == type);
}

/* must be called with lock held */
static void ucs_vfs_node_increase_refcount(ucs_vfs_node_t *node)
{
    ++node->refcount;
}

/* must be called with lock held */
static void ucs_vfs_node_decrease_refcount(ucs_vfs_node_t *node);

/* must be called with lock held */
static void ucs_vfs_node_remove_children(ucs_vfs_node_t *node)
{
    ucs_vfs_node_t *child_node, *tmp_node;

    ucs_list_for_each_safe(child_node, tmp_node, &node->children, list) {
        child_node->parent = NULL; /* prevent children from destroying me */
        ucs_vfs_node_decrease_refcount(child_node);
    }
}

static void ucs_vfs_node_decrease_refcount(ucs_vfs_node_t *node)
{
    ucs_vfs_node_t *parent_node = node->parent;
    ucs_vfs_node_t *tmp_node, *link_node;

    if (--node->refcount > 0) {
        return;
    }

    /* If reference count is 0, then remove node. */

    /* recursively remove children */
    ucs_vfs_node_remove_children(node);

    /* Remove symbolic link nodes targeting to the node.
       This is a workaround for the following scenario:
       1. Link to target is created.
       2. Target is removed.
       3. A new directory is created with the same path as the original target.
       This is not an issue for regular file system. However, it can lead to an
       incorrect linking in case of representing UCX objects in VFS. */
    ucs_list_for_each_safe(link_node, tmp_node, &node->links, link_list) {
        ucs_vfs_node_decrease_refcount(link_node);
    }

    /* remove from object hash */
    if (node->obj != NULL) {
        ucs_vfs_kh_del_key(vfs_obj, &ucs_vfs_obj_context.obj_hash,
                           (uintptr_t)node->obj);
    }

    /* remove from path hash */
    ucs_vfs_kh_del_key(vfs_path, &ucs_vfs_obj_context.path_hash, node->path);

    /* remove from parent's list */
    ucs_list_del(&node->list);

    /* for symbolic link: remove from target's list */
    if (node->type == UCS_VFS_NODE_TYPE_SYM_LINK) {
        ucs_list_del(&node->link_list);
    }

    ucs_free(node);

    /* recursively remove all empty parent subdirs */
    if ((parent_node != NULL) && ucs_list_is_empty(&parent_node->children) &&
        (parent_node->type == UCS_VFS_NODE_TYPE_SUBDIR)) {
        ucs_vfs_node_decrease_refcount(parent_node);
    }
}

/* must be called with lock held */
int ucs_vfs_check_node_dir(ucs_vfs_node_t *node)
{
    return ucs_vfs_check_node(node, UCS_VFS_NODE_TYPE_DIR) ||
           ucs_vfs_check_node(node, UCS_VFS_NODE_TYPE_SUBDIR);
}

/* must be called with lock held and incremented refcount */
static void ucs_vfs_refresh_dir(ucs_vfs_node_t *node)
{
    ucs_vfs_refresh_cb_t refresh_cb;
    void *obj;

    ucs_assert(ucs_vfs_check_node(node, UCS_VFS_NODE_TYPE_DIR) ||
               ucs_vfs_check_node(node, UCS_VFS_NODE_TYPE_SUBDIR));

    if (!(node->flags & UCS_VFS_FLAGS_DIRTY)) {
        return;
    }

    ucs_assert(node->refcount >= 2);

    refresh_cb = node->refresh_cb;
    obj        = node->obj;

    ucs_spin_unlock(&ucs_vfs_obj_context.lock);
    refresh_cb(obj);
    ucs_spin_lock(&ucs_vfs_obj_context.lock);

    node->flags &= ~UCS_VFS_FLAGS_DIRTY;
}

/* must be called with lock held */
static ucs_vfs_node_t *ucs_vfs_get_parent_dir(ucs_vfs_node_t *node)
{
    ucs_vfs_node_t *parent_node = node->parent;

    while (ucs_vfs_check_node(parent_node, UCS_VFS_NODE_TYPE_SUBDIR)) {
        parent_node = parent_node->parent;
    }

    return parent_node;
}

/* must be called with lock held */
int ucs_vfs_check_node_file(ucs_vfs_node_t *node)
{
    return ucs_vfs_check_node(node, UCS_VFS_NODE_TYPE_RO_FILE) ||
           ucs_vfs_check_node(node, UCS_VFS_NODE_TYPE_RW_FILE);
}

/* must be called with lock held */
static void ucs_vfs_read_file(ucs_vfs_node_t *node, ucs_string_buffer_t *strb)
{
    ucs_vfs_node_t *parent_node;

    ucs_assert(ucs_vfs_check_node_file(node));

    parent_node = ucs_vfs_get_parent_dir(node);

    ucs_spin_unlock(&ucs_vfs_obj_context.lock);

    node->read_cb(parent_node->obj, strb, node->arg_ptr, node->arg_u64);

    ucs_spin_lock(&ucs_vfs_obj_context.lock);
}

/* must be called with lock held */
static ucs_status_t
ucs_vfs_write_file(ucs_vfs_node_t *node, const char *buffer, size_t size)
{
    ucs_vfs_node_t *parent_node;
    ucs_status_t status;

    ucs_assert(ucs_vfs_check_node(node, UCS_VFS_NODE_TYPE_RW_FILE));

    parent_node = ucs_vfs_get_parent_dir(node);

    ucs_spin_unlock(&ucs_vfs_obj_context.lock);

    status = node->write_cb(parent_node->obj, buffer, size, node->arg_ptr,
                            node->arg_u64);

    ucs_spin_lock(&ucs_vfs_obj_context.lock);

    return status;
}

/* must be called with lock held */
static void ucs_vfs_path_list_dir_cb(ucs_vfs_node_t *node,
                                     ucs_vfs_list_dir_cb_t dir_cb, void *arg)
{
    ucs_vfs_node_t *child_node;

    ucs_list_for_each(child_node, &node->children, list) {
        dir_cb(ucs_basename(child_node->path), arg);
    }
}

/* must be called with lock held */
static void
ucs_vfs_get_link_path(ucs_vfs_node_t *node, ucs_string_buffer_t *strb)
{
    size_t i, n;

    ucs_assert(ucs_vfs_check_node(node, UCS_VFS_NODE_TYPE_SYM_LINK));

    n = ucs_string_count_char(node->path, '/');
    for (i = 1; i < n; ++i) {
        ucs_string_buffer_appendf(strb, "../");
    }

    if (node->target != NULL) {
        ucs_string_buffer_appendf(strb, "%s", &node->target->path[1]);
    }
}

ucs_status_t
ucs_vfs_obj_add_dir(void *parent_obj, void *obj, const char *rel_path, ...)
{
    ucs_vfs_node_t *node;
    va_list ap;
    ucs_status_t status;

    ucs_vfs_global_init();

    ucs_spin_lock(&ucs_vfs_obj_context.lock);

    va_start(ap, rel_path);
    status = ucs_vfs_node_add(parent_obj, UCS_VFS_NODE_TYPE_DIR, obj, rel_path,
                              ap, &node);
    va_end(ap);

    ucs_spin_unlock(&ucs_vfs_obj_context.lock);

    return status;
}

ucs_status_t ucs_vfs_obj_add_ro_file(void *obj, ucs_vfs_file_read_cb_t read_cb,
                                     void *arg_ptr, uint64_t arg_u64,
                                     const char *rel_path, ...)
{
    ucs_vfs_node_t *node;
    va_list ap;
    ucs_status_t status;

    ucs_vfs_global_init();

    ucs_spin_lock(&ucs_vfs_obj_context.lock);

    va_start(ap, rel_path);
    status = ucs_vfs_node_add(obj, UCS_VFS_NODE_TYPE_RO_FILE, NULL, rel_path,
                              ap, &node);
    va_end(ap);

    if (status == UCS_OK) {
        node->read_cb = read_cb;
        node->arg_ptr = arg_ptr;
        node->arg_u64 = arg_u64;
    }

    ucs_spin_unlock(&ucs_vfs_obj_context.lock);

    return status;
}

ucs_status_t ucs_vfs_obj_add_rw_file(void *obj, ucs_vfs_file_read_cb_t read_cb,
                                     ucs_vfs_file_write_cb_t write_cb,
                                     void *arg_ptr, uint64_t arg_u64,
                                     const char *rel_path, ...)
{
    ucs_vfs_node_t *node;
    va_list ap;
    ucs_status_t status;

    ucs_vfs_global_init();

    ucs_spin_lock(&ucs_vfs_obj_context.lock);

    va_start(ap, rel_path);
    status = ucs_vfs_node_add(obj, UCS_VFS_NODE_TYPE_RW_FILE, NULL, rel_path,
                              ap, &node);
    va_end(ap);

    if (status == UCS_OK) {
        node->read_cb  = read_cb;
        node->write_cb = write_cb;
        node->arg_ptr  = arg_ptr;
        node->arg_u64  = arg_u64;
    }

    ucs_spin_unlock(&ucs_vfs_obj_context.lock);

    return status;
}

ucs_status_t
ucs_vfs_obj_add_sym_link(void *obj, void *target_obj, const char *rel_path, ...)
{
    ucs_vfs_node_t *link_node;
    ucs_vfs_node_t *target_node;
    va_list ap;
    ucs_status_t status;

    ucs_vfs_global_init();

    ucs_spin_lock(&ucs_vfs_obj_context.lock);

    target_node = ucs_vfs_node_find_by_obj(target_obj);
    if (target_node == NULL) {
        status = UCS_ERR_INVALID_PARAM;
        goto out;
    }

    va_start(ap, rel_path);
    status = ucs_vfs_node_add(obj, UCS_VFS_NODE_TYPE_SYM_LINK, NULL, rel_path,
                              ap, &link_node);
    va_end(ap);

    if (status == UCS_OK) {
        link_node->target = target_node;
        ucs_list_add_head(&target_node->links, &link_node->link_list);
    }

out:
    ucs_spin_unlock(&ucs_vfs_obj_context.lock);

    return status;
}

void ucs_vfs_obj_remove(void *obj)
{
    ucs_vfs_node_t *node;

    ucs_vfs_global_init();

    ucs_spin_lock(&ucs_vfs_obj_context.lock);

    node = ucs_vfs_node_find_by_obj(obj);
    if (node != NULL) {
        ucs_vfs_node_decrease_refcount(node);
    }

    ucs_spin_unlock(&ucs_vfs_obj_context.lock);
}

void ucs_vfs_obj_set_dirty(void *obj, ucs_vfs_refresh_cb_t refresh_cb)
{
    ucs_vfs_node_t *node;

    ucs_vfs_global_init();

    ucs_spin_lock(&ucs_vfs_obj_context.lock);

    node = ucs_vfs_node_find_by_obj(obj);
    if (node != NULL) {
        node->flags     |= UCS_VFS_FLAGS_DIRTY;
        node->refresh_cb = refresh_cb;
    }

    ucs_spin_unlock(&ucs_vfs_obj_context.lock);
}

ucs_status_t ucs_vfs_path_get_info(const char *path, ucs_vfs_path_info_t *info)
{
    ucs_string_buffer_t strb;
    ucs_vfs_node_t *node;
    ucs_status_t status;

    ucs_vfs_global_init();

    ucs_spin_lock(&ucs_vfs_obj_context.lock);

    node = ucs_vfs_node_find_by_path(path);
    if (node == NULL) {
        status = UCS_ERR_NO_ELEM;
        goto out_unlock;
    }

    ucs_vfs_node_increase_refcount(node);

    switch (node->type) {
    case UCS_VFS_NODE_TYPE_RO_FILE:
    case UCS_VFS_NODE_TYPE_RW_FILE:
        ucs_string_buffer_init(&strb);
        ucs_vfs_read_file(node, &strb);
        info->size = ucs_string_buffer_length(&strb);
        ucs_string_buffer_cleanup(&strb);

        info->mode = S_IFREG | S_IRUSR;
        if (node->type == UCS_VFS_NODE_TYPE_RW_FILE) {
            info->mode |= S_IWUSR;
        }
        status = UCS_OK;
        break;
    case UCS_VFS_NODE_TYPE_DIR:
    case UCS_VFS_NODE_TYPE_SUBDIR:
        ucs_vfs_refresh_dir(node);
        info->mode = S_IFDIR | S_IRUSR | S_IXUSR;
        info->size = ucs_list_length(&node->children);
        status     = UCS_OK;
        break;
    case UCS_VFS_NODE_TYPE_SYM_LINK:
        ucs_string_buffer_init(&strb);
        ucs_vfs_get_link_path(node, &strb);
        info->mode = S_IFLNK | S_IRUSR | S_IXUSR;
        info->size = ucs_string_buffer_length(&strb);
        ucs_string_buffer_cleanup(&strb);
        status = UCS_OK;
        break;
    default:
        status = UCS_ERR_NO_ELEM;
        break;
    }

    ucs_vfs_node_decrease_refcount(node);

out_unlock:
    ucs_spin_unlock(&ucs_vfs_obj_context.lock);

    return status;
}

ucs_status_t ucs_vfs_path_read_file(const char *path, ucs_string_buffer_t *strb)
{
    ucs_vfs_node_t *node;
    ucs_status_t status;

    ucs_vfs_global_init();

    ucs_spin_lock(&ucs_vfs_obj_context.lock);

    node = ucs_vfs_node_find_by_path(path);
    if (!ucs_vfs_check_node_file(node)) {
        status = UCS_ERR_NO_ELEM;
        goto out_unlock;
    }

    ucs_vfs_node_increase_refcount(node);

    ucs_vfs_read_file(node, strb);
    status = UCS_OK;

    ucs_vfs_node_decrease_refcount(node);

out_unlock:
    ucs_spin_unlock(&ucs_vfs_obj_context.lock);

    return status;
}

ucs_status_t
ucs_vfs_path_write_file(const char *path, const char *buffer, size_t size)
{
    ucs_vfs_node_t *node;
    ucs_status_t status;

    ucs_vfs_global_init();

    ucs_spin_lock(&ucs_vfs_obj_context.lock);

    node = ucs_vfs_node_find_by_path(path);
    if (!ucs_vfs_check_node(node, UCS_VFS_NODE_TYPE_RW_FILE)) {
        status = UCS_ERR_NO_ELEM;
        goto out_unlock;
    }

    status = ucs_vfs_write_file(node, buffer, size);

out_unlock:
    ucs_spin_unlock(&ucs_vfs_obj_context.lock);

    return status;
}

ucs_status_t
ucs_vfs_path_list_dir(const char *path, ucs_vfs_list_dir_cb_t dir_cb, void *arg)
{
    ucs_vfs_node_t *node;
    ucs_status_t status;

    ucs_vfs_global_init();

    ucs_spin_lock(&ucs_vfs_obj_context.lock);

    if (!strcmp(path, "/")) {
        ucs_vfs_path_list_dir_cb(&ucs_vfs_obj_context.root, dir_cb, arg);
        status = UCS_OK;
        goto out_unlock;
    }

    node = ucs_vfs_node_find_by_path(path);
    if (!ucs_vfs_check_node_dir(node)) {
        status = UCS_ERR_NO_ELEM;
        goto out_unlock;
    }

    ucs_vfs_node_increase_refcount(node);

    ucs_vfs_refresh_dir(node);
    ucs_vfs_path_list_dir_cb(node, dir_cb, arg);
    status = UCS_OK;

    ucs_vfs_node_decrease_refcount(node);

out_unlock:
    ucs_spin_unlock(&ucs_vfs_obj_context.lock);

    return status;
}

ucs_status_t ucs_vfs_path_get_link(const char *path, ucs_string_buffer_t *strb)
{
    ucs_vfs_node_t *node;
    ucs_status_t status;

    ucs_vfs_global_init();

    ucs_spin_lock(&ucs_vfs_obj_context.lock);

    node = ucs_vfs_node_find_by_path(path);
    if (!ucs_vfs_check_node(node, UCS_VFS_NODE_TYPE_SYM_LINK)) {
        status = UCS_ERR_NO_ELEM;
        goto out_unlock;
    }

    ucs_vfs_get_link_path(node, strb);
    status = UCS_OK;

out_unlock:
    ucs_spin_unlock(&ucs_vfs_obj_context.lock);

    return status;
}

UCS_STATIC_CLEANUP
{
    UCS_CLEANUP_ONCE(&ucs_vfs_init_once) {
        ucs_vfs_node_remove_children(&ucs_vfs_obj_context.root);

        kh_destroy_inplace(vfs_path, &ucs_vfs_obj_context.path_hash);
        kh_destroy_inplace(vfs_obj, &ucs_vfs_obj_context.obj_hash);
        ucs_spinlock_destroy(&ucs_vfs_obj_context.lock);
    }
}
