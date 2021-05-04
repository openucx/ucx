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
#include <ucs/debug/memtrack.h>
#include <ucs/type/spinlock.h>
#include <ucs/sys/string.h>
#include <stdarg.h>
#include <stdint.h>
#include <sys/stat.h>


typedef enum {
    UCS_VFS_NODE_TYPE_DIR,
    UCS_VFS_NODE_TYPE_RO_FILE,
    UCS_VFS_NODE_TYPE_SUBDIR,
    UCS_VFS_NODE_TYPE_LAST
} ucs_vfs_node_type_t;


#define UCS_VFS_FLAGS_DIRTY UCS_BIT(0)


typedef struct ucs_vfs_node ucs_vfs_node_t;
struct ucs_vfs_node {
    ucs_vfs_node_type_t    type;
    int                    refcount;
    uint8_t                flags;
    void                   *obj;
    ucs_vfs_node_t         *parent;
    ucs_list_link_t        children;
    ucs_vfs_file_show_cb_t text_cb;
    ucs_vfs_refresh_cb_t   refresh_cb;
    ucs_list_link_t        list;
    void                   *arg_ptr;
    uint64_t               arg_u64;
    char                   path[0];
};

KHASH_MAP_INIT_STR(vfs_path, ucs_vfs_node_t*);
KHASH_MAP_INIT_INT64(vfs_obj, ucs_vfs_node_t*);

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

    UCS_STATIC_ASSERT(UCS_VFS_TYPE_UNSIGNED >= UCS_VFS_TYPE_LAST);
    UCS_STATIC_ASSERT(UCS_VFS_TYPE_HEX >= UCS_VFS_TYPE_LAST);

    if (type == UCS_VFS_TYPE_POINTER) {
        ucs_string_buffer_appendf(strb, "%p\n", *(void**)arg_ptr);
    } else if (type == UCS_VFS_TYPE_STRING) {
        ucs_string_buffer_appendf(strb, "%s\n", (char*)arg_ptr);
    } else {
        switch (type & ~(UCS_VFS_TYPE_UNSIGNED | UCS_VFS_TYPE_HEX)) {
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
            return;
        }

        if (type & UCS_VFS_TYPE_HEX) {
            ucs_string_buffer_appendf(strb, "%lx\n", ulvalue);
        } else if (type & UCS_VFS_TYPE_UNSIGNED) {
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
    node->flags      = 0;
    node->obj        = obj;
    node->parent     = parent_node;
    node->text_cb    = NULL;
    node->refresh_cb = NULL;
    node->arg_ptr    = NULL;
    node->arg_u64    = 0;
    ucs_list_head_init(&node->children);
}

/* must be called with lock held */
static ucs_vfs_node_t *ucs_vfs_node_create(ucs_vfs_node_t *parent_node,
                                           const char *name,
                                           ucs_vfs_node_type_t type, void *obj)
{
    char path_buf[PATH_MAX];
    ucs_vfs_node_t *node;

    if (parent_node == &ucs_vfs_obj_context.root) {
        ucs_snprintf_safe(path_buf, sizeof(path_buf), "/%s", name);
    } else {
        ucs_snprintf_safe(path_buf, sizeof(path_buf), "%s/%s",
                          parent_node->path, name);
    }

    node = ucs_vfs_node_find_by_path(path_buf);
    if (node != NULL) {
        return node;
    }

    node = ucs_malloc(sizeof(*node) + strlen(path_buf) + 1, "vfs_node");
    if (node == NULL) {
        ucs_error("Failed to allocate vfs_node");
        return NULL;
    }

    /* initialize node */
    ucs_vfs_node_init(node, type, obj, parent_node);
    strcpy(node->path, path_buf);

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
static ucs_vfs_node_t *ucs_vfs_node_add(void *parent_obj,
                                        ucs_vfs_node_type_t type, void *obj,
                                        const char *rel_path, va_list ap)
{
    ucs_vfs_node_t *parent_node;
    char rel_path_buf[PATH_MAX];
    char *token, *next_token;

    if (parent_obj == NULL) {
        parent_node = &ucs_vfs_obj_context.root;
    } else {
        parent_node = ucs_vfs_node_find_by_obj(parent_obj);
        if (parent_node == NULL) {
            return NULL;
        }
    }

    /* generate the relative path */
    vsnprintf(rel_path_buf, sizeof(rel_path_buf), rel_path, ap);

    /* Build parent nodes along the rel_path, without associated object */
    next_token = rel_path_buf;
    token      = strsep(&next_token, "/");
    while (next_token != NULL) {
        parent_node = ucs_vfs_node_create(parent_node, token,
                                          UCS_VFS_NODE_TYPE_SUBDIR, NULL);
        token       = strsep(&next_token, "/");
    }

    return ucs_vfs_node_create(parent_node, token, type, obj);
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
static void ucs_vfs_node_decrease_refcount(ucs_vfs_node_t *node)
{
    ucs_vfs_node_t *parent_node = node->parent;
    ucs_vfs_node_t *child_node, *tmp_node;

    if (--node->refcount > 0) {
        return;
    }

    /* If reference count is 0, then remove node. */

    /* recursively remove children */
    ucs_list_for_each_safe(child_node, tmp_node, &node->children, list) {
        child_node->parent = NULL; /* prevent children from destroying me */
        ucs_vfs_node_decrease_refcount(child_node);
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

    ucs_free(node);

    /* recursively remove all empty parent subdirs */
    if ((parent_node != NULL) && ucs_list_is_empty(&parent_node->children) &&
        (parent_node->type == UCS_VFS_NODE_TYPE_SUBDIR)) {
        ucs_vfs_node_decrease_refcount(parent_node);
    }
}

/* must be called with lock held and incremented refcount */
static void ucs_vfs_refresh_dir(ucs_vfs_node_t *node)
{
    ucs_assert(ucs_vfs_check_node(node, UCS_VFS_NODE_TYPE_DIR) ||
               ucs_vfs_check_node(node, UCS_VFS_NODE_TYPE_SUBDIR));

    if (!(node->flags & UCS_VFS_FLAGS_DIRTY)) {
        return;
    }

    ucs_assert(node->refcount >= 2);

    ucs_spin_unlock(&ucs_vfs_obj_context.lock);

    node->refresh_cb(node->obj);

    ucs_spin_lock(&ucs_vfs_obj_context.lock);

    node->flags &= ~UCS_VFS_FLAGS_DIRTY;
}

/* must be called with lock held */
static void
ucs_vfs_read_ro_file(ucs_vfs_node_t *node, ucs_string_buffer_t *strb)
{
    ucs_vfs_node_t *parent_node = node->parent;

    ucs_assert(ucs_vfs_check_node(node, UCS_VFS_NODE_TYPE_RO_FILE) == 1);

    while (ucs_vfs_check_node(parent_node, UCS_VFS_NODE_TYPE_SUBDIR)) {
        parent_node = parent_node->parent;
    }

    ucs_spin_unlock(&ucs_vfs_obj_context.lock);

    node->text_cb(parent_node->obj, strb, node->arg_ptr, node->arg_u64);

    ucs_spin_lock(&ucs_vfs_obj_context.lock);
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

void ucs_vfs_obj_add_dir(void *parent_obj, void *obj, const char *rel_path, ...)
{
    va_list ap;

    ucs_spin_lock(&ucs_vfs_obj_context.lock);

    va_start(ap, rel_path);
    ucs_vfs_node_add(parent_obj, UCS_VFS_NODE_TYPE_DIR, obj, rel_path, ap);
    va_end(ap);

    ucs_spin_unlock(&ucs_vfs_obj_context.lock);
}

void ucs_vfs_obj_add_ro_file(void *obj, ucs_vfs_file_show_cb_t text_cb,
                             void *arg_ptr, uint64_t arg_u64,
                             const char *rel_path, ...)
{
    ucs_vfs_node_t *node;
    va_list ap;

    ucs_spin_lock(&ucs_vfs_obj_context.lock);

    va_start(ap, rel_path);
    node = ucs_vfs_node_add(obj, UCS_VFS_NODE_TYPE_RO_FILE, NULL, rel_path, ap);
    va_end(ap);

    if (node != NULL) {
        node->text_cb = text_cb;
        node->arg_ptr = arg_ptr;
        node->arg_u64 = arg_u64;
    }

    ucs_spin_unlock(&ucs_vfs_obj_context.lock);
}

void ucs_vfs_obj_remove(void *obj)
{
    ucs_vfs_node_t *node;

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

    ucs_spin_lock(&ucs_vfs_obj_context.lock);

    node = ucs_vfs_node_find_by_path(path);
    if (node == NULL) {
        status = UCS_ERR_NO_ELEM;
        goto out_unlock;
    }

    ucs_vfs_node_increase_refcount(node);

    switch (node->type) {
    case UCS_VFS_NODE_TYPE_RO_FILE:
        ucs_string_buffer_init(&strb);
        ucs_vfs_read_ro_file(node, &strb);
        info->mode = S_IFREG | S_IRUSR;
        info->size = ucs_string_buffer_length(&strb);
        ucs_string_buffer_cleanup(&strb);
        status = UCS_OK;
        break;
    case UCS_VFS_NODE_TYPE_DIR:
    case UCS_VFS_NODE_TYPE_SUBDIR:
        ucs_vfs_refresh_dir(node);
        info->mode = S_IFDIR | S_IRUSR | S_IXUSR;
        info->size = ucs_list_length(&node->children);
        status     = UCS_OK;
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

    ucs_spin_lock(&ucs_vfs_obj_context.lock);

    node = ucs_vfs_node_find_by_path(path);
    if (!ucs_vfs_check_node(node, UCS_VFS_NODE_TYPE_RO_FILE)) {
        status = UCS_ERR_NO_ELEM;
        goto out_unlock;
    }

    ucs_vfs_node_increase_refcount(node);

    ucs_vfs_read_ro_file(node, strb);
    status = UCS_OK;

    ucs_vfs_node_decrease_refcount(node);

out_unlock:
    ucs_spin_unlock(&ucs_vfs_obj_context.lock);

    return status;
}

ucs_status_t
ucs_vfs_path_list_dir(const char *path, ucs_vfs_list_dir_cb_t dir_cb, void *arg)
{
    ucs_vfs_node_t *node;
    ucs_status_t status;

    ucs_spin_lock(&ucs_vfs_obj_context.lock);

    if (!strcmp(path, "/")) {
        ucs_vfs_path_list_dir_cb(&ucs_vfs_obj_context.root, dir_cb, arg);
        status = UCS_OK;
        goto out_unlock;
    }

    node = ucs_vfs_node_find_by_path(path);

    if (!ucs_vfs_check_node(node, UCS_VFS_NODE_TYPE_DIR) &&
        !ucs_vfs_check_node(node, UCS_VFS_NODE_TYPE_SUBDIR)) {
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

UCS_STATIC_INIT
{
    ucs_spinlock_init(&ucs_vfs_obj_context.lock, 0);
    ucs_spin_lock(&ucs_vfs_obj_context.lock);
    ucs_vfs_node_init(&ucs_vfs_obj_context.root, UCS_VFS_NODE_TYPE_DIR, NULL,
                      NULL);
    ucs_spin_unlock(&ucs_vfs_obj_context.lock);
    kh_init_inplace(vfs_obj, &ucs_vfs_obj_context.obj_hash);
    kh_init_inplace(vfs_path, &ucs_vfs_obj_context.path_hash);
}

UCS_STATIC_CLEANUP
{
    kh_destroy_inplace(vfs_path, &ucs_vfs_obj_context.path_hash);
    kh_destroy_inplace(vfs_obj, &ucs_vfs_obj_context.obj_hash);
    ucs_spinlock_destroy(&ucs_vfs_obj_context.lock);
}
