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


typedef enum {
    UCS_VFS_NODE_TYPE_DIR,
    UCS_VFS_NODE_TYPE_RO_FILE,
    UCS_VFS_NODE_TYPE_SUBDIR,
    UCS_VFS_NODE_TYPE_LAST
} ucs_vfs_node_type_t;

typedef struct ucs_vfs_node ucs_vfs_node_t;
struct ucs_vfs_node {
    ucs_vfs_node_type_t    type;
    void                   *obj;
    ucs_vfs_node_t         *parent;
    ucs_list_link_t        children;
    ucs_vfs_file_show_cb_t text_cb;
    ucs_list_link_t        list;
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
    node->type    = type;
    node->obj     = obj;
    node->parent  = parent_node;
    node->text_cb = NULL;
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
    ucs_snprintf_zero(rel_path_buf, sizeof(rel_path_buf), rel_path, ap);

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
static void ucs_vfs_node_remove(ucs_vfs_node_t *node)
{
    ucs_vfs_node_t *parent_node = node->parent;
    ucs_vfs_node_t *child_node, *tmp_node;

    /* recursively remove children empty parent subdirs */
    ucs_list_for_each_safe(child_node, tmp_node, &node->children, list) {
        child_node->parent = NULL; /* prevent children from destroying me */
        ucs_vfs_node_remove(child_node);
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
        ucs_vfs_node_remove(parent_node);
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
    }

    ucs_spin_unlock(&ucs_vfs_obj_context.lock);
}

void ucs_vfs_obj_remove(void *obj)
{
    ucs_vfs_node_t *node;

    ucs_spin_lock(&ucs_vfs_obj_context.lock);

    node = ucs_vfs_node_find_by_obj(obj);
    if (node != NULL) {
        ucs_vfs_node_remove(node);
    }

    ucs_spin_unlock(&ucs_vfs_obj_context.lock);
}

UCS_STATIC_INIT
{
    ucs_spinlock_init(&ucs_vfs_obj_context.lock, 0);
    ucs_vfs_node_init(&ucs_vfs_obj_context.root, UCS_VFS_NODE_TYPE_DIR, NULL,
                      NULL);
    kh_init_inplace(vfs_obj, &ucs_vfs_obj_context.obj_hash);
    kh_init_inplace(vfs_path, &ucs_vfs_obj_context.path_hash);
}

UCS_STATIC_CLEANUP
{
    kh_destroy_inplace(vfs_path, &ucs_vfs_obj_context.path_hash);
    kh_destroy_inplace(vfs_obj, &ucs_vfs_obj_context.obj_hash);
    ucs_spinlock_destroy(&ucs_vfs_obj_context.lock);
}
