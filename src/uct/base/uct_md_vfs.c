/**
 * Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "uct_md.h"
#include "uct_vfs_attr.h"
#include <ucs/debug/log_def.h>
#include <ucs/sys/sys.h>
#include <ucs/vfs/base/vfs_cb.h>
#include <ucs/vfs/base/vfs_obj.h>
#include <unistd.h>


static const uct_vfs_flag_info_t uct_md_vfs_flag_infos[] = {
    {UCT_MD_FLAG_ALLOC, "alloc"},
    {UCT_MD_FLAG_REG, "reg"},
    {UCT_MD_FLAG_NEED_MEMH, "need_memh"},
    {UCT_MD_FLAG_NEED_RKEY, "need_rkey"},
    {UCT_MD_FLAG_ADVISE, "advise"},
    {UCT_MD_FLAG_FIXED, "fixed"},
    {UCT_MD_FLAG_RKEY_PTR, "rkey_ptr"},
    {UCT_MD_FLAG_SOCKADDR, "sockaddr"},
    {UCT_MD_FLAG_INVALIDATE, "invalidate"},
};


typedef struct {
    unsigned long offset;
    uint64_t      type;
    const char    *name;
} uct_md_vfs_cap_info_t;

#define UCT_MD_VFS_CAP_INFO(_attr, _type) \
    { \
        ucs_offsetof(uct_md_attr_t, cap._attr), _type, "capability/" #_attr \
    }

#define UCT_MD_VFS_ATTR_INFO(_attr, _type) \
    { \
        ucs_offsetof(uct_md_attr_t, _attr), _type, #_attr \
    }

static const uct_md_vfs_cap_info_t uct_md_vfs_cap_infos[] = {
    UCT_MD_VFS_CAP_INFO(max_alloc, UCS_VFS_TYPE_ULONG),
    UCT_MD_VFS_CAP_INFO(max_reg, UCS_VFS_TYPE_SIZET),
    UCT_MD_VFS_CAP_INFO(reg_mem_types, UCS_VFS_TYPE_ULONG),
    UCT_MD_VFS_CAP_INFO(detect_mem_types, UCS_VFS_TYPE_ULONG),
    UCT_MD_VFS_CAP_INFO(alloc_mem_types, UCS_VFS_TYPE_ULONG),
    UCT_MD_VFS_CAP_INFO(access_mem_types, UCS_VFS_TYPE_ULONG),
    UCT_MD_VFS_ATTR_INFO(rkey_packed_size, UCS_VFS_TYPE_SIZET)
};


static ucs_status_t uct_md_vfs_get_attr(uct_md_h md, ucs_string_buffer_t *strb,
                                        uct_md_attr_t *md_attr)
{
    ucs_status_t status = uct_md_query(md, md_attr);

    if (status != UCS_OK) {
        ucs_string_buffer_appendf(strb, "<failed to query md attributes>\n");
    }
    return status;
}

static void uct_md_vfs_read_cap(void *obj, ucs_string_buffer_t *strb,
                                void *arg_ptr, uint64_t arg_u64)
{
    uct_md_attr_t md_attr;
    void *attr;

    if (uct_md_vfs_get_attr(obj, strb, &md_attr) != UCS_OK) {
        return;
    }

    attr = UCS_PTR_BYTE_OFFSET(&md_attr, *(unsigned long*)arg_ptr);
    ucs_vfs_show_primitive(NULL, strb, attr, arg_u64);
}

static void uct_md_vfs_read_reg_cost(void *obj, ucs_string_buffer_t *strb,
                                     void *arg_ptr, uint64_t arg_u64)
{
    uct_md_attr_t md_attr;

    if (uct_md_vfs_get_attr(obj, strb, &md_attr) != UCS_OK) {
        return;
    }

    ucs_string_buffer_appendf(strb, "f(x) = %e + x * %e\n", md_attr.reg_cost.c,
                              md_attr.reg_cost.m);
}

static void uct_md_vfs_read_local_cpus(void *obj, ucs_string_buffer_t *strb,
                                       void *arg_ptr, uint64_t arg_u64)
{
    uct_md_attr_t md_attr;
    long i, num_cpus;

    if (uct_md_vfs_get_attr(obj, strb, &md_attr) != UCS_OK) {
        return;
    }

    num_cpus = ucs_sys_get_num_cpus();
    if (num_cpus == -1) {
        ucs_string_buffer_appendf(strb, "<failed to get number of CPUs: %m>\n");
        return;
    }

    for (i = 0; i < num_cpus; ++i) {
        ucs_string_buffer_appendf(strb, "%d,",
                                  ucs_cpu_is_set(i, &md_attr.local_cpus));
    }
    ucs_string_buffer_rtrim(strb, ",");
    ucs_string_buffer_appendf(strb, "\n");
}

void uct_md_vfs_init(uct_component_h component, uct_md_h md,
                     const char *md_name)
{
    size_t i;
    uct_md_attr_t md_attr;

    ucs_vfs_obj_add_dir(component, md, "memory_domain/%s", md_name);

    if (uct_md_query(md, &md_attr) == UCS_OK) {
        uct_vfs_init_flags(md, md_attr.cap.flags, uct_md_vfs_flag_infos,
                           ucs_static_array_size(uct_md_vfs_flag_infos));
    } else {
        ucs_debug("failed to query md attributes");
    }

    for (i = 0; i < ucs_static_array_size(uct_md_vfs_cap_infos); ++i) {
        ucs_vfs_obj_add_ro_file(md, uct_md_vfs_read_cap,
                                (void*)&uct_md_vfs_cap_infos[i].offset,
                                uct_md_vfs_cap_infos[i].type, "%s",
                                uct_md_vfs_cap_infos[i].name);
    }

    ucs_vfs_obj_add_ro_file(md, uct_md_vfs_read_reg_cost, NULL, 0, "reg_cost");
    ucs_vfs_obj_add_ro_file(md, uct_md_vfs_read_local_cpus, NULL, 0,
                            "local_cpus");

    component->md_vfs_init(md);
}
