/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "mm_md.h"

#include <ucs/debug/log.h>
#include <inttypes.h>
#include <limits.h>


ucs_config_field_t uct_mm_md_config_table[] = {
  {"", "", NULL,
   ucs_offsetof(uct_mm_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

  {"HUGETLB_MODE", "try",
   "Enable using huge pages for internal buffers. "
   "Possible values are:\n"
   " y   - Allocate memory using huge pages only.\n"
   " n   - Allocate memory using regular pages only.\n"
   " try - Try to allocate memory using huge pages and if it fails, allocate regular pages.\n",
   ucs_offsetof(uct_mm_md_config_t, hugetlb_mode), UCS_CONFIG_TYPE_TERNARY},

  {NULL}
};

ucs_status_t uct_mm_query_md_resources(uct_component_t *component,
                                       uct_md_resource_desc_t **resources_p,
                                       unsigned *num_resources_p)
{
    ucs_status_t status;
    int UCS_V_UNUSED attach_shm_file;

    status = uct_mm_mdc_mapper_ops(component)->query(&attach_shm_file);
    switch (status) {
    case UCS_OK:
        return uct_md_query_single_md_resource(component, resources_p,
                                               num_resources_p);
    case UCS_ERR_UNSUPPORTED:
        return uct_md_query_empty_md_resource(resources_p, num_resources_p);
    default:
        return status;
    }
}

ucs_status_t uct_mm_seg_new(void *address, size_t length, uct_mm_seg_t **seg_p)
{
    uct_mm_seg_t *seg;

    seg = ucs_malloc(sizeof(*seg), "mm_seg");
    if (seg == NULL) {
        ucs_error("failed to allocate mm segment");
        return UCS_ERR_NO_MEMORY;
    }

    seg->address = address;
    seg->length  = length;
    seg->seg_id  = 0;
    *seg_p       = seg;
    return UCS_OK;
}

void uct_mm_md_query(uct_md_h md, uct_md_attr_t *md_attr, uint64_t max_alloc)
{
    memset(md_attr, 0, sizeof(*md_attr));

    md_attr->cap.flags            = UCT_MD_FLAG_RKEY_PTR |
                                    UCT_MD_FLAG_NEED_RKEY;
    md_attr->cap.max_reg          = 0;
    md_attr->cap.max_alloc        = 0;
    md_attr->cap.alloc_mem_types  = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    md_attr->cap.access_mem_types = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    md_attr->cap.detect_mem_types = 0;

    if (max_alloc > 0) {
        md_attr->cap.flags       |= UCT_MD_FLAG_ALLOC | UCT_MD_FLAG_FIXED;
        md_attr->cap.max_alloc    = max_alloc;
    }

    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
}

ucs_status_t uct_mm_rkey_ptr(uct_component_t *component, uct_rkey_t rkey,
                             void *handle, uint64_t raddr, void **laddr_p)
{
    /* rkey stores offset from the remote va */
    *laddr_p = UCS_PTR_BYTE_OFFSET(raddr, (ptrdiff_t)rkey);
    return UCS_OK;
}

ucs_status_t uct_mm_md_open(uct_component_t *component, const char *md_name,
                            const uct_md_config_t *config, uct_md_h *md_p)
{
    uct_mm_component_t *mmc = ucs_derived_of(component, uct_mm_component_t);
    ucs_status_t status;
    uct_mm_md_t *md;

    md = ucs_malloc(sizeof(*md), "uct_mm_md_t");
    if (md == NULL) {
        ucs_error("Failed to allocate memory for uct_mm_md_t");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    md->config = ucs_malloc(mmc->super.md_config.size, "mm_md config");
    if (md->config == NULL) {
        ucs_error("Failed to allocate memory for mm_md config");
        status = UCS_ERR_NO_MEMORY;
        goto err_free_mm_md;
    }

    status = ucs_config_parser_clone_opts(config, md->config,
                                          mmc->super.md_config.table);
    if (status != UCS_OK) {
        ucs_error("Failed to clone opts");
        goto err_free_mm_md_config;
    }

    md->super.ops       = &mmc->md_ops->super;
    md->super.component = &mmc->super;
    md->iface_addr_len  = mmc->md_ops->iface_addr_length(md);

    /* cppcheck-suppress autoVariables */
    *md_p = &md->super;
    return UCS_OK;

err_free_mm_md_config:
    ucs_free(md->config);
err_free_mm_md:
    ucs_free(md);
err:
    return status;
}

void uct_mm_md_close(uct_md_h md)
{
    uct_mm_md_t *mm_md = ucs_derived_of(md, uct_mm_md_t);

    ucs_config_parser_release_opts(mm_md->config,
                                   md->component->md_config.table);
    ucs_free(mm_md->config);
    ucs_free(mm_md);
}
