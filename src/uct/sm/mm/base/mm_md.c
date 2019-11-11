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
    uct_mm_component_t *mmc = ucs_derived_of(component, uct_mm_component_t);
    ucs_status_t status;

    status = mmc->md_ops->query();
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

ucs_status_t uct_mm_mem_alloc(uct_md_h md, size_t *length_p, void **address_p,
                              unsigned flags, const char *alloc_name,
                              uct_mem_h *memh_p)
{
    ucs_status_t status;
    uct_mm_seg_t *seg;

    seg = ucs_calloc(1, sizeof(*seg), "mm_seg");
    if (NULL == seg) {
        ucs_error("Failed to allocate memory for mm segment");
        return UCS_ERR_NO_MEMORY;
    }

    status = uct_mm_md_mapper_call(md, alloc, md, length_p, UCS_TRY, flags,
                                   alloc_name, address_p, &seg->seg_id,
                                   &seg->path);
    if (status != UCS_OK) {
        ucs_free(seg);
        return status;
    }

    seg->length  = *length_p;
    seg->address = *address_p;
    *memh_p      = seg;

    ucs_debug("mm allocated address %p length %zu mmid %"PRIu64,
              seg->address, seg->length, seg->seg_id);
    return UCS_OK;
}

ucs_status_t uct_mm_mem_free(uct_md_h md, uct_mem_h memh)
{
    uct_mm_seg_t *seg = memh;
    ucs_status_t status;

    status = uct_mm_md_mapper_call(md, free, seg->address, seg->seg_id,
                                   seg->length, seg->path);
    if (status != UCS_OK) {
        return status;
    }

    ucs_free(seg);
    return UCS_OK;
}

ucs_status_t uct_mm_mem_reg(uct_md_h md, void *address, size_t length,
                            unsigned flags, uct_mem_h *memh_p)
{
    ucs_status_t status;
    uct_mm_seg_t *seg;

    seg = ucs_calloc(1, sizeof(*seg), "mm_seg");
    if (NULL == seg) {
        ucs_error("Failed to allocate memory for mm segment");
        return UCS_ERR_NO_MEMORY;
    }

    status = uct_mm_md_mapper_call(md, reg, address, length, &seg->seg_id);
    if (status != UCS_OK) {
        ucs_free(seg);
        return status;
    }

    seg->length  = length;
    seg->address = address;
    *memh_p      = seg;

    ucs_debug("mm registered address %p length %zu mmid %"PRIu64,
              address, length, seg->seg_id);
    return UCS_OK;
}

ucs_status_t uct_mm_mem_dereg(uct_md_h md, uct_mem_h memh)
{
    uct_mm_seg_t *seg = memh;
    ucs_status_t status;

    status = uct_mm_md_mapper_call(md, dereg, seg->seg_id);
    if (status != UCS_OK) {
        return status;
    }

    ucs_free(seg);
    return UCS_OK;
}

void uct_mm_md_query(uct_md_h md, uct_md_attr_t *md_attr, int support_alloc)
{
    memset(md_attr, 0, sizeof(*md_attr));

    md_attr->cap.flags            = UCT_MD_FLAG_RKEY_PTR |
                                    UCT_MD_FLAG_NEED_RKEY;
    md_attr->cap.max_reg          = 0;
    md_attr->cap.max_alloc        = 0;
    md_attr->cap.access_mem_type  = UCS_MEMORY_TYPE_HOST;
    md_attr->cap.detect_mem_types = 0;

    if (support_alloc) {
        md_attr->cap.flags       |= UCT_MD_FLAG_ALLOC | UCT_MD_FLAG_FIXED;
        md_attr->cap.max_alloc    = ULONG_MAX;
    }

    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
}

ucs_status_t uct_mm_mkey_pack(uct_md_h md, uct_mem_h memh, void *rkey_buffer)
{
    uct_mm_packed_rkey_t *rkey = rkey_buffer;
    uct_mm_seg_t *seg = memh;

    rkey->mmid      = seg->seg_id;
    rkey->owner_ptr = (uintptr_t)seg->address;
    rkey->length    = seg->length;

    if (seg->path != NULL) {
        strcpy(rkey->path, seg->path);
    }

    ucs_trace("packed rkey: mmid %"PRIu64" owner_ptr %"PRIxPTR,
              rkey->mmid, rkey->owner_ptr);
    return UCS_OK;
}

ucs_status_t uct_mm_rkey_unpack(uct_component_t *component,
                                const void *rkey_buffer, uct_rkey_t *rkey_p,
                                void **handle_p)
{
    /* user is responsible to free rkey_buffer */
    const uct_mm_packed_rkey_t *rkey = rkey_buffer;
    uct_mm_remote_seg_t *mm_desc;
    ucs_status_t status;

    ucs_trace("unpacking rkey: mmid %"PRIu64" owner_ptr %"PRIxPTR,
              rkey->mmid, rkey->owner_ptr);

    mm_desc = ucs_malloc(sizeof(*mm_desc), "mm_desc");
    if (mm_desc == NULL) {
        return UCS_ERR_NO_RESOURCE;
    }

    status = uct_mm_mdc_mapper_ops(component)->attach(rkey->mmid, rkey->length,
                                                      (void *)rkey->owner_ptr,
                                                      &mm_desc->address,
                                                      &mm_desc->cookie,
                                                      rkey->path);
    if (status != UCS_OK) {
        ucs_free(mm_desc);
        return status;
    }

    mm_desc->length = rkey->length;
    mm_desc->mmid   = rkey->mmid;
    /* store the offset of the addresses, this can be used directly to translate
     * the remote VA to local VA of the attached segment */
    *handle_p = mm_desc;
    *rkey_p   = (uintptr_t)mm_desc->address - rkey->owner_ptr;
    return UCS_OK;
}

ucs_status_t uct_mm_rkey_ptr(uct_component_t *component, uct_rkey_t rkey,
                             void *handle, uint64_t raddr, void **laddr_p)
{
    uct_mm_remote_seg_t *mm_desc = handle;

    /* rkey stores offset from the remote va */
    *laddr_p = UCS_PTR_BYTE_OFFSET(raddr, (ptrdiff_t)rkey);
    if ((*laddr_p < mm_desc->address) ||
        (*laddr_p >= UCS_PTR_BYTE_OFFSET(mm_desc->address, mm_desc->length))) {
       return UCS_ERR_INVALID_ADDR;
    }
    return UCS_OK;
}

ucs_status_t uct_mm_rkey_release(uct_component_t *component, uct_rkey_t rkey,
                                 void *handle)
{
    ucs_status_t status;
    uct_mm_remote_seg_t *mm_desc = handle;

    status = uct_mm_mdc_mapper_ops(component)->detach(mm_desc);
    ucs_free(mm_desc);
    return status;
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
