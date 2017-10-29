/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "mm_md.h"

ucs_config_field_t uct_mm_md_config_table[] = {
  {"", "", NULL,
   ucs_offsetof(uct_mm_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

  {"HUGETLB_MODE", "yes",
   "Enable using huge pages for internal buffers. "
   "Possible values are:\n"
   " y   - Allocate memory using huge pages only.\n"
   " n   - Allocate memory using regular pages only.\n"
   " try - Try to allocate memory using huge pages and if it fails, allocate regular pages.\n",
   ucs_offsetof(uct_mm_md_config_t, hugetlb_mode), UCS_CONFIG_TYPE_TERNARY},

  {NULL}
};

ucs_status_t uct_mm_mem_alloc(uct_md_h md, size_t *length_p, void **address_p,
                              unsigned flags, uct_mem_h *memh_p UCS_MEMTRACK_ARG)
{
    ucs_status_t status;
    uct_mm_seg_t *seg;

    seg = ucs_calloc(1, sizeof(*seg), "mm_seg");
    if (NULL == seg) {
        ucs_error("Failed to allocate memory for mm segment");
        return UCS_ERR_NO_MEMORY;
    }


    status = uct_mm_md_mapper_ops(md)->alloc(md, length_p, UCS_TRY, flags,
                                             address_p, &seg->mmid, &seg->path
                                             UCS_MEMTRACK_VAL);
    if (status != UCS_OK) {
        ucs_free(seg);
        return status;
    }

    seg->length  = *length_p;
    seg->address = *address_p;
    *memh_p      = seg;

    ucs_debug("mm allocated address %p length %zu mmid %"PRIu64,
              seg->address, seg->length, seg->mmid);
    return UCS_OK;
}

ucs_status_t uct_mm_mem_free(uct_md_h md, uct_mem_h memh)
{
    uct_mm_seg_t *seg = memh;
    ucs_status_t status;

    status = uct_mm_md_mapper_ops(md)->free(seg->address, seg->mmid, seg->length,
                                            seg->path);
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

    status = uct_mm_md_mapper_ops(md)->reg(address, length, 
                                           &seg->mmid);
    if (status != UCS_OK) {
        ucs_free(seg);
        return status;
    }

    seg->length  = length;
    seg->address = address;
    *memh_p      = seg;

    ucs_debug("mm registered address %p length %zu mmid %"PRIu64,
              address, length, seg->mmid);
    return UCS_OK;
}

ucs_status_t uct_mm_mem_dereg(uct_md_h md, uct_mem_h memh)
{
    uct_mm_seg_t *seg = memh;
    ucs_status_t status;

    status = uct_mm_md_mapper_ops(md)->dereg(seg->mmid);
    if (status != UCS_OK) {
        return status;
    }

    ucs_free(seg);
    return UCS_OK;
}

ucs_status_t uct_mm_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    md_attr->cap.flags     = 0;
    if (uct_mm_md_mapper_ops(md)->alloc != NULL) {
        md_attr->cap.flags |= UCT_MD_FLAG_ALLOC;
    }
    if (uct_mm_md_mapper_ops(md)->attach != NULL) {
        md_attr->cap.flags |= UCT_MD_FLAG_RKEY_PTR;
    }
    if (uct_mm_md_mapper_ops(md)->reg != NULL) {
        md_attr->cap.flags |= UCT_MD_FLAG_REG;
        md_attr->reg_cost.overhead = 1000.0e-9;
        md_attr->reg_cost.growth   = 0.007e-9;
    }
    md_attr->cap.flags        |= UCT_MD_FLAG_NEED_RKEY;
    md_attr->cap.reg_mem_types = UCS_BIT(UCT_MD_MEM_TYPE_HOST);
    md_attr->cap.mem_type     = UCT_MD_MEM_TYPE_HOST;
    /* all mm md(s) support fixed memory alloc */
    md_attr->cap.flags        |= UCT_MD_FLAG_FIXED;
    md_attr->cap.max_alloc    = ULONG_MAX;
    md_attr->cap.max_reg      = 0;
    md_attr->rkey_packed_size = sizeof(uct_mm_packed_rkey_t) +
                                uct_mm_md_mapper_ops(md)->get_path_size(md);
    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

ucs_status_t uct_mm_mkey_pack(uct_md_h md, uct_mem_h memh, void *rkey_buffer)
{
    uct_mm_packed_rkey_t *rkey = rkey_buffer;
    uct_mm_seg_t *seg = memh;

    rkey->mmid      = seg->mmid;
    rkey->owner_ptr = (uintptr_t)seg->address;
    rkey->length    = seg->length;

    if (seg->path != NULL) {
        strcpy(rkey->path, seg->path);
    }

    ucs_trace("packed rkey: mmid %"PRIu64" owner_ptr %"PRIxPTR,
              rkey->mmid, rkey->owner_ptr);
    return UCS_OK;
}

ucs_status_t uct_mm_rkey_unpack(uct_md_component_t *mdc, const void *rkey_buffer,
                                uct_rkey_t *rkey_p, void **handle_p)
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

    status = uct_mm_mdc_mapper_ops(mdc)->attach(rkey->mmid, rkey->length,
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

ucs_status_t uct_mm_rkey_ptr(uct_md_component_t *mdc, uct_rkey_t rkey,
                             void *handle, uint64_t raddr, void **laddr_p)
{
    uct_mm_remote_seg_t *mm_desc = handle;

    /* rkey stores offset from the remote va */
    *laddr_p = (void *)(raddr + (uint64_t)rkey);
    if ((*laddr_p < mm_desc->address) ||
        (*laddr_p >= mm_desc->address + mm_desc->length)) {
       return UCS_ERR_INVALID_ADDR;
    }
    return UCS_OK;
}

ucs_status_t uct_mm_rkey_release(uct_md_component_t *mdc, uct_rkey_t rkey, void *handle)
{
    ucs_status_t status;
    uct_mm_remote_seg_t *mm_desc = handle;

    status = uct_mm_mdc_mapper_ops(mdc)->detach(mm_desc);
    ucs_free(mm_desc);
    return status;
}

static void uct_mm_md_close(uct_md_h md)
{
    uct_mm_md_t *mm_md = ucs_derived_of(md, uct_mm_md_t);

    ucs_config_parser_release_opts(mm_md->config, md->component->md_config_table);
    ucs_free(mm_md->config);
    ucs_free(mm_md);
}

uct_md_ops_t uct_mm_md_ops = {
    .close        = uct_mm_md_close,
    .query        = uct_mm_md_query,
    .mem_alloc    = uct_mm_mem_alloc,
    .mem_free     = uct_mm_mem_free,
    .mem_reg      = uct_mm_mem_reg,
    .mem_dereg    = uct_mm_mem_dereg,
    .mkey_pack    = uct_mm_mkey_pack,
    .is_mem_type_owned = (void *)ucs_empty_function_return_zero,
};

ucs_status_t uct_mm_md_open(const char *md_name, const uct_md_config_t *md_config,
                            uct_md_h *md_p, uct_md_component_t *mdc)
{
    uct_mm_md_t *mm_md;
    ucs_status_t status;

    mm_md = ucs_malloc(sizeof(*mm_md), "uct_mm_md_t");
    if (mm_md == NULL) {
        ucs_error("Failed to allocate memory for uct_mm_md_t");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    mm_md->config = ucs_malloc(mdc->md_config_size, "mm_md config");
    if (mm_md->config == NULL) {
        ucs_error("Failed to allocate memory for mm_md config");
        status = UCS_ERR_NO_MEMORY;
        goto err_free_mm_md;
    }

    status = ucs_config_parser_clone_opts(md_config, mm_md->config,
                                          mdc->md_config_table);
    if (status != UCS_OK) {
        ucs_error("Failed to clone opts");
        goto err_free_mm_md_config;
    }

    mdc->rkey_ptr = uct_mm_rkey_ptr;

    mm_md->super.ops = &uct_mm_md_ops;
    mm_md->super.component = mdc;

    *md_p = &mm_md->super;
    return UCS_OK;

err_free_mm_md_config:
    ucs_free(mm_md->config);
err_free_mm_md:
    ucs_free(mm_md);
err:
    return status;
}
