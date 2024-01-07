/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2019. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <uct/ib/mlx5/ib_mlx5.h>

#include <ucs/arch/bitops.h>
#include <ucs/profile/profile.h>

/* max log value to store in uint8_t */
#define UCT_IB_MLX5_MD_MAX_DCI_CHANNELS 8

#define UCT_IB_MLX5_MD_UMEM_ACCESS \
    (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE)


static uint32_t uct_ib_mlx5_flush_rkey_make()
{
    return ((getpid() & 0xff) << 8) | UCT_IB_MD_INVALID_FLUSH_RKEY;
}

#if HAVE_DEVX
static const char uct_ib_mkey_token[] = "uct_ib_mkey_token";

typedef struct uct_ib_mlx5_dbrec_page {
    uct_ib_mlx5_devx_umem_t    mem;
} uct_ib_mlx5_dbrec_page_t;


static size_t uct_ib_mlx5_calc_mkey_inlen(int list_size)
{
    return UCT_IB_MLX5DV_ST_SZ_BYTES(create_mkey_in) +
           UCT_IB_MLX5DV_ST_SZ_BYTES(klm) * list_size;
}

static ucs_status_t uct_ib_mlx5_alloc_mkey_inbox(int list_size, char **in_p)
{
    size_t inlen;
    char *in;

    inlen = uct_ib_mlx5_calc_mkey_inlen(list_size);
    in    = ucs_calloc(1, inlen, "mkey mailbox");
    if (in == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    *in_p = in;
    return UCS_OK;
}

static ucs_status_t
uct_ib_mlx5_devx_reg_ksm(uct_ib_mlx5_md_t *md, int atomic, uint64_t address,
                         size_t length, int list_size, size_t entity_size,
                         char *in, uint32_t mkey_index, const char *reason,
                         struct mlx5dv_devx_obj **mr_p, uint32_t *mkey)
{
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(create_mkey_out)] = {};
    struct mlx5dv_devx_obj *mr;
    void *mkc;

    UCT_IB_MLX5DV_SET(create_mkey_in, in, opcode, UCT_IB_MLX5_CMD_OP_CREATE_MKEY);
    UCT_IB_MLX5DV_SET(create_mkey_in, in, input_mkey_index, mkey_index);
    mkc = UCT_IB_MLX5DV_ADDR_OF(create_mkey_in, in, memory_key_mkey_entry);
    UCT_IB_MLX5DV_SET(mkc, mkc, access_mode_1_0, UCT_IB_MLX5_MKC_ACCESS_MODE_KSM);
    UCT_IB_MLX5DV_SET(mkc, mkc, a, !!atomic);
    UCT_IB_MLX5DV_SET(mkc, mkc, rw, 1);
    UCT_IB_MLX5DV_SET(mkc, mkc, rr, 1);
    UCT_IB_MLX5DV_SET(mkc, mkc, lw, 1);
    UCT_IB_MLX5DV_SET(mkc, mkc, lr, 1);
    UCT_IB_MLX5DV_SET(mkc, mkc, pd, uct_ib_mlx5_devx_md_get_pdn(md));
    UCT_IB_MLX5DV_SET(mkc, mkc, translations_octword_size, list_size);
    UCT_IB_MLX5DV_SET(mkc, mkc, log_entity_size, ucs_ilog2(entity_size));
    UCT_IB_MLX5DV_SET(mkc, mkc, qpn, 0xffffff);
    UCT_IB_MLX5DV_SET(mkc, mkc, mkey_7_0, md->mkey_tag);
    UCT_IB_MLX5DV_SET64(mkc, mkc, start_addr, address);
    UCT_IB_MLX5DV_SET64(mkc, mkc, len, length);
    UCT_IB_MLX5DV_SET(create_mkey_in, in, translations_octword_actual_size, list_size);

    mr = UCS_PROFILE_NAMED_CALL_ALWAYS("devx_create_mkey",
                                       mlx5dv_devx_obj_create,
                                       md->super.dev.ibv_context, in,
                                       uct_ib_mlx5_calc_mkey_inlen(list_size),
                                       out, sizeof(out));
    if (mr == NULL) {
        if (reason != NULL) {
            ucs_debug("mlx5dv_devx_obj_create(CREATE_MKEY, mode=KSM) failed, "
                      "syndrome 0x%x: %m",
                      UCT_IB_MLX5DV_GET(create_mkey_out, out, syndrome));
        }

        return UCS_ERR_UNSUPPORTED;
    }

    *mr_p = mr;
    *mkey = (UCT_IB_MLX5DV_GET(create_mkey_out, out, mkey_index) << 8) |
            md->mkey_tag;

    if (reason != NULL) {
        ucs_trace("%s: registered KSM%s for %s %lx..%lx with %d entries of %zu "
                  "bytes, mkey=0x%x mr=%p",
                  uct_ib_device_name(&md->super.dev), atomic ? " atomic" : "",
                  reason, address, address + length, list_size, entity_size,
                  *mkey, mr);
    }

    return UCS_OK;
}

static ucs_status_t
uct_ib_mlx5_devx_reg_ksm_data(uct_ib_mlx5_md_t *md, int atomic, void *address,
                              uct_ib_mlx5_devx_ksm_data_t *ksm_data,
                              size_t length, uint64_t iova, uint32_t mkey_index,
                              const char *reason, struct mlx5dv_devx_obj **mr_p,
                              uint32_t *mkey)
{
    void *mr_address = address;
    ucs_status_t status;
    char *in;
    void *klm;
    int i;

    status = uct_ib_mlx5_alloc_mkey_inbox(ksm_data->mr_num, &in);
    if (status != UCS_OK) {
        return UCS_ERR_NO_MEMORY;
    }

    klm = UCT_IB_MLX5DV_ADDR_OF(create_mkey_in, in, klm_pas_mtt);
    for (i = 0; i < ksm_data->mr_num; i++) {
        UCT_IB_MLX5DV_SET64(klm, klm, address, (uintptr_t)mr_address);
        UCT_IB_MLX5DV_SET(klm, klm, byte_count, ksm_data->mrs[i]->length);
        UCT_IB_MLX5DV_SET(klm, klm, mkey, ksm_data->mrs[i]->lkey);
        klm        = UCS_PTR_BYTE_OFFSET(klm, UCT_IB_MLX5DV_ST_SZ_BYTES(klm));
        mr_address = UCS_PTR_BYTE_OFFSET(mr_address, ksm_data->mrs[i]->length);
    }

    status = uct_ib_mlx5_devx_reg_ksm(md, atomic, iova, length,
                                      ksm_data->mr_num,
                                      ksm_data->mrs[0]->length, in, mkey_index,
                                      reason, mr_p, mkey);
    ucs_free(in);
    return status;
}

static ucs_status_t
uct_ib_mlx5_devx_reg_ksm_data_addr(uct_ib_mlx5_md_t *md, struct ibv_mr *mr,
                                   uint64_t address, size_t length,
                                   uint64_t iova, int atomic, int list_size,
                                   uint32_t mkey_index, const char *reason,
                                   struct mlx5dv_devx_obj **mr_p,
                                   uint32_t *mkey)
{
    int i;
    char *in;
    void *klm;
    ucs_status_t status;

    status = uct_ib_mlx5_alloc_mkey_inbox(list_size, &in);
    if (status != UCS_OK) {
        return status;
    }

    klm = UCT_IB_MLX5DV_ADDR_OF(create_mkey_in, in, klm_pas_mtt);
    for (i = 0; i < list_size; i++) {
        UCT_IB_MLX5DV_SET(klm, klm, mkey, mr->lkey);
        UCT_IB_MLX5DV_SET64(klm, klm, address,
                            address + (i * UCT_IB_MD_MAX_MR_SIZE));
        klm = UCS_PTR_BYTE_OFFSET(klm, UCT_IB_MLX5DV_ST_SZ_BYTES(klm));
    }

    status = uct_ib_mlx5_devx_reg_ksm(md, atomic, iova, length, list_size,
                                      UCT_IB_MD_MAX_MR_SIZE, in, mkey_index,
                                      reason, mr_p, mkey);
    ucs_free(in);
    return status;
}

static ucs_status_t uct_ib_mlx5_devx_reg_ksm_data_contig(
        uct_ib_mlx5_md_t *md, uct_ib_mlx5_devx_mr_t *mr, void *address,
        uint64_t iova, int atomic, uint32_t mkey_index, const char *reason,
        struct mlx5dv_devx_obj **mr_p, uint32_t *mkey)
{
    size_t mr_length = mr->super.ib->length;
    uint64_t ksm_address;
    uint64_t ksm_iova;
    size_t ksm_length;
    int list_size;

    /* FW requires indirect atomic MR address and length to be aligned
     * to max supported atomic argument size */
    ksm_address = ucs_align_down_pow2((uint64_t)address, UCT_IB_MD_MAX_MR_SIZE);
    ksm_iova    = iova + ksm_address - (uint64_t)address;
    ksm_length  = mr_length + (uint64_t)address - ksm_address;
    ksm_length  = ucs_align_up(ksm_length, md->super.dev.atomic_align);

    /* Add offset to workaround CREATE_MKEY range check issue */
    list_size = ucs_div_round_up(ksm_length + ucs_get_page_size(),
                                 UCT_IB_MD_MAX_MR_SIZE);

    return uct_ib_mlx5_devx_reg_ksm_data_addr(md, mr->super.ib, ksm_address,
                                              ksm_length, ksm_iova, atomic,
                                              list_size, mkey_index, reason,
                                              mr_p, mkey);
}

static void *
uct_ib_mlx5_devx_memh_base_address(const uct_ib_mlx5_devx_mem_t *memh)
{
#if HAVE_IBV_DM
    if (memh->dm != NULL) {
        /* Device memory memory key is zero based */
        return NULL;
    }
#endif
    return memh->address;
}

/**
 * Pop MR LRU-entry from @a md cash
 */
static void
uct_ib_mlx5_devx_md_mr_lru_pop(uct_ib_mlx5_md_t *md, const char *reason)
{
    uct_ib_mlx5_mem_lru_entry_t *head;
    struct mlx5dv_devx_obj *mr;
    khint_t iter;

    ucs_assert(!ucs_list_is_empty(&md->lru_rkeys.list));
    head = ucs_list_extract_head(&md->lru_rkeys.list,
                                 uct_ib_mlx5_mem_lru_entry_t, list);
    ucs_trace("%s: pop mkey 0x%x from LRU because of %s",
              uct_ib_device_name(&md->super.dev), head->rkey, reason);

    iter = kh_get(rkeys, &md->lru_rkeys.hash, head->rkey);
    ucs_assertv_always(iter != kh_end(&md->lru_rkeys.hash),
                       "%s: LRU mkey 0x%x not found",
                       uct_ib_device_name(&md->super.dev), head->rkey);

    mr = kh_val(&md->lru_rkeys.hash, iter)->indirect_mr;
    if ((mr != NULL) && head->is_dummy) {
        ucs_debug("%s: destroy dvmr %p with key 0x%x",
                  uct_ib_device_name(&md->super.dev), mr, head->rkey);

        uct_ib_mlx5_devx_obj_destroy(mr, "MKEY, LRU_INDIRECT");
    }

    kh_del(rkeys, &md->lru_rkeys.hash, iter);
    ucs_free(head);
}

static void
uct_ib_md_mlx5_devx_mr_lru_entry_update(uct_ib_mlx5_md_t *md,
                                        uct_ib_mlx5_mem_lru_entry_t *entry,
                                        struct mlx5dv_devx_obj *mr)
{
    /* 2nd state of an entry must be resetting to NULL since the mr is
     * destroyed or vice-versa */
    ucs_assertv((entry->indirect_mr == NULL) != (mr == NULL),
                "indirect_mr=%p mr=%p", entry->indirect_mr, mr);

    entry->indirect_mr = mr;
    /* move to the end of the list */
    ucs_list_del(&entry->list);
    ucs_list_add_tail(&md->lru_rkeys.list, &entry->list);
}

/**
 * Cash @a mr with @a rkey on @a md.
 * @param [in]  md        Memory domain.
 * @param [in]  rkey      Remote key.
 * @param [in]  mr        Memory region handle.
 * @return Error code.
 */
static ucs_status_t
uct_ib_md_mlx5_devx_mr_lru_push(uct_ib_mlx5_md_t *md, uint32_t rkey, void *mr)
{
    uct_ib_mlx5_mem_lru_entry_t *entry;
    khint_t iter;
    ucs_kh_put_t res;

    ucs_assert(rkey != UCT_IB_INVALID_MKEY);

    iter = kh_put(rkeys, &md->lru_rkeys.hash, rkey, &res);
    if (ucs_unlikely(res == UCS_KH_PUT_FAILED)) {
        ucs_error("Cannot allocate rkey LRU hash entry");
        return UCS_ERR_NO_MEMORY;
    }

    if (res == UCS_KH_PUT_KEY_PRESENT) {
        ucs_trace("%s: mr lru size=%d reset: %x->%p",
                  uct_ib_device_name(&md->super.dev),
                  kh_size(&md->lru_rkeys.hash), rkey, mr);
        entry = kh_val(&md->lru_rkeys.hash, iter);
        ucs_assertv(entry->rkey == rkey, "entry_rkey=0x%x, rkey=0x%x",
                    entry->rkey, rkey);
        uct_ib_md_mlx5_devx_mr_lru_entry_update(md, entry, mr);
        entry->is_dummy = 1;
        return UCS_ERR_ALREADY_EXISTS;
    }

    if (mr == NULL) {
        ucs_trace("%s: mr lru size=%d miss: %x",
                  uct_ib_device_name(&md->super.dev),
                  kh_size(&md->lru_rkeys.hash), rkey);
        /* trying to reset non-exist entry, del empty iter */
        kh_del(rkeys, &md->lru_rkeys.hash, iter);
        return UCS_ERR_NO_ELEM;
    }

    if (kh_size(&md->lru_rkeys.hash) >= md->super.config.max_idle_rkey_count) {
        uct_ib_mlx5_devx_md_mr_lru_pop(md, "limit");
    }

    entry = ucs_malloc(sizeof(*entry), "rkey_lru_entry");
    if (entry == NULL) {
        ucs_error("Cannot allocate rkey LRU entry");
        return UCS_ERR_NO_MEMORY;
    }

    entry->indirect_mr = mr;
    entry->rkey        = rkey;
    entry->is_dummy    = 0;
    ucs_list_add_tail(&md->lru_rkeys.list, &entry->list);
    kh_val(&md->lru_rkeys.hash, iter) = entry;
    ucs_trace("%s: push mkey 0x%x mr %p to LRU",
              uct_ib_device_name(&md->super.dev), rkey, mr);

    ucs_trace("%s: mr lru size=%d push: %x->%p", uct_ib_device_name(&md->super.dev),
              kh_size(&md->lru_rkeys.hash), rkey, mr);

    if ((++md->lru_rkeys.count % md->super.config.max_idle_rkey_count) == 0) {
        /* modulo prime number to avoid resonance with mkey */
        md->mkey_tag = (md->mkey_tag + 1) % 251;
    }

    return UCS_OK;
}

static void uct_ib_mlx5_devx_mr_lru_init(uct_ib_mlx5_md_t *md)
{
    ucs_list_head_init(&md->lru_rkeys.list);
    kh_init_inplace(rkeys, &md->lru_rkeys.hash);
    md->lru_rkeys.count = 0;
}

static void uct_ib_mlx5_devx_mr_lru_cleanup(uct_ib_mlx5_md_t *md)
{
    while (!ucs_list_is_empty(&md->lru_rkeys.list)) {
        uct_ib_mlx5_devx_md_mr_lru_pop(md, "cleanup");
    }

    ucs_assertv(kh_size(&md->lru_rkeys.hash) == 0,
                "%s: %d LRU cache entries are leaked",
                uct_ib_device_name(&md->super.dev),
                kh_size(&md->lru_rkeys.hash));

    kh_destroy_inplace(rkeys, &md->lru_rkeys.hash);
}

UCS_PROFILE_FUNC_ALWAYS(ucs_status_t, uct_ib_mlx5_devx_reg_indirect_key,
                        (md, memh), uct_ib_mlx5_md_t *md,
                        uct_ib_mlx5_devx_mem_t *memh)
{
    ucs_status_t status;

    ucs_assertv(md->flags & UCT_IB_MLX5_MD_FLAG_KSM, "md %p: name %s", md,
                md->super.name);

    do {
        status = uct_ib_mlx5_devx_reg_ksm_data_contig(
                md, &memh->mrs[UCT_IB_MR_DEFAULT],
                uct_ib_mlx5_devx_memh_base_address(memh),
                (uint64_t)memh->address, 0, 0, "indirect key",
                &memh->indirect_dvmr, &memh->indirect_rkey);
        if (status != UCS_OK) {
            break;
        }

        /* This loop is guaranteed to finish because eventually all entries in
         * the LRU will have an associated indirect_mr object, so the next key
         * we will get from HW will be a new value not in the LRU. */
        status = uct_ib_md_mlx5_devx_mr_lru_push(md, memh->indirect_rkey,
                                                 memh->indirect_dvmr);
    } while (status == UCS_ERR_ALREADY_EXISTS);

    if (status != UCS_OK) {
        ucs_error("%s: LRU push returned %s",
                  uct_ib_device_name(&md->super.dev),
                  ucs_status_string(status));
        return status;
    }

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE uint32_t uct_ib_mlx5_mkey_index(uint32_t mkey)
{
    return mkey >> 8;
}

static UCS_F_ALWAYS_INLINE uct_ib_mr_type_t uct_ib_devx_get_atomic_mr_type(
        uct_ib_md_t *md, const uct_ib_mlx5_devx_mem_t *memh)
{
#if HAVE_IBV_DM
    /* Device memory only supports default mr */
    if (memh->dm != NULL) {
        return UCT_IB_MR_DEFAULT;
    }
#endif
    return uct_ib_md_get_atomic_mr_type(md);
}

UCS_PROFILE_FUNC_ALWAYS(ucs_status_t, uct_ib_mlx5_devx_reg_atomic_key,
                        (md, memh), uct_ib_mlx5_md_t *md,
                        uct_ib_mlx5_devx_mem_t *memh)
{
    uct_ib_mr_type_t mr_type = uct_ib_devx_get_atomic_mr_type(&md->super, memh);
    uct_ib_mlx5_devx_mr_t *mr = &memh->mrs[mr_type];
    uint8_t mr_id             = uct_ib_md_get_atomic_mr_id(&md->super);
    uint32_t atomic_offset    = uct_ib_md_atomic_offset(mr_id);
    uint32_t mkey_index;
    uint64_t iova;
    ucs_status_t status;
    int is_atomic;

    if (memh->smkey_mr != NULL) {
        mkey_index = uct_ib_mlx5_mkey_index(memh->super.rkey) +
                     md->super.mkey_by_name_reserve.size;
    } else {
        mkey_index = 0;
    }

    is_atomic = memh->super.flags & UCT_IB_MEM_ACCESS_REMOTE_ATOMIC;
    iova      = (uint64_t)memh->address + atomic_offset;

    if (memh->super.flags & UCT_IB_MEM_MULTITHREADED) {
        return uct_ib_mlx5_devx_reg_ksm_data(md, is_atomic, memh->address,
                                             mr->ksm_data, mr->ksm_data->length,
                                             iova, mkey_index,
                                             "multi-thread atomic key",
                                             &memh->atomic_dvmr,
                                             &memh->atomic_rkey);
    }

    status = uct_ib_mlx5_devx_reg_ksm_data_contig(
            md, mr, uct_ib_mlx5_devx_memh_base_address(memh), iova, is_atomic,
            mkey_index, "atomic key", &memh->atomic_dvmr, &memh->atomic_rkey);
    if (status != UCS_OK) {
        return status;
    }

    ucs_debug("KSM registered memory %p..%p lkey 0x%x offset 0x%x%s on %s rkey "
              "0x%x",
              memh->address,
              UCS_PTR_BYTE_OFFSET(memh->address, mr->super.ib->length),
              mr->super.ib->lkey, atomic_offset, is_atomic ? " atomic" : "",
              uct_ib_device_name(&md->super.dev), memh->atomic_rkey);
    return UCS_OK;
}

static ucs_status_t
uct_ib_mlx5_devx_reg_mt(uct_ib_mlx5_md_t *md, void *address, size_t length,
                        int is_atomic, const uct_md_mem_reg_params_t *params,
                        uint64_t access_flags, uint32_t *mkey_p,
                        uct_ib_mlx5_devx_ksm_data_t **ksm_data_p)
{
    size_t chunk = md->super.config.mt_reg_chunk;
    int mr_num   = ucs_div_round_up(length, chunk);
    uct_ib_mlx5_devx_ksm_data_t *ksm_data;
    ucs_status_t status;
    int dmabuf_fd;

    if (!(md->flags & UCT_IB_MLX5_MD_FLAG_KSM) ||
        (is_atomic && !(md->flags & UCT_IB_MLX5_MD_FLAG_INDIRECT_ATOMICS))) {
        return UCS_ERR_UNSUPPORTED;
    }

    /* Multi-threaded registration does not support dmabuf */
    dmabuf_fd = UCS_PARAM_VALUE(UCT_MD_MEM_REG_FIELD, params, dmabuf_fd,
                                DMABUF_FD, UCT_DMABUF_FD_INVALID);
    if (dmabuf_fd != UCT_DMABUF_FD_INVALID) {
        return UCS_ERR_UNSUPPORTED;
    }

    ucs_trace("multithreaded register memory %p..%p chunks %d", address,
              UCS_PTR_BYTE_OFFSET(address, length), mr_num);

    ksm_data = ucs_malloc((mr_num * sizeof(*ksm_data->mrs)) + sizeof(*ksm_data),
                          "ksm_data");
    if (ksm_data == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    ksm_data->mr_num = mr_num;
    ksm_data->length = length;

    status = uct_ib_md_handle_mr_list_mt(&md->super, address, length, params,
                                         access_flags, ksm_data->mrs);
    if (status != UCS_OK) {
        goto err_free;
    }

    status = uct_ib_mlx5_devx_reg_ksm_data(md, is_atomic, address, ksm_data,
                                           length, (uint64_t)address, 0,
                                           "multi-thread key", &ksm_data->dvmr,
                                           mkey_p);
    if (status != UCS_OK) {
        goto err_dereg;
    }

    *ksm_data_p = ksm_data;
    return UCS_OK;

err_dereg:
    uct_ib_md_handle_mr_list_mt(&md->super, address, length, NULL, 0,
                                ksm_data->mrs);
err_free:
    ucs_free(ksm_data);
err:
    return status;
}

static ucs_status_t
uct_ib_mlx5_devx_dereg_mt(uct_ib_mlx5_md_t *md,
                          uct_ib_mlx5_devx_ksm_data_t *ksm_data)
{
    ucs_status_t status;
    struct ibv_mr **mr;

    ucs_trace("%s: destroy KSM %p", uct_ib_device_name(&md->super.dev),
              ksm_data->dvmr);

    status = uct_ib_mlx5_devx_obj_destroy(ksm_data->dvmr, "MKEY, KSM");
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ib_md_handle_mr_list_mt(&md->super, 0, ksm_data->length, NULL,
                                         0, ksm_data->mrs);
    if (status == UCS_ERR_UNSUPPORTED) {
        /* Fallback to direct deregistration */
        ucs_carray_for_each(mr, ksm_data->mrs, ksm_data->mr_num) {
            status = uct_ib_dereg_mr(*mr);
            if (status != UCS_OK) {
                return status;
            }
        }
    } else if (status != UCS_OK) {
        return status;
    }

    ucs_free(ksm_data);
    return status;
}

static void uct_ib_mlx5_devx_reg_symmetric(uct_ib_mlx5_md_t *md,
                                           uct_ib_mlx5_devx_mem_t *memh,
                                           void *address)
{
    uint32_t start = md->smkey_index;
    struct mlx5dv_devx_obj *smkey_mr;
    uint32_t symmetric_rkey;
    ucs_status_t status;

    ucs_assert(!(memh->super.flags & UCT_IB_MEM_MULTITHREADED));

    /* Best effort, only allocate in the range below the atomic keys. */
    while (md->smkey_index < md->super.mkey_by_name_reserve.size) {
        status = uct_ib_mlx5_devx_reg_ksm_data_contig(
                md, &memh->mrs[UCT_IB_MR_DEFAULT], address, (uint64_t)address,
                (memh->super.flags & UCT_IB_MEM_ACCESS_REMOTE_ATOMIC),
                md->super.mkey_by_name_reserve.base + md->smkey_index, NULL,
                &smkey_mr, &symmetric_rkey);
        if (status == UCS_OK) {
            ucs_trace("%s: symmetric rkey created for addr=%p "
                      "mkey_index=0x%x smkey_mr=%p rkey=0x%x",
                      uct_ib_device_name(&md->super.dev), address,
                      md->super.mkey_by_name_reserve.base + start, smkey_mr,
                      symmetric_rkey);

            memh->smkey_mr   = smkey_mr;
            memh->super.rkey = symmetric_rkey;
            md->smkey_index++;
            return;
        }

        /* Use blocks of 8 mkeys, first mkey creation gives block ownership.
         * Try from the start of the next block if any failure.
         */
        md->smkey_index = ucs_align_up_pow2(md->smkey_index + 1,
                                            md->super.config.smkey_block_size);
    }

    ucs_debug("%s: symmetric rkey create failed for addr=%p mkey_index=0x%x",
              uct_ib_device_name(&md->super.dev), address,
              md->super.mkey_by_name_reserve.base + start);
}

static UCS_F_ALWAYS_INLINE int
uct_ib_mlx5_devx_symmetric_rkey(const uct_ib_mlx5_md_t *md, unsigned flags)
{
    return (flags & UCT_MD_MEM_SYMMETRIC_RKEY) &&
           (md->flags & UCT_IB_MLX5_MD_FLAG_MKEY_BY_NAME_RESERVE);
}

static ucs_status_t
uct_ib_mlx5_devx_reg_mr(uct_ib_mlx5_md_t *md, uct_ib_mlx5_devx_mem_t *memh,
                        void *address, size_t length,
                        const uct_md_mem_reg_params_t *params,
                        uct_ib_mr_type_t mr_type, uint64_t access_mask,
                        uint32_t *lkey_p, uint32_t *rkey_p)
{
    uint64_t access_flags = uct_ib_memh_access_flags(&md->super, &memh->super) &
                            access_mask;
    unsigned flags        = UCT_MD_MEM_REG_FIELD_VALUE(params, flags,
                                                       FIELD_FLAGS, 0);
    ucs_status_t status;
    uint32_t mkey;

    if ((length >= md->super.config.min_mt_reg) &&
        !(access_flags & IBV_ACCESS_ON_DEMAND) &&
        !uct_ib_mlx5_devx_symmetric_rkey(md, flags)) {
        /* Verbs transports can issue atomic operations to the default key */
        status = uct_ib_mlx5_devx_reg_mt(md, address, length,
                                         (memh->super.flags &
                                          UCT_IB_MEM_ACCESS_REMOTE_ATOMIC),
                                         params, access_flags, &mkey,
                                         &memh->mrs[mr_type].ksm_data);
        if (status == UCS_OK) {
            *rkey_p = *lkey_p = mkey;
            memh->super.flags |= UCT_IB_MEM_MULTITHREADED;
            return UCS_OK;
        } else if (status != UCS_ERR_UNSUPPORTED) {
            return status;
        }

        /* Fallback if multi-thread registration is unsupported */
    }

    status = uct_ib_reg_mr(&md->super, address, length, params, access_flags,
                           NULL, &memh->mrs[mr_type].super.ib);
    if (status != UCS_OK) {
        return status;
    }

    *lkey_p = memh->mrs[mr_type].super.ib->lkey;
    *rkey_p = memh->mrs[mr_type].super.ib->rkey;
    return UCS_OK;
}

static ucs_status_t uct_ib_mlx5_devx_dereg_mr(uct_ib_mlx5_md_t *md,
                                              uct_ib_mlx5_devx_mem_t *memh,
                                              uct_ib_mr_type_t mr_type)
{
    if (memh->super.flags & UCT_IB_MEM_MULTITHREADED) {
        return uct_ib_mlx5_devx_dereg_mt(md, memh->mrs[mr_type].ksm_data);
    } else {
        return uct_ib_dereg_mr(memh->mrs[mr_type].super.ib);
    }
}

static ucs_status_t
uct_ib_mlx5_devx_mem_reg(uct_md_h uct_md, void *address, size_t length,
                         const uct_md_mem_reg_params_t *params,
                         uct_mem_h *memh_p)
{
    uct_ib_mlx5_md_t *md = ucs_derived_of(uct_md, uct_ib_mlx5_md_t);
    unsigned flags = UCT_MD_MEM_REG_FIELD_VALUE(params, flags, FIELD_FLAGS, 0);
    uct_ib_mlx5_devx_mem_t *memh;
    uct_ib_mem_t *ib_memh;
    ucs_status_t status;
    uint32_t dummy_mkey;

    status = uct_ib_memh_alloc(&md->super, length, flags, sizeof(*memh),
                               sizeof(memh->mrs[0]), &ib_memh);
    if (status != UCS_OK) {
        goto err;
    }

    memh                = ucs_derived_of(ib_memh, uct_ib_mlx5_devx_mem_t);
    memh->exported_lkey = UCT_IB_INVALID_MKEY;
    memh->atomic_rkey   = UCT_IB_INVALID_MKEY;
    memh->indirect_rkey = UCT_IB_INVALID_MKEY;

    status = uct_ib_mlx5_devx_reg_mr(md, memh, address, length, params,
                                     UCT_IB_MR_DEFAULT, UINT64_MAX,
                                     &memh->super.lkey, &memh->super.rkey);
    if (status != UCS_OK) {
        goto err_memh_free;
    }

    if (uct_ib_mlx5_devx_symmetric_rkey(md, flags)) {
        uct_ib_mlx5_devx_reg_symmetric(md, memh, address);
    }

    if (md->super.relaxed_order) {
        status = uct_ib_mlx5_devx_reg_mr(md, memh, address, length, params,
                                         UCT_IB_MR_STRICT_ORDER,
                                         ~IBV_ACCESS_RELAXED_ORDERING,
                                         &dummy_mkey, &dummy_mkey);
        if (status != UCS_OK) {
            goto err_dereg_default;
        }
    }

    if (md->super.config.odp.prefetch) {
        uct_ib_mem_prefetch(&md->super, &memh->super, address, length);
    }

    memh->address = address;
    *memh_p       = memh;
    return UCS_OK;

err_dereg_default:
    uct_ib_mlx5_devx_dereg_mr(md, memh, UCT_IB_MR_DEFAULT);
err_memh_free:
    ucs_free(memh);
err:
    return status;
}

static ucs_status_t
uct_ib_devx_dereg_invalidate_rkey_check(uct_ib_mlx5_md_t *md,
                                        uct_ib_mlx5_devx_mem_t *memh,
                                        uint32_t rkey, unsigned flags_mask,
                                        uint64_t cap_mask, const char *name)
{
    if (!(memh->super.flags & flags_mask)) {
        return UCS_OK;
    }

    if (!(md->super.cap_flags & cap_mask)) {
        ucs_debug("%s: invalidate %s is not supported (rkey=0x%x)",
                  uct_ib_device_name(&md->super.dev), name, rkey);
        return UCS_ERR_UNSUPPORTED;
    }

    if (rkey == UCT_IB_INVALID_MKEY) {
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

static ucs_status_t uct_ib_devx_dereg_invalidate_params_check(
        uct_ib_mlx5_md_t *md, const uct_md_mem_dereg_params_t *params)
{
    uct_ib_mlx5_devx_mem_t *memh;
    ucs_status_t status;
    unsigned flags;

    flags = UCT_MD_MEM_DEREG_FIELD_VALUE(params, flags, FIELD_FLAGS, 0);
    if (!(flags & UCT_MD_MEM_DEREG_FLAG_INVALIDATE)) {
        return UCS_OK;
    }

    memh   = UCT_MD_MEM_DEREG_FIELD_VALUE(params, memh, FIELD_MEMH, NULL);
    status = uct_ib_devx_dereg_invalidate_rkey_check(
            md, memh, memh->indirect_rkey, UCT_IB_MEM_ACCESS_REMOTE_RMA,
            UCT_MD_FLAG_INVALIDATE_RMA, "RMA");
    if (status != UCS_OK) {
        return status;
    }

    return uct_ib_devx_dereg_invalidate_rkey_check(
            md, memh, memh->atomic_rkey, UCT_IB_MEM_ACCESS_REMOTE_ATOMIC,
            UCT_MD_FLAG_INVALIDATE_AMO, "AMO");
}

static ucs_status_t
uct_ib_mlx5_devx_dereg_keys(uct_ib_mlx5_md_t *md, uct_ib_mlx5_devx_mem_t *memh)
{
    ucs_status_t status;

    if (memh->atomic_dvmr != NULL) {
        /* TODO atomic_dvmr should also be pushed to LRU since it can be used
           to invalidate AMO or RMA with relaxed-order */
        status = uct_ib_mlx5_devx_obj_destroy(memh->atomic_dvmr,
                                              "MKEY, ATOMIC");
        if (status != UCS_OK) {
            return status;
        }
    }

    if (memh->indirect_dvmr != NULL) {
        uct_ib_md_mlx5_devx_mr_lru_push(md, memh->indirect_rkey, NULL);
        ucs_trace("%s: destroy indirect_dvmr %p with key %x",
                  uct_ib_device_name(&md->super.dev), memh->indirect_dvmr,
                  memh->indirect_rkey);
        status = uct_ib_mlx5_devx_obj_destroy(memh->indirect_dvmr,
                                              "MKEY, INDIRECT");
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

static ucs_status_t
uct_ib_mlx5_devx_mem_dereg(uct_md_h uct_md,
                           const uct_md_mem_dereg_params_t *params)
{
    uct_ib_mlx5_md_t *md = ucs_derived_of(uct_md, uct_ib_mlx5_md_t);
    uct_ib_mlx5_devx_mem_t *memh;
    ucs_status_t status;
    int ret;

    UCT_MD_MEM_DEREG_CHECK_PARAMS(params, 1);
    if (ENABLE_PARAMS_CHECK) {
        status = uct_ib_devx_dereg_invalidate_params_check(md, params);
        if (status != UCS_OK) {
            return status;
        }
    }

    memh   = ucs_derived_of(params->memh, uct_ib_mlx5_devx_mem_t);
    status = uct_ib_mlx5_devx_dereg_keys(md, memh);
    if (status != UCS_OK) {
        return status;
    }

    if (memh->smkey_mr != NULL) {
        ucs_trace("%s: destroy smkey_mr %p with key %x",
                  uct_ib_device_name(&md->super.dev), memh->smkey_mr,
                  memh->super.rkey);
        status = uct_ib_mlx5_devx_obj_destroy(memh->smkey_mr,
                                              "MKEY, SYMMETRIC");
        if (status != UCS_OK) {
            return status;
        }
    }

    if (memh->cross_mr != NULL) {
        ucs_trace("%s: destroy cross_mr %p with key %x",
                  uct_ib_device_name(&md->super.dev), memh->cross_mr,
                  memh->exported_lkey);
        status = uct_ib_mlx5_devx_obj_destroy(memh->cross_mr, "CROSS_MR");
        if (status != UCS_OK) {
            return status;
        }
    }

    if (memh->umem != NULL) {
        ret = mlx5dv_devx_umem_dereg(memh->umem);
        if (ret < 0) {
            ucs_error("mlx5dv_devx_umem_dereg(crossmr) failed: %m");
            return UCS_ERR_IO_ERROR;
        }
    }

    if (!(memh->super.flags & UCT_IB_MEM_IMPORTED)) {
        if (md->super.relaxed_order) {
            status = uct_ib_mlx5_devx_dereg_mr(md, memh,
                                               UCT_IB_MR_STRICT_ORDER);
            if (status != UCS_OK) {
                return status;
            }
        }

        status = uct_ib_mlx5_devx_dereg_mr(md, memh, UCT_IB_MR_DEFAULT);
        if (status != UCS_OK) {
            return status;
        }
    }

    if (UCT_MD_MEM_DEREG_FIELD_VALUE(params, flags, FIELD_FLAGS, 0) &
        UCT_MD_MEM_DEREG_FLAG_INVALIDATE) {
        ucs_assert(params->comp != NULL); /* suppress coverity false-positive */
        uct_invoke_completion(params->comp, UCS_OK);
    }

    ucs_free(memh);
    return UCS_OK;
}

static ucs_status_t uct_ib_mlx5_add_page(ucs_mpool_t *mp, size_t *size_p, void **page_p)
{
    uct_ib_mlx5_md_t *md = ucs_container_of(mp, uct_ib_mlx5_md_t, dbrec_pool);
    uct_ib_mlx5_dbrec_page_t *page;
    size_t size = ucs_align_up(*size_p + sizeof(*page), ucs_get_page_size());
    uct_ib_mlx5_devx_umem_t mem;
    ucs_status_t status;

    status = uct_ib_mlx5_md_buf_alloc(md, size, 1, (void**)&page, &mem, 0,
                                      "devx dbrec");
    if (status != UCS_OK) {
        return status;
    }

    page->mem = mem;
    *size_p   = size - sizeof(*page);
    *page_p   = page + 1;
    return UCS_OK;
}

static void uct_ib_mlx5_init_dbrec(ucs_mpool_t *mp, void *obj, void *chunk)
{
    uct_ib_mlx5_dbrec_page_t *page = (uct_ib_mlx5_dbrec_page_t*)chunk - 1;
    uct_ib_mlx5_dbrec_t *dbrec     = obj;

    dbrec->mem_id = page->mem.mem->umem_id;
    dbrec->offset = UCS_PTR_BYTE_DIFF(chunk, obj) + sizeof(*page);
}

static void uct_ib_mlx5_free_page(ucs_mpool_t *mp, void *chunk)
{
    uct_ib_mlx5_md_t *md = ucs_container_of(mp, uct_ib_mlx5_md_t, dbrec_pool);
    uct_ib_mlx5_dbrec_page_t *page = (uct_ib_mlx5_dbrec_page_t*)chunk - 1;
    uct_ib_mlx5_md_buf_free(md, page, &page->mem);
}

static ucs_mpool_ops_t uct_ib_mlx5_dbrec_ops = {
    .chunk_alloc   = uct_ib_mlx5_add_page,
    .chunk_release = uct_ib_mlx5_free_page,
    .obj_init      = uct_ib_mlx5_init_dbrec,
    .obj_cleanup   = NULL,
    .obj_str       = NULL
};

static void uct_ib_mlx5_devx_check_odp(uct_ib_mlx5_md_t *md,
                                       const uct_ib_md_config_t *md_config,
                                       void *cap)
{
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(query_hca_cap_out)] = {};
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(query_hca_cap_in)]   = {};
    ucs_status_t status;
    const void *odp_cap;
    const char *reason;
    uint8_t version;

    if (IBV_ACCESS_ON_DEMAND == 0) {
        reason = "IBV_ACCESS_ON_DEMAND is not supported";
        goto no_odp;
    }

    if (!IBV_DEVICE_HAS_ODP(&md->super.dev)) {
        reason = "device does not support ODP";
        goto no_odp;
    }

    if (!UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, pg)) {
        reason = "cap.pg is not supported";
        goto no_odp;
    }

    if (!UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, umr_extended_translation_offset)) {
        reason = "cap.umr_extended_translation_offset is not supported";
        goto no_odp;
    }

    UCT_IB_MLX5DV_SET(query_hca_cap_in, in, opcode, UCT_IB_MLX5_CMD_OP_QUERY_HCA_CAP);
    UCT_IB_MLX5DV_SET(query_hca_cap_in, in, op_mod, UCT_IB_MLX5_HCA_CAP_OPMOD_GET_CUR |
                                                   (UCT_IB_MLX5_CAP_ODP << 1));
    status = uct_ib_mlx5_devx_general_cmd(md->super.dev.ibv_context, in,
                                          sizeof(in), out, sizeof(out),
                                          "QUERY_HCA_CAP, ODP", 0);
    if (status != UCS_OK) {
        reason = "faied to query HCA capabilities";
        goto no_odp;
    }

    if (UCT_IB_MLX5DV_GET(query_hca_cap_out, out,
                          capability.odp_cap.mem_page_fault)) {
        odp_cap = UCT_IB_MLX5DV_ADDR_OF(
                query_hca_cap_out, out,
                capability.odp_cap.memory_page_fault_scheme_cap);
        version = 2;
    } else {
        if (md_config->devx_objs & UCS_BIT(UCT_IB_DEVX_OBJ_RCQP)) {
            reason = "version 1 is not supported for DevX QP";
            goto no_odp;
        }

        odp_cap = UCT_IB_MLX5DV_ADDR_OF(
                query_hca_cap_out, out,
                capability.odp_cap.transport_page_fault_scheme_cap);
        version = 1;
    }

    if (!UCT_IB_MLX5DV_GET(odp_scheme_cap, odp_cap, ud_odp_caps.send) ||
        !UCT_IB_MLX5DV_GET(odp_scheme_cap, odp_cap, rc_odp_caps.send) ||
        !UCT_IB_MLX5DV_GET(odp_scheme_cap, odp_cap, rc_odp_caps.write) ||
        !UCT_IB_MLX5DV_GET(odp_scheme_cap, odp_cap, rc_odp_caps.read)) {
        reason = "it's not suported for UD/RC transports";
        goto no_odp;
    }

    if ((md->super.dev.flags & UCT_IB_DEVICE_FLAG_DC) &&
        (!UCT_IB_MLX5DV_GET(odp_scheme_cap, odp_cap, dc_odp_caps.send) ||
         !UCT_IB_MLX5DV_GET(odp_scheme_cap, odp_cap, dc_odp_caps.write) ||
         !UCT_IB_MLX5DV_GET(odp_scheme_cap, odp_cap, dc_odp_caps.read))) {
        reason = "it's not suported for DC transport";
        goto no_odp;
    }

    ucs_debug("%s: ODP is supported, version %d",
              uct_ib_device_name(&md->super.dev), version);
    md->super.reg_nonblock_mem_types = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    return;

no_odp:
    ucs_debug("%s: ODP is disabled because %s",
              uct_ib_device_name(&md->super.dev), reason);
}

static uct_ib_port_select_mode_t
uct_ib_mlx5_devx_query_port_select(uct_ib_mlx5_md_t *md)
{
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(query_lag_out)] = {};
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(query_lag_in)]   = {};
    void *lag;
    uint8_t port_select_mode;
    ucs_status_t status;

    lag = UCT_IB_MLX5DV_ADDR_OF(query_lag_out, out, lag_context);
    UCT_IB_MLX5DV_SET(query_lag_in, in, opcode, UCT_IB_MLX5_CMD_OP_QUERY_LAG);
    status = uct_ib_mlx5_devx_general_cmd(md->super.dev.ibv_context, in,
                                          sizeof(in), out, sizeof(out),
                                          "QUERY_LAG", 0);
    if (status != UCS_OK) {
        return UCT_IB_MLX5_LAG_INVALID_MODE;
    }

    port_select_mode = UCT_IB_MLX5DV_GET(lag_context, lag, port_select_mode);

    if (port_select_mode > UCT_IB_MLX5_LAG_MULTI_PORT_ESW) {
        return UCT_IB_MLX5_LAG_INVALID_MODE;
    }

    return port_select_mode;
}

static ucs_status_t
uct_ib_mlx5_devx_query_lag(uct_ib_mlx5_md_t *md, uint8_t *state)
{
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(query_lag_out)] = {};
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(query_lag_in)]  = {};
    void *lag;
    ucs_status_t status;

    lag = UCT_IB_MLX5DV_ADDR_OF(query_lag_out, out, lag_context);
    UCT_IB_MLX5DV_SET(query_lag_in, in, opcode, UCT_IB_MLX5_CMD_OP_QUERY_LAG);
    status = uct_ib_mlx5_devx_general_cmd(md->super.dev.ibv_context, in,
                                          sizeof(in), out, sizeof(out),
                                          "QUERY_LAG", 0);
    if (status != UCS_OK) {
        return status;
    }

    *state = UCT_IB_MLX5DV_GET(lag_context, lag, lag_state);
    return UCS_OK;
}

static struct ibv_context *
uct_ib_mlx5_devx_open_device(struct ibv_device *ibv_device)
{
    struct mlx5dv_context_attr dv_attr = {};
    struct mlx5dv_devx_event_channel *event_channel;
    struct ibv_context *ctx;
    struct ibv_cq *cq;

    dv_attr.flags |= MLX5DV_CONTEXT_FLAGS_DEVX;
    ctx = mlx5dv_open_device(ibv_device, &dv_attr);
    if (ctx == NULL) {
        ucs_debug("mlx5dv_open_device(%s) failed: %m",
                  ibv_get_device_name(ibv_device));
        return NULL;
    }

    cq = ibv_create_cq(ctx, 1, NULL, NULL, 0);
    if (cq == NULL) {
        uct_ib_check_memlock_limit_msg(UCS_LOG_LEVEL_DEBUG,
                                       "%s: ibv_create_cq()",
                                       ibv_get_device_name(ibv_device));
        goto close_ctx;
    }

    ibv_destroy_cq(cq);

    event_channel = mlx5dv_devx_create_event_channel(
            ctx, MLX5_IB_UAPI_DEVX_CR_EV_CH_FLAGS_OMIT_DATA);
    if (event_channel == NULL) {
        ucs_diag("mlx5dv_devx_create_event_channel(%s) failed: %m",
                 ibv_get_device_name(ibv_device));
        goto close_ctx;
    }

    mlx5dv_devx_destroy_event_channel(event_channel);

    return ctx;

close_ctx:
    ibv_close_device(ctx);
    return NULL;
}

static uct_ib_md_ops_t uct_ib_mlx5_devx_md_ops;

static void uct_ib_mlx5_devx_init_flush_mr(uct_ib_mlx5_md_t *md)
{
    uct_md_mem_reg_params_t params;
    ucs_status_t status;

    if (!(md->flags & UCT_IB_MLX5_MD_FLAG_KSM)) {
        md->super.flush_rkey = UCT_IB_MD_INVALID_FLUSH_RKEY;
        return;
    }

    ucs_assert(UCT_IB_MD_FLUSH_REMOTE_LENGTH <= ucs_get_page_size());

    params.field_mask = UCT_MD_MEM_REG_FIELD_FLAGS;
    params.flags      = UCT_MD_MEM_FLAG_HIDE_ERRORS;

    status = uct_ib_reg_mr(&md->super, md->zero_buf,
                           UCT_IB_MD_FLUSH_REMOTE_LENGTH, &params,
                           UCT_IB_MEM_ACCESS_FLAGS, NULL, &md->flush_mr);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_ib_mlx5_devx_reg_ksm_data_addr(md, md->flush_mr,
                                                (uintptr_t)md->zero_buf,
                                                UCT_IB_MD_FLUSH_REMOTE_LENGTH,
                                                0, 0, 1, 0, "flush mr",
                                                &md->flush_dvmr,
                                                &md->super.flush_rkey);
    if (status != UCS_OK) {
        ucs_error("failed to create flush_remote rkey: %s",
                  ucs_status_string(status));
        goto err_dereg_mr;
    }

    ucs_debug("created indirect rkey 0x%x for remote flush",
              md->super.flush_rkey);
    ucs_assert((md->super.flush_rkey & 0xff) == 0);
    return;

err_dereg_mr:
    uct_ib_dereg_mr(md->flush_mr);
    md->flush_mr = NULL;
err:
    md->super.flush_rkey = uct_ib_mlx5_flush_rkey_make();
}

static ucs_status_t
uct_ib_mlx5_devx_query_cap_2(struct ibv_context *ctx, void *out, size_t size)
{
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(query_hca_cap_in)] = {};

    /* query HCA CAP 2 */
    UCT_IB_MLX5DV_SET(query_hca_cap_in, in, opcode,
                      UCT_IB_MLX5_CMD_OP_QUERY_HCA_CAP);
    UCT_IB_MLX5DV_SET(query_hca_cap_in, in, op_mod,
                      UCT_IB_MLX5_HCA_CAP_OPMOD_GET_CUR |
                              (UCT_IB_MLX5_CAP_2_GENERAL << 1));
    return uct_ib_mlx5_devx_general_cmd(ctx, in, sizeof(in), out, size,
                                        "QUERY_HCA_CAP, CAP2", 1);
}

static void uct_ib_mlx5_devx_check_xgvmi(uct_ib_mlx5_md_t *md, void *cap_2,
                                         uct_ib_device_t *dev)
{
    uint64_t object_for_other_vhca;
    uint32_t object_to_object;

    object_to_object      = UCT_IB_MLX5DV_GET(cmd_hca_cap_2, cap_2,
                                              cross_vhca_object_to_object_supported);
    object_for_other_vhca = UCT_IB_MLX5DV_GET64(
            cmd_hca_cap_2, cap_2, allowed_object_for_other_vhca_access);

    if ((object_to_object &
         UCT_IB_MLX5_HCA_CAPS_2_CROSS_VHCA_OBJ_TO_OBJ_LOCAL_MKEY_TO_REMOTE_MKEY) &&
        (object_for_other_vhca &
         UCT_IB_MLX5_HCA_CAPS_2_ALLOWED_OBJ_FOR_OTHER_VHCA_ACCESS_MKEY)) {
        md->flags           |= UCT_IB_MLX5_MD_FLAG_INDIRECT_XGVMI;
        md->super.cap_flags |= UCT_MD_FLAG_EXPORTED_MKEY;
        ucs_debug("%s: cross gvmi alias mkey is supported",
                  uct_ib_device_name(dev));
    } else {
        ucs_debug("%s: crossing_vhca_mkey is not supported",
                  uct_ib_device_name(dev));
    }
}

static void uct_ib_mlx5_devx_check_mkey_by_name(uct_ib_mlx5_md_t *md,
                                                void *cap_2,
                                                uct_ib_device_t *dev)
{
    int log_size;

    ucs_assertv(md->super.mkey_by_name_reserve.size == 0,
                "mkey_by_name_reserve.size=%u",
                md->super.mkey_by_name_reserve.size);

    if (UCT_IB_MLX5DV_GET(cmd_hca_cap_2, cap_2, mkey_by_name_reserve) == 0) {
        ucs_debug("%s: mkey_by_name_reserve is not supported",
                  uct_ib_device_name(dev));
        return;
    }

    log_size = UCT_IB_MLX5DV_GET(cmd_hca_cap_2, cap_2,
                                 mkey_by_name_reserve_log_size);

    md->super.mkey_by_name_reserve.base =
            UCT_IB_MLX5DV_GET(cmd_hca_cap_2, cap_2, mkey_by_name_reserve_base);

    /* The direct key and atomic key must have constant offset so that remote
     * keys can eventually compare equal.
     *
     * So first half of the range is used for allocation of smallest mkey,
     * other half will be used to create atomic key explicitly.
     */
    md->super.mkey_by_name_reserve.size = UCS_BIT(log_size) / 2;
    md->flags           |= UCT_IB_MLX5_MD_FLAG_MKEY_BY_NAME_RESERVE;

    ucs_debug("%s: mkey_by_name_reserve is supported, base=0x%x size=%u",
              uct_ib_device_name(dev), md->super.mkey_by_name_reserve.base,
              md->super.mkey_by_name_reserve.size);
}

static void uct_ib_mlx5_md_port_counter_set_id_init(uct_ib_mlx5_md_t *md)
{
    uint8_t *counter_set_id;

    ucs_carray_for_each(counter_set_id, md->port_counter_set_ids,
                        sizeof(md->port_counter_set_ids)) {
        *counter_set_id = UCT_IB_COUNTER_SET_ID_INVALID;
    }
}

static int uct_ib_mlx5_check_uar(uct_ib_mlx5_md_t *md)
{
    uct_ib_mlx5_devx_uar_t uar;
    ucs_status_t status;

    status = uct_ib_mlx5_devx_uar_init(&uar, md, 0);
    if (status != UCS_OK) {
        return UCS_ERR_UNSUPPORTED;
    }

    uct_ib_mlx5_devx_uar_cleanup(&uar);
    return UCS_OK;
}

static ucs_status_t
uct_ib_mlx5_devx_device_mem_alloc(uct_md_h uct_md, size_t *length_p,
                                  void **address_p, ucs_memory_type_t mem_type,
                                  unsigned flags, const char *alloc_name,
                                  uct_mem_h *memh_p)
{
#if HAVE_IBV_DM
    uct_ib_md_t *md           = ucs_derived_of(uct_md, uct_ib_md_t);
    uct_ib_mlx5_md_t *devx_md = ucs_derived_of(md, uct_ib_mlx5_md_t);
    struct ibv_alloc_dm_attr dm_attr;
    uct_md_mem_reg_params_t reg_params;
    unsigned mem_flags;
    struct ibv_dm *dm;
    uct_ib_mlx5_devx_mem_t *memh;
    void *address;
    ucs_status_t status;
    uint32_t mkey;

    if (mem_type != UCS_MEMORY_TYPE_RDMA) {
        return UCS_ERR_UNSUPPORTED;
    }

    /* Align the allocation to a potential use of registration cache */
    dm_attr.length        = ucs_align_up_pow2(*length_p, md->dev.atomic_align);
    dm_attr.log_align_req = ucs_ilog2(md->dev.atomic_align);
    dm_attr.comp_mask     = 0;

    if (dm_attr.length > md->dev.dev_attr.max_dm_size) {
        ucs_error("%s: device memory allocation length (%zu) exceeds maximal "
                  "supported size (%zu)",
                  uct_ib_device_name(&md->dev), dm_attr.length,
                  md->dev.dev_attr.max_dm_size);
        return UCS_ERR_NO_MEMORY;
    }

    mem_flags = UCT_MD_MEM_ACCESS_REMOTE_GET | UCT_MD_MEM_ACCESS_REMOTE_PUT |
                UCT_MD_MEM_ACCESS_REMOTE_ATOMIC;

    status = uct_ib_memh_alloc(md, dm_attr.length, mem_flags,
                               sizeof(uct_ib_mlx5_devx_mem_t),
                               sizeof(struct ibv_mr), (uct_ib_mem_t**)&memh);
    if (status != UCS_OK) {
        return UCS_ERR_NO_MEMORY;
    }

    dm = UCS_PROFILE_CALL(ibv_alloc_dm, md->dev.ibv_context, &dm_attr);
    if (dm == NULL) {
        ucs_debug("%s: ibv_alloc_dm(length=%zu) failed: %m",
                  uct_ib_device_name(&md->dev), dm_attr.length);
        status = UCS_ERR_NO_MEMORY;
        goto err_free_memh;
    }

    /* Reserve non-accessible address range of the same length as the allocated dm */
    address = mmap(NULL, dm_attr.length, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS,
                   -1, 0);
    if (address == MAP_FAILED) {
        ucs_debug("failed to reserve virtual address for dm, length: %zu",
                  dm_attr.length);
        status = UCS_ERR_NO_MEMORY;
        goto err_free_dm;
    }

    reg_params.field_mask = 0;
    status = uct_ib_reg_mr(md, address, dm_attr.length, &reg_params,
                           UCT_IB_MEM_ACCESS_FLAGS, dm,
                           &memh->mrs[UCT_IB_MR_DEFAULT].super.ib);
    if (status != UCS_OK) {
        goto err_munmap_address;
    }

    /* Mapping address to the dm allocation (which is zero based) using ksm,
       enabling put / get operations using that address range */
    status = uct_ib_mlx5_devx_reg_ksm_data_addr(
            devx_md, memh->mrs[UCT_IB_MR_DEFAULT].super.ib, 0, dm_attr.length,
            (uintptr_t)address, 0, 1, 0, alloc_name, &memh->dm_addr_dvmr, &mkey);
    if (status != UCS_OK) {
        goto err_dereg_dm;
    }

    memh->dm      = dm;
    memh->address = address;
    *length_p     = dm_attr.length;
    *address_p    = address;
    *memh_p       = memh;
    return UCS_OK;

err_dereg_dm:
    uct_ib_dereg_mr(memh->mrs[UCT_IB_MR_DEFAULT].super.ib);
err_munmap_address:
    munmap(address, dm_attr.length);
err_free_dm:
    ibv_free_dm(dm);
err_free_memh:
    ucs_free(memh);
    return status;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

static ucs_status_t
uct_ib_mlx5_devx_device_mem_free(uct_md_h uct_md, uct_mem_h tl_memh)
{
#if HAVE_IBV_DM
    uct_ib_mlx5_md_t *md         = ucs_derived_of(uct_md, uct_ib_mlx5_md_t);
    uct_ib_mlx5_devx_mem_t *memh = tl_memh;
    struct ibv_dm *dm            = memh->dm;
    size_t length = memh->mrs[UCT_IB_MR_DEFAULT].super.ib->length;
    ucs_status_t status;
    int ret;

    uct_ib_mlx5_devx_obj_destroy(memh->dm_addr_dvmr, "DM-KSM");

    ret = munmap(memh->address, length);
    if (ret != 0) {
        ucs_warn("munmap(address=%p, length=%zu) failed: %m", memh->address,
                 length);
    }

    status = uct_ib_mlx5_devx_dereg_keys(md, memh);
    if (status != UCS_OK) {
        ucs_warn("%s: uct_ib_mlx5_devx_dereg_keys() failed",
                 ucs_status_string(status));
    }

    status = uct_ib_mlx5_devx_dereg_mr(md, tl_memh, UCT_IB_MR_DEFAULT);
    if (status != UCS_OK) {
        return status;
    }

    ret = UCS_PROFILE_CALL(ibv_free_dm, dm);
    if (ret) {
        ucs_warn("ibv_free_dm() failed: %m");
        status = UCS_ERR_BUSY;
    }

    ucs_free(memh);
    return status;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

static void uct_ib_mlx5dv_check_dm_ksm_reg(uct_ib_mlx5_md_t *md)
{
#if HAVE_IBV_DM
    size_t length   = 1;
    uct_md_h uct_md = (uct_md_h)&md->super;
    uct_ib_mlx5_devx_mem_t *devx_memh;
    void *address;
    uct_mem_h memh;
    ucs_status_t status;

    if (md->super.dev.dev_attr.max_dm_size == 0) {
        return;
    }

    status = uct_ib_mlx5_devx_device_mem_alloc(uct_md, &length, &address,
                                               UCS_MEMORY_TYPE_RDMA, 0,
                                               "check dm ksm registration",
                                               &memh);
    if (status != UCS_OK) {
        ucs_debug("%s: KSM over device memory is not supported",
                  ucs_status_string(status));
        return;
    }

    devx_memh = ucs_derived_of(memh, uct_ib_mlx5_devx_mem_t);
    status    = uct_ib_mlx5_devx_reg_atomic_key(md, devx_memh);
    if (status == UCS_OK) {
        /* Enable device memory only if atomics are available*/
        md->super.cap_flags |= UCT_MD_FLAG_ALLOC;
    }

    status = uct_ib_mlx5_devx_device_mem_free(uct_md, memh);
    if (status != UCS_OK) {
        ucs_diag("%s: failed to free dm allocated in check_dm_ksm_reg",
                 ucs_status_string(status));
        return;
    }
#endif
}

static ucs_status_t uct_ib_mlx5_devx_md_open(struct ibv_device *ibv_device,
                                             const uct_ib_md_config_t *md_config,
                                             uct_ib_md_t **p_md)
{
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(query_hca_cap_out)] = {};
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(query_hca_cap_in)]   = {};
    char cap_2_out[UCT_IB_MLX5DV_ST_SZ_BYTES(query_hca_cap_out)] = {};
    ucs_status_t status                                    = UCS_OK;
    uint8_t lag_state                                      = 0;
    void *cap_2;
    uint8_t log_max_qp;
    uint16_t vhca_id;
    struct ibv_context *ctx;
    uct_ib_device_t *dev;
    uct_ib_mlx5_md_t *md;
    unsigned max_rd_atomic_dc;
    void *cap;
    int ret;
    ucs_log_level_t log_level;
    ucs_mpool_params_t mp_params;
    int ksm_atomic;

    if (!mlx5dv_is_supported(ibv_device)) {
        status = UCS_ERR_UNSUPPORTED;
        goto err;
    }

    if (md_config->devx == UCS_NO) {
        status = UCS_ERR_UNSUPPORTED;
        goto err;
    }

    ctx = uct_ib_mlx5_devx_open_device(ibv_device);
    if (ctx == NULL) {
        status = UCS_ERR_UNSUPPORTED;
        goto err;
    }

    md = ucs_derived_of(uct_ib_md_alloc(sizeof(*md), "ib_mlx5_devx_md", ctx),
                        uct_ib_mlx5_md_t);
    if (md == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_context;
    }

    status = uct_ib_mlx5_check_uar(md);
    if (status != UCS_OK) {
        goto err_free_md;
    }

    dev          = &md->super.dev;
    md->mkey_tag = 0;
    uct_ib_mlx5_devx_mr_lru_init(md);

    status = uct_ib_device_query(dev, ibv_device);
    if (status != UCS_OK) {
        goto err_lru_cleanup;
    }

    cap = UCT_IB_MLX5DV_ADDR_OF(query_hca_cap_out, out, capability);
    UCT_IB_MLX5DV_SET(query_hca_cap_in, in, opcode, UCT_IB_MLX5_CMD_OP_QUERY_HCA_CAP);
    UCT_IB_MLX5DV_SET(query_hca_cap_in, in, op_mod, UCT_IB_MLX5_HCA_CAP_OPMOD_GET_CUR |
                                                   (UCT_IB_MLX5_CAP_GENERAL << 1));
    ret = mlx5dv_devx_general_cmd(ctx, in, sizeof(in), out, sizeof(out));
    if (ret != 0) {
        if ((errno == EPERM) || (errno == EPROTONOSUPPORT) ||
            (errno == EOPNOTSUPP)) {
            status    = UCS_ERR_UNSUPPORTED;
            log_level = UCS_LOG_LEVEL_DEBUG;
        } else {
            status    = UCS_ERR_IO_ERROR;
            log_level = UCS_LOG_LEVEL_ERROR;
        }
        ucs_log(log_level,
                "mlx5dv_devx_general_cmd(QUERY_HCA_CAP) failed,"
                " syndrome 0x%x: %m",
                UCT_IB_MLX5DV_GET(query_hca_cap_out, out, syndrome));
        goto err_lru_cleanup;
    }

    UCS_STATIC_ASSERT(UCS_MASK(UCT_IB_MLX5_MD_MAX_DCI_CHANNELS) <= UINT8_MAX);
    md->log_max_dci_stream_channels = UCT_IB_MLX5DV_GET(cmd_hca_cap, cap,
                                                        log_max_dci_stream_channels);
    md->log_max_dci_stream_channels = ucs_min(md->log_max_dci_stream_channels,
                                              UCT_IB_MLX5_MD_MAX_DCI_CHANNELS);

    if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, log_max_msg) !=
        UCT_IB_MLX5_LOG_MAX_MSG_SIZE) {
        status = UCS_ERR_UNSUPPORTED;
        ucs_debug("Unexpected QUERY_HCA_CAP.log_max_msg %d\n",
                  UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, log_max_msg));
        goto err_lru_cleanup;
    }

    status = uct_ib_mlx5_devx_query_lag(md, &lag_state);
    if ((status != UCS_OK) || (lag_state == 0)) {
        dev->lag_level = 1;
    } else {
        dev->lag_level = UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, num_lag_ports);
    }

    md->port_select_mode = uct_ib_mlx5_devx_query_port_select(md);

    log_max_qp       = UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, log_max_qp);
    max_rd_atomic_dc = 1 << UCT_IB_MLX5DV_GET(cmd_hca_cap, cap,
                                              log_max_ra_req_dc);
    ucs_assertv(max_rd_atomic_dc < UINT8_MAX, "max_rd_atomic_dc=%u",
                max_rd_atomic_dc);
    md->max_rd_atomic_dc = max_rd_atomic_dc;

    if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, dct) &&
         (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, lag_dct) || (lag_state == 0))) {
         /* Either DCT supports LAG, or LAG is off */
         dev->flags |= UCT_IB_DEVICE_FLAG_DC;
    }

    if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, rndv_offload_dc)) {
        md->flags |= UCT_IB_MLX5_MD_FLAG_DC_TM;
    }

    if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, compact_address_vector)) {
        dev->flags |= UCT_IB_DEVICE_FLAG_AV;
    }

    if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, ece)) {
        md->super.ece_enable = 1;
    }

    if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, fixed_buffer_size)) {
        md->flags |= UCT_IB_MLX5_MD_FLAG_KSM;
    }

    if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, ext_stride_num_range)) {
        md->flags |= UCT_IB_MLX5_MD_FLAG_MP_RQ;
    }

    if (!UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, umr_modify_atomic_disabled)) {
        md->flags |= UCT_IB_MLX5_MD_FLAG_INDIRECT_ATOMICS;
    }

    if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, log_max_rmp) > 0) {
        md->flags |= UCT_IB_MLX5_MD_FLAG_RMP;
    }

    if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, ooo_sl_mask)) {
        md->flags |= UCT_IB_MLX5_MD_FLAG_OOO_SL_MASK;
    }

    if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, init2_lag_tx_port_affinity)) {
        md->flags |= UCT_IB_MLX5_MD_FLAG_LAG;
    }

    if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, cqe_version)) {
        md->flags |= UCT_IB_MLX5_MD_FLAG_CQE_V1;
    }

    if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, enhanced_cqe_compression)) {
        if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, cqe_compression_128b)) {
            md->flags |= UCT_IB_MLX5_MD_FLAG_CQE128_ZIP;
        }

        if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, cqe_compression)) {
            md->flags |= UCT_IB_MLX5_MD_FLAG_CQE64_ZIP;
        }
    }

    if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap,
                          dci_no_rdma_wr_optimized_performance)) {
        md->flags |= UCT_IB_MLX5_MD_FLAG_NO_RDMA_WR_OPTIMIZED;
    }

    if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap,
                          ib_striding_wq_cq_first_indication)) {
        md->flags |= UCT_IB_MLX5_MD_FLAG_MP_XRQ_FIRST_MSG;
    }

    vhca_id = UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, vhca_id);

    status = uct_ib_mlx5_devx_query_cap_2(ctx, cap_2_out, sizeof(cap_2_out));
    if (status == UCS_OK) {
        cap_2 = UCT_IB_MLX5DV_ADDR_OF(query_hca_cap_out, cap_2_out, capability);

        uct_ib_mlx5_devx_check_xgvmi(md, cap_2, dev);
        uct_ib_mlx5_devx_check_mkey_by_name(md, cap_2, dev);
    }

    uct_ib_mlx5_devx_check_odp(md, md_config, cap);

    if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, atomic)) {
        int ops = UCT_IB_MLX5_ATOMIC_OPS_CMP_SWAP |
                  UCT_IB_MLX5_ATOMIC_OPS_FETCH_ADD;
        uint8_t arg_size;
        int cap_ops, mode8b;

        UCT_IB_MLX5DV_SET(query_hca_cap_in, in, op_mod, UCT_IB_MLX5_HCA_CAP_OPMOD_GET_CUR |
                                                       (UCT_IB_MLX5_CAP_ATOMIC << 1));
        status = uct_ib_mlx5_devx_general_cmd(ctx, in, sizeof(in), out,
                                              sizeof(out),
                                              "QUERY_HCA_CAP, ATOMIC", 0);
        if (status != UCS_OK) {
            goto err_lru_cleanup;
        }

        arg_size = UCT_IB_MLX5DV_GET(atomic_caps, cap, atomic_size_qp);
        cap_ops  = UCT_IB_MLX5DV_GET(atomic_caps, cap, atomic_operations);
        mode8b   = UCT_IB_MLX5DV_GET(atomic_caps, cap, atomic_req_8B_endianness_mode);

        if ((cap_ops & ops) == ops) {
            dev->atomic_arg_sizes = sizeof(uint64_t);
            if (!mode8b) {
                dev->atomic_arg_sizes_be = sizeof(uint64_t);
            }
        }

        dev->atomic_align = ucs_rounddown_pow2(arg_size);

        ops |= UCT_IB_MLX5_ATOMIC_OPS_MASKED_CMP_SWAP |
               UCT_IB_MLX5_ATOMIC_OPS_MASKED_FETCH_ADD;

        arg_size &= UCT_IB_MLX5DV_GET(query_hca_cap_out, out,
                                      capability.atomic_caps.atomic_size_dc);

        if ((cap_ops & ops) == ops) {
            dev->ext_atomic_arg_sizes = arg_size;
            if (mode8b) {
                arg_size &= ~(sizeof(uint64_t));
            }
            dev->ext_atomic_arg_sizes_be = arg_size;
        }

        dev->pci_fadd_arg_sizes  = UCT_IB_MLX5DV_GET(atomic_caps, cap, fetch_add_pci_atomic) << 2;
        dev->pci_cswap_arg_sizes = UCT_IB_MLX5DV_GET(atomic_caps, cap, compare_swap_pci_atomic) << 2;
    }

    md->super.super.ops = &uct_ib_mlx5_devx_md_ops.super;

    status = uct_ib_md_open_common(&md->super, ibv_device, md_config);
    if (status != UCS_OK) {
        goto err_lru_cleanup;
    }

    uct_ib_mlx5_md_port_counter_set_id_init(md);
    ucs_recursive_spinlock_init(&md->dbrec_lock, 0);
    ucs_mpool_params_reset(&mp_params);
    mp_params.elem_size       = sizeof(uct_ib_mlx5_dbrec_t);
    mp_params.elems_per_chunk = ucs_get_page_size() / UCS_SYS_CACHE_LINE_SIZE - 1;
    mp_params.ops             = &uct_ib_mlx5_dbrec_ops;
    mp_params.name            = "devx dbrec";
    status = ucs_mpool_init(&mp_params, &md->dbrec_pool);
    if (status != UCS_OK) {
        goto err_lock_destroy;
    }

    status = uct_ib_mlx5_md_buf_alloc(md, ucs_get_page_size(), 0, &md->zero_buf,
                                      &md->zero_mem, 0, "zero umem");
    if (status != UCS_OK) {
        goto err_dbrec_mpool_cleanup;
    }

    ucs_debug("%s: opened DEVX md log_max_qp=%d",
              uct_ib_device_name(dev), log_max_qp);

    dev->flags          |= UCT_IB_DEVICE_FLAG_MLX5_PRM;
    md->flags           |= UCT_IB_MLX5_MD_FLAG_DEVX;
    md->flags           |= UCT_IB_MLX5_MD_FLAGS_DEVX_OBJS(md_config->devx_objs);
    md->super.name       = UCT_IB_MD_NAME(mlx5);
    md->super.vhca_id    = vhca_id;

    ksm_atomic = 0;
    if (md->flags & UCT_IB_MLX5_MD_FLAG_KSM) {
        md->super.cap_flags |= UCT_MD_FLAG_INVALIDATE_RMA;

        if (md->flags & UCT_IB_MLX5_MD_FLAG_INDIRECT_ATOMICS) {
            md->super.cap_flags |= UCT_MD_FLAG_INVALIDATE_AMO |
                                   UCT_MD_FLAG_INVALIDATE;
            ksm_atomic           = 1;
        }

        uct_ib_mlx5dv_check_dm_ksm_reg(md);
    }

    /* Enable relaxed order only if we would be able to create an indirect key
       (with offset) for strict order access */
    uct_ib_md_parse_relaxed_order(&md->super, md_config, ksm_atomic);

    uct_ib_mlx5_devx_init_flush_mr(md);

    *p_md = &md->super;
    return UCS_OK;

err_dbrec_mpool_cleanup:
    ucs_mpool_cleanup(&md->dbrec_pool, 0);
err_lock_destroy:
    ucs_recursive_spinlock_destroy(&md->dbrec_lock);
    uct_ib_md_close_common(&md->super);
err_lru_cleanup:
    uct_ib_mlx5_devx_mr_lru_cleanup(md);
err_free_md:
    uct_ib_md_free(&md->super);
err_free_context:
    uct_ib_md_device_context_close(ctx);
err:
    if ((status == UCS_ERR_UNSUPPORTED) && (md_config->devx == UCS_YES)) {
        ucs_error("DEVX requested but not supported by %s",
                  ibv_get_device_name(ibv_device));
        status = UCS_ERR_IO_ERROR;
    } else {
        ucs_debug("%s: DEVX is not supported", ibv_get_device_name(ibv_device));
    }

    return status;
}

static void uct_ib_mlx5_devx_cleanup_flush_mr(uct_ib_mlx5_md_t *md)
{
    ucs_status_t status;

    ucs_debug("%s: md=%p md->flags=0x%x flush_rkey=0x%x",
              uct_ib_device_name(&md->super.dev), md, md->flags,
              md->super.flush_rkey);

    if (!(md->flags & UCT_IB_MLX5_MD_FLAG_KSM) ||
        !uct_ib_md_is_flush_rkey_valid(md->super.flush_rkey)) {
        return;
    }

    uct_ib_mlx5_devx_obj_destroy(md->flush_dvmr, "flush_dvmr");

    status = uct_ib_dereg_mr(md->flush_mr);
    if (status != UCS_OK) {
        ucs_warn("uct_ib_dereg_mr(flush_mr) failed: %m");
    }
}

static void uct_ib_mlx5_devx_md_close(uct_md_h tl_md)
{
    uct_ib_mlx5_md_t *md    = ucs_derived_of(tl_md, uct_ib_mlx5_md_t);
    struct ibv_context *ctx = md->super.dev.ibv_context;

    uct_ib_mlx5_devx_cleanup_flush_mr(md);
    uct_ib_mlx5_md_buf_free(md, md->zero_buf, &md->zero_mem);
    ucs_mpool_cleanup(&md->dbrec_pool, 1);
    ucs_recursive_spinlock_destroy(&md->dbrec_lock);
    uct_ib_md_close_common(&md->super);
    uct_ib_mlx5_devx_mr_lru_cleanup(md);
    uct_ib_md_free(&md->super);
    uct_ib_md_device_context_close(ctx);
}

uint32_t uct_ib_mlx5_devx_md_get_pdn(uct_ib_mlx5_md_t *md)
{
    struct mlx5dv_pd dvpd = {0};
    struct mlx5dv_obj dv  = {{0}};
    int ret;

    /* obtain pdn */
    dv.pd.in  = md->super.pd;
    dv.pd.out = &dvpd;
    ret       = mlx5dv_init_obj(&dv, MLX5DV_OBJ_PD);
    if (ret) {
        ucs_fatal("mlx5dv_init_obj(%s, PD) failed: %m",
                  uct_ib_device_name(&md->super.dev));
    }

    return dvpd.pdn;
}

uint8_t
uct_ib_mlx5_devx_md_get_counter_set_id(uct_ib_mlx5_md_t *md, uint8_t port_num)
{
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(query_qp_in)]   = {};
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(query_qp_out)] = {};
    struct ibv_qp_init_attr qp_init_attr              = {};
    struct ibv_qp_attr qp_attr                        = {};
    uint8_t *counter_set_id;
    struct ibv_qp *dummy_qp;
    struct ibv_cq *dummy_cq;
    void *qpc;
    int ret;

    counter_set_id = &md->port_counter_set_ids[port_num - UCT_IB_FIRST_PORT];
    if (*counter_set_id != UCT_IB_COUNTER_SET_ID_INVALID) {
        return *counter_set_id;
    }

    dummy_cq = ibv_create_cq(md->super.dev.ibv_context, 1, NULL, NULL, 0);
    if (dummy_cq == NULL) {
        uct_ib_check_memlock_limit_msg(UCS_LOG_LEVEL_DEBUG,
                                       "%s: ibv_create_cq()",
                                       uct_ib_device_name(&md->super.dev));
        goto err;
    }

    qp_init_attr.send_cq          = dummy_cq;
    qp_init_attr.recv_cq          = dummy_cq;
    qp_init_attr.qp_type          = IBV_QPT_RC;
    qp_init_attr.cap.max_send_wr  = 1;
    qp_init_attr.cap.max_recv_wr  = 1;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;

    dummy_qp = ibv_create_qp(md->super.pd, &qp_init_attr);
    if (dummy_qp == NULL) {
        uct_ib_check_memlock_limit_msg(UCS_LOG_LEVEL_DEBUG,
                                       "%s: ibv_create_qp()",
                                       uct_ib_device_name(&md->super.dev));
        goto err_free_cq;
    }

    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.port_num = port_num;

    ret = ibv_modify_qp(dummy_qp, &qp_attr,
                        IBV_QP_STATE | IBV_QP_PORT | IBV_QP_PKEY_INDEX |
                        IBV_QP_ACCESS_FLAGS);
    if (ret) {
        ucs_diag("failed to modify dummy QP 0x%x to INIT on %s:%d: %m",
                 dummy_qp->qp_num, uct_ib_device_name(&md->super.dev),
                 port_num);
        goto err_destroy_qp;
    }

    UCT_IB_MLX5DV_SET(query_qp_in, in, opcode, UCT_IB_MLX5_CMD_OP_QUERY_QP);
    UCT_IB_MLX5DV_SET(query_qp_in, in, qpn, dummy_qp->qp_num);

    ret = mlx5dv_devx_qp_query(dummy_qp, in, sizeof(in), out, sizeof(out));
    if (ret) {
        ucs_diag("mlx5dv_devx_qp_query(%s:%d, DUMMY_QP, QPN=0x%x) failed, "
                 "syndrome 0x%x: %m",
                 uct_ib_device_name(&md->super.dev), port_num, dummy_qp->qp_num,
                 UCT_IB_MLX5DV_GET(query_qp_out, out, syndrome));
        goto err_destroy_qp;
    }

    qpc             = UCT_IB_MLX5DV_ADDR_OF(query_qp_out, out, qpc);
    *counter_set_id = UCT_IB_MLX5DV_GET(qpc, qpc, counter_set_id);
    ibv_destroy_qp(dummy_qp);
    ibv_destroy_cq(dummy_cq);

    ucs_debug("counter_set_id on %s:%d is 0x%x",
              uct_ib_device_name(&md->super.dev), port_num, *counter_set_id);
    return *counter_set_id;

err_destroy_qp:
    ibv_destroy_qp(dummy_qp);
err_free_cq:
    ibv_destroy_cq(dummy_cq);
err:
    *counter_set_id = 0;
    ucs_debug("using zero counter_set_id on %s:%d",
              uct_ib_device_name(&md->super.dev), port_num);
    return 0;
}

static ucs_status_t
uct_ib_mlx5_devx_allow_xgvmi_access(uct_ib_mlx5_md_t *md,
                                    uct_ib_mlx5_devx_mem_t *memh,
                                    uint32_t exported_lkey, int silent)
{
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(allow_other_vhca_access_in)]   = {0};
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(allow_other_vhca_access_out)] = {0};
    void *access_key;

    UCT_IB_MLX5DV_SET(allow_other_vhca_access_in, in, opcode,
                      UCT_IB_MLX5_CMD_OP_ALLOW_OTHER_VHCA_ACCESS);
    UCT_IB_MLX5DV_SET(allow_other_vhca_access_in, in,
                      object_type_to_be_accessed, UCT_IB_MLX5_OBJ_TYPE_MKEY);
    UCT_IB_MLX5DV_SET(allow_other_vhca_access_in, in, object_id_to_be_accessed,
                      uct_ib_mlx5_mkey_index(exported_lkey));
    access_key = UCT_IB_MLX5DV_ADDR_OF(allow_other_vhca_access_in, in,
                                       access_key);
    ucs_strncpy_zero(access_key, uct_ib_mkey_token,
                     UCT_IB_MLX5DV_FLD_SZ_BYTES(alias_context, access_key));

    return uct_ib_mlx5_devx_general_cmd(md->super.dev.ibv_context, in,
                                        sizeof(in), out, sizeof(out),
                                        "ALLOW_OTHER_VHCA_ACCESS", silent);
}

static ucs_status_t uct_ib_mlx5_devx_xgvmi_umem_mr(uct_ib_mlx5_md_t *md,
                                                   uct_ib_mlx5_devx_mem_t *memh)
{
#if HAVE_DECL_MLX5DV_DEVX_UMEM_REG_EX
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(create_mkey_in)]   = {0};
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(create_mkey_out)] = {0};
    struct mlx5dv_devx_umem_in umem_in;
    struct mlx5dv_devx_obj *cross_mr;
    struct mlx5dv_devx_umem *umem;
    uint32_t exported_lkey;
    ucs_status_t status;
    void *aligned_address;
    size_t length;
    void *mkc;

    if (memh->dm != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    length  = memh->mrs[UCT_IB_MR_DEFAULT].super.ib->length;

    /* register umem */
    umem_in.addr        = memh->address;
    umem_in.size        = length;
    umem_in.access      = UCT_IB_MLX5_MD_UMEM_ACCESS;
    aligned_address     = ucs_align_down_pow2_ptr(memh->address,
                                                  ucs_get_page_size());
    umem_in.pgsz_bitmap = UCS_MASK(ucs_ffs64((uint64_t)aligned_address) + 1);
    umem_in.comp_mask   = 0;

    umem = mlx5dv_devx_umem_reg_ex(md->super.dev.ibv_context, &umem_in);
    if (umem == NULL) {
        uct_ib_md_log_mem_reg_error(&md->super, 0,
                                    "mlx5dv_devx_umem_reg_ex() failed: %m");
        status = UCS_ERR_NO_MEMORY;
        goto err_out;
    }

    /* create mkey */
    mkc = UCT_IB_MLX5DV_ADDR_OF(create_mkey_in, in, memory_key_mkey_entry);
    UCT_IB_MLX5DV_SET(create_mkey_in, in, opcode,
                      UCT_IB_MLX5_CMD_OP_CREATE_MKEY);
    UCT_IB_MLX5DV_SET(create_mkey_in, in, translations_octword_actual_size, 1);
    UCT_IB_MLX5DV_SET(create_mkey_in, in, mkey_umem_id, umem->umem_id);
    UCT_IB_MLX5DV_SET64(create_mkey_in, in, mkey_umem_offset, 0);
    UCT_IB_MLX5DV_SET(mkc, mkc, access_mode_1_0,
                      UCT_IB_MLX5_MKC_ACCESS_MODE_MTT);
    UCT_IB_MLX5DV_SET(mkc, mkc, a, 1);
    UCT_IB_MLX5DV_SET(mkc, mkc, rw, 1);
    UCT_IB_MLX5DV_SET(mkc, mkc, rr, 1);
    UCT_IB_MLX5DV_SET(mkc, mkc, lw, 1);
    UCT_IB_MLX5DV_SET(mkc, mkc, lr, 1);
    UCT_IB_MLX5DV_SET(mkc, mkc, crossing_target_mkey, 1);
    UCT_IB_MLX5DV_SET(mkc, mkc, qpn, 0xffffff);
    UCT_IB_MLX5DV_SET(mkc, mkc, pd, uct_ib_mlx5_devx_md_get_pdn(md));
    UCT_IB_MLX5DV_SET(mkc, mkc, mkey_7_0, md->mkey_tag);
    UCT_IB_MLX5DV_SET64(mkc, mkc, start_addr, (uintptr_t)memh->address);
    UCT_IB_MLX5DV_SET64(mkc, mkc, len, length);

    cross_mr = uct_ib_mlx5_devx_obj_create(md->super.dev.ibv_context, in,
                                           sizeof(in), out, sizeof(out), "MKEY",
                                           uct_md_reg_log_lvl(0));
    if (cross_mr == NULL) {
        status = UCS_ERR_IO_ERROR;
        goto err_umem_dereg;
    }

    exported_lkey = (UCT_IB_MLX5DV_GET(create_mkey_out, out, mkey_index) << 8) |
                    md->mkey_tag;

    status = uct_ib_mlx5_devx_allow_xgvmi_access(md, memh, exported_lkey, 0);
    if (status != UCS_OK) {
        goto err_cross_mr_destroy;
    }

    memh->umem          = umem;
    memh->cross_mr      = cross_mr;
    memh->exported_lkey = exported_lkey;
    return UCS_OK;

err_cross_mr_destroy:
    mlx5dv_devx_obj_destroy(cross_mr);
err_umem_dereg:
    mlx5dv_devx_umem_dereg(umem);
err_out:
    return status;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

UCS_PROFILE_FUNC_ALWAYS(ucs_status_t, uct_ib_mlx5_devx_reg_exported_key,
                        (md, memh), uct_ib_mlx5_md_t *md,
                        uct_ib_mlx5_devx_mem_t *memh)
{
    struct mlx5dv_devx_obj *cross_mr;
    uint32_t exported_lkey;
    ucs_status_t status;

    if (memh->dm != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    ucs_assertv(memh->cross_mr == NULL,
                "memh=%p cross_mr=%p exported_lkey=0x%x", memh, memh->cross_mr,
                memh->exported_lkey);
    if (!(md->flags & UCT_IB_MLX5_MD_FLAG_INDIRECT_XGVMI)) {
        goto out_umem_mr;
    }

    status = uct_ib_mlx5_devx_reg_ksm_data_contig(md,
                                                  &memh->mrs[UCT_IB_MR_DEFAULT],
                                                  memh->address,
                                                  (uint64_t)memh->address, 0, 0,
                                                  "exported key", &cross_mr,
                                                  &exported_lkey);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ib_mlx5_devx_allow_xgvmi_access(md, memh, exported_lkey, 1);
    if (status == UCS_OK) {
        goto out;
    }

    ucs_debug("%s: indirect cross gvmi not supported, fallback to DEVX UMEM",
              uct_ib_device_name(&md->super.dev));
    mlx5dv_devx_obj_destroy(cross_mr);
    md->flags &= ~UCT_IB_MLX5_MD_FLAG_INDIRECT_XGVMI;

out_umem_mr:
    return uct_ib_mlx5_devx_xgvmi_umem_mr(md, memh);

out:
    memh->cross_mr      = cross_mr;
    memh->exported_lkey = exported_lkey;
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE int
uct_ib_mlx5_devx_mkey_pack_invalidate_param_check(unsigned flags)
{
    return ENABLE_PARAMS_CHECK &&
           (flags & (UCT_MD_MKEY_PACK_FLAG_INVALIDATE_RMA |
                     UCT_MD_MKEY_PACK_FLAG_INVALIDATE_AMO));
}

static ucs_status_t
uct_ib_mlx5_devx_mkey_pack(uct_md_h uct_md, uct_mem_h uct_memh,
                           void *address, size_t length,
                           const uct_md_mkey_pack_params_t *params,
                           void *mkey_buffer)
{
    uct_ib_mlx5_md_t *md         = ucs_derived_of(uct_md, uct_ib_mlx5_md_t);
    uct_ib_mlx5_devx_mem_t *memh = uct_memh;
    ucs_status_t status;
    unsigned flags;
    uint32_t rkey;

    flags = UCS_PARAM_VALUE(UCT_MD_MKEY_PACK_FIELD, params, flags, FLAGS, 0);
    if (flags & UCT_MD_MKEY_PACK_FLAG_EXPORT) {
#if HAVE_IBV_DM
        if (memh->dm != NULL) {
            ucs_error("%s: cannot export memory allocated on the device "
                      "(address %p length %zu)",
                      uct_ib_device_name(&md->super.dev), memh->address,
                      memh->mrs[UCT_IB_MR_DEFAULT].super.ib->length);
            return UCS_ERR_INVALID_PARAM;
        }
#endif
        if (uct_ib_mlx5_devx_mkey_pack_invalidate_param_check(flags)) {
            ucs_error("packing a memory key that supports invalidation "
                      "and exporting is unsupported");
            return UCS_ERR_INVALID_PARAM;
        }

        if (ENABLE_PARAMS_CHECK && (memh->super.flags & UCT_IB_MEM_IMPORTED)) {
            ucs_error("exporting an imported key is unsupported");
            return UCS_ERR_INVALID_PARAM;
        }

        if (memh->cross_mr == NULL) {
            status = uct_ib_mlx5_devx_reg_exported_key(md, memh);
            if (status != UCS_OK) {
                return status;
            }
        }

        uct_ib_md_pack_exported_mkey(&md->super, memh->exported_lkey,
                                     mkey_buffer);
        return UCS_OK;
    }

    if (uct_ib_mlx5_devx_mkey_pack_invalidate_param_check(flags) &&
        (memh->super.flags & UCT_IB_MEM_IMPORTED)) {
        ucs_error("invalidating an imported key is unsupported");
        return UCS_ERR_INVALID_PARAM;
    }

    /* Create atomic key on-demand only if a user requested atomic access to the
     * memory region and the hardware supports it.
     */
    if (ucs_unlikely(memh->atomic_dvmr == NULL) &&
        ((memh->super.flags & UCT_IB_MEM_ACCESS_REMOTE_ATOMIC) ||
         md->super.relaxed_order) &&
        !(memh->super.flags & UCT_IB_MEM_IMPORTED) &&
        ucs_test_all_flags(md->flags,
                           UCT_IB_MLX5_MD_FLAG_KSM |
                           UCT_IB_MLX5_MD_FLAG_INDIRECT_ATOMICS)) {
        status = uct_ib_mlx5_devx_reg_atomic_key(md, memh);
        if (status == UCS_OK) {
            ucs_assertv(memh->atomic_rkey != UCT_IB_INVALID_MKEY,
                        "dev=%s memh=%p", uct_ib_device_name(&md->super.dev),
                        memh);
        } else if (status != UCS_ERR_UNSUPPORTED) {
            return status;
        }
    }

    if (ENABLE_PARAMS_CHECK && (flags & UCT_MD_MKEY_PACK_FLAG_INVALIDATE_AMO) &&
        (memh->atomic_rkey == UCT_IB_INVALID_MKEY)) {
        ucs_error("%s: cannot invalidate AMO without creating an atomic key",
                  uct_ib_device_name(&md->super.dev));
        return UCS_ERR_INVALID_PARAM;
    }

    if ((flags & UCT_MD_MKEY_PACK_FLAG_INVALIDATE_RMA) || (memh->dm != NULL)) {
        if (ucs_unlikely(memh->indirect_dvmr == NULL)) {
            status = uct_ib_mlx5_devx_reg_indirect_key(md, memh);
            if (status != UCS_OK) {
                return status;
            }
        }

        rkey = memh->indirect_rkey;
    } else {
        rkey = memh->super.rkey;
    }

    uct_ib_md_pack_rkey(rkey, memh->atomic_rkey, mkey_buffer);
    return UCS_OK;
}

ucs_status_t uct_ib_mlx5_devx_mem_attach(uct_md_h uct_md,
                                         const void *mkey_buffer,
                                         uct_md_mem_attach_params_t *params,
                                         uct_mem_h *memh_p)
{
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(create_alias_obj_in)]   = {0};
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(create_alias_obj_out)] = {0};
    uct_ib_mlx5_md_t *md = ucs_derived_of(uct_md, uct_ib_mlx5_md_t);
    const uint64_t flags = UCT_MD_MEM_ATTACH_FIELD_VALUE(params, flags,
                                                         FIELD_FLAGS, 0);
    const uct_ib_md_packed_mkey_t *packed_mkey = mkey_buffer;
    uct_ib_mlx5_devx_mem_t *memh;
    uct_ib_mem_t *ib_memh;
    void *hdr, *alias_ctx;
    ucs_status_t status;
    void *access_key;
    int ret;

    status = uct_ib_memh_alloc(&md->super, 0, 0, sizeof(*memh), 0, &ib_memh);
    if (status != UCS_OK) {
        goto err;
    }

    memh      = ucs_derived_of(ib_memh, uct_ib_mlx5_devx_mem_t);
    hdr       = UCT_IB_MLX5DV_ADDR_OF(create_alias_obj_in, in, hdr);
    alias_ctx = UCT_IB_MLX5DV_ADDR_OF(create_alias_obj_in, in, alias_ctx);

    /* create alias */
    UCT_IB_MLX5DV_SET(general_obj_in_cmd_hdr, hdr, opcode,
                      UCT_IB_MLX5_CMD_OP_CREATE_GENERAL_OBJECT);
    UCT_IB_MLX5DV_SET(general_obj_in_cmd_hdr, hdr, obj_type,
                      UCT_IB_MLX5_OBJ_TYPE_MKEY);
    UCT_IB_MLX5DV_SET(general_obj_in_cmd_hdr, hdr, alias_object, 1);

    UCT_IB_MLX5DV_SET(alias_context, alias_ctx, vhca_id_to_be_accessed,
                      packed_mkey->vhca_id);
    UCT_IB_MLX5DV_SET(alias_context, alias_ctx, object_id_to_be_accessed,
                      uct_ib_mlx5_mkey_index(packed_mkey->lkey));
    UCT_IB_MLX5DV_SET(alias_context, alias_ctx, metadata_1,
                      uct_ib_mlx5_devx_md_get_pdn(md));
    access_key = UCT_IB_MLX5DV_ADDR_OF(alias_context, alias_ctx, access_key);
    ucs_strncpy_zero(access_key, uct_ib_mkey_token,
                     UCT_IB_MLX5DV_FLD_SZ_BYTES(alias_context, access_key));

    memh->cross_mr = uct_ib_mlx5_devx_obj_create(md->super.dev.ibv_context, in,
                                                 sizeof(in), out, sizeof(out),
                                                 "MKEY_ALIAS",
                                                 uct_md_attach_log_lvl(flags));
    if (memh->cross_mr == NULL) {
        status = UCS_ERR_IO_ERROR;
        goto err_memh_free;
    }

    ret = UCT_IB_MLX5DV_GET(create_alias_obj_out, out, alias_ctx.status);
    if (ret) {
        uct_md_log_mem_attach_error(flags,
                                    "created MR alias object in a bad state");
        status = UCS_ERR_IO_ERROR;
        goto err_cross_mr_destroy;
    }

    memh->super.lkey   = (UCT_IB_MLX5DV_GET(create_alias_obj_out, out,
                                            hdr.obj_id) << 8) |
                         md->mkey_tag;
    memh->super.rkey   = memh->super.lkey;
    memh->super.flags |= UCT_IB_MEM_IMPORTED;
    *memh_p            = ib_memh;
    return UCS_OK;

err_cross_mr_destroy:
    mlx5dv_devx_obj_destroy(memh->cross_mr);
err_memh_free:
    ucs_free(ib_memh);
err:
    return status;
}

static ucs_status_t
uct_ib_mlx5_devx_md_query(uct_md_h uct_md, uct_md_attr_v2_t *md_attr)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);
    ucs_status_t status;

    status = uct_ib_md_query(uct_md, md_attr);
    if (status != UCS_OK) {
        return status;
    }

#if HAVE_IBV_DM
    if (md->cap_flags & UCT_MD_FLAG_ALLOC) {
        md_attr->alloc_mem_types |= UCS_BIT(UCS_MEMORY_TYPE_RDMA);
        md_attr->max_alloc        = md->dev.dev_attr.max_dm_size;
    }

#endif
    return UCS_OK;
}

static uct_ib_md_ops_t uct_ib_mlx5_devx_md_ops = {
    .super = {
        .close              = uct_ib_mlx5_devx_md_close,
        .query              = uct_ib_mlx5_devx_md_query,
        .mem_alloc          = uct_ib_mlx5_devx_device_mem_alloc,
        .mem_free           = uct_ib_mlx5_devx_device_mem_free,
        .mem_reg            = uct_ib_mlx5_devx_mem_reg,
        .mem_dereg          = uct_ib_mlx5_devx_mem_dereg,
        .mem_attach         = uct_ib_mlx5_devx_mem_attach,
        .mem_advise         = uct_ib_mem_advise,
        .mkey_pack          = uct_ib_mlx5_devx_mkey_pack,
        .detect_memory_type = ucs_empty_function_return_unsupported,
    },
    .open = uct_ib_mlx5_devx_md_open,
};

UCT_IB_MD_DEFINE_ENTRY(devx, uct_ib_mlx5_devx_md_ops);

#endif

static void uct_ib_mlx5dv_check_dc(uct_ib_device_t *dev)
{
#if HAVE_DC_DV
    struct ibv_context *ctx            = dev->ibv_context;
    uct_ib_qp_init_attr_t qp_init_attr = {};
    struct mlx5dv_qp_init_attr dv_attr = {};
    struct ibv_qp_attr attr            = {};
    uct_ib_mlx5dv_qp_tmp_objs_t qp_tmp_objs;
    struct ibv_pd *pd;
    struct ibv_qp *qp;
    ucs_status_t status;
    int ret;

    pd = ibv_alloc_pd(ctx);
    if (pd == NULL) {
        ucs_debug("%s: ibv_alloc_pd() failed: %m", uct_ib_device_name(dev));
        goto out;
    }

    status = uct_ib_mlx5dv_qp_tmp_objs_create(dev, pd, &qp_tmp_objs, 1);
    if (status != UCS_OK) {
        goto out_dealloc_pd;
    }

    uct_ib_mlx5dv_dct_qp_init_attr(&qp_init_attr, &dv_attr, pd, qp_tmp_objs.cq,
                                   qp_tmp_objs.srq);

    /* create DCT qp successful means DC is supported */
    qp = UCS_PROFILE_CALL_ALWAYS(mlx5dv_create_qp, ctx, &qp_init_attr,
                                 &dv_attr);
    if (qp == NULL) {
        ucs_debug("%s: mlx5dv_create_qp(DCT) failed: %m",
                  uct_ib_device_name(dev));
        goto out_qp_tmp_objs_close;
    }

    attr.qp_state        = IBV_QPS_INIT;
    attr.port_num        = 1;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ  |
                           IBV_ACCESS_REMOTE_ATOMIC;
    ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE |
                                   IBV_QP_PKEY_INDEX |
                                   IBV_QP_PORT |
                                   IBV_QP_ACCESS_FLAGS);
    if (ret != 0) {
        ucs_debug("failed to ibv_modify_qp(DCT, INIT) on %s: %m",
                  uct_ib_device_name(dev));
        goto out_destroy_qp;
    }

    /* always set global address parameters, in case the port is RoCE or SRIOV */
    attr.qp_state                  = IBV_QPS_RTR;
    attr.min_rnr_timer             = 1;
    attr.path_mtu                  = IBV_MTU_256;
    attr.ah_attr.port_num          = 1;
    attr.ah_attr.sl                = 0;
    attr.ah_attr.is_global         = 1;
    attr.ah_attr.grh.hop_limit     = 1;
    attr.ah_attr.grh.traffic_class = 0;
    attr.ah_attr.grh.sgid_index    = 0;

    ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE |
                                   IBV_QP_MIN_RNR_TIMER |
                                   IBV_QP_AV |
                                   IBV_QP_PATH_MTU);

    if (ret != 0) {
        ucs_debug("%s: failed to ibv_modify_qp(DCT, RTR): %m",
                  uct_ib_device_name(dev));
        goto out_destroy_qp;
    }

    dev->flags |= UCT_IB_DEVICE_FLAG_DC;

out_destroy_qp:
    uct_ib_destroy_qp(qp);
out_qp_tmp_objs_close:
    uct_ib_mlx5dv_qp_tmp_objs_destroy(&qp_tmp_objs);
out_dealloc_pd:
    ibv_dealloc_pd(pd);
out:
#endif
    ucs_debug("%s: DC %s supported", uct_ib_device_name(dev),
              (dev->flags & UCT_IB_DEVICE_FLAG_DC) ? "is" : "is not");
}

static uct_ib_md_ops_t uct_ib_mlx5_md_ops;

static ucs_status_t uct_ib_mlx5dv_md_open(struct ibv_device *ibv_device,
                                          const uct_ib_md_config_t *md_config,
                                          uct_ib_md_t **p_md)
{
    ucs_status_t status = UCS_OK;
    struct ibv_context *ctx;
    uct_ib_device_t *dev;
    uct_ib_mlx5_md_t *md;

    if ((md_config->mlx5dv == UCS_NO) || !mlx5dv_is_supported(ibv_device)) {
        return UCS_ERR_UNSUPPORTED;
    }

    ctx = ibv_open_device(ibv_device);
    if (ctx == NULL) {
        ucs_diag("ibv_open_device(%s) failed: %m",
                 ibv_get_device_name(ibv_device));
        status = UCS_ERR_UNSUPPORTED;
        goto err;
    }

    md = ucs_derived_of(uct_ib_md_alloc(sizeof(*md), "ib_mlx5dv_md", ctx),
                        uct_ib_mlx5_md_t);
    if (md == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_context;
    }

    dev = &md->super.dev;

    status = uct_ib_device_query(dev, ibv_device);
    if (status != UCS_OK) {
        goto err_md_free;
    }

    if (IBV_DEVICE_ATOMIC_HCA(dev)) {
        dev->atomic_arg_sizes = sizeof(uint64_t);

#if HAVE_STRUCT_IBV_DEVICE_ATTR_EX_PCI_ATOMIC_CAPS
        dev->pci_fadd_arg_sizes  = dev->dev_attr.pci_atomic_caps.fetch_add << 2;
        dev->pci_cswap_arg_sizes = dev->dev_attr.pci_atomic_caps.compare_swap << 2;
#endif
    }

    uct_ib_mlx5dv_check_dc(dev);

    md->super.super.ops  = &uct_ib_mlx5_md_ops.super;
    md->max_rd_atomic_dc = IBV_DEV_ATTR(dev, max_qp_rd_atom);
    status               = uct_ib_md_open_common(&md->super, ibv_device,
                                                 md_config);
    if (status != UCS_OK) {
        goto err_md_free;
    }

    dev->flags    |= UCT_IB_DEVICE_FLAG_MLX5_PRM;
    md->super.name = UCT_IB_MD_NAME(mlx5);

    uct_ib_md_parse_relaxed_order(&md->super, md_config, 0);
    uct_ib_md_ece_check(&md->super);

    md->super.flush_rkey = uct_ib_mlx5_flush_rkey_make();

    /* cppcheck-suppress autoVariables */
    *p_md = &md->super;
    return UCS_OK;

err_md_free:
    uct_ib_md_free(&md->super);
err_free_context:
    uct_ib_md_device_context_close(ctx);
err:
    return status;
}

static uct_ib_md_ops_t uct_ib_mlx5_md_ops = {
    .super = {
        .close              = uct_ib_md_close,
        .query              = uct_ib_md_query,
        .mem_reg            = uct_ib_verbs_mem_reg,
        .mem_dereg          = uct_ib_verbs_mem_dereg,
        .mem_attach         = ucs_empty_function_return_unsupported,
        .mem_advise         = uct_ib_mem_advise,
        .mkey_pack          = uct_ib_verbs_mkey_pack,
        .detect_memory_type = ucs_empty_function_return_unsupported,
    },
    .open = uct_ib_mlx5dv_md_open,
};

UCT_IB_MD_DEFINE_ENTRY(dv, uct_ib_mlx5_md_ops);
