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


typedef struct {
    struct mlx5dv_devx_obj     *dvmr;
    int                        mr_num;
    size_t                     length;
    struct ibv_mr              *mrs[];
} uct_ib_mlx5_ksm_data_t;

typedef union uct_ib_mlx5_mr {
    uct_ib_mr_t                super;
    uct_ib_mlx5_ksm_data_t     *ksm_data;
} uct_ib_mlx5_mr_t;

typedef struct uct_ib_mlx5_mem {
    uct_ib_mem_t               super;
#if HAVE_DEVX
    struct mlx5dv_devx_obj     *atomic_dvmr;
    struct mlx5dv_devx_obj     *indirect_dvmr;
    struct mlx5dv_devx_umem    *umem;
    struct mlx5dv_devx_obj     *cross_mr;
#endif
    uct_ib_mlx5_mr_t           mrs[];
} uct_ib_mlx5_mem_t;


static const char uct_ib_mkey_token[] = "uct_ib_mkey_token";

static ucs_status_t
uct_ib_mlx5_reg_key(uct_ib_md_t *md, void *address, size_t length,
                    uint64_t access_flags, int dmabuf_fd, size_t dmabuf_offset,
                    uct_ib_mem_t *ib_memh, uct_ib_mr_type_t mr_type, int silent)
{
    uct_ib_mlx5_mem_t *memh = ucs_derived_of(ib_memh, uct_ib_mlx5_mem_t);

    return uct_ib_reg_key_impl(md, address, length, access_flags, dmabuf_fd,
                               dmabuf_offset, ib_memh,
                               &memh->mrs[mr_type].super, mr_type, silent);
}

static ucs_status_t uct_ib_mlx5_dereg_key(uct_ib_md_t *md,
                                          uct_ib_mem_t *ib_memh,
                                          uct_ib_mr_type_t mr_type)
{
    uct_ib_mlx5_mem_t *memh = ucs_derived_of(ib_memh, uct_ib_mlx5_mem_t);

    return uct_ib_dereg_mr(memh->mrs[mr_type].super.ib);
}

static ucs_status_t uct_ib_mlx5_reg_atomic_key(uct_ib_md_t *ibmd,
                                               uct_ib_mem_t *ib_memh)
{
    uct_ib_mr_type_t mr_type = uct_ib_memh_get_atomic_base_mr_type(ib_memh);
    uct_ib_mlx5_mem_t *memh = ucs_derived_of(ib_memh, uct_ib_mlx5_mem_t);

    if (mr_type != UCT_IB_MR_STRICT_ORDER) {
        return UCS_ERR_UNSUPPORTED;
    }

    memh->super.atomic_rkey = memh->mrs[mr_type].super.ib->rkey;
    return UCS_OK;
}

static ucs_status_t
uct_ib_mlx5_mem_prefetch(uct_ib_md_t *md, uct_ib_mem_t *ib_memh, void *addr,
                         size_t length)
{
#if HAVE_DECL_IBV_ADVISE_MR
    struct ibv_sge sg_list;
    int ret;

    if (!(ib_memh->flags & UCT_IB_MEM_FLAG_ODP)) {
        return UCS_OK;
    }

    ucs_debug("memh %p prefetch %p length %zu", ib_memh, addr, length);

    sg_list.lkey   = ib_memh->lkey;
    sg_list.addr   = (uintptr_t)addr;
    sg_list.length = length;

    ret = UCS_PROFILE_CALL(ibv_advise_mr, md->pd,
                           IBV_ADVISE_MR_ADVICE_PREFETCH_WRITE,
                           IB_UVERBS_ADVISE_MR_FLAG_FLUSH, &sg_list, 1);
    if (ret) {
        ucs_error("ibv_advise_mr(addr=%p length=%zu) returned %d: %m",
                  addr, length, ret);
        return UCS_ERR_IO_ERROR;
    }
#endif
    return UCS_OK;
}

static int uct_ib_mlx5_has_roce_port(uct_ib_device_t *dev)
{
    int port_num;

    for (port_num = dev->first_port;
         port_num < dev->first_port + dev->num_ports;
         port_num++)
    {
        if (uct_ib_device_is_port_roce(dev, port_num)) {
            return 1;
        }
    }

    return 0;
}

static uint32_t uct_ib_mlx5_flush_rkey_make()
{
    return ((getpid() & 0xff) << 8) | UCT_IB_MD_INVALID_FLUSH_RKEY;
}

#if HAVE_DEVX

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

static ucs_status_t uct_ib_mlx5_devx_reg_ksm(uct_ib_mlx5_md_t *md, int atomic,
                                             intptr_t addr, size_t length,
                                             int list_size, size_t entity_size,
                                             char *in,
                                             struct mlx5dv_devx_obj **mr_p,
                                             uint32_t *mkey)
{
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(create_mkey_out)] = {};
    struct mlx5dv_devx_obj *mr;
    void *mkc;

    UCT_IB_MLX5DV_SET(create_mkey_in, in, opcode, UCT_IB_MLX5_CMD_OP_CREATE_MKEY);
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
    UCT_IB_MLX5DV_SET64(mkc, mkc, start_addr, addr);
    UCT_IB_MLX5DV_SET64(mkc, mkc, len, length);
    UCT_IB_MLX5DV_SET(create_mkey_in, in, translations_octword_actual_size, list_size);

    mr = UCS_PROFILE_NAMED_CALL_ALWAYS("devx_create_mkey",
                                       mlx5dv_devx_obj_create,
                                       md->super.dev.ibv_context, in,
                                       uct_ib_mlx5_calc_mkey_inlen(list_size),
                                       out, sizeof(out));
    if (mr == NULL) {
        ucs_debug("mlx5dv_devx_obj_create(CREATE_MKEY, mode=KSM) failed, "
                  "syndrome 0x%x: %m",
                  UCT_IB_MLX5DV_GET(create_mkey_out, out, syndrome));
        return UCS_ERR_UNSUPPORTED;
    }

    *mr_p = mr;
    *mkey = (UCT_IB_MLX5DV_GET(create_mkey_out, out, mkey_index) << 8) |
            md->mkey_tag;

    return UCS_OK;
}

static ucs_status_t
uct_ib_mlx5_devx_reg_ksm_data(uct_ib_mlx5_md_t *md, int atomic,
                              uct_ib_mlx5_ksm_data_t *ksm_data,
                              size_t length, off_t off,
                              struct mlx5dv_devx_obj **mr_p,
                              uint32_t *mkey)
{
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
        UCT_IB_MLX5DV_SET64(klm, klm, address, (intptr_t)ksm_data->mrs[i]->addr);
        UCT_IB_MLX5DV_SET(klm, klm, byte_count, ksm_data->mrs[i]->length);
        UCT_IB_MLX5DV_SET(klm, klm, mkey, ksm_data->mrs[i]->lkey);
        klm = UCS_PTR_BYTE_OFFSET(klm, UCT_IB_MLX5DV_ST_SZ_BYTES(klm));
    }

    status = uct_ib_mlx5_devx_reg_ksm(md, atomic,
                                      (intptr_t)ksm_data->mrs[0]->addr + off,
                                      length, ksm_data->mr_num,
                                      ksm_data->mrs[0]->length, in, mr_p,
                                      mkey);
    ucs_free(in);
    return status;
}

static ucs_status_t
uct_ib_mlx5_devx_reg_ksm_data_addr(uct_ib_mlx5_md_t *md, struct ibv_mr *mr,
                                   intptr_t addr, size_t length,
                                   uintptr_t iova, int atomic, int list_size,
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
                            addr + (i * UCT_IB_MD_MAX_MR_SIZE));
        klm = UCS_PTR_BYTE_OFFSET(klm, UCT_IB_MLX5DV_ST_SZ_BYTES(klm));
    }

    status = uct_ib_mlx5_devx_reg_ksm(md, atomic, iova, length,
                                      list_size, UCT_IB_MD_MAX_MR_SIZE, in,
                                      mr_p, mkey);
    ucs_free(in);
    return status;
}

static ucs_status_t
uct_ib_mlx5_devx_reg_ksm_data_contig(uct_ib_mlx5_md_t *md,
                                     uct_ib_mlx5_mr_t *mr, off_t off,
                                     int atomic, struct mlx5dv_devx_obj **mr_p,
                                     uint32_t *mkey)
{
    intptr_t addr = (intptr_t)mr->super.ib->addr & ~(UCT_IB_MD_MAX_MR_SIZE - 1);
    /* FW requires indirect atomic MR addr and length to be aligned
     * to max supported atomic argument size */
    size_t length = ucs_align_up(mr->super.ib->length +
                                 (intptr_t)mr->super.ib->addr - addr,
                                 md->super.dev.atomic_align);
    /* add off to workaround CREATE_MKEY range check issue */
    int list_size = ucs_div_round_up(length + off, UCT_IB_MD_MAX_MR_SIZE);

    return uct_ib_mlx5_devx_reg_ksm_data_addr(md, mr->super.ib, addr, length,
                                              addr + off, atomic, list_size,
                                              mr_p, mkey);
}

/**
 * Pop MR LRU-entry from @a md cash
 */
static void uct_ib_mlx5_devx_md_mr_lru_pop(uct_ib_mlx5_md_t *md,
                                           const char *reason)
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

static UCS_F_ALWAYS_INLINE int
uct_ib_mlx5_devx_md_is_ksm_atomic_supported(uct_ib_mlx5_md_t *md)
{
    return ucs_test_all_flags(md->flags, UCT_IB_MLX5_MD_FLAG_KSM |
                                         UCT_IB_MLX5_MD_FLAG_INDIRECT_ATOMICS);
}

static ucs_status_t uct_ib_mlx5_devx_reg_indirect_key(uct_ib_md_t *ibmd,
                                                      uct_ib_mem_t *ib_memh)
{
    uct_ib_mlx5_md_t *md    = ucs_derived_of(ibmd, uct_ib_mlx5_md_t);
    uct_ib_mlx5_mem_t *memh = ucs_derived_of(ib_memh, uct_ib_mlx5_mem_t);
    ucs_status_t status;

    ucs_assert(uct_ib_mlx5_devx_md_is_ksm_atomic_supported(md));

    do {
        status = uct_ib_mlx5_devx_reg_ksm_data_contig(
                md, &memh->mrs[UCT_IB_MR_DEFAULT], 0, 0, &memh->indirect_dvmr,
                &memh->super.indirect_rkey);
        if (status != UCS_OK) {
            break;
        }

        /* This loop is guaranteed to finish because eventually all entries in
         * the LRU will have an associated indirect_mr object, so the next key
         * we will get from HW will be a new value not in the LRU. */
        status = uct_ib_md_mlx5_devx_mr_lru_push(md, memh->super.indirect_rkey,
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

static ucs_status_t uct_ib_mlx5_devx_dereg_key(uct_ib_md_t *ibmd,
                                               uct_ib_mem_t *ib_memh,
                                               uct_ib_mr_type_t mr_type)
{
    uct_ib_mlx5_md_t *md    = ucs_derived_of(ibmd, uct_ib_mlx5_md_t);
    uct_ib_mlx5_mem_t *memh = ucs_derived_of(ib_memh, uct_ib_mlx5_mem_t);
    ucs_status_t ret_status = UCS_OK;
    ucs_status_t status;
    int ret;

    if (memh->super.indirect_rkey != UCT_IB_INVALID_MKEY) {
        uct_ib_md_mlx5_devx_mr_lru_push(md, memh->super.indirect_rkey, NULL);
        ucs_debug("%s: destroy dvmr %p with key %x",
                  uct_ib_device_name(&ibmd->dev), memh->indirect_dvmr,
                  memh->super.indirect_rkey);
        memh->super.indirect_rkey = UCT_IB_INVALID_MKEY;
        ret_status = uct_ib_mlx5_devx_obj_destroy(memh->indirect_dvmr,
                                                  "MKEY, INDIRECT");
    }

    if (memh->cross_mr != NULL) {
        ret = mlx5dv_devx_obj_destroy(memh->cross_mr);
        if (ret < 0) {
            ucs_warn("mlx5dv_devx_obj_destroy(crossmr) failed: %m");
            ret_status = UCS_ERR_IO_ERROR;
        }

        memh->cross_mr = NULL;
    }

    if (memh->umem != NULL) {
        ret = mlx5dv_devx_umem_dereg(memh->umem);
        if (ret < 0) {
            ucs_warn("mlx5dv_devx_umem_dereg(crossmr) failed: %m");
            ret_status = UCS_ERR_IO_ERROR;
        }

        memh->umem = NULL;
    }

    status = uct_ib_mlx5_dereg_key(ibmd, ib_memh, mr_type);
    if (ret_status == UCS_OK) {
        ret_status = status;
    }

    return ret_status;
}

static ucs_status_t uct_ib_mlx5_devx_reg_atomic_key(uct_ib_md_t *ibmd,
                                                    uct_ib_mem_t *ib_memh)
{
    uct_ib_mr_type_t mr_type = uct_ib_memh_get_atomic_base_mr_type(ib_memh);
    uct_ib_mlx5_mem_t *memh  = ucs_derived_of(ib_memh, uct_ib_mlx5_mem_t);
    uct_ib_mlx5_md_t *md     = ucs_derived_of(ibmd, uct_ib_mlx5_md_t);
    uct_ib_mlx5_mr_t *mr     = &memh->mrs[mr_type];
    ucs_status_t status;
    uint8_t mr_id;

    if (!uct_ib_mlx5_devx_md_is_ksm_atomic_supported(md)) {
        return uct_ib_mlx5_reg_atomic_key(ibmd, ib_memh);
    }

    status = uct_ib_mlx5_md_get_atomic_mr_id(ibmd, &mr_id);
    if (status != UCS_OK) {
        return status;
    }

    if (memh->super.flags & UCT_IB_MEM_MULTITHREADED) {
        return uct_ib_mlx5_devx_reg_ksm_data(md, 1, mr->ksm_data,
                                             mr->ksm_data->length,
                                             uct_ib_md_atomic_offset(mr_id),
                                             &memh->atomic_dvmr,
                                             &memh->super.atomic_rkey);
    }

    status = uct_ib_mlx5_devx_reg_ksm_data_contig(
            md, mr, uct_ib_md_atomic_offset(mr_id), 1, &memh->atomic_dvmr,
            &memh->super.atomic_rkey);
    if (status != UCS_OK) {
        return status;
    }

    ucs_debug("KSM registered memory %p..%p offset 0x%x on %s rkey 0x%x",
              mr->super.ib->addr, UCS_PTR_BYTE_OFFSET(mr->super.ib->addr,
              mr->super.ib->length), uct_ib_md_atomic_offset(mr_id),
              uct_ib_device_name(&md->super.dev), memh->super.atomic_rkey);
    return status;
}

static ucs_status_t uct_ib_mlx5_devx_dereg_atomic_key(uct_ib_md_t *ibmd,
                                                      uct_ib_mem_t *ib_memh)
{
    uct_ib_mlx5_mem_t *memh = ucs_derived_of(ib_memh, uct_ib_mlx5_mem_t);
    uct_ib_mlx5_md_t *md    = ucs_derived_of(ibmd, uct_ib_mlx5_md_t);

    if (!uct_ib_mlx5_devx_md_is_ksm_atomic_supported(md)) {
        return UCS_OK;
    }

    return uct_ib_mlx5_devx_obj_destroy(memh->atomic_dvmr, "MKEY, ATOMIC");
}

static ucs_status_t uct_ib_mlx5_devx_reg_multithreaded(uct_ib_md_t *ibmd,
                                                       void *address, size_t length,
                                                       uint64_t access_flags,
                                                       uct_ib_mem_t *ib_memh,
                                                       uct_ib_mr_type_t mr_type,
                                                       int silent)
{
    uct_ib_mlx5_md_t *md    = ucs_derived_of(ibmd, uct_ib_mlx5_md_t);
    uct_ib_mlx5_mem_t *memh = ucs_derived_of(ib_memh, uct_ib_mlx5_mem_t);
    uct_ib_mlx5_mr_t *mr    = &memh->mrs[mr_type];
    size_t chunk            = md->super.config.mt_reg_chunk;
    uct_ib_mlx5_ksm_data_t *ksm_data;
    size_t ksm_data_size;
    ucs_status_t status;
    uint32_t mkey;
    int mr_num;

    if (!uct_ib_mlx5_devx_md_is_ksm_atomic_supported(md)) {
        return UCS_ERR_UNSUPPORTED;
    }

    mr_num        = ucs_div_round_up(length, chunk);
    ksm_data_size = (mr_num * sizeof(*ksm_data->mrs)) + sizeof(*ksm_data);
    ksm_data      = ucs_calloc(1, ksm_data_size, "ksm_data");
    if (!ksm_data) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    ucs_trace("multithreaded register memory %p..%p chunks %d",
              address, UCS_PTR_BYTE_OFFSET(address, length), mr_num);

    ksm_data->mr_num = mr_num;
    status = uct_ib_md_handle_mr_list_multithreaded(ibmd, address, length,
                                                    access_flags, chunk,
                                                    ksm_data->mrs, silent);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_ib_mlx5_devx_reg_ksm_data(
            md, memh->super.flags & UCT_IB_MEM_ACCESS_REMOTE_ATOMIC,
            ksm_data, length, 0, &ksm_data->dvmr, &mkey);
    if (status != UCS_OK) {
        goto err_dereg;
    }

    ksm_data->length = length;
    mr->ksm_data     = ksm_data;

    if (mr_type == UCT_IB_MR_DEFAULT) {
        uct_ib_memh_init_keys(ib_memh, mkey, mkey);
    }
    return UCS_OK;

err_dereg:
    uct_ib_md_handle_mr_list_multithreaded(ibmd, address, length, UCT_IB_MEM_DEREG,
                                           chunk, ksm_data->mrs, 1);
err:
    ucs_free(ksm_data);
    return status;
}

static ucs_status_t uct_ib_mlx5_devx_dereg_multithreaded(uct_ib_md_t *ibmd,
                                                         uct_ib_mem_t *ib_memh,
                                                         uct_ib_mr_type_t mr_type)
{
    uct_ib_mlx5_mem_t *memh = ucs_derived_of(ib_memh, uct_ib_mlx5_mem_t);
    uct_ib_mlx5_mr_t *mr    = &memh->mrs[mr_type];
    size_t chunk            = ibmd->config.mt_reg_chunk;
    ucs_status_t s, status  = UCS_OK;

    s = uct_ib_md_handle_mr_list_multithreaded(ibmd, 0, mr->ksm_data->length,
                                               UCT_IB_MEM_DEREG, chunk,
                                               mr->ksm_data->mrs, 1);
    if (s == UCS_ERR_UNSUPPORTED) {
        s = uct_ib_dereg_mrs(mr->ksm_data->mrs, mr->ksm_data->mr_num);
        if (s != UCS_OK) {
            status = s;
        }
    } else if (s != UCS_OK) {
        status = s;
    }

    s = uct_ib_mlx5_devx_obj_destroy(mr->ksm_data->dvmr, "MKEY, KSM");
    if (s != UCS_OK) {
        status = s;
    }

    ucs_free(mr->ksm_data);

    return status;
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

static ucs_status_t
uct_ib_mlx5_devx_check_odp(uct_ib_mlx5_md_t *md,
                           const uct_ib_md_config_t *md_config, void *cap)
{
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(query_hca_cap_out)] = {};
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(query_hca_cap_in)]   = {};
    void *odp;
    ucs_status_t status;

    if (md_config->devx_objs & UCS_BIT(UCT_IB_DEVX_OBJ_RCQP)) {
        ucs_debug("%s: disable ODP because it's not supported for DEVX QP",
                  uct_ib_device_name(&md->super.dev));
        goto no_odp;
    }

    if (uct_ib_mlx5_has_roce_port(&md->super.dev)) {
        ucs_debug("%s: disable ODP on RoCE", uct_ib_device_name(&md->super.dev));
        goto no_odp;
    }

    if (!UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, pg)) {
        goto no_odp;
    }

    odp = UCT_IB_MLX5DV_ADDR_OF(query_hca_cap_out, out, capability);
    UCT_IB_MLX5DV_SET(query_hca_cap_in, in, opcode, UCT_IB_MLX5_CMD_OP_QUERY_HCA_CAP);
    UCT_IB_MLX5DV_SET(query_hca_cap_in, in, op_mod, UCT_IB_MLX5_HCA_CAP_OPMOD_GET_CUR |
                                                   (UCT_IB_MLX5_CAP_ODP << 1));
    status = uct_ib_mlx5_devx_general_cmd(md->super.dev.ibv_context, in,
                                          sizeof(in), out, sizeof(out),
                                          "QUERY_HCA_CAP, ODP", 0);
    if (status != UCS_OK) {
        return status;
    }

    if (!UCT_IB_MLX5DV_GET(odp_cap, odp, ud_odp_caps.send) ||
        !UCT_IB_MLX5DV_GET(odp_cap, odp, rc_odp_caps.send) ||
        !UCT_IB_MLX5DV_GET(odp_cap, odp, rc_odp_caps.write) ||
        !UCT_IB_MLX5DV_GET(odp_cap, odp, rc_odp_caps.read)) {
        goto no_odp;
    }

    if ((md->super.dev.flags & UCT_IB_DEVICE_FLAG_DC) &&
        (!UCT_IB_MLX5DV_GET(odp_cap, odp, dc_odp_caps.send) ||
         !UCT_IB_MLX5DV_GET(odp_cap, odp, dc_odp_caps.write) ||
         !UCT_IB_MLX5DV_GET(odp_cap, odp, dc_odp_caps.read))) {
        goto no_odp;
    }

    if (md->super.config.odp.max_size == UCS_MEMUNITS_AUTO) {
        if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, umr_extended_translation_offset)) {
            md->super.config.odp.max_size = 1ul << 55;
        } else {
            md->super.config.odp.max_size = 1ul << 28;
        }
    }

    if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, fixed_buffer_size) &&
        UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, null_mkey) &&
        UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, umr_extended_translation_offset)) {
        md->super.dev.flags |= UCT_IB_DEVICE_FLAG_ODP_IMPLICIT;
    }

    return UCS_OK;

no_odp:
    md->super.config.odp.max_size = 0;
    return UCS_OK;
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
    struct mlx5dv_devx_event_channel UCS_V_UNUSED *event_channel;
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
    ucs_status_t status;

    if (!(md->flags & UCT_IB_MLX5_MD_FLAG_KSM)) {
        md->super.flush_rkey = UCT_IB_MD_INVALID_FLUSH_RKEY;
        return;
    }

    ucs_assert(UCT_IB_MD_FLUSH_REMOTE_LENGTH <= ucs_get_page_size());

    status = uct_ib_reg_mr(md->super.pd, md->zero_buf,
                           UCT_IB_MD_FLUSH_REMOTE_LENGTH,
                           UCT_IB_MEM_ACCESS_FLAGS, UCT_DMABUF_FD_INVALID, 0,
                           &md->flush_mr, 1);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_ib_mlx5_devx_reg_ksm_data_addr(md, md->flush_mr,
                                                (intptr_t)md->zero_buf,
                                                UCT_IB_MD_FLUSH_REMOTE_LENGTH,
                                                0, 0, 1,
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

static int uct_ib_mlx5_is_xgvmi_alias_supported(struct ibv_context *ctx)
{
#if HAVE_DECL_MLX5DV_DEVX_UMEM_REG_EX
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(query_hca_cap_out)] = {};
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(query_hca_cap_in)]   = {};
    uint64_t object_for_other_vhca;
    uint32_t object_to_object;
    void *cap;
    ucs_status_t status;

    cap = UCT_IB_MLX5DV_ADDR_OF(query_hca_cap_out, out, capability);

    /* query HCA CAP 2 */
    UCT_IB_MLX5DV_SET(query_hca_cap_in, in, opcode,
                      UCT_IB_MLX5_CMD_OP_QUERY_HCA_CAP);
    UCT_IB_MLX5DV_SET(query_hca_cap_in, in, op_mod,
                      UCT_IB_MLX5_HCA_CAP_OPMOD_GET_CUR |
                              (UCT_IB_MLX5_CAP_2_GENERAL << 1));
    status = uct_ib_mlx5_devx_general_cmd(ctx, in, sizeof(in), out, sizeof(out),
                                          "QUERY_HCA_CAP", 1);
    if (status != UCS_OK) {
        return 0;
    }

    object_to_object      = UCT_IB_MLX5DV_GET(
            cmd_hca_cap_2, cap, cross_vhca_object_to_object_supported);
    object_for_other_vhca = UCT_IB_MLX5DV_GET64(
            cmd_hca_cap_2, cap, allowed_object_for_other_vhca_access);

    return (object_to_object &
            UCT_IB_MLX5_HCA_CAPS_2_CROSS_VHCA_OBJ_TO_OBJ_LOCAL_MKEY_TO_REMOTE_MKEY) &&
           (object_for_other_vhca &
            UCT_IB_MLX5_HCA_CAPS_2_ALLOWED_OBJ_FOR_OTHER_VHCA_ACCESS_MKEY);
#else
    return 0;
#endif
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

static ucs_status_t uct_ib_mlx5_devx_md_open(struct ibv_device *ibv_device,
                                             const uct_ib_md_config_t *md_config,
                                             uct_ib_md_t **p_md)
{
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(query_hca_cap_out)] = {};
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(query_hca_cap_in)]   = {};
    ucs_status_t status                                    = UCS_OK;
    uint8_t lag_state                                      = 0;
    uint64_t cap_flags                                     = 0;
    uct_ib_uint128_t vhca_id;
    struct ibv_context *ctx;
    uct_ib_device_t *dev;
    uct_ib_mlx5_md_t *md;
    unsigned max_rd_atomic_dc;
    void *cap;
    int ret;
    ucs_log_level_t log_level;
    ucs_mpool_params_t mp_params;

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
    if (status != UCS_OK) {
        dev->lag_level = 0;
    } else if (lag_state == 0) {
        dev->lag_level = 1;
    } else {
        dev->lag_level = UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, num_lag_ports);
    }

    md->port_select_mode = uct_ib_mlx5_devx_query_port_select(md);

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

    memcpy(vhca_id, UCT_IB_MLX5DV_ADDR_OF(cmd_hca_cap, cap, vhca_id),
           sizeof(vhca_id));

    if (uct_ib_mlx5_is_xgvmi_alias_supported(ctx)) {
        cap_flags |= UCT_MD_FLAG_EXPORTED_MKEY;
        ucs_debug("%s: cross gvmi alias mkey is supported",
                  uct_ib_device_name(dev));
    } else {
        ucs_debug("%s: crossing_vhca_mkey is not supported",
                  uct_ib_device_name(dev));
    }

    status = uct_ib_mlx5_devx_check_odp(md, md_config, cap);
    if (status != UCS_OK) {
        goto err_lru_cleanup;
    }

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

    md->super.ops = &uct_ib_mlx5_devx_md_ops;
    status        = uct_ib_md_open_common(&md->super, ibv_device, md_config,
                                          sizeof(uct_ib_mlx5_mem_t),
                                          sizeof(uct_ib_mlx5_mr_t));
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

    ucs_debug("%s: opened DEVX md", ibv_get_device_name(ibv_device));

    dev->flags          |= UCT_IB_DEVICE_FLAG_MLX5_PRM;
    md->flags           |= UCT_IB_MLX5_MD_FLAG_DEVX;
    md->flags           |= UCT_IB_MLX5_MD_FLAGS_DEVX_OBJS(md_config->devx_objs);
    md->super.name       = UCT_IB_MD_NAME(mlx5);
    md->super.cap_flags |= cap_flags;

    memcpy(md->super.vhca_id, vhca_id, sizeof(vhca_id));

    if (uct_ib_mlx5_devx_md_is_ksm_atomic_supported(md)) {
        md->super.cap_flags |= UCT_MD_FLAG_INVALIDATE;
    }

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

static void uct_ib_mlx5_devx_md_cleanup(uct_ib_md_t *ibmd)
{
    uct_ib_mlx5_md_t *md    = ucs_derived_of(ibmd, uct_ib_mlx5_md_t);
    struct ibv_context *ctx = ibmd->dev.ibv_context;

    uct_ib_mlx5_devx_cleanup_flush_mr(md);
    uct_ib_mlx5_md_buf_free(md, md->zero_buf, &md->zero_mem);
    ucs_mpool_cleanup(&md->dbrec_pool, 1);
    ucs_recursive_spinlock_destroy(&md->dbrec_lock);
    uct_ib_md_close_common(&md->super);
    uct_ib_mlx5_devx_mr_lru_cleanup(md);
    uct_ib_md_free(ibmd);
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
uct_ib_mlx5_devx_reg_exported_key(uct_ib_md_t *ib_md, uct_ib_mem_t *ib_memh)
{
#if HAVE_DECL_MLX5DV_DEVX_UMEM_REG_EX
    uct_ib_mlx5_md_t *md    = ucs_derived_of(ib_md, uct_ib_mlx5_md_t);
    uct_ib_mlx5_mem_t *memh = ucs_derived_of(ib_memh, uct_ib_mlx5_mem_t);
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(create_mkey_in)]                = {0};
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(create_mkey_out)]              = {0};
    char ein[UCT_IB_MLX5DV_ST_SZ_BYTES(allow_other_vhca_access_in)]   = {0};
    char eout[UCT_IB_MLX5DV_ST_SZ_BYTES(allow_other_vhca_access_out)] = {0};
    struct mlx5dv_devx_umem_in umem_in;
    ucs_status_t status;
    void *access_key;
    void *address, *aligned_address;
    size_t length;
    void *mkc;

    address = memh->mrs[UCT_IB_MR_DEFAULT].super.ib->addr;
    length  = memh->mrs[UCT_IB_MR_DEFAULT].super.ib->length;

    /* register umem */
    umem_in.addr        = address;
    umem_in.size        = length;
    umem_in.access      = UCT_IB_MLX5_MD_UMEM_ACCESS;
    aligned_address     = ucs_align_down_pow2_ptr(address, ucs_get_page_size());
    umem_in.pgsz_bitmap = UCS_MASK(ucs_ffs64((uint64_t)aligned_address) + 1);
    umem_in.comp_mask   = 0;

    ucs_assertv(memh->umem == NULL, "memh %p umem %p", memh, memh->umem);
    memh->umem = mlx5dv_devx_umem_reg_ex(md->super.dev.ibv_context, &umem_in);
    if (memh->umem == NULL) {
        uct_ib_md_log_mem_reg_error(ib_md, 0,
                                    "mlx5dv_devx_umem_reg_ex() failed: %m");
        status = UCS_ERR_NO_MEMORY;
        goto err_out;
    }

    /* create mkey */
    mkc = UCT_IB_MLX5DV_ADDR_OF(create_mkey_in, in, memory_key_mkey_entry);
    UCT_IB_MLX5DV_SET(create_mkey_in, in, opcode,
                      UCT_IB_MLX5_CMD_OP_CREATE_MKEY);
    UCT_IB_MLX5DV_SET(create_mkey_in, in, translations_octword_actual_size, 1);
    UCT_IB_MLX5DV_SET(create_mkey_in, in, mkey_umem_id, memh->umem->umem_id);
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
    UCT_IB_MLX5DV_SET64(mkc, mkc, start_addr, (intptr_t)address);
    UCT_IB_MLX5DV_SET64(mkc, mkc, len, length);

    ucs_assertv(memh->cross_mr == NULL, "memh %p cross_mr %p", memh,
                memh->cross_mr);
    memh->cross_mr = uct_ib_mlx5_devx_obj_create(md->super.dev.ibv_context, in,
                                                 sizeof(in), out, sizeof(out),
                                                 "MKEY",
                                                 uct_md_reg_log_lvl(0));
    if (memh->cross_mr == NULL) {
        status = UCS_ERR_IO_ERROR;
        goto err_umem_dereg;
    }

    ucs_assertv(memh->super.exported_lkey == UCT_IB_INVALID_MKEY,
                "memh %p exported_lkey 0x%" PRIx32, memh,
                memh->super.exported_lkey);
    memh->super.exported_lkey = (UCT_IB_MLX5DV_GET(create_mkey_out, out,
                                                   mkey_index) << 8) |
                                md->mkey_tag;

    UCT_IB_MLX5DV_SET(allow_other_vhca_access_in, ein, opcode,
                      UCT_IB_MLX5_CMD_OP_ALLOW_OTHER_VHCA_ACCESS);
    UCT_IB_MLX5DV_SET(allow_other_vhca_access_in, ein,
                      object_type_to_be_accessed, UCT_IB_MLX5_OBJ_TYPE_MKEY);
    UCT_IB_MLX5DV_SET(allow_other_vhca_access_in, ein,
                      object_id_to_be_accessed,
                      memh->super.exported_lkey >> 8);
    access_key = UCT_IB_MLX5DV_ADDR_OF(allow_other_vhca_access_in, ein,
                                       access_key);
    ucs_strncpy_zero(access_key, uct_ib_mkey_token,
                     UCT_IB_MLX5DV_FLD_SZ_BYTES(alias_context, access_key));

    status = uct_ib_mlx5_devx_general_cmd(md->super.dev.ibv_context, ein,
                                          sizeof(ein), eout, sizeof(eout),
                                          "ALLOW_OTHER_VHCA_ACCESS", 0);
    if (status != UCS_OK) {
        goto err_cross_mr_destroy;
    }

    return UCS_OK;

err_cross_mr_destroy:
    mlx5dv_devx_obj_destroy(memh->cross_mr);
    memh->cross_mr = NULL;
err_umem_dereg:
    mlx5dv_devx_umem_dereg(memh->umem);
    memh->umem = NULL;
err_out:
    memh->super.exported_lkey = UCT_IB_INVALID_MKEY;
    return status;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

static ucs_status_t
uct_ib_mlx5_devx_import_exported_key(uct_ib_md_t *ib_md, uint64_t flags,
                                     const uct_ib_uint128_t *target_vhca_id,
                                     uint32_t target_mkey,
                                     uct_ib_mem_t *ib_memh)
{
    uct_ib_mlx5_md_t *md    = ucs_derived_of(ib_md, uct_ib_mlx5_md_t);
    uct_ib_mlx5_mem_t *memh = ucs_derived_of(ib_memh, uct_ib_mlx5_mem_t);
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(create_alias_obj_in)]   = {0};
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(create_alias_obj_out)] = {0};
    void *hdr       = UCT_IB_MLX5DV_ADDR_OF(create_alias_obj_in, in, hdr);
    void *alias_ctx = UCT_IB_MLX5DV_ADDR_OF(create_alias_obj_in, in,
                                            alias_ctx);
    void *access_key;
    void *target_vhca_id_p;
    int ret;

    /* create alias */
    UCT_IB_MLX5DV_SET(general_obj_in_cmd_hdr, hdr, opcode,
                      UCT_IB_MLX5_CMD_OP_CREATE_GENERAL_OBJECT);
    UCT_IB_MLX5DV_SET(general_obj_in_cmd_hdr, hdr, obj_type,
                      UCT_IB_MLX5_OBJ_TYPE_MKEY);
    UCT_IB_MLX5DV_SET(general_obj_in_cmd_hdr, hdr, alias_object, 1);

    target_vhca_id_p = UCT_IB_MLX5DV_ADDR_OF(alias_context, alias_ctx,
                                             vhca_id_to_be_accessed);
    memcpy(target_vhca_id_p, *target_vhca_id, sizeof(*target_vhca_id));

    UCT_IB_MLX5DV_SET(alias_context, alias_ctx, object_id_to_be_accessed,
                      target_mkey >> 8);
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
        goto err_out;
    }

    ret = UCT_IB_MLX5DV_GET(create_alias_obj_out, out, alias_ctx.status);
    if (ret) {
        uct_md_log_mem_attach_error(flags,
                                    "created MR alias object in a bad state");
        goto err_cross_mr_destroy;
    }

    memh->super.lkey = (UCT_IB_MLX5DV_GET(create_alias_obj_out, out,
                                          hdr.obj_id) << 8) |
                        md->mkey_tag;
    memh->super.rkey = memh->super.lkey;

    return UCS_OK;

err_cross_mr_destroy:
    mlx5dv_devx_obj_destroy(memh->cross_mr);
    memh->cross_mr = NULL;
err_out:
    return UCS_ERR_IO_ERROR;
}

static uct_ib_md_ops_t uct_ib_mlx5_devx_md_ops = {
    .open                = uct_ib_mlx5_devx_md_open,
    .cleanup             = uct_ib_mlx5_devx_md_cleanup,
    .reg_key             = uct_ib_mlx5_reg_key,
    .reg_indirect_key    = uct_ib_mlx5_devx_reg_indirect_key,
    .dereg_key           = uct_ib_mlx5_devx_dereg_key,
    .reg_atomic_key      = uct_ib_mlx5_devx_reg_atomic_key,
    .dereg_atomic_key    = uct_ib_mlx5_devx_dereg_atomic_key,
    .reg_multithreaded   = uct_ib_mlx5_devx_reg_multithreaded,
    .dereg_multithreaded = uct_ib_mlx5_devx_dereg_multithreaded,
    .mem_prefetch        = uct_ib_mlx5_mem_prefetch,
    .get_atomic_mr_id    = uct_ib_mlx5_md_get_atomic_mr_id,
    .reg_exported_key    = uct_ib_mlx5_devx_reg_exported_key,
    .import_exported_key = uct_ib_mlx5_devx_import_exported_key
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

    if (!mlx5dv_is_supported(ibv_device)) {
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

    if (UCT_IB_HAVE_ODP_IMPLICIT(&dev->dev_attr) &&
        !uct_ib_mlx5_has_roce_port(dev)) {
        dev->flags |= UCT_IB_DEVICE_FLAG_ODP_IMPLICIT;
    }

    if (IBV_HAVE_ATOMIC_HCA(&dev->dev_attr)) {
        dev->atomic_arg_sizes = sizeof(uint64_t);

#if HAVE_STRUCT_IBV_DEVICE_ATTR_EX_PCI_ATOMIC_CAPS
        dev->pci_fadd_arg_sizes  = dev->dev_attr.pci_atomic_caps.fetch_add << 2;
        dev->pci_cswap_arg_sizes = dev->dev_attr.pci_atomic_caps.compare_swap << 2;
#endif
    }

    uct_ib_mlx5dv_check_dc(dev);

    md->super.ops        = &uct_ib_mlx5_md_ops;
    md->max_rd_atomic_dc = IBV_DEV_ATTR(dev, max_qp_rd_atom);
    status               = uct_ib_md_open_common(&md->super, ibv_device,
                                                 md_config,
                                                 sizeof(uct_ib_mlx5_mem_t),
                                                 sizeof(uct_ib_mlx5_mr_t));
    if (status != UCS_OK) {
        goto err_md_free;
    }

    dev->flags    |= UCT_IB_DEVICE_FLAG_MLX5_PRM;
    md->super.name = UCT_IB_MD_NAME(mlx5);

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
    .open                = uct_ib_mlx5dv_md_open,
    .cleanup             = uct_ib_md_cleanup,
    .reg_key             = uct_ib_mlx5_reg_key,
    .reg_indirect_key    = (uct_ib_md_reg_indirect_key_func_t)
            ucs_empty_function_return_unsupported,
    .dereg_key           = uct_ib_mlx5_dereg_key,
    .reg_atomic_key      = uct_ib_mlx5_reg_atomic_key,
    .dereg_atomic_key    = (uct_ib_md_dereg_atomic_key_func_t)
            ucs_empty_function_return_success,
    .reg_multithreaded   = (uct_ib_md_reg_multithreaded_func_t)
            ucs_empty_function_return_unsupported,
    .dereg_multithreaded = (uct_ib_md_dereg_multithreaded_func_t)
            ucs_empty_function_return_unsupported,
    .mem_prefetch        = uct_ib_mlx5_mem_prefetch,
    .get_atomic_mr_id    = (uct_ib_md_get_atomic_mr_id_func_t)
            ucs_empty_function_return_unsupported,
    .reg_exported_key    = (uct_ib_md_reg_exported_key_func_t )
            ucs_empty_function_return_unsupported,
    .import_exported_key = (uct_ib_md_import_key_func_t )
            ucs_empty_function_return_unsupported
};

UCT_IB_MD_DEFINE_ENTRY(dv, uct_ib_mlx5_md_ops);
