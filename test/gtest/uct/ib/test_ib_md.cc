/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2016. ALL RIGHTS RESERVED.
* Copyright (C) Advanced Micro Devices, Inc. 2016 - 2017. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <uct/api/uct.h>
#include <ucs/time/time.h>
#include <uct/ib/base/ib_md.h>
#ifdef HAVE_MLX5_DV
#include <uct/ib/mlx5/ib_mlx5.h>
#endif

#include <common/test.h>
#include <uct/test_md.h>

class test_ib_md : public test_md
{
protected:
    void init() override;
    const uct_ib_md_t &ib_md() const;
    void ib_md_umr_check(void *rkey_buffer, bool amo_access,
                         size_t size = 8192, bool aligned = false);
    bool has_ksm() const;

    ucs_status_t reg_smkey_pack(void *buffer, size_t size, uct_ib_mem_t **memh,
                                uct_rkey_t *rkey_p = NULL);
    void check_smkeys(uct_rkey_t rkey1, uct_rkey_t rkey2);

    void test_mkey_pack_mt(bool invalidate);
    void test_mkey_pack_mt_internal(unsigned access_mask, bool invalidate);
    void test_smkey_reg_atomic(void);

private:
#ifdef HAVE_MLX5_DV
    uint32_t m_mlx5_flags = 0;
#endif
    void check_mlx5_mr(uct_ib_mem_t *ib_memh, bool is_expected);
};

void test_ib_md::init() {
    test_md::init();

#ifdef HAVE_MLX5_DV
    /* Save mlx5 IB md flags because failed atomic registration will modify it */
    if (ib_md().dev.flags & UCT_IB_DEVICE_FLAG_MLX5_PRM) {
        m_mlx5_flags = ucs_derived_of(md(), uct_ib_mlx5_md_t)->flags;
    }
#endif
}

const uct_ib_md_t &test_ib_md::ib_md() const {
    return *ucs_derived_of(md(), uct_ib_md_t);
}

void test_ib_md::check_mlx5_mr(uct_ib_mem_t *ib_memh, bool is_expected)
{
#if HAVE_DEVX
    uct_ib_mlx5_devx_mem_t *memh = ucs_derived_of(ib_memh,
                                                  uct_ib_mlx5_devx_mem_t);
    if (is_expected) {
        EXPECT_NE(nullptr, memh->atomic_dvmr);
        EXPECT_NE(UCT_IB_INVALID_MKEY, memh->atomic_rkey);
    } else {
        EXPECT_EQ(nullptr, memh->atomic_dvmr);
        EXPECT_EQ(UCT_IB_INVALID_MKEY, memh->atomic_rkey);
    }

    EXPECT_EQ(nullptr, memh->smkey_mr);
#endif
}

/*
 * Test that ib md does not create umr region if
 * UCT_MD_MEM_ACCESS_REMOTE_ATOMIC is not set
 */
void test_ib_md::ib_md_umr_check(void *rkey_buffer, bool amo_access,
                                 size_t size, bool aligned)
{
    ucs_status_t status;
    size_t alloc_size;
    void *buffer;
    int ret;

    if (amo_access && (IBV_DEV_ATTR(&ib_md().dev, vendor_part_id) < 4123)) { /* <CX6 */
        UCS_TEST_SKIP_R("HW does not support atomic indirect MR");
    }

    if (ucs_get_phys_mem_size() < size * 8) {
        UCS_TEST_SKIP_R("not enough physical memory");
    }
    if (ucs_get_memfree_size() < size * 4) {
        UCS_TEST_SKIP_R("not enough free memory");
    }

    buffer     = NULL;
    alloc_size = size;
    if (aligned) {
        ret = ucs_posix_memalign(&buffer, alloc_size, alloc_size, "test_umr");
        ASSERT_TRUE(ret == 0);
    } else {
        status = ucs_mmap_alloc(&alloc_size, &buffer, 0, "test_umr");
        ASSERT_UCS_OK(status);
    }

    uct_mem_h memh;
    status = uct_md_mem_reg(md(), buffer, size,
                            amo_access ? UCT_MD_MEM_ACCESS_REMOTE_ATOMIC :
                                         UCT_MD_MEM_ACCESS_RMA,
                            &memh);
    ASSERT_UCS_OK(status, << " buffer=" << buffer << " size=" << size);
    ASSERT_TRUE(memh != UCT_MEM_HANDLE_NULL);

    uct_ib_mem_t *ib_memh = (uct_ib_mem_t *)memh;

    if (amo_access) {
        EXPECT_TRUE(ib_memh->flags & UCT_IB_MEM_ACCESS_REMOTE_ATOMIC);
    } else {
        EXPECT_FALSE(ib_memh->flags & UCT_IB_MEM_ACCESS_REMOTE_ATOMIC);
    }

    check_mlx5_mr(ib_memh, false);

    status = uct_md_mkey_pack(md(), memh, rkey_buffer);
    EXPECT_UCS_OK(status);

    status = uct_md_mkey_pack(md(), memh, rkey_buffer);
    EXPECT_UCS_OK(status);
    check_mlx5_mr(ib_memh, (amo_access && has_ksm()) || ib_md().relaxed_order);

    status = uct_md_mem_dereg(md(), memh);
    EXPECT_UCS_OK(status);

    if (aligned) {
        ucs_free(buffer);
    } else {
        ucs_mmap_free(buffer, alloc_size);
    }
}

bool test_ib_md::has_ksm() const {
#if HAVE_DEVX
    return m_mlx5_flags & UCT_IB_MLX5_MD_FLAG_KSM;
#else
    return false;
#endif
}

UCS_TEST_P(test_ib_md, ib_md_umr_ksm) {
    std::string rkey_buffer(md_attr().rkey_packed_size, '\0');
    ib_md_umr_check(&rkey_buffer[0], has_ksm(), UCT_IB_MD_MAX_MR_SIZE + 0x1000);
}

UCS_TEST_P(test_ib_md, relaxed_order, "IB_PCI_RELAXED_ORDERING=try") {
    std::string rkey_buffer(md_attr().rkey_packed_size, '\0');

    ib_md_umr_check(&rkey_buffer[0], false);
    ib_md_umr_check(&rkey_buffer[0], true);
}

UCS_TEST_P(test_ib_md, aligned) {
    std::string rkey_buffer(md_attr().rkey_packed_size, '\0');
    size_t size = RUNNING_ON_VALGRIND ? 8192 : UCT_IB_MD_MAX_MR_SIZE;
    ib_md_umr_check(&rkey_buffer[0], true, size, true);
}

ucs_status_t test_ib_md::reg_smkey_pack(void *buffer, size_t size,
                                        uct_ib_mem_t **memh, uct_rkey_t *rkey_p)
{
    std::string rkey_buffer(md_attr().rkey_packed_size, '\0');
    uct_rkey_bundle_t bundle;

    ucs_status_t status = uct_md_mem_reg(md(), buffer, size,
                                         UCT_MD_MEM_ACCESS_ALL |
                                                 UCT_MD_MEM_SYMMETRIC_RKEY,
                                         (void**)memh);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_md_mkey_pack(md(), *memh, &rkey_buffer[0]);
    if ((status == UCS_OK) && (rkey_p != NULL)) {
        status = uct_rkey_unpack(md()->component, &rkey_buffer[0], &bundle);
        if (status == UCS_OK) {
            *rkey_p = bundle.rkey;
            status  = uct_rkey_release(md()->component, &bundle);
        }
    }

    return status;
}

// Symmetric keys properties for the generic IB transport
void test_ib_md::check_smkeys(uct_rkey_t rkey1, uct_rkey_t rkey2)
{
    EXPECT_NE(UCT_IB_INVALID_MKEY, uct_ib_md_atomic_rkey(rkey1));
    EXPECT_NE(UCT_IB_INVALID_MKEY, uct_ib_md_direct_rkey(rkey1));
    EXPECT_NE(UCT_IB_INVALID_MKEY, uct_ib_md_atomic_rkey(rkey2));
    EXPECT_NE(UCT_IB_INVALID_MKEY, uct_ib_md_direct_rkey(rkey2));
    EXPECT_EQ(uct_ib_md_atomic_rkey(rkey1) - uct_ib_md_direct_rkey(rkey1),
              uct_ib_md_atomic_rkey(rkey2) - uct_ib_md_direct_rkey(rkey2));
}

void test_ib_md::test_smkey_reg_atomic(void)
{
    static const size_t size = 8192;
    void *buffer;
    uct_ib_mem_t *memh1, *memh2, *memh3;
    uct_rkey_t rkey1, rkey2, rkey3;

    if (ib_md().mkey_by_name_reserve.size == 0) {
        UCS_TEST_SKIP_R("HW does not allow symmetric rkey");
    }

    ASSERT_EQ(0, ucs_posix_memalign(&buffer, size, size, "smkey_reg_atomic"));
    ASSERT_UCS_OK(reg_smkey_pack(buffer, size, &memh1, &rkey1));
    ASSERT_UCS_OK(reg_smkey_pack(buffer, size, &memh2, &rkey2));
    ASSERT_UCS_OK(reg_smkey_pack(buffer, size, &memh3, &rkey3));

    check_smkeys(rkey1, rkey2);
    check_smkeys(rkey2, rkey3);

    EXPECT_UCS_OK(uct_md_mem_dereg(md(), memh1));
    EXPECT_UCS_OK(uct_md_mem_dereg(md(), memh2));
    EXPECT_UCS_OK(uct_md_mem_dereg(md(), memh3));
    ucs_mmap_free(buffer, size);
}

void test_ib_md::test_mkey_pack_mt_internal(unsigned access_mask,
                                            bool invalidate)
{
    constexpr size_t size = UCS_MBYTE;
    unsigned pack_flags, dereg_flags;
    void *buffer;
    uct_mem_h memh;

    if (!check_invalidate_support(access_mask)) {
        UCS_TEST_SKIP_R("mkey invalidation isn't supported");
    }

    if (!has_ksm()) {
        UCS_TEST_SKIP_R("KSM is required for MT registration");
    }

    buffer = ucs_malloc(size, "mkey_pack_mt");
    ASSERT_NE(buffer, nullptr) << "Allocation failed";

    if (invalidate) {
        pack_flags  = UCT_MD_MKEY_PACK_FLAG_INVALIDATE_RMA;
        dereg_flags = UCT_MD_MEM_DEREG_FLAG_INVALIDATE;
    } else {
        pack_flags = dereg_flags = 0;
    }

    ASSERT_UCS_OK(reg_mem(access_mask, buffer, size, &memh));

    uct_ib_mem_t *ib_memh = (uct_ib_mem_t*)memh;
    EXPECT_TRUE(ib_memh->flags & UCT_IB_MEM_MULTITHREADED);

    std::vector<uint8_t> rkey(md_attr().rkey_packed_size);
    uct_md_mkey_pack_params_t pack_params;
    pack_params.field_mask = UCT_MD_MKEY_PACK_FIELD_FLAGS;
    pack_params.flags      = pack_flags;
    ASSERT_UCS_OK(uct_md_mkey_pack_v2(md(), memh, buffer, size,
                                      &pack_params, rkey.data()));

    uct_md_mem_dereg_params_t params;
    params.field_mask  = UCT_MD_MEM_DEREG_FIELD_MEMH |
                         UCT_MD_MEM_DEREG_FIELD_COMPLETION |
                         UCT_MD_MEM_DEREG_FIELD_FLAGS;
    params.memh        = memh;
    params.flags       = dereg_flags;
    comp().comp.func   = dereg_cb;
    comp().comp.count  = 1;
    comp().comp.status = UCS_OK;
    comp().self        = this;
    params.comp        = &comp().comp;
    ASSERT_UCS_OK(uct_md_mem_dereg_v2(md(), &params));

    ucs_free(buffer);
}

void test_ib_md::test_mkey_pack_mt(bool invalidate)
{
    test_mkey_pack_mt_internal(UCT_MD_MEM_ACCESS_REMOTE_ATOMIC, invalidate);
    test_mkey_pack_mt_internal(UCT_MD_MEM_ACCESS_RMA, invalidate);
    test_mkey_pack_mt_internal(UCT_MD_MEM_ACCESS_ALL, invalidate);
}

UCS_TEST_P(test_ib_md, pack_mkey_mt, "IB_REG_MT_THRESH=128K",
           "IB_REG_MT_CHUNK=128K")
{
    test_mkey_pack_mt(false);
}

UCS_TEST_P(test_ib_md, pack_mkey_mt_invalidate, "IB_REG_MT_THRESH=128K",
           "IB_REG_MT_CHUNK=128K")
{
    test_mkey_pack_mt(true);
}

UCS_TEST_P(test_ib_md, smkey_reg_atomic)
{
    test_smkey_reg_atomic();
}

UCS_TEST_P(test_ib_md, smkey_reg_atomic_mt, "IB_REG_MT_THRESH=1k",
           "IB_REG_MT_CHUNK=1k")
{
    test_smkey_reg_atomic();
}

_UCT_MD_INSTANTIATE_TEST_CASE(test_ib_md, ib)

class test_ib_md_non_blocking : public test_md_non_blocking {
};

UCS_TEST_P(test_ib_md_non_blocking, reg_advise_odp_no_devx, "IB_MLX5_DEVX=no")
{
    test_nb_reg_advise();
}

UCS_TEST_P(test_ib_md_non_blocking, reg_odp_no_devx, "IB_MLX5_DEVX=no")
{
    test_nb_reg();
}

_UCT_MD_INSTANTIATE_TEST_CASE(test_ib_md_non_blocking, ib)
