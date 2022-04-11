
/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
* Copyright (C) Advanced Micro Devices, Inc. 2016 - 2017. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <uct/api/uct.h>
#include <ucs/time/time.h>
#include <uct/ib/base/ib_md.h>
#ifdef HAVE_MLX5_HW
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
    bool check_umr() const;

private:
#ifdef HAVE_DEVX
    uint32_t m_mlx5_flags = 0;
#endif
};

void test_ib_md::init() {
    test_md::init();

#ifdef HAVE_DEVX
    /* Save mlx5 IB md flags because failed atomic registration will modify it */
    if (ib_md().dev.flags & UCT_IB_DEVICE_FLAG_MLX5_PRM) {
        m_mlx5_flags = ucs_derived_of(md(), uct_ib_mlx5_md_t)->flags;
    }
#endif
}

const uct_ib_md_t &test_ib_md::ib_md() const {
    return *ucs_derived_of(md(), uct_ib_md_t);
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

#ifdef HAVE_MLX5_HW
    EXPECT_FALSE(ib_memh->flags & UCT_IB_MEM_FLAG_ATOMIC_MR);
#endif

    status = uct_md_mkey_pack(md(), memh, rkey_buffer);
    EXPECT_UCS_OK(status);

    status = uct_md_mkey_pack(md(), memh, rkey_buffer);
    EXPECT_UCS_OK(status);

#ifdef HAVE_MLX5_HW
    if ((amo_access && check_umr()) || ib_md().relaxed_order) {
        EXPECT_TRUE(ib_memh->flags & UCT_IB_MEM_FLAG_ATOMIC_MR);
        EXPECT_NE(UCT_IB_INVALID_MKEY, ib_memh->atomic_rkey);
    } else {
        EXPECT_FALSE(ib_memh->flags & UCT_IB_MEM_FLAG_ATOMIC_MR);
        EXPECT_EQ(UCT_IB_INVALID_MKEY, ib_memh->atomic_rkey);
    }
#endif

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
#elif defined(HAVE_EXP_UMR_KSM)
    return ib_md().dev.dev_attr.exp_device_cap_flags &
           IBV_EXP_DEVICE_UMR_FIXED_SIZE;
#else
    return false;
#endif
}

bool test_ib_md::check_umr() const {
#if HAVE_DEVX
    return has_ksm();
#elif HAVE_EXP_UMR
    if (ib_md().dev.flags & UCT_IB_DEVICE_FLAG_MLX5_PRM) {
        uct_ib_mlx5_md_t *mlx5_md = ucs_derived_of(&ib_md(), uct_ib_mlx5_md_t);
        return mlx5_md->umr_qp != NULL;
    }
    return false;
#else
    return false;
#endif
}

UCS_TEST_P(test_ib_md, ib_md_umr_rcache, "REG_METHODS=rcache") {
    std::string rkey_buffer(md_attr().rkey_packed_size, '\0');

    /* The order is important here because
     * of registration cache. A cached region will
     * be promoted to atomic access but it will never be demoted
     */
    ib_md_umr_check(&rkey_buffer[0], false);
    ib_md_umr_check(&rkey_buffer[0], true);
}

UCS_TEST_P(test_ib_md, ib_md_umr_direct, "REG_METHODS=direct") {
    std::string rkey_buffer(md_attr().rkey_packed_size, '\0');

    /* without rcache the order is not really important */
    ib_md_umr_check(&rkey_buffer[0], true);
    ib_md_umr_check(&rkey_buffer[0], false);
    ib_md_umr_check(&rkey_buffer[0], true);
    ib_md_umr_check(&rkey_buffer[0], false);
}

UCS_TEST_P(test_ib_md, ib_md_umr_ksm) {
    std::string rkey_buffer(md_attr().rkey_packed_size, '\0');
    ib_md_umr_check(&rkey_buffer[0], has_ksm(), UCT_IB_MD_MAX_MR_SIZE + 0x1000);
}

UCS_TEST_P(test_ib_md, relaxed_order, "PCI_RELAXED_ORDERING=on") {
    std::string rkey_buffer(md_attr().rkey_packed_size, '\0');

    ib_md_umr_check(&rkey_buffer[0], false);
    ib_md_umr_check(&rkey_buffer[0], true);
}

#if HAVE_UMR_KSM
UCS_TEST_P(test_ib_md, umr_noninline_klm, "MAX_INLINE_KLM_LIST=1") {

    /* KLM list size would be 2, and setting MAX_INLINE_KLM_LIST=1 would force
     * using non-inline UMR post_send.
     */
    std::string rkey_buffer(md_attr().rkey_packed_size, '\0');
    ib_md_umr_check(&rkey_buffer[0], has_ksm(), UCT_IB_MD_MAX_MR_SIZE + 0x1000);
}
#endif

UCS_TEST_P(test_ib_md, aligned) {
    std::string rkey_buffer(md_attr().rkey_packed_size, '\0');
    ib_md_umr_check(&rkey_buffer[0], true, UCT_IB_MD_MAX_MR_SIZE, true);
}

_UCT_MD_INSTANTIATE_TEST_CASE(test_ib_md, ib)
