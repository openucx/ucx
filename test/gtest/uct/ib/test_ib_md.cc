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

private:
#ifdef HAVE_MLX5_DV
    uint32_t m_mlx5_flags = 0;
#endif
    void check_mlx5_atomic_mr(uct_ib_mem_t *ib_memh, bool is_expected);
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

void test_ib_md::check_mlx5_atomic_mr(uct_ib_mem_t *ib_memh, bool is_expected)
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

    check_mlx5_atomic_mr(ib_memh, false);

    status = uct_md_mkey_pack(md(), memh, rkey_buffer);
    EXPECT_UCS_OK(status);

    status = uct_md_mkey_pack(md(), memh, rkey_buffer);
    EXPECT_UCS_OK(status);
    check_mlx5_atomic_mr(ib_memh,
                         (amo_access && has_ksm()) || ib_md().relaxed_order);

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

UCS_TEST_P(test_ib_md, relaxed_order, "PCI_RELAXED_ORDERING=try") {
    std::string rkey_buffer(md_attr().rkey_packed_size, '\0');

    ib_md_umr_check(&rkey_buffer[0], false);
    ib_md_umr_check(&rkey_buffer[0], true);
}

UCS_TEST_P(test_ib_md, aligned) {
    std::string rkey_buffer(md_attr().rkey_packed_size, '\0');
    size_t size = RUNNING_ON_VALGRIND ? 8192 : UCT_IB_MD_MAX_MR_SIZE;
    ib_md_umr_check(&rkey_buffer[0], true, size, true);
}

_UCT_MD_INSTANTIATE_TEST_CASE(test_ib_md, ib)
