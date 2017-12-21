
/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
* Copyright (C) Advanced Micro Devices, Inc. 2016 - 2017. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

extern "C" {
#include <uct/api/uct.h>
#include <ucs/time/time.h>
#include <uct/ib/base/ib_md.h>
}
#include <common/test.h>
#include <uct/test_md.h>

class test_ib_md : public test_md
{
protected:
    void ib_md_umr_check(void *rkey_buffer, bool amo_access);
};


/*
 * Test that ib md does not create umr region if 
 * UCT_MD_MEM_ACCESS_REMOTE_ATOMIC is not set
 */

void test_ib_md::ib_md_umr_check(void *rkey_buffer, bool amo_access) {

    ucs_status_t status;
    size_t size = 8192;
    void *buffer = malloc(size);
    ASSERT_TRUE(buffer != NULL);

    uct_mem_h memh;
    status = uct_md_mem_reg(pd(), buffer, size, 
                            amo_access ? UCT_MD_MEM_ACCESS_REMOTE_ATOMIC :
                                         UCT_MD_MEM_ACCESS_RMA,
                            &memh);
    ASSERT_UCS_OK(status, << " buffer=" << buffer << " size=" << size);
    ASSERT_TRUE(memh != UCT_MEM_HANDLE_NULL);

    uct_ib_mem_t *ib_memh = (uct_ib_mem_t *)memh;
    uct_ib_md_t  *ib_md = (uct_ib_md_t *)pd();

    if (amo_access) {
        EXPECT_TRUE(ib_memh->flags & UCT_IB_MEM_ACCESS_REMOTE_ATOMIC);
        EXPECT_FALSE(ib_memh->flags & UCT_IB_MEM_FLAG_ATOMIC_MR);
    } else {
        EXPECT_FALSE(ib_memh->flags & UCT_IB_MEM_ACCESS_REMOTE_ATOMIC);
        EXPECT_FALSE(ib_memh->flags & UCT_IB_MEM_FLAG_ATOMIC_MR);
    }

    status = uct_md_mkey_pack(pd(), memh, rkey_buffer);
    EXPECT_UCS_OK(status);

    if (amo_access) {
        if (ib_md->umr_qp != NULL) {
            EXPECT_TRUE(ib_memh->flags & UCT_IB_MEM_FLAG_ATOMIC_MR);
            EXPECT_TRUE(ib_memh->atomic_mr != NULL);
        } else {
            EXPECT_FALSE(ib_memh->flags & UCT_IB_MEM_FLAG_ATOMIC_MR);
            EXPECT_TRUE(ib_memh->atomic_mr == NULL);
        }
    } else {
        EXPECT_FALSE(ib_memh->flags & UCT_IB_MEM_FLAG_ATOMIC_MR);
        EXPECT_TRUE(ib_memh->atomic_mr == NULL);
    }

    status = uct_md_mem_dereg(pd(), memh);
    EXPECT_UCS_OK(status);
    free(buffer);
}

UCS_TEST_P(test_ib_md, ib_md_umr_rcache, "REG_METHODS=rcache") {

    ucs_status_t status;
    uct_md_attr_t md_attr;
    void *rkey_buffer;

    status = uct_md_query(pd(), &md_attr);
    ASSERT_UCS_OK(status);
    rkey_buffer = malloc(md_attr.rkey_packed_size);
    ASSERT_TRUE(rkey_buffer != NULL);

    /* The order is important here because
     * of registration cache. A cached region will
     * be promoted to atomic access but it will never be demoted 
     */
    ib_md_umr_check(rkey_buffer, false);
    ib_md_umr_check(rkey_buffer, true);

    free(rkey_buffer);
}

UCS_TEST_P(test_ib_md, ib_md_umr_direct, "REG_METHODS=direct") {

    ucs_status_t status;
    uct_md_attr_t md_attr;
    void *rkey_buffer;

    status = uct_md_query(pd(), &md_attr);
    ASSERT_UCS_OK(status);
    rkey_buffer = malloc(md_attr.rkey_packed_size);
    ASSERT_TRUE(rkey_buffer != NULL);

    /* without rcache the order is not really important */
    ib_md_umr_check(rkey_buffer, true);
    ib_md_umr_check(rkey_buffer, false);
    ib_md_umr_check(rkey_buffer, true);
    ib_md_umr_check(rkey_buffer, false);

    free(rkey_buffer);
}

_UCT_MD_INSTANTIATE_TEST_CASE(test_ib_md, ib)
