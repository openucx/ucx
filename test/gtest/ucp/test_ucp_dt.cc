/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
extern "C" {
#include <ucp/dt/dt.h>
}

class test_ucp_dt_iov : public ucs::test{
protected:
    size_t calc_iov_offset(const ucp_dt_iov_t *iov, size_t iov_indx, size_t iov_offs) {
        size_t offset = iov_offs;;
        for (size_t i = 0; i < iov_indx; ++i) {
            offset += iov[i].length;
        }
        return offset;
   }
};

UCS_TEST_F(test_ucp_dt_iov, seek)
{
    for (int count = 0; count < 100; ++count) {
        size_t iovcnt = (ucs::rand() % 20) + 1;
        std::vector<ucp_dt_iov_t> iov(iovcnt);

        size_t total_size = 0;
        for (size_t i = 0; i < iovcnt; ++i) {
            iov[i].length = (ucs::rand() % 1000) + 1;
            total_size   += iov[i].length;
        }

        ASSERT_EQ(total_size, calc_iov_offset(&iov[0], iovcnt, 0));

        size_t offset = 0;
        size_t iov_offs = 0, iov_indx = 0;
        for (int j = 0; j < 100; ++j) {
            size_t new_offset = ucs::rand() % total_size;
            ucp_dt_iov_seek(&iov[0], iovcnt,
                            (ptrdiff_t)new_offset - (ptrdiff_t)offset,
                            &iov_offs, &iov_indx);
            EXPECT_EQ(new_offset, calc_iov_offset(&iov[0], iov_indx, iov_offs));
            offset = new_offset;
        }
    }
}
