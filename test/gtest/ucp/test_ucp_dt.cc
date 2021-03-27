/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>

#include "ucp_datatype.h"

extern "C" {
#include <ucp/dt/dt.h>
#include <ucp/dt/datatype_iter.inl>
}

class test_ucp_dt_iov : public ucs::test {
protected:
    size_t calc_iov_offset(const ucp_dt_iov_t *iov, size_t iov_indx, size_t iov_offs) {
        size_t offset = iov_offs;
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
};

class test_ucp_dt_iter : public ucs::test_with_param<ucp_datatype_t> {
protected:
    virtual void init() {
        ucp_params_t ctx_params;
        ctx_params.field_mask = UCP_PARAM_FIELD_FEATURES;
        ctx_params.features   = UCP_FEATURE_TAG;
        UCS_TEST_CREATE_HANDLE(ucp_context_h, m_ucph, ucp_cleanup, ucp_init,
                               &ctx_params, NULL);
    }

    virtual void cleanup() {
        m_ucph.reset();
    }

    void do_test(size_t size, bool is_pack) {
        std::string dt_buffer(size, 0), packed_buffer(size, 0);
        ucs::fill_random(dt_buffer);

        size_t iovcnt = 1;
        if (GetParam() == UCP_DATATYPE_IOV) {
            iovcnt = std::min(static_cast<size_t>((ucs::rand() % 20) + 1),
                              dt_buffer.size());
        }

        ucp::data_type_desc_t dt_desc(GetParam(), &dt_buffer[0],
                                      dt_buffer.size(), iovcnt);

        ucp_datatype_iter_t dt_iter = {};
        uint8_t sg_count;
        ucp_datatype_iter_init(m_ucph.get(), dt_desc.buf(), dt_desc.count(),
                               dt_desc.dt(), dt_buffer.size(), &dt_iter, &sg_count);
        EXPECT_EQ(iovcnt, sg_count);

        size_t offset = 0;
        ucs::fill_random(packed_buffer);

        while (!ucp_datatype_iter_is_end(&dt_iter)) {
            void *packed_ptr = UCS_PTR_BYTE_OFFSET(&packed_buffer[0], offset);
            size_t seg_size  = (ucs::rand() % (size / 2));
            ucp_datatype_iter_t next_iter;
            /* TODO create non-NULL worker when using memtype */
            if (is_pack) {
                ucp_datatype_iter_next_pack(&dt_iter, NULL, seg_size,
                                            &next_iter, packed_ptr);
            } else {
                size_t unpack_size = std::min(seg_size, size - offset);
                ucp_datatype_iter_next_unpack(&dt_iter, NULL, unpack_size,
                                              &next_iter, packed_ptr);
            }
            ucp_datatype_iter_copy_from_next(&dt_iter, &next_iter, UINT_MAX);
            offset += seg_size;
        }

        EXPECT_EQ(dt_buffer, packed_buffer);

        ucp_datatype_iter_cleanup(&dt_iter, UINT_MAX);
    }

public:
    static std::vector<ucp_datatype_t> enum_dt_generic_params() {
        ucp_datatype_t datatype;
        ucs_status_t status;

        if (dt_gen == 0) {
            status = ucp_dt_create_generic(&ucp::test_dt_copy_ops, NULL, &datatype);
            if (status != UCS_OK) {
                return std::vector<ucp_datatype_t>();
            }

            /* keep global pointer to dt_gen to silence valgrind leak checker */
            dt_gen = ucp_dt_to_generic(datatype);
        }

        return std::vector<ucp_datatype_t>(1, ucp_dt_from_generic(dt_gen));
    }

private:
    static ucp_dt_generic_t* dt_gen;

    ucs::handle<ucp_context_h> m_ucph;
};

ucp_dt_generic_t* test_ucp_dt_iter::dt_gen = 0;

UCS_TEST_P(test_ucp_dt_iter, pack_100b) {
    do_test(100, true);
}

UCS_TEST_P(test_ucp_dt_iter, pack_1MB) {
    do_test(UCS_MBYTE + (ucs::rand() % UCS_KBYTE), true);
}

UCS_TEST_P(test_ucp_dt_iter, unpack_100b) {
    do_test(100, false);
}

UCS_TEST_P(test_ucp_dt_iter, unpack_1MB) {
    do_test(UCS_MBYTE + (ucs::rand() % UCS_KBYTE), false);
}

INSTANTIATE_TEST_CASE_P(contig, test_ucp_dt_iter,
                        testing::Values(ucp_dt_make_contig(1),
                                        ucp_dt_make_contig(8),
                                        ucp_dt_make_contig(39)));

INSTANTIATE_TEST_CASE_P(iov, test_ucp_dt_iter,
                        testing::Values((ucp_datatype_t)ucp_dt_make_iov()));

INSTANTIATE_TEST_CASE_P(generic, test_ucp_dt_iter,
                        testing::ValuesIn(test_ucp_dt_iter::enum_dt_generic_params()));
