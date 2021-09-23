/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
#include <algorithm>

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

    void init_dt_iter(size_t size, bool is_pack)
    {
        m_dt_buffer.resize(size, 0);
        ucs::fill_random(m_dt_buffer);

        m_packed_buffer.resize(size, 0);
        ucs::fill_random(m_packed_buffer);

        size_t iovcnt = 1;
        if (GetParam() == UCP_DATATYPE_IOV) {
            iovcnt = std::min(static_cast<size_t>((ucs::rand() % 20) + 1),
                              m_dt_buffer.size());
        }

        m_dt_desc.make(GetParam(), &m_dt_buffer[0], m_dt_buffer.size(), iovcnt);

        uint8_t sg_count;
        ucp_datatype_iter_init(m_ucph.get(), m_dt_desc.buf(), m_dt_desc.count(),
                               m_dt_desc.dt(), m_dt_buffer.size(), is_pack,
                               &m_dt_iter, &sg_count);
        if (!UCP_DT_IS_GENERIC(GetParam())) {
            EXPECT_EQ(iovcnt, sg_count);
        }

        UCS_STRING_BUFFER_ONSTACK(strb, 64);
        ucp_datatype_iter_str(&m_dt_iter, &strb);
        UCS_TEST_MESSAGE << ucs_string_buffer_cstr(&strb);
    }

    void finalize_dt_iter()
    {
        EXPECT_EQ(m_dt_buffer, m_packed_buffer);
        ucp_datatype_iter_cleanup(&m_dt_iter, UINT_MAX);
    }

    size_t random_seg_size() const
    {
        return ucs::rand() % (m_dt_buffer.size() / 2);
    }

    void test_pack(size_t size)
    {
        init_dt_iter(size, true);

        ucp_datatype_iter_t next_iter;
        do {
            EXPECT_FALSE(ucp_datatype_iter_is_end(&m_dt_iter));
            size_t seg_size  = random_seg_size();
            void *packed_ptr = UCS_PTR_BYTE_OFFSET(&m_packed_buffer[0],
                                                   m_dt_iter.offset);
            /* TODO create non-NULL worker when using memtype */
            ucp_datatype_iter_next_pack(&m_dt_iter, NULL, seg_size, &next_iter,
                                        packed_ptr);
            ucp_datatype_iter_copy_position(&m_dt_iter, &next_iter, UINT_MAX);
        } while (!ucp_datatype_iter_is_end(&m_dt_iter));

        finalize_dt_iter();
    }

    void test_unpack(size_t size)
    {
        init_dt_iter(size, false);

        typedef std::vector< std::pair<size_t, size_t> > segment_vector_t;
        segment_vector_t segments;
        size_t offset = 0;
        while (offset < m_dt_buffer.size()) {
            size_t seg_size = ucs_min(random_seg_size(),
                                      m_dt_buffer.size() - offset);
            segments.push_back(std::make_pair(offset, seg_size));
            offset += seg_size;
        }
        std::random_shuffle(segments.begin(), segments.end(), ucs::rand_range);

        for (segment_vector_t::iterator it = segments.begin();
             it != segments.end(); ++it) {
            size_t offset    = it->first;
            void *packed_ptr = UCS_PTR_BYTE_OFFSET(&m_packed_buffer[0], offset);
            /* TODO create non-NULL worker when using memtype */
            ucp_datatype_iter_unpack(&m_dt_iter, NULL, it->second, offset,
                                     packed_ptr);
        }

        finalize_dt_iter();
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
    static ucp_dt_generic_t   *dt_gen;
    ucs::handle<ucp_context_h> m_ucph;
    std::string                m_dt_buffer;
    std::string                m_packed_buffer;
    ucp::data_type_desc_t      m_dt_desc;
    ucp_datatype_iter_t        m_dt_iter;
};

ucp_dt_generic_t* test_ucp_dt_iter::dt_gen = 0;

UCS_TEST_P(test_ucp_dt_iter, pack_100b) {
    test_pack(100);
}

UCS_TEST_P(test_ucp_dt_iter, pack_1MB) {
    test_pack(UCS_MBYTE + (ucs::rand() % UCS_KBYTE));
}

UCS_TEST_P(test_ucp_dt_iter, unpack_100b) {
    test_unpack(100);
}

UCS_TEST_P(test_ucp_dt_iter, unpack_1MB) {
    test_unpack(UCS_MBYTE + (ucs::rand() % UCS_KBYTE));
}

INSTANTIATE_TEST_SUITE_P(contig, test_ucp_dt_iter,
                        testing::Values(ucp_dt_make_contig(1),
                                        ucp_dt_make_contig(8),
                                        ucp_dt_make_contig(39)));

INSTANTIATE_TEST_SUITE_P(iov, test_ucp_dt_iter,
                        testing::Values(ucp_dt_make_iov()));

INSTANTIATE_TEST_SUITE_P(generic, test_ucp_dt_iter,
                        testing::ValuesIn(test_ucp_dt_iter::enum_dt_generic_params()));
