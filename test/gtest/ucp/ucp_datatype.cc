/**
* Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <common/test.h>

#include "ucp_datatype.h"
#include "ucp_test.h"

extern "C" {
#include <ucp/dt/dt.inl>
}

namespace ucp {


data_type_desc_t &
data_type_desc_t::make(ucp_datatype_t datatype, const void *buf, size_t length,
                       size_t iov_cnt)
{
    if (m_length == 0) {
        m_length = length;
    }

    if (m_origin == uintptr_t(NULL)) {
        m_origin = uintptr_t(buf);
    }

    m_dt = datatype;
    memset(m_iov, 0, sizeof(m_iov));

    switch (m_dt & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        m_buf   = buf;
        m_count = length / ucp_contig_dt_elem_size(datatype);
        break;
    case UCP_DATATYPE_IOV:
    {
        const size_t iov_length = (length > iov_cnt) ?
            ucs::rand() % (length / iov_cnt) : 0;
        size_t iov_length_it = 0;
        for (size_t iov_it = 0; iov_it < iov_cnt - 1; ++iov_it) {
            m_iov[iov_it].buffer = (char *)(buf) + iov_length_it;
            m_iov[iov_it].length = iov_length;
            iov_length_it += iov_length;
        }

        /* Last entry */
        m_iov[iov_cnt - 1].buffer = (char *)(buf) + iov_length_it;
        m_iov[iov_cnt - 1].length = length - iov_length_it;

        m_buf   = m_iov;
        m_count = iov_cnt;
        break;
    }
    case UCP_DATATYPE_GENERIC:
        m_buf   = buf;
        m_count = length;
        break;
    default:
        m_buf   = NULL;
        m_count = 0;
        EXPECT_TRUE(false) << "Unsupported datatype";
        break;
    }

    return *this;
}

int dt_gen_start_count  = 0;
int dt_gen_finish_count = 0;

static void* dt_common_start(void *context, void *buffer, size_t count)
{
    dt_gen_state *dt_state = new dt_gen_state;

    dt_state->count   = count;
    dt_state->started = 1;
    dt_state->magic   = ucp::MAGIC;
    dt_state->context = context;
    dt_state->buffer  = buffer;
    dt_gen_start_count++;

    return dt_state;
}

static void* dt_common_start_pack(void *context, const void *buffer,
                                  size_t count)
{
    return dt_common_start(NULL, (void*)buffer, count);
}

static void* dt_common_start_unpack(void *context, void *buffer, size_t count)
{
    return dt_common_start(context, buffer, count);
}

template <typename T>
size_t dt_packed_size(void *state)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;

    return dt_state->count * sizeof(T);
}

template <typename T>
size_t dt_pack(void *state, size_t offset, void *dest, size_t max_length)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;
    T *p = reinterpret_cast<T*> (dest);
    uint32_t count;

    EXPECT_GT(dt_gen_start_count, dt_gen_finish_count);
    EXPECT_EQ(1, dt_state->started);
    EXPECT_EQ(uint32_t(MAGIC), dt_state->magic);

    ucs_assert((offset % sizeof(T)) == 0);

    count = std::min(max_length / sizeof(T),
                     dt_state->count - (offset / sizeof(T)));
    for (unsigned i = 0; i < count; ++i) {
        p[i] = (offset / sizeof(T)) + i;
    }
    return count * sizeof(T);
}

template <typename T>
ucs_status_t dt_unpack(void *state, size_t offset, const void *src,
                       size_t length)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;
    std::vector<T> *ctx;
    uint32_t count;

    EXPECT_GT(dt_gen_start_count, dt_gen_finish_count);
    EXPECT_EQ(1, dt_state->started);
    EXPECT_EQ(uint32_t(MAGIC), dt_state->magic);

    ctx = reinterpret_cast<std::vector<T>*>(dt_state->context);
    count = length / sizeof(T);
    for (unsigned i = 0; i < count; ++i) {
        T expected = ctx ? (*ctx)[offset / sizeof(T) + i] :
                     (offset / sizeof(T)) + i;
        T actual   = ((T*)src)[i];
        if (actual != expected) {
            UCS_TEST_ABORT("Invalid data at index " << i << ". expected: " <<
                           expected << " actual: " << actual << " offset: " <<
                           offset << ".");
        }
    }
    return UCS_OK;
}

static ucs_status_t dt_err_unpack(void *state, size_t offset, const void *src,
                                  size_t length)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;

    EXPECT_GT(dt_gen_start_count, dt_gen_finish_count);
    EXPECT_EQ(1, dt_state->started);
    EXPECT_EQ(uint32_t(MAGIC), dt_state->magic);

    return UCS_ERR_NO_MEMORY;
}

static size_t
dt_pack_copy(void *state, size_t offset, void *dest, size_t max_length)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;
    size_t length;

    ucs_assertv(offset <= dt_state->count, "offset=%zu count=%zu", offset,
                dt_state->count);
    length = std::min(max_length, dt_state->count - offset);
    memcpy(dest, UCS_PTR_BYTE_OFFSET(dt_state->buffer, offset), length);

    return length;
}

static ucs_status_t
dt_unpack_copy(void *state, size_t offset, const void *src, size_t length)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;

    ucs_assertv(offset + length <= dt_state->count,
                "offset=%zu length=%zu count=%zu", offset, length,
                dt_state->count);
    memcpy(UCS_PTR_BYTE_OFFSET(dt_state->buffer, offset), src, length);

    return UCS_OK;
}

static void dt_common_finish(void *state)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;

    --dt_state->started;
    EXPECT_EQ(0, dt_state->started);
    dt_gen_finish_count++;
    delete dt_state;
}

ucp_generic_dt_ops test_dt_uint32_ops = {
    dt_common_start_pack,
    dt_common_start_unpack,
    dt_packed_size<uint32_t>,
    dt_pack<uint32_t>,
    dt_unpack<uint32_t>,
    dt_common_finish
};

ucp_generic_dt_ops test_dt_uint8_ops = {
    dt_common_start_pack,
    dt_common_start_unpack,
    dt_packed_size<uint8_t>,
    dt_pack<uint8_t>,
    dt_unpack<uint8_t>,
    dt_common_finish
};

ucp_generic_dt_ops test_dt_uint32_err_ops = {
    dt_common_start_pack,
    dt_common_start_unpack,
    dt_packed_size<uint32_t>,
    dt_pack<uint32_t>,
    dt_err_unpack,
    dt_common_finish
};

ucp_generic_dt_ops test_dt_copy_ops = {
    dt_common_start_pack,
    dt_common_start_unpack,
    dt_packed_size<uint8_t>,
    dt_pack_copy,
    dt_unpack_copy,
    dt_common_finish
};

std::string datatype_name(ucp_datatype_t dt)
{
    return ucp_datatype_class_names[ucp_datatype_class(dt)];
}

std::vector<std::vector<ucp_datatype_t> >
datatype_pairs(const ucp_generic_dt_ops_t *ops, size_t contig_elem_size)
{
    ucp_datatype_t dt;
    ASSERT_UCS_OK(ucp_dt_create_generic(ops, NULL, &dt));

    ucp_datatype_t dt_arr[] = { ucp_dt_make_contig(contig_elem_size),
                                ucp_dt_make_iov(),
                                dt };
    std::vector<ucp_datatype_t> dts(dt_arr, dt_arr + 3);

    return ucs::make_pairs(dts);
}

} // ucp
