/**
* Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef TEST_UCP_DATATYPE_H_
#define TEST_UCP_DATATYPE_H_

#include <common/gtest.h>

#include <ucp/api/ucp.h>
extern "C" {
#include <ucp/dt/dt_contig.h>
#include <ucp/dt/dt_generic.h>
#include <ucp/dt/dt_iov.h>
}

#include <string.h>

namespace ucp {

/* Can't be destroyed before related UCP request is completed */
class data_type_desc_t {
public: 
    enum {
        MAX_IOV = 40
    };

    data_type_desc_t()
        : m_origin(uintptr_t(NULL)), m_length(0), m_dt(0), m_buf(NULL),
          m_count(0), m_iov_cnt_limit(sizeof(m_iov) / sizeof(m_iov[0])) {
        memset(m_iov, 0, sizeof(m_iov));
    };

    data_type_desc_t(ucp_datatype_t datatype, const void *buf, size_t length)
        : m_origin(uintptr_t(buf)), m_length(length), m_dt(0), m_buf(NULL),
          m_iov_cnt_limit(sizeof(m_iov) / sizeof(m_iov[0])) {
        make(datatype, buf, length);
    }

    data_type_desc_t(ucp_datatype_t datatype, const void *buf, size_t length,
                     size_t iov_count)
        : m_origin(uintptr_t(buf)), m_length(length), m_dt(0), m_buf(NULL),
          m_iov_cnt_limit(sizeof(m_iov) / sizeof(m_iov[0])) {
        make(datatype, buf, length, iov_count);
    };

    data_type_desc_t &make(ucp_datatype_t datatype, const void *buf,
                           size_t length) {
        return make(datatype, buf, length, m_iov_cnt_limit);
    };

    data_type_desc_t &forward_to(size_t offset) {
        EXPECT_LE(offset, m_length);
        invalidate();
        return make(m_dt, (const void *)(m_origin + offset), m_length - offset,
                    m_iov_cnt_limit);
    };

    ucp_datatype_t dt() const {
        EXPECT_TRUE(is_valid());
        return m_dt;
    };

    void *buf() const {
        EXPECT_TRUE(is_valid());
        return const_cast<void *>(m_buf);
    };

    ssize_t buf_length() const {
        EXPECT_TRUE(is_valid());
        if (UCP_DT_IS_CONTIG(m_dt) || UCP_DT_IS_GENERIC(m_dt)) {
            return m_length - (uintptr_t(m_buf) - m_origin);
        } else if (UCP_DT_IS_IOV(m_dt)) {
            size_t length = 0;
            for (size_t i = 0; i < count(); ++i) {
                length += m_iov[i].length;
            }
            return length;
        }
        ADD_FAILURE() << "Not supported datatype";
        return -1;
    }

    size_t count() const {
        EXPECT_TRUE(is_valid());
        return m_count;
    };

    bool is_valid() const {
        return (m_buf != NULL) && (m_count != 0) &&
               (UCP_DT_IS_IOV(m_dt) ? (m_count <= m_iov_cnt_limit) :
               (UCP_DT_IS_CONTIG(m_dt) || UCP_DT_IS_GENERIC(m_dt)));
    }

private:
    data_type_desc_t &make(ucp_datatype_t datatype, const void *buf,
                           size_t length, size_t iov_count);

    void invalidate() {
        EXPECT_TRUE(is_valid());
        m_buf   = NULL;
        m_count = 0;
    }

    uintptr_t       m_origin;
    size_t          m_length;

    ucp_datatype_t  m_dt;
    const void     *m_buf;
    size_t          m_count;

    const size_t    m_iov_cnt_limit;
    ucp_dt_iov_t    m_iov[MAX_IOV];
};

struct dt_gen_state {
    size_t              count;
    int                 started;
    uint32_t            magic;
    void                *context;
};

extern int dt_gen_start_count;
extern int dt_gen_finish_count;
extern ucp_generic_dt_ops test_dt_uint32_ops;
extern ucp_generic_dt_ops test_dt_uint32_err_ops;
extern ucp_generic_dt_ops test_dt_uint8_ops;

} // ucp

#endif /* TEST_UCP_DATATYPE_H_ */
