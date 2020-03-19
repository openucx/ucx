/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>

extern "C" {
#include <ucs/sys/iovec.h>
#include <ucs/sys/iovec.inl>
}


class test_ucs_iov : public ucs::test {
protected:
    struct iov1_t {
        char      length_padding[128];
        size_t    length;
        char      buffer_padding[64];
        void      *buffer;
    };

    struct iov2_t {
        char      length_padding[64];
        size_t    length;
        char      buffer_padding[256];
        void      *buffer;
    };

    template <typename T>
    void iov_set_length(T *iov, size_t length) {
        iov->length = length;
    }

    template <typename T>
    void iov_set_buffer(T *iov, void *buffer) {
        iov->buffer = buffer;
    }

    template <typename T>
    size_t iov_get_length(T *iov) {
        return iov->length;
    }

    template <typename T>
    void *iov_get_buffer(T *iov) {
        return iov->buffer;
    }

    template <typename T1, typename T2>
    size_t iov_converter(T1 *src_iov, size_t *src_iov_cnt_p,
                         T2 *dst_iov, size_t dst_iov_cnt,
                         size_t max_length, ucs_iov_iter_t *iov_iter_p) {
        return ucs_iov_converter(src_iov, src_iov_cnt_p,
                                 iov_set_buffer, iov_set_length,
                                 dst_iov, dst_iov_cnt,
                                 iov_get_buffer, iov_get_length,
                                 max_length, iov_iter_p);
    }

    void expect_zero_changes(size_t res_cnt, size_t res_length,
                             const ucs_iov_iter_t *iov_iter) {
        EXPECT_EQ(0lu, res_cnt);
        EXPECT_EQ(0lu, res_length);
        EXPECT_EQ(0lu, iov_iter->iov_index);
        EXPECT_EQ(0lu, iov_iter->buffer_offset);
    }

    template<typename T1, typename T2>
    void test_iov_type_pair(T1 *iov1, size_t iov1_cnt,
                            T2 *iov2, size_t iov2_cnt,
                            size_t max_length) {
        size_t res_total_length = 0;
        size_t exp_total_length = 0;
        size_t cnt, length;
        ucs_iov_iter_t iov_iter;

        iov1 = new T1[iov1_cnt];
        ASSERT_TRUE(iov1 != NULL);
        iov2  = new T2[iov2_cnt];
        ASSERT_TRUE(iov2 != NULL);

        for (size_t i = 0; i < iov2_cnt; i++) {
            iov_set_buffer(&iov2[i], (void*)0x1);
            iov_set_length(&iov2[i], i);
            exp_total_length += iov_get_length(&iov2[i]);
        }

        ucs_iov_iter_init(&iov_iter);

        while (iov_iter.iov_index < iov2_cnt) {
            cnt    = iov1_cnt;
            length = iov_converter(iov1, &cnt,
                                   iov2, iov2_cnt,
                                   max_length, &iov_iter);
            EXPECT_TRUE((iov_iter.iov_index == iov2_cnt) ||
                        (length == max_length) || (cnt == iov1_cnt));
            res_total_length += length;
        }

        EXPECT_EQ(exp_total_length, res_total_length);

        ucs_iov_iter_init(&iov_iter);
        cnt    = 0;
        length = iov_converter((T1*)NULL, &cnt,
                               iov2, iov2_cnt,
                               max_length, &iov_iter);
        expect_zero_changes(cnt, length, &iov_iter);

        ucs_iov_iter_init(&iov_iter);
        cnt    = iov1_cnt;
        length = iov_converter(iov1, &cnt,
                               (T2*)NULL, 0,
                               max_length, &iov_iter);
        expect_zero_changes(cnt, length, &iov_iter);

        ucs_iov_iter_init(&iov_iter);
        cnt    = iov1_cnt;
        length = iov_converter(iov1, &cnt,
                               iov2, iov2_cnt,
                               0, &iov_iter);
        expect_zero_changes(cnt, length, &iov_iter);

        delete[] iov1;
        delete[] iov2;
    }
};

UCS_TEST_F(test_ucs_iov, total_length) {
    const size_t iov_cnt = 1024;
    size_t total_length  = 0;
    struct iovec *iov;

    iov = new struct iovec[iov_cnt];
    ASSERT_TRUE(iov != NULL);

    for (size_t i = 0; i < iov_cnt; i++) {
        iov[i].iov_len = i;
        total_length  += iov[i].iov_len;
    }

    EXPECT_EQ(total_length, ucs_iovec_total_length(iov, iov_cnt));

    delete[] iov;
}

UCS_TEST_F(test_ucs_iov, iov_to_iov) {
    const size_t iov1_cnt   = 16;
    const size_t iov2_cnt   = 1024;
    const size_t max_length = 1024;
    void *iov_buf1          = NULL;
    void *iov_buf2          = NULL;

    test_iov_type_pair(static_cast<iov1_t*>(iov_buf1), iov1_cnt,
                       static_cast<iov2_t*>(iov_buf2), iov2_cnt,
                       max_length);
    test_iov_type_pair(static_cast<iov2_t*>(iov_buf1), iov1_cnt,
                       static_cast<iov1_t*>(iov_buf2), iov2_cnt,
                       max_length);
    test_iov_type_pair(static_cast<iov1_t*>(iov_buf1), iov1_cnt,
                       static_cast<iov1_t*>(iov_buf2), iov2_cnt,
                       max_length);
    test_iov_type_pair(static_cast<iov2_t*>(iov_buf1), iov1_cnt,
                       static_cast<iov2_t*>(iov_buf2), iov2_cnt,
                       max_length);
}
