/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ucp_test.h"
extern "C" {
#include <ucp/proto/ucp_int.h>
}

class test_ucp_rma : public ucp_test {
public:

    /*
     * @param [in]  max_size       Maximal size of data to send.
     * @param [in]  memheap_addr   VA to perform the RMA operation to,
     * @param [in]  rkey           Memheap remote key.
     * @param [out] expected_data  What should the memheap contain at the given
     *                             address after the operation.
     */
    typedef void (test_ucp_rma::* blocking_send_func_t)(entity *e,
                                                        size_t max_size,
                                                        void *memheap_addr,
                                                        ucp_rkey_h rkey,
                                                        std::string& expected_data);

    void blocking_put(entity *e, size_t max_size, void *memheap_addr,
                      ucp_rkey_h rkey, std::string& expected_data)
    {
        ucs_status_t status;
        std::string send_data(max_size, 0);
        ucs::fill_random(send_data.begin(), send_data.end());
        status = ucp_rma_put(e->ep(), &send_data[0], send_data.length(),
                             (uintptr_t)memheap_addr, rkey);
        expected_data = send_data;
        std::fill(send_data.begin(), send_data.end(), 0);
        ASSERT_UCS_OK(status);
    }

    void blocking_get(entity *e, size_t max_size, void *memheap_addr,
                      ucp_rkey_h rkey, std::string& expected_data)
    {
        ucs_status_t status;
        std::string reply_buffer;

        ucs::fill_random((char*)memheap_addr, (char*)memheap_addr + max_size);
        reply_buffer.resize(max_size);
        status = ucp_rma_get(e->ep(), &reply_buffer[0], reply_buffer.length(),
                             (uintptr_t)memheap_addr, rkey);
        expected_data.clear();
        ASSERT_UCS_OK(status);

        EXPECT_EQ(std::string((char*)memheap_addr, reply_buffer.length()),
                  reply_buffer);
    }

    template <typename T>
    void blocking_add(entity *e,  size_t max_size, void *memheap_addr,
                      ucp_rkey_h rkey, std::string& expected_data)
    {
        ucs_status_t status;
        T add, prev;

        prev = *(T*)memheap_addr;
        add  = (T)rand() * (T)rand();

        if (sizeof(T) == sizeof(uint32_t)) {
            status = ucp_rma_add32(e->ep(), add, (uintptr_t)memheap_addr, rkey);
        } else if (sizeof(T) == sizeof(uint64_t)) {
            status = ucp_rma_add64(e->ep(), add, (uintptr_t)memheap_addr, rkey);
        } else {
            status = UCS_ERR_UNSUPPORTED;
        }
        ASSERT_UCS_OK(status);

        expected_data.resize(sizeof(T));
        *(T*)&expected_data[0] = add + prev;
    }

    void unaligned_blocking_add64(entity *e,  size_t max_size, void *memheap_addr,
                             ucp_rkey_h rkey, std::string& expected_data)
    {
        /* Test that unaligned addresses generate error */
        ucs_status_t status;
        status = ucp_rma_add64(e->ep(), 0, (uintptr_t)memheap_addr + 1, rkey);
        EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
        expected_data.clear();
    }

    template <typename T>
    void blocking_fadd(entity *e,  size_t max_size, void *memheap_addr,
                      ucp_rkey_h rkey, std::string& expected_data)
    {
        ucs_status_t status;
        T add, prev, result;

        prev = *(T*)memheap_addr;
        add  = (T)rand() * (T)rand();

        if (sizeof(T) == sizeof(uint32_t)) {
            status = ucp_rma_fadd32(e->ep(), add, (uintptr_t)memheap_addr, rkey,
                                    (uint32_t*)(void*)&result);
        } else if (sizeof(T) == sizeof(uint64_t)) {
            status = ucp_rma_fadd64(e->ep(), add, (uintptr_t)memheap_addr, rkey,
                                    (uint64_t*)(void*)&result);
        } else {
            status = UCS_ERR_UNSUPPORTED;
        }
        ASSERT_UCS_OK(status);

        EXPECT_EQ(prev, result);

        expected_data.resize(sizeof(T));
        *(T*)&expected_data[0] = add + prev;
    }

    template <typename T>
    void blocking_swap(entity *e,  size_t max_size, void *memheap_addr,
                       ucp_rkey_h rkey, std::string& expected_data)
    {
        ucs_status_t status;
        T swap, prev, result;

        prev = *(T*)memheap_addr;
        swap = (T)rand() * (T)rand();

        if (sizeof(T) == sizeof(uint32_t)) {
            status = ucp_rma_swap32(e->ep(), swap, (uintptr_t)memheap_addr, rkey,
                                    (uint32_t*)(void*)&result);
        } else if (sizeof(T) == sizeof(uint64_t)) {
            status = ucp_rma_swap64(e->ep(), swap, (uintptr_t)memheap_addr, rkey,
                                    (uint64_t*)(void*)&result);
        } else {
            status = UCS_ERR_UNSUPPORTED;
        }
        ASSERT_UCS_OK(status);

        EXPECT_EQ(prev, result);

        expected_data.resize(sizeof(T));
        *(T*)&expected_data[0] = swap;
    }

    template <typename T>
    void blocking_cswap(entity *e,  size_t max_size, void *memheap_addr,
                        ucp_rkey_h rkey, std::string& expected_data)
    {
        ucs_status_t status;
        T compare, swap, prev, result;

        prev = *(T*)memheap_addr;
        if ((rand() % 2) == 0) {
            compare = prev; /* success mode */
        } else {
            compare = ~prev; /* fail mode */
        }
        swap = (T)rand() * (T)rand();

        if (sizeof(T) == sizeof(uint32_t)) {
            status = ucp_rma_cswap32(e->ep(), compare, swap,
                                     (uintptr_t)memheap_addr, rkey,
                                     (uint32_t*)(void*)&result);
        } else if (sizeof(T) == sizeof(uint64_t)) {
            status = ucp_rma_cswap64(e->ep(), compare, swap,
                                     (uintptr_t)memheap_addr, rkey,
                                     (uint64_t*)(void*)&result);
        } else {
            status = UCS_ERR_UNSUPPORTED;
        }
        ASSERT_UCS_OK(status);

        EXPECT_EQ(prev, result);

        expected_data.resize(sizeof(T));
        if (compare == prev) {
            *(T*)&expected_data[0] = swap;
        } else {
            *(T*)&expected_data[0] = prev;
        }
    }

protected:
    void test_mapped_memory(entity *e, ucp_mem_h memh,
                            void *ptr, size_t size)
    {
        EXPECT_EQ(ptr, memh->address);
        EXPECT_GE(memh->length, size);
        EXPECT_NE(0ull, memh->pd_map);

        size_t rkey_size;
        void *rkey_buffer;
        ucs_status_t status;

        status = ucp_rkey_pack(e->ucph(), memh, &rkey_buffer, &rkey_size);
        ASSERT_UCS_OK(status);

        ucp_rkey_h rkey;
        status = ucp_ep_rkey_unpack(e->ep(), rkey_buffer, &rkey);
        ASSERT_UCS_OK(status);

        ucp_rkey_buffer_release(rkey_buffer);
        ucp_rkey_destroy(rkey);
    }

    void test_blocking_xfer(blocking_send_func_t send, size_t alignment) {
        static const size_t memheap_size = 512 * 1024;
        entity *pe0 = create_entity();
        entity *pe1 = create_entity();
        ucs_status_t status;
        size_t size;

        ucp_mem_h memh;
        void *memheap = NULL;
        status = ucp_mem_map(pe1->ucph(), &memheap, memheap_size, 0, &memh);
        ASSERT_UCS_OK(status);

        memset(memheap, 0, memheap_size);

        void *rkey_buffer;
        size_t rkey_buffer_size;
        status = ucp_rkey_pack(pe1->ucph(), memh, &rkey_buffer, &rkey_buffer_size);
        ASSERT_UCS_OK(status);

        pe0->connect(pe1);
        pe1->connect(pe0);

        ucp_rkey_h rkey;
        status = ucp_ep_rkey_unpack(pe0->ep(), rkey_buffer, &rkey);
        ASSERT_UCS_OK(status);

        for (int i = 0; i < 300 / ucs::test_time_multiplier(); ++i) {

            size = ucs_max(rand() % memheap_size, alignment);

            size_t offset = rand() % (memheap_size - size - alignment);
            offset = ucs_align_up(offset, alignment);

            ucs_assert(((((uintptr_t)memheap + offset)) % alignment) == 0);

            std::string expected_data;
            (this->*send)(pe0, size, (void*)((uintptr_t)memheap + offset),
                          rkey, expected_data);

            status = ucp_rma_flush(pe0->worker());
            ASSERT_UCS_OK(status);

            EXPECT_EQ(expected_data,
                      std::string((char*)memheap + offset, expected_data.length()));
        }

        ucp_rkey_destroy(rkey);

        status = ucp_rma_flush(pe1->worker());
        ASSERT_UCS_OK(status);

        pe0->disconnect();
        pe1->disconnect();

        ucp_rkey_buffer_release(rkey_buffer);
        status = ucp_mem_unmap(pe1->ucph(), memh);
        ASSERT_UCS_OK(status);
    }
};

UCS_TEST_F(test_ucp_rma, mem_alloc) {

    ucs_status_t status;
    entity *e = create_entity();

    e->connect(e);

    for (int i = 0; i < 1000 / ucs::test_time_multiplier(); ++i) {
        size_t size = rand() % (1024 * 1024);

        ucp_mem_h memh;
        void *ptr = NULL;
        status = ucp_mem_map(e->ucph(), &ptr, size, 0, &memh);
        if (size == 0) {
            EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
            continue;
        }

        ASSERT_UCS_OK(status);

        test_mapped_memory(e, memh, ptr, size);

        status = ucp_mem_unmap(e->ucph(), memh);
        ASSERT_UCS_OK(status);
    }
}

UCS_TEST_F(test_ucp_rma, mem_reg) {

    ucs_status_t status;
    entity *e = create_entity();

    e->connect(e);

    for (int i = 0; i < 1000 / ucs::test_time_multiplier(); ++i) {
        size_t size = rand() % (1024 * 1024);

        void *ptr = malloc(size);

        ucp_mem_h memh;
        status = ucp_mem_map(e->ucph(), &ptr, size, 0, &memh);
        if (size == 0) {
            EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
            continue;
        }

        ASSERT_UCS_OK(status);

        test_mapped_memory(e, memh, ptr, size);

        status = ucp_mem_unmap(e->ucph(), memh);
        ASSERT_UCS_OK(status);

        free(ptr);
    }
}

UCS_TEST_F(test_ucp_rma, blocking_put) {
    test_blocking_xfer(&test_ucp_rma::blocking_put, 1);
}

UCS_TEST_F(test_ucp_rma, blocking_get) {
    test_blocking_xfer(&test_ucp_rma::blocking_get, 1);
}

UCS_TEST_F(test_ucp_rma, atomic_add) {
    test_blocking_xfer(&test_ucp_rma::blocking_add<uint32_t>, sizeof(uint32_t));
    test_blocking_xfer(&test_ucp_rma::blocking_add<uint64_t>, sizeof(uint64_t));
}

UCS_TEST_F(test_ucp_rma, atomic_fadd) {
    test_blocking_xfer(&test_ucp_rma::blocking_fadd<uint32_t>, sizeof(uint32_t));
    test_blocking_xfer(&test_ucp_rma::blocking_fadd<uint64_t>, sizeof(uint64_t));
}

UCS_TEST_F(test_ucp_rma, atomic_swap) {
    test_blocking_xfer(&test_ucp_rma::blocking_swap<uint32_t>, sizeof(uint32_t));
    test_blocking_xfer(&test_ucp_rma::blocking_swap<uint64_t>, sizeof(uint64_t));
}

UCS_TEST_F(test_ucp_rma, atomic_cswap) {
    test_blocking_xfer(&test_ucp_rma::blocking_cswap<uint32_t>, sizeof(uint32_t));
    test_blocking_xfer(&test_ucp_rma::blocking_cswap<uint64_t>, sizeof(uint64_t));
}

UCS_TEST_F(test_ucp_rma, unaligned_atomic_add) {
    test_blocking_xfer(&test_ucp_rma::unaligned_blocking_add64, sizeof(uint64_t));
}

