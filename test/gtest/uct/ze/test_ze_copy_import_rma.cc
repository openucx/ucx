/**
 * Copyright (C) Intel Corporation, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "test_ze_copy_import.h"
#include "uct/uct_p2p_test.h"

#include <common/mem_buffer.h>


/*
 * Copies whose host side crosses the boundary of imported host memory. The
 * driver serves such a copy without the import, so the data must still land
 * correctly on both sides of the boundary.
 */
class test_ze_copy_import_rma : public uct_p2p_test {
public:
    test_ze_copy_import_rma() : uct_p2p_test(0)
    {
    }

protected:
    static constexpr size_t   CHUNK      = 64 * UCS_KBYTE;
    static constexpr uint64_t SEED_GUARD = 0x4444444444444444lu;
    static constexpr uint64_t SEED_H2D   = 0x1111111111111111lu;
    static constexpr uint64_t SEED_D2H   = 0x2222222222222222lu;

    /* [offset, offset + length) in CHUNK units */
    using range = std::pair<size_t, size_t>;

    void init() override
    {
        uct_p2p_test::init();
        ze_copy_host_import_or_skip();

        if (!mem_buffer::is_mem_type_supported(UCS_MEMORY_TYPE_ZE_DEVICE)) {
            UCS_TEST_SKIP_R("ZE device memory is not available");
        }

        check_caps_skip(UCT_IFACE_FLAG_PUT_ZCOPY | UCT_IFACE_FLAG_GET_ZCOPY);
    }

    /* Register a host range with the receiver's MD, which imports it */
    uct_allocated_memory_t reg_host(void *ptr, size_t length)
    {
        const uct_ze_copy_host_import_t *hi = uct_ze_copy_md_get_host_import();
        uct_allocated_memory_t mem          = {};
        void *base                          = NULL;

        mem.address    = ptr;
        mem.length     = length;
        mem.mem_type   = UCS_MEMORY_TYPE_HOST;
        mem.method     = UCT_ALLOC_METHOD_LAST;
        mem.memh       = UCT_MEM_HANDLE_NULL;
        mem.sys_device = UCS_SYS_DEVICE_ID_UNKNOWN;
        receiver().mem_type_reg(&mem, UCT_MD_MEM_ACCESS_ALL);

        EXPECT_EQ(ZE_RESULT_SUCCESS, hi->get_base(hi->driver, ptr, &base));
        EXPECT_EQ(ptr, base);
        return mem;
    }

    /* Copy the device buffer to or from host_ptr and wait for completion */
    void copy(bool to_host, const mapped_buffer &dev, void *host_ptr,
              uct_rkey_t rkey)
    {
        uint64_t remote_addr = reinterpret_cast<uintptr_t>(host_ptr);
        ucs_status_t status;

        do {
            if (to_host) {
                status = uct_ep_put_zcopy(sender_ep(), dev.iov(), 1,
                                          remote_addr, rkey, NULL);
            } else {
                status = uct_ep_get_zcopy(sender_ep(), dev.iov(), 1,
                                          remote_addr, rkey, NULL);
            }

            if (status == UCS_ERR_NO_RESOURCE) {
                progress();
            }
        } while (status == UCS_ERR_NO_RESOURCE);

        ASSERT_TRUE((status == UCS_OK) || (status == UCS_INPROGRESS))
                << ucs_status_string(status);
        flush();
    }

    /*
     * Import the given ranges of a four-chunk host buffer, then copy its two
     * middle chunks to and from the device and compare every byte, including
     * the untouched chunks on both sides.
     */
    void test_copy(const std::vector<range> &ranges)
    {
        std::vector<char> host(4 * CHUNK);
        char *middle = host.data() + CHUNK;
        char *last   = host.data() + (3 * CHUNK);
        std::vector<uct_allocated_memory_t> regs;
        uct_rkey_bundle_t rkey;

        for (const range &r : ranges) {
            regs.push_back(reg_host(host.data() + (r.first * CHUNK),
                                    r.second * CHUNK));
        }

        receiver().rkey_unpack(&regs.front(), &rkey);
        mapped_buffer dev(2 * CHUNK, sender(), 0, UCS_MEMORY_TYPE_ZE_DEVICE);

        mem_buffer::pattern_fill(host.data(), CHUNK, SEED_GUARD);
        mem_buffer::pattern_fill(last, CHUNK, SEED_GUARD);

        mem_buffer::pattern_fill(middle, 2 * CHUNK, SEED_H2D);
        dev.memset(0);
        copy(false, dev, middle, rkey.rkey);
        dev.pattern_check(SEED_H2D);

        dev.pattern_fill(SEED_D2H);
        copy(true, dev, middle, rkey.rkey);
        mem_buffer::pattern_check(middle, 2 * CHUNK, SEED_D2H);

        mem_buffer::pattern_check(host.data(), CHUNK, SEED_GUARD);
        mem_buffer::pattern_check(last, CHUNK, SEED_GUARD);

        receiver().rkey_release(&rkey);
        for (uct_allocated_memory_t &reg : regs) {
            receiver().mem_type_dereg(&reg);
        }
    }
};


UCS_TEST_P(test_ze_copy_import_rma, inside_import) {
    test_copy({{0, 4}});
}

UCS_TEST_P(test_ze_copy_import_rma, across_two_imports) {
    test_copy({{0, 2}, {2, 2}});
}

UCS_TEST_P(test_ze_copy_import_rma, across_import_end) {
    test_copy({{0, 2}});
}

UCT_INSTANTIATE_ZE_TEST_CASE(test_ze_copy_import_rma)
