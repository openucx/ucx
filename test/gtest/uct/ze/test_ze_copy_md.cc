/**
 * Copyright (C) Intel Corporation, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <common/mem_buffer.h>
#include <uct/test_md.h>

#include <functional>
#include <pthread.h>
#include <thread>

extern "C" {
#include <uct/api/uct.h>
#include <uct/ze/copy/ze_copy_md.h>
#include <ucs/memory/memtype_cache.h>
}


/* Host pointer import entry points; skips the test when unavailable */
static const uct_ze_copy_host_import_t *ze_copy_host_import_or_skip()
{
    const uct_ze_copy_host_import_t *hi = uct_ze_copy_md_get_host_import();

    if (hi == NULL) {
        UCS_TEST_SKIP_R("Level Zero host pointer import is not available");
    }

    return hi;
}


/*
 * Smoke tests for the ze_copy MD: lifecycle, capability query, allocation,
 * memory-type detection and host memory import. The ze_copy MD is the path
 * used by UCP for VRAM<->host staging on Intel XPU.
 */
class test_ze_copy_md : public test_md {
protected:
    /* Allocate ZE_DEVICE memory through the MD, which advertises it */
    static void alloc_device(uct_md_h md, size_t *length_p, void **addr_p,
                             uct_mem_h *memh_p)
    {
        ASSERT_UCS_OK(md->ops->mem_alloc(md, length_p, addr_p,
                                         UCS_MEMORY_TYPE_ZE_DEVICE,
                                         UCS_SYS_DEVICE_ID_UNKNOWN, 0,
                                         "test_ze_copy", memh_p));
    }

    /* Base of the import that contains ptr, or NULL when it is not imported */
    static void *import_base(const uct_ze_copy_host_import_t *hi, void *ptr)
    {
        void *base      = NULL;
        ze_result_t ret = hi->get_base(hi->driver, ptr, &base);

        if (ret == ZE_RESULT_SUCCESS) {
            return base;
        }

        EXPECT_EQ(ZE_RESULT_ERROR_INVALID_ARGUMENT, ret);
        return NULL;
    }

    static ucs_status_t
    reg_status(uct_md_h md, void *ptr, size_t length, uct_mem_h *memh_p)
    {
        uct_md_mem_reg_params_t params = {};

        *memh_p = NULL;
        return uct_md_mem_reg_v2(md, ptr, length, &params, memh_p);
    }

    static ucs_status_t dereg_status(uct_md_h md, uct_mem_h memh)
    {
        uct_md_mem_dereg_params_t params = {};

        params.field_mask = UCT_MD_MEM_DEREG_FIELD_MEMH;
        params.memh       = memh;
        return uct_md_mem_dereg_v2(md, &params);
    }

    static uct_mem_h mem_reg(uct_md_h md, void *ptr, size_t length)
    {
        uct_mem_h memh;

        EXPECT_UCS_OK(reg_status(md, ptr, length, &memh));
        return memh;
    }

    static void mem_dereg(uct_md_h md, uct_mem_h memh)
    {
        EXPECT_UCS_OK(dereg_status(md, memh));
    }

    /* Run func(0) and func(1) in two threads released at the same time, and
       return after both of them finished */
    static void run_in_two_threads(const std::function<void(int)> &func)
    {
        pthread_barrier_t barrier;
        std::vector<std::thread> threads;

        ASSERT_EQ(0, pthread_barrier_init(&barrier, NULL, 2));
        for (int i = 0; i < 2; ++i) {
            threads.emplace_back([&barrier, &func, i]() {
                pthread_barrier_wait(&barrier);
                func(i);
            });
        }

        for (std::thread &thread : threads) {
            thread.join();
        }

        pthread_barrier_destroy(&barrier);
    }

    static constexpr size_t CHUNK = 64 * UCS_KBYTE;
};


UCS_TEST_P(test_ze_copy_md, open_close_md) {
    auto *ze_md = ucs_derived_of(md(), uct_ze_copy_md_t);
    EXPECT_NE(nullptr, ze_md->ze_device);
    EXPECT_NE(nullptr, ze_md->ze_context);
}


UCS_TEST_P(test_ze_copy_md, md_attr_advertises_alloc_caps) {
    /* ze_copy MD must be able to allocate ZE memory. */
    EXPECT_NE(0u, md_attr().alloc_mem_types);
    EXPECT_GT(md_attr().max_alloc, 0u);
}


/*
 * Allocate ZE_DEVICE memory through the MD, then free it.
 */
UCS_TEST_P(test_ze_copy_md, mem_alloc_free_device) {
    size_t length  = 4096;
    void *addr     = NULL;
    uct_mem_h memh = NULL;

    alloc_device(md(), &length, &addr, &memh);
    EXPECT_TRUE(addr != NULL);
    EXPECT_GE(length, 4096u);

    EXPECT_UCS_OK(md()->ops->mem_free(md(), memh));
}


/*
 * detect_memory_type must classify ZE device pointers as ZE_DEVICE.
 */
UCS_TEST_P(test_ze_copy_md, detect_memory_type_device) {
    size_t length  = 4096;
    void *addr     = NULL;
    uct_mem_h memh = NULL;
    alloc_device(md(), &length, &addr, &memh);

    ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_HOST;
    EXPECT_UCS_OK(md()->ops->detect_memory_type(md(), addr, length, &mem_type));
    EXPECT_EQ(UCS_MEMORY_TYPE_ZE_DEVICE, mem_type);

    EXPECT_UCS_OK(md()->ops->mem_free(md(), memh));
}


/*
 * mem_query reports base address and allocation length back.
 */
UCS_TEST_P(test_ze_copy_md, mem_query_base_and_length) {
    size_t length  = 8192;
    void *addr     = NULL;
    uct_mem_h memh = NULL;
    alloc_device(md(), &length, &addr, &memh);

    uct_md_mem_attr_v2_t mem_attr = {};
    mem_attr.field_mask           = UCT_MD_MEM_ATTR_V2_FIELD_MEM_TYPE |
                                    UCT_MD_MEM_ATTR_V2_FIELD_BASE_ADDRESS |
                                    UCT_MD_MEM_ATTR_V2_FIELD_ALLOC_LENGTH;
    EXPECT_UCS_OK(md()->ops->mem_query(md(), addr, length, &mem_attr));
    EXPECT_EQ(UCS_MEMORY_TYPE_ZE_DEVICE, mem_attr.mem_type);
    EXPECT_TRUE(mem_attr.base_address != NULL);
    EXPECT_GE(mem_attr.alloc_length, length);

    EXPECT_UCS_OK(md()->ops->mem_free(md(), memh));
}


/*
 * Registering host memory imports it into the driver; deregistering releases
 * the import.
 */
UCS_TEST_P(test_ze_copy_md, host_import_reg_dereg, "ZE_COPY_HOST_IMPORT=try") {
    const uct_ze_copy_host_import_t *hi = ze_copy_host_import_or_skip();
    std::vector<char> buf(2 * CHUNK);
    char *ptr = buf.data();

    uct_mem_h memh = mem_reg(md(), ptr, buf.size());
    EXPECT_EQ(ptr, import_base(hi, ptr));
    EXPECT_EQ(ptr, import_base(hi, ptr + buf.size() - 1));

    mem_dereg(md(), memh);
    EXPECT_TRUE(import_base(hi, ptr) == NULL);
}


/*
 * Registrations of the same range, and of a range inside it, share one
 * import that stays until the last of them is deregistered, in any order.
 */
UCS_TEST_P(test_ze_copy_md, host_import_shared, "ZE_COPY_HOST_IMPORT=try") {
    const uct_ze_copy_host_import_t *hi = ze_copy_host_import_or_skip();
    std::vector<char> buf(2 * CHUNK);
    char *ptr = buf.data();

    for (int reverse = 0; reverse < 2; ++reverse) {
        uct_mem_h whole = mem_reg(md(), ptr, buf.size());
        uct_mem_h again = mem_reg(md(), ptr, buf.size());
        uct_mem_h inner = mem_reg(md(), ptr + (CHUNK / 2), CHUNK);
        EXPECT_EQ(whole, again);
        EXPECT_EQ(whole, inner);

        mem_dereg(md(), reverse ? inner : whole);
        EXPECT_EQ(ptr, import_base(hi, ptr));
        mem_dereg(md(), again);
        EXPECT_EQ(ptr, import_base(hi, ptr));
        mem_dereg(md(), reverse ? whole : inner);
        EXPECT_TRUE(import_base(hi, ptr) == NULL);
    }
}


/*
 * Imports are process-wide: two MDs registering the same range share one
 * import.
 */
UCS_TEST_P(test_ze_copy_md, host_import_shared_by_mds,
           "ZE_COPY_HOST_IMPORT=try") {
    const uct_ze_copy_host_import_t *hi = ze_copy_host_import_or_skip();
    std::vector<char> buf(2 * CHUNK);
    char *ptr = buf.data();
    ucs::handle<uct_md_h> md2;

    ASSERT_NO_FATAL_FAILURE(open_md(md2));

    uct_mem_h memh1 = mem_reg(md(), ptr, buf.size());
    uct_mem_h memh2 = mem_reg(md2, ptr, buf.size());
    EXPECT_EQ(memh1, memh2);

    mem_dereg(md(), memh1);
    EXPECT_EQ(ptr, import_base(hi, ptr));
    mem_dereg(md2, memh2);
    EXPECT_TRUE(import_base(hi, ptr) == NULL);
}


/*
 * Two threads registering the same range at the same time, each through its
 * own MD, share one import. It stays until the last registration is released,
 * including when both are released at the same time.
 */
UCS_TEST_P(test_ze_copy_md, host_import_shared_mt, "ZE_COPY_HOST_IMPORT=try") {
    const uct_ze_copy_host_import_t *hi = ze_copy_host_import_or_skip();
    std::vector<char> buf(2 * CHUNK);
    char *ptr = buf.data();
    ucs::handle<uct_md_h> md2;
    ucs_status_t status[2];
    uct_mem_h memh[2];

    ASSERT_NO_FATAL_FAILURE(open_md(md2));
    const uct_md_h mds[] = {md(), md2};

    auto reg   = [&](int i) {
        status[i] = reg_status(mds[i], ptr, buf.size(), &memh[i]);
    };
    auto dereg = [&](int i) { status[i] = dereg_status(mds[i], memh[i]); };

    for (int dereg_concurrently = 0; dereg_concurrently < 2;
         ++dereg_concurrently) {
        ASSERT_NO_FATAL_FAILURE(run_in_two_threads(reg));
        ASSERT_UCS_OK(status[0]);
        ASSERT_UCS_OK(status[1]);
        EXPECT_EQ(memh[0], memh[1]);
        EXPECT_EQ(ptr, import_base(hi, ptr));

        if (dereg_concurrently) {
            ASSERT_NO_FATAL_FAILURE(run_in_two_threads(dereg));
            EXPECT_UCS_OK(status[0]);
            EXPECT_UCS_OK(status[1]);
        } else {
            mem_dereg(mds[0], memh[0]);
            EXPECT_EQ(ptr, import_base(hi, ptr));
            mem_dereg(mds[1], memh[1]);
        }

        EXPECT_TRUE(import_base(hi, ptr) == NULL);
    }
}


/*
 * A range that shares only one end with an existing import is left
 * unimported, and deregistering it must not release that import.
 */
UCS_TEST_P(test_ze_copy_md, host_import_partial_overlap,
           "ZE_COPY_HOST_IMPORT=try") {
    const uct_ze_copy_host_import_t *hi = ze_copy_host_import_or_skip();
    std::vector<char> buf(4 * CHUNK);
    char *ptr    = buf.data();
    char *middle = ptr + CHUNK;

    uct_mem_h middle_memh = mem_reg(md(), middle, 2 * CHUNK);
    EXPECT_EQ(middle, import_base(hi, middle));

    /* Ends inside the import */
    uct_mem_h head = mem_reg(md(), ptr, 2 * CHUNK);
    /* Starts inside the import */
    uct_mem_h tail = mem_reg(md(), ptr + (2 * CHUNK), 2 * CHUNK);
    EXPECT_NE(middle_memh, head);
    EXPECT_NE(middle_memh, tail);
    EXPECT_TRUE(import_base(hi, ptr) == NULL);
    EXPECT_TRUE(import_base(hi, ptr + buf.size() - 1) == NULL);
    EXPECT_EQ(middle, import_base(hi, middle));

    mem_dereg(md(), head);
    mem_dereg(md(), tail);
    EXPECT_EQ(middle, import_base(hi, middle));

    mem_dereg(md(), middle_memh);
    EXPECT_TRUE(import_base(hi, middle) == NULL);
}


/*
 * An import of ours inside a later, larger import of ours: both are owned
 * and released independently, in either order.
 */
UCS_TEST_P(test_ze_copy_md, host_import_nested, "ZE_COPY_HOST_IMPORT=try") {
    const uct_ze_copy_host_import_t *hi = ze_copy_host_import_or_skip();
    std::vector<char> buf(3 * CHUNK);
    char *ptr       = buf.data();
    char *inner_ptr = ptr + CHUNK;

    for (int reverse = 0; reverse < 2; ++reverse) {
        uct_mem_h inner = mem_reg(md(), inner_ptr, CHUNK);
        uct_mem_h outer = mem_reg(md(), ptr, buf.size());
        EXPECT_NE(inner, outer);
        EXPECT_EQ(inner_ptr, import_base(hi, inner_ptr));
        EXPECT_EQ(ptr, import_base(hi, ptr));

        /* A range with one end in each import is left unimported, as the
           driver would reject importing it, and does not borrow either */
        uct_mem_h straddling = mem_reg(md(), ptr + (CHUNK / 2), CHUNK);
        EXPECT_NE(straddling, inner);
        EXPECT_NE(straddling, outer);
        mem_dereg(md(), straddling);
        EXPECT_EQ(inner_ptr, import_base(hi, inner_ptr));
        EXPECT_EQ(ptr, import_base(hi, ptr));

        mem_dereg(md(), reverse ? outer : inner);
        if (reverse) {
            EXPECT_TRUE(import_base(hi, ptr) == NULL);
            EXPECT_EQ(inner_ptr, import_base(hi, inner_ptr));
        } else {
            /* The inner range is now served by the outer import */
            EXPECT_EQ(ptr, import_base(hi, ptr));
            EXPECT_EQ(ptr, import_base(hi, inner_ptr));
        }

        mem_dereg(md(), reverse ? inner : outer);
        EXPECT_TRUE(import_base(hi, ptr) == NULL);
        EXPECT_TRUE(import_base(hi, inner_ptr) == NULL);
    }
}


/*
 * Imports made by the application are used but never released: a range
 * inside one borrows it, a range around one gets an import of its own.
 */
UCS_TEST_P(test_ze_copy_md, host_import_foreign, "ZE_COPY_HOST_IMPORT=try") {
    const uct_ze_copy_host_import_t *hi = ze_copy_host_import_or_skip();
    std::vector<char> buf(3 * CHUNK);
    char *ptr     = buf.data();
    char *foreign = ptr + CHUNK;

    ASSERT_EQ(ZE_RESULT_SUCCESS, hi->import(hi->driver, foreign, CHUNK));

    uct_mem_h inner = mem_reg(md(), foreign + (CHUNK / 4), CHUNK / 2);
    EXPECT_EQ(foreign, import_base(hi, foreign + (CHUNK / 4)));
    mem_dereg(md(), inner);
    EXPECT_EQ(foreign, import_base(hi, foreign));

    uct_mem_h outer = mem_reg(md(), ptr, buf.size());
    EXPECT_EQ(ptr, import_base(hi, ptr));
    EXPECT_EQ(foreign, import_base(hi, foreign));
    mem_dereg(md(), outer);
    EXPECT_TRUE(import_base(hi, ptr) == NULL);
    EXPECT_EQ(foreign, import_base(hi, foreign));

    EXPECT_EQ(ZE_RESULT_SUCCESS, hi->release(hi->driver, foreign));
    EXPECT_TRUE(import_base(hi, foreign) == NULL);
}


/*
 * Another user of the extension can drop one of our imports by releasing a
 * pointer inside it. Registering the range again leaves it unimported, and
 * deregistering both registrations must not release anything twice.
 */
UCS_TEST_P(test_ze_copy_md, host_import_released_by_foreign,
           "ZE_COPY_HOST_IMPORT=try") {
    const uct_ze_copy_host_import_t *hi = ze_copy_host_import_or_skip();
    std::vector<char> buf(2 * CHUNK);
    char *ptr   = buf.data();
    char *inner = ptr + CHUNK;

    uct_mem_h first = mem_reg(md(), ptr, buf.size());
    EXPECT_EQ(ptr, import_base(hi, ptr));

    /* An import inside ours is not recorded, but its release drops ours */
    EXPECT_EQ(ZE_RESULT_SUCCESS, hi->import(hi->driver, inner, CHUNK));
    EXPECT_EQ(ZE_RESULT_SUCCESS, hi->release(hi->driver, inner));
    EXPECT_TRUE(import_base(hi, ptr) == NULL);

    uct_mem_h second = mem_reg(md(), ptr, buf.size());
    EXPECT_NE(first, second);
    EXPECT_TRUE(import_base(hi, ptr) == NULL);
    mem_dereg(md(), first);
    mem_dereg(md(), second);
    EXPECT_TRUE(import_base(hi, ptr) == NULL);
}


/*
 * Level Zero allocations are not imported.
 */
UCS_TEST_P(test_ze_copy_md, host_import_skips_ze_memory,
           "ZE_COPY_HOST_IMPORT=try") {
    const uct_ze_copy_host_import_t *hi = ze_copy_host_import_or_skip();

    if (!mem_buffer::is_mem_type_supported(UCS_MEMORY_TYPE_ZE_HOST)) {
        UCS_TEST_SKIP_R("ZE host memory is not available");
    }

    mem_buffer buffer(CHUNK, UCS_MEMORY_TYPE_ZE_HOST);
    uct_mem_h memh = mem_reg(md(), buffer.ptr(), CHUNK);
    EXPECT_TRUE(import_base(hi, buffer.ptr()) == NULL);
    mem_dereg(md(), memh);
}


/*
 * HOST_IMPORT=n keeps registrations working without importing.
 */
UCS_TEST_P(test_ze_copy_md, host_import_disabled, "ZE_COPY_HOST_IMPORT=n") {
    const uct_ze_copy_host_import_t *hi = ze_copy_host_import_or_skip();
    std::vector<char> buf(2 * CHUNK);
    char *ptr = buf.data();

    uct_mem_h memh = mem_reg(md(), ptr, buf.size());
    EXPECT_TRUE(import_base(hi, ptr) == NULL);
    mem_dereg(md(), memh);
}


/*
 * HOST_IMPORT=y opens the MD only when the driver can import host memory.
 */
UCS_TEST_P(test_ze_copy_md, host_import_required, "ZE_COPY_HOST_IMPORT=try") {
    uct_md_h md_y;
    ucs_status_t status;

    ASSERT_UCS_OK(uct_config_modify(m_md_config, "ZE_COPY_HOST_IMPORT", "y"));
    {
        scoped_log_handler slh(hide_errors_logger);
        status = uct_md_open(GetParam().component, GetParam().md_name.c_str(),
                             m_md_config, &md_y);
    }

    if (uct_ze_copy_md_get_host_import() == NULL) {
        EXPECT_EQ(UCS_ERR_UNSUPPORTED, status);
        return;
    }

    ASSERT_UCS_OK(status);
    EXPECT_TRUE(ucs_derived_of(md_y, uct_ze_copy_md_t)->host_import);
    uct_md_close(md_y);
}

_UCT_MD_INSTANTIATE_TEST_CASE(test_ze_copy_md, ze_cpy)


class test_ze_copy_memtype_cache : public test_ze_copy_md {
protected:
    void init() override
    {
        /* Drop the cache, so that the MD open has to create it again */
        ucs_memtype_cache_cleanup();
        ucs_memtype_cache_global_init();
        ASSERT_EQ(nullptr, ucs_memtype_cache_global_instance);

        test_ze_copy_md::init();
    }
};


/*
 * An application allocation made after the MD is opened must be tracked in
 * the memtype cache.
 */
UCS_TEST_P(test_ze_copy_memtype_cache, memtype_cache_created_at_md_open,
           "MEMTYPE_CACHE=y") {
    ASSERT_NE(nullptr, ucs_memtype_cache_global_instance);

    const size_t length = 4096;
    ucs_memory_info_t mem_info;
    void *addr;

    {
        mem_buffer buffer(length, UCS_MEMORY_TYPE_ZE_DEVICE);

        addr = buffer.ptr();
        ASSERT_UCS_OK(ucs_memtype_cache_lookup(addr, length, &mem_info));
        EXPECT_EQ(UCS_MEMORY_TYPE_ZE_DEVICE, mem_info.type);
    }

    EXPECT_EQ(UCS_ERR_NO_ELEM,
              ucs_memtype_cache_lookup(addr, length, &mem_info));
}

_UCT_MD_INSTANTIATE_TEST_CASE(test_ze_copy_memtype_cache, ze_cpy)
