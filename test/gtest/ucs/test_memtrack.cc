/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <common/test.h>

extern "C" {
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/sys.h>
}

#include <stdio.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <limits>


class test_memtrack : public ucs::test {
protected:
    static const size_t ALLOC_SIZE = 10000;
    static const char ALLOC_NAME[];

    void init() {
        ucs_memtrack_cleanup();
        push_config();
        modify_config("MEMTRACK_DEST", "/dev/null");
        ucs_memtrack_init();
        init_count_size();
    }

    void cleanup() {
        ucs_memtrack_cleanup();
        pop_config();
        ucs_memtrack_init();
    }

    void test_total(unsigned count, size_t size)
    {
        ucs_memtrack_entry_t total;

        ucs_memtrack_total(&total);
        EXPECT_EQ(m_count, total.count);
        EXPECT_EQ(m_count + count, total.peak_count);
        EXPECT_EQ(m_size + size, total.peak_size);
    }

    void test_total_memalign_realloc(unsigned count, size_t size)
    {
        /* ucs_posix_memalign_realloc() call may hold two buffers
           for a short time if malloc() returned an unaligned pointer */
        ucs_memtrack_entry_t total;

        ucs_memtrack_total(&total);
        EXPECT_EQ(m_count, total.count);

        unsigned peak_count = total.peak_count - m_count;
        size_t peak_size    = total.peak_size - m_size;

        EXPECT_EQ(peak_count % count, 0);
        EXPECT_EQ(peak_size % size, 0);
        EXPECT_GT(peak_count / count, 0);
        EXPECT_GT(peak_size / size, 0);
        EXPECT_LT(peak_count / count, 3);
        EXPECT_LT(peak_size / size, 3);
    }

private:
    void init_count_size()
    {
        ucs_memtrack_entry_t entry;

        ucs_memtrack_total(&entry);
        m_count = entry.count;
        m_size  = entry.size;
    }

    unsigned m_count{0};
    size_t m_size{0};
};

const char test_memtrack::ALLOC_NAME[] = "memtrack_test";


UCS_TEST_F(test_memtrack, sanity) {
    ucs_memtrack_entry_t entry;
    void *a, *b;
    int i;

    ucs_memtrack_total(&entry);
    i = entry.count;

    b = ucs_malloc(1, ALLOC_NAME);
    ucs_free(b);

    b = ucs_malloc(1, ALLOC_NAME);
    a = ucs_malloc(3, ALLOC_NAME);
    ucs_free(b);
    ucs_memtrack_total(&entry);
    if (ucs_memtrack_is_enabled()) {
        EXPECT_EQ((size_t)(i + 1), entry.count);
    }

    b = ucs_malloc(4, ALLOC_NAME);
    ucs_free(b);
    ucs_memtrack_total(&entry);
    if (ucs_memtrack_is_enabled()) {
        EXPECT_EQ((size_t)(i + 1), entry.count);
    }
    ucs_free(a);

    for (i = 0; i < 101; i++) {
        a = ucs_malloc(i, ALLOC_NAME);
        ucs_free(a);
    }
}

UCS_TEST_F(test_memtrack, parse_dump) {
    char *buf;
    size_t size;

    /* Dump */
    {
        FILE* tempf = open_memstream(&buf, &size);
        ucs_memtrack_dump(tempf);
        fclose(tempf);
    }

    /* Parse */
    ASSERT_NE((void*)NULL, strstr(buf, "TOTAL"));
    free(buf);
}

UCS_TEST_F(test_memtrack, malloc_realloc) {
    void* ptr;

    ptr = ucs_malloc(ALLOC_SIZE, ALLOC_NAME);
    ASSERT_NE((void*)NULL, ptr);

    ptr = ucs_realloc(ptr, 2 * ALLOC_SIZE, ALLOC_NAME);
    ASSERT_NE((void*)NULL, ptr);
    ucs_free(ptr);

    test_total(1, 2 * ALLOC_SIZE);
}

UCS_TEST_F(test_memtrack, realloc_null) {
    void* ptr;

    ptr = ucs_realloc(NULL, ALLOC_SIZE, ALLOC_NAME);
    ASSERT_NE((void*)NULL, ptr);
    ucs_free(ptr);

    test_total(1, ALLOC_SIZE);
}

UCS_TEST_F(test_memtrack, calloc) {
    void* ptr;

    ptr = ucs_calloc(1, ALLOC_SIZE, ALLOC_NAME);
    ASSERT_NE((void*)NULL, ptr);
    ucs_free(ptr);

    ptr = ucs_calloc(ALLOC_SIZE, 1, ALLOC_NAME);
    ASSERT_NE((void*)NULL, ptr);
    ucs_free(ptr);

    test_total(1, ALLOC_SIZE);
}

UCS_TEST_F(test_memtrack, sysv) {
    ucs_status_t status;
    void* ptr = NULL;
    int shmid;
    size_t size;

    size = ALLOC_SIZE;

    status = ucs_sysv_alloc(&size, std::numeric_limits<size_t>::max(), &ptr, 0,
                            ALLOC_NAME, &shmid);
    ASSERT_UCS_OK(status);
    ASSERT_NE((void*)NULL, ptr);

    memset(ptr, 0xAA, size);
    ucs_sysv_free(ptr);

    test_total(1, size);
}

UCS_TEST_F(test_memtrack, memalign_realloc) {
    void* ptr;
    int ret;

    ret = ucs_posix_memalign(&ptr, 8, ALLOC_SIZE, ALLOC_NAME);
    ucs_free(ptr);
    ASSERT_EQ(0, ret);
    ASSERT_NE((void*)NULL, ptr);
    /* Silence coverity warning. */
    ptr = NULL;

    ret = ucs_posix_memalign(&ptr, UCS_KBYTE, ALLOC_SIZE, ALLOC_NAME);
    ASSERT_EQ(0, ret);
    ASSERT_NE((void*)NULL, ptr);
    ASSERT_EQ(0, (uintptr_t)ptr % UCS_KBYTE);

    ptr = ucs_realloc(ptr, 2 * ALLOC_SIZE, ALLOC_NAME);
    ASSERT_NE((void*)NULL, ptr);

    ptr = ucs_realloc(ptr, ALLOC_SIZE, ALLOC_NAME);
    ASSERT_NE((void*)NULL, ptr);

    ucs_free(ptr);
    /* Silence coverity warning. */
    ptr = NULL;

    test_total_memalign_realloc(1, 2 * ALLOC_SIZE);

    ret = ucs_posix_memalign(&ptr, UCS_KBYTE, ALLOC_SIZE, ALLOC_NAME);
    ASSERT_EQ(0, ret);
    ASSERT_NE((void*)NULL, ptr);
    ASSERT_EQ(0, (uintptr_t)ptr % UCS_KBYTE);

    ret = ucs_posix_memalign_realloc(&ptr, UCS_KBYTE, 4 * ALLOC_SIZE,
                                     ALLOC_NAME);
    ASSERT_EQ(0, ret);
    ASSERT_NE((void*)NULL, ptr);
    ASSERT_EQ(0, (uintptr_t)ptr % UCS_KBYTE);

    ret = ucs_posix_memalign_realloc(&ptr, UCS_MBYTE, 2 * ALLOC_SIZE,
                                     ALLOC_NAME);
    ASSERT_EQ(0, ret);
    ASSERT_NE((void*)NULL, ptr);
    ASSERT_EQ(0, (uintptr_t)ptr % UCS_MBYTE);

    ucs_free(ptr);
}

UCS_TEST_F(test_memtrack, mmap) {
    void* ptr;

    ptr = ucs_mmap(NULL, ALLOC_SIZE, PROT_READ|PROT_WRITE,
                   MAP_PRIVATE|MAP_ANONYMOUS, -1, 0, ALLOC_NAME);
    ASSERT_NE((void*)NULL, ptr);
    ucs_munmap(ptr, ALLOC_SIZE);

    test_total(1, ALLOC_SIZE);
}

UCS_TEST_F(test_memtrack, custom) {
    void *ptr, *initial_ptr;

    initial_ptr = ptr = malloc(ALLOC_SIZE);
    ucs_memtrack_allocated(ptr, ALLOC_SIZE, ALLOC_NAME);

    memset(ptr, 0, ALLOC_SIZE);

    ucs_memtrack_releasing(ptr);
    ASSERT_EQ(initial_ptr, ptr);
    free(ptr);

    test_total(1, ALLOC_SIZE);
}
