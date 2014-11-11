/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#include <ucs/gtest/test.h>

extern "C" {
#include <ucs/debug/memtrack.h>
}

#include <stdio.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#if ENABLE_MEMTRACK

class test_memtrack : public ucs::test {
protected:
    static const size_t ALLOC_SIZE = 100;

    void init() {
        ucs_memtrack_cleanup();
        push_config();
        set_config("MEMTRACK_DEST", "/dev/null");
        ucs_memtrack_init();
    }

    void cleanup() {
        ucs_memtrack_cleanup();
        pop_config();
        ucs_memtrack_init();
    }
};


UCS_TEST_F(test_memtrack, sanity) {
    char test_str[] = "memtrack_test";
    ucs_memtrack_entry_t entry;
    void *a, *b;
    int i;

    ucs_memtrack_total(&entry);
    i = entry.current_count;

    b = ucs_malloc(1, test_str);
    ucs_free(b);

    b = ucs_malloc(1, test_str);
    a = ucs_malloc(3, test_str);
    ucs_free(b);
    ucs_memtrack_total(&entry);
    if (ucs_memtrack_is_enabled()) {
        ASSERT_EQ((size_t)(i + 1), entry.current_count);
    }

    b = ucs_malloc(4, test_str);
    ucs_free(b);
    ucs_memtrack_total( &entry);
    if (ucs_memtrack_is_enabled()) {
        ASSERT_EQ((size_t)1, entry.current_count);
    }
    ucs_free(a);

    for (i = 0; i < 101; i++) {
        a = ucs_malloc(i, test_str);
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
    ASSERT_NE((void *)NULL, strstr(buf, "TOTAL"));

    free(buf);
}

UCS_TEST_F(test_memtrack, alloc_types) {
    char test_str[] = "memtrack_test";
    void* ptr;

    ptr = ucs_malloc(ALLOC_SIZE, test_str);
    ASSERT_NE((void *)NULL, ptr);
    ptr = ucs_realloc(ptr, 2 * ALLOC_SIZE);
    ASSERT_NE((void *)NULL, ptr);
    ucs_free(ptr);

    ptr = ucs_calloc(1, ALLOC_SIZE, test_str);
    ASSERT_NE((void *)NULL, ptr);
    ucs_free(ptr);

    ptr = ucs_calloc(ALLOC_SIZE, 1, test_str);
    ASSERT_NE((void *)NULL, ptr);
    ucs_free(ptr);

    ptr = ucs_memalign(10, ALLOC_SIZE, test_str);
    ASSERT_NE((void *)NULL, ptr);
    ucs_free(ptr);

    ptr = ucs_memalign(1000, ALLOC_SIZE, test_str);
    ASSERT_NE((void *)NULL, ptr);
    ucs_free(ptr);
}

#endif
