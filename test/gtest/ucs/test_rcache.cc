/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
extern "C" {
#include <ucs/arch/atomic.h>
#include <ucs/sys/math.h>
#include <ucs/sys/rcache.h>
#include <ucs/sys/sys.h>
}


class test_rcache : public ucs::test {
protected:

    struct region {
        ucs_rcache_region_t super;
        uint32_t            magic;
        uint32_t            id;
    };

    test_rcache() : m_reg_count(0), m_ptr(NULL) {
    }

    virtual void init() {
        ucs::test::init();
        static const ucs_rcache_ops_t ops = {
            mem_reg_cb,
            mem_dereg_cb,
            dump_region_cb
        };
        ucs_rcache_params_t params = {
            sizeof(region),
            UCS_PGT_ADDR_ALIGN,
            ucs_get_page_size(),
            1000,
            &ops,
            reinterpret_cast<void*>(this)
        };
        UCS_TEST_CREATE_HANDLE(ucs_rcache_t*, m_rcache, ucs_rcache_destroy,
                               ucs_rcache_create, &params, "test", NULL);
    }

    virtual void cleanup() {
        m_rcache.reset();
        EXPECT_EQ(0u, m_reg_count);
        ucs::test::cleanup();
    }

    region *get(void *address, size_t length, int prot = PROT_READ|PROT_WRITE) {
        ucs_status_t status;
        ucs_rcache_region_t *r;
        status = ucs_rcache_get(m_rcache, address, length, prot, NULL, &r);
        ASSERT_UCS_OK(status);
        EXPECT_TRUE(r != NULL);
        struct region *region = ucs_derived_of(r, struct region);
        EXPECT_EQ(uint32_t(MAGIC), region->magic);
        EXPECT_TRUE(ucs_test_all_flags(region->super.prot, prot));
        return region;
    }

    void put(region *r) {
        ucs_rcache_region_put(m_rcache, &r->super);
    }

    virtual ucs_status_t mem_reg(region *region)
    {
        int mem_prot = ucs_get_mem_prot(region->super.super.start, region->super.super.end);
        if (!ucs_test_all_flags(mem_prot, region->super.prot)) {
            ucs_debug("protection error mem_prot " UCS_RCACHE_PROT_FMT " wanted " UCS_RCACHE_PROT_FMT,
                      UCS_RCACHE_PROT_ARG(mem_prot),
                      UCS_RCACHE_PROT_ARG(region->super.prot));
            return UCS_ERR_IO_ERROR;
        }

        mlock((const void*)region->super.super.start,
              region->super.super.end - region->super.super.start);
        EXPECT_NE(uint32_t(MAGIC), region->magic);
        region->magic = MAGIC;
        region->id    = ucs_atomic_fadd32(&next_id, 1);

        ucs_atomic_add32(&m_reg_count, +1);
        return UCS_OK;
    }

    virtual void mem_dereg(region *region)
    {
        munlock((const void*)region->super.super.start,
                region->super.super.end - region->super.super.start);
        EXPECT_EQ(uint32_t(MAGIC), region->magic);
        region->magic = 0;
        uint32_t prev = ucs_atomic_fadd32(&m_reg_count, -1);
        EXPECT_GT(prev, 0u);
    }

    virtual void dump_region(region *region, char *buf, size_t max)
    {
        snprintf(buf, max, "magic 0x%x id %u", region->magic, region->id);
    }

    void* shared_malloc(size_t size)
    {
        if (barrier()) {
            m_ptr = malloc(size);
        }
        barrier();
        return m_ptr;
    }

    void shared_free(void *ptr)
    {
        if (barrier()) {
            free(ptr);
        }
    }

    static void* alloc_pages(size_t size, int prot)
    {
        void *ptr = mmap(NULL, size, prot, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        EXPECT_NE(MAP_FAILED, ptr) << strerror(errno);
        return ptr;
    }

    static const uint32_t MAGIC = 0x05e905e9;
    static volatile uint32_t next_id;
    volatile uint32_t m_reg_count;
    ucs::handle<ucs_rcache_t*> m_rcache;
    void * volatile m_ptr;

private:

    static ucs_status_t mem_reg_cb(void *context, ucs_rcache_t *rcache,
                                   void *arg, ucs_rcache_region_t *r)
    {
        return reinterpret_cast<test_rcache*>(context)->mem_reg(
                        ucs_derived_of(r, struct region));
    }

    static void mem_dereg_cb(void *context, ucs_rcache_t *rcache,
                             ucs_rcache_region_t *r)
    {
        reinterpret_cast<test_rcache*>(context)->mem_dereg(
                        ucs_derived_of(r, struct region));
    }

    static void dump_region_cb(void *context, ucs_rcache_t *rcache,
                               ucs_rcache_region_t *r, char *buf, size_t max)
    {
        reinterpret_cast<test_rcache*>(context)->dump_region(
                        ucs_derived_of(r, struct region), buf, max);
    }
};

volatile uint32_t test_rcache::next_id = 1;


static uintptr_t virt_to_phys(uintptr_t address)
{
    static const char *pagemap_file = "/proc/self/pagemap";
    const size_t page_size = ucs_get_page_size();
    uint64_t entry, pfn;
    ssize_t offset, ret;
    uintptr_t pa;
    int fd;

    /* See https://www.kernel.org/doc/Documentation/vm/pagemap.txt */
    fd = open(pagemap_file, O_RDONLY);
    if (fd < 0) {
        ucs_error("failed to open %s: %m", pagemap_file);
        pa = -1;
        goto out;
    }

    offset = (address / page_size) * sizeof(entry);
    ret = lseek(fd, offset, SEEK_SET);
    if (ret != offset) {
        ucs_error("failed to seek in %s to offset %zu: %m", pagemap_file, offset);
        pa = -1;
        goto out_close;
    }

    ret = read(fd, &entry, sizeof(entry));
    if (ret != sizeof(entry)) {
        ucs_error("read from %s at offset %zu returned %ld: %m", pagemap_file,
                  offset, ret);
        pa = -1;
        goto out_close;
    }

    if (entry & (1ULL << 63)) {
        pfn = entry & ((1ULL << 54) - 1);
        pa = (pfn * page_size) | (address & (page_size - 1));
    } else {
        pa = -1; /* Page not present */
    }

out_close:
    close(fd);
out:
    return pa;
}

UCS_MT_TEST_F(test_rcache, basic, 10) {
    static const size_t size = 1 * 1024 * 1024;
    void *ptr = malloc(size);
    region *region = get(ptr, size);
    put(region);
    free(ptr);
}

UCS_MT_TEST_F(test_rcache, get_unmapped, 6) {
    /*
     *  - allocate, get, put, get again -> should be same id
     *  - release, get again -> should be different id
     */
    static const size_t size = 1 * 1024 * 1024;
    region *region;
    uintptr_t pa, new_pa;
    uint32_t id;
    void *ptr;

    ptr = malloc(size);
    region = get(ptr, size);
    id = region->id;
    pa = virt_to_phys(region->super.super.start);
    put(region);

    region = get(ptr, size);
    put(region);
    free(ptr);

    ptr = malloc(size);
    region = get(ptr, size);
    ucs_debug("got region id %d", region->id);
    new_pa = virt_to_phys(region->super.super.start);
    if (pa != new_pa) {
        ucs_debug("physical address changed (0x%lx->0x%lx)",
                  pa, new_pa);
        ucs_debug("id=%d region->id=%d", id, region->id);
        EXPECT_NE(id, region->id);
    } else {
        ucs_debug("physical address not changed (0x%lx)", pa);
    }
    put(region);
    free(ptr);
}

UCS_MT_TEST_F(test_rcache, merge, 6) {
    /*
     * +---------+-----+---------+
     * | region1 | pad | region2 |
     * +---+-----+-----+----+----+
     *     |   region3      |
     *     +----------------+
     */
    static const size_t size1 = 256 * ucs_get_page_size();
    static const size_t size2 = 512 * ucs_get_page_size();
    static const size_t pad   = 64 * ucs_get_page_size();
    region *region1, *region2, *region3, *region1_2;
    void *ptr1, *ptr2, *ptr3, *mem;
    size_t size3;

    mem = alloc_pages(size1 + pad + size2, PROT_READ|PROT_WRITE);

    /* Create region1 */
    ptr1 = (char*)mem;
    region1 = get(ptr1, size1);

    /* Get same region as region1 - should be same one */
    region1_2 = get(ptr1, size1);
    EXPECT_EQ(region1, region1_2);
    put(region1_2);

    /* Create region2 */
    ptr2 = (char*)mem + pad + size1;
    region2 = get(ptr2, size2);

    /* Create region3 which should merge region1 and region2 */
    ptr3 = (char*)mem + pad;
    size3 = size1 + size2 - pad;
    region3 = get(ptr3, size3);

    /* Get the same area as region1 - should be a different region now */
    region1_2 = get(ptr1, size1);
    EXPECT_NE(region1, region1_2); /* should be different region because was merged */
    EXPECT_EQ(region3, region1_2); /* it should be the merged region */
    put(region1_2);

    put(region1);
    put(region2);
    put(region3);

    munmap(mem, size1 + pad + size2);
}

UCS_MT_TEST_F(test_rcache, merge_inv, 6) {
    /*
     * Merge with another region which causes immediate invalidation of the
     * other region.
     * +---------+
     * | region1 |
     * +---+-----+----------+
     *     |   region2      |
     *     +----------------+
     */
    static const size_t size1 = 256 * 1024;
    static const size_t size2 = 512 * 1024;
    static const size_t pad   = 64 * 1024;
    region *region1, *region2;
    void *ptr1, *ptr2, *mem;
    uint32_t id1;

    mem = alloc_pages(pad + size2, PROT_READ|PROT_WRITE);

    /* Create region1 */
    ptr1 = (char*)mem;
    region1 = get(ptr1, size1);
    id1 = region1->id;
    put(region1);

    /* Create overlapping region - should destroy region1 */
    ptr2 = (char*)mem + pad;
    region2 = get(ptr2, size2);
    EXPECT_NE(id1, region2->id);
    put(region2);

    munmap(mem, pad + size2);
}

UCS_MT_TEST_F(test_rcache, release_inuse, 6) {
    static const size_t size = 1 * 1024 * 1024;

    void *ptr1 = malloc(size);
    region *region1 = get(ptr1, size);
    free(ptr1);

    void *ptr2 = malloc(size);
    region *region2 = get(ptr2, size);
    put(region2);
    free(ptr2);

    /* key should still be valid */
    EXPECT_EQ(uint32_t(MAGIC), region1->magic);

    put(region1);
}

/*
 * +-------------+-------------+
 * | region1 -r  | region2 -w  |
 * +---+---------+------+------+
 *     |   region3 r    |
 *     +----------------+
 *
 * don't merge with inaccessible pages
 */
UCS_MT_TEST_F(test_rcache, merge_with_unwritable, 6) {
    static const size_t size1 = 10 * ucs_get_page_size();
    static const size_t size2 =  8 * ucs_get_page_size();

    void *mem = alloc_pages(size1 + size2, PROT_READ);
    void *ptr1 = mem;

    /* Set region1 to map all of 1-st part of the 2-nd */
    region *region1 = get(ptr1, size1 + size2 / 2, PROT_READ);
    EXPECT_EQ(PROT_READ, region1->super.prot);

    /* Set 2-nd part as write-only */
    void *ptr2 = (char*)mem + size1;
    int ret = mprotect(ptr2, size2, PROT_WRITE);
    ASSERT_EQ(0, ret) << strerror(errno);

    /* Get 2-nd part - should not merge with region1 */
    region *region2 = get(ptr2, size2, PROT_WRITE);
    EXPECT_GE(region2->super.super.start, (uintptr_t)ptr2);
    EXPECT_EQ(PROT_WRITE, region2->super.prot);

    EXPECT_TRUE(!(region1->super.flags & UCS_RCACHE_REGION_FLAG_PGTABLE));
    put(region1);

    put(region2);
    munmap(mem, size1 + size2);
}

/* don't expand prot of our region if our pages cant support it */
UCS_MT_TEST_F(test_rcache, merge_merge_unwritable, 6) {
    static const size_t size1 = 10 * ucs_get_page_size();
    static const size_t size2 =  8 * ucs_get_page_size();

    void *mem = alloc_pages(size1 + size2, PROT_READ|PROT_WRITE);
    ASSERT_NE(MAP_FAILED, mem) << strerror(errno);

    void *ptr1 = mem;

    /* Set region1 to map all of 1-st part of the 2-nd */
    region *region1 = get(ptr1, size1 + size2 / 2, PROT_READ|PROT_WRITE);
    EXPECT_EQ(PROT_READ|PROT_WRITE, region1->super.prot);

    /* Set 2-nd part as read-only */
    void *ptr2 = (char*)mem + size1;
    int ret = mprotect(ptr2, size2, PROT_READ);
    ASSERT_EQ(0, ret) << strerror(errno);

    /* Get 2-nd part - should not merge because we are read-only */
    region *region2 = get(ptr2, size2, PROT_READ);
    EXPECT_GE(region2->super.super.start, (uintptr_t)ptr2);
    EXPECT_EQ(PROT_READ, region2->super.prot);

    put(region1);
    put(region2);
    munmap(mem, size1 + size2);
}

/* expand prot of new region to support existing regions */
UCS_MT_TEST_F(test_rcache, merge_expand_prot, 6) {
    static const size_t size1 = 10 * ucs_get_page_size();
    static const size_t size2 =  8 * ucs_get_page_size();

    void *mem = alloc_pages(size1 + size2, PROT_READ|PROT_WRITE);
    ASSERT_NE(MAP_FAILED, mem) << strerror(errno);

    void *ptr1 = mem;

    /* Set region1 to map all of 1-st part of the 2-nd */
    region *region1 = get(ptr1, size1 + size2 / 2, PROT_READ);
    EXPECT_EQ(PROT_READ, region1->super.prot);

    /* Get 2-nd part - should merge with region1 with full protection */
    void *ptr2 = (char*)mem + size1;
    region *region2 = get(ptr2, size2, PROT_WRITE);
    if (region1->super.flags & UCS_RCACHE_REGION_FLAG_PGTABLE) {
        EXPECT_LE(region2->super.super.start, (uintptr_t)ptr1);
        EXPECT_TRUE(region2->super.prot & PROT_READ);
    }
    EXPECT_TRUE(region2->super.prot & PROT_WRITE);
    EXPECT_GE(region2->super.super.end, (uintptr_t)ptr2 + size2);

    put(region1);
    put(region2);
    munmap(mem, size1 + size2);
}

/*
 * Test flow:
 * +---------------------+
 * |       r+w           |  1. memory allocated with R+W prot
 * +---------+-----------+
 * | region1 |           |  2. region1 is created in part of the memory
 * +-----+---+-----------+
 * | r   |     r+w       |  3. region1 is freed, some of the region memory changed to R
 * +-----+---------------+
 * |     |    region2    |  4. region2 is created. region1 must be invalidated and
 * +-----+---------------+     kicked out of pagetable.
 */
UCS_MT_TEST_F(test_rcache, merge_invalid_prot, 6)
{
    static const size_t size1 = 10 * ucs_get_page_size();
    static const size_t size2 =  8 * ucs_get_page_size();
    int ret;

    void *mem = alloc_pages(size1+size2, PROT_READ|PROT_WRITE);
    void *ptr1 = mem;

    region *region1 = get(ptr1, size1, PROT_READ|PROT_WRITE);
    EXPECT_EQ(PROT_READ|PROT_WRITE, region1->super.prot);
    put(region1);

    ret = mprotect(ptr1, ucs_get_page_size(), PROT_READ);
    ASSERT_EQ(0, ret) << strerror(errno);

    void *ptr2 = (char*)mem+size1 - 1024 ;
    region *region2 = get(ptr2, size2, PROT_READ|PROT_WRITE);

    /* check permissions and that the region is not merged */
    EXPECT_EQ(PROT_READ|PROT_WRITE, region2->super.prot);
    EXPECT_EQ(region2->super.super.start, (uintptr_t)ptr2);

    barrier();
    EXPECT_EQ(6u, m_reg_count);
    barrier();
    put(region2);
    munmap(mem, size1+size2);
}

UCS_MT_TEST_F(test_rcache, shared_region, 6) {
    static const size_t size = 1 * 1024 * 1024;

    void *mem = shared_malloc(size);

    void *ptr1 = mem;
    size_t size1 = size * 2 / 3;

    void *ptr2 = (char*)mem + size - size1;
    size_t size2 = size1;

    region *region1 = get(ptr1, size1);
    usleep(100);
    put(region1);

    region *region2 = get(ptr2, size2);
    usleep(100);
    put(region2);

    shared_free(mem);
}

class test_rcache_no_register : public test_rcache {
protected:
    bool m_fail_reg;
    virtual ucs_status_t mem_reg(region *region) {
        if (m_fail_reg) {
            return UCS_ERR_IO_ERROR;
        }
        return test_rcache::mem_reg(region);
    }

    virtual void init() {
        test_rcache::init();
        ucs_log_push_handler(log_handler);
        m_fail_reg = true;
    }

    virtual void cleanup() {
        ucs_log_pop_handler();
        test_rcache::cleanup();
    }

    static ucs_log_func_rc_t
    log_handler(const char *file, unsigned line, const char *function,
                ucs_log_level_t level, const char *message, va_list ap)
    {
        /* Ignore warnings about empty memory pool */
        if ((level == UCS_LOG_LEVEL_WARN) && strstr(message, "failed to register")) {
            UCS_TEST_MESSAGE << format_message(message, ap);
            return UCS_LOG_FUNC_RC_STOP;
        }

        return UCS_LOG_FUNC_RC_CONTINUE;
    }
};

UCS_MT_TEST_F(test_rcache_no_register, register_failure, 10) {
    static const size_t size = 1 * 1024 * 1024;
    void *ptr = malloc(size);

    ucs_status_t status;
    ucs_rcache_region_t *r;
    status = ucs_rcache_get(m_rcache, ptr, size, PROT_READ|PROT_WRITE, NULL, &r);
    EXPECT_EQ(UCS_ERR_IO_ERROR, status);
    EXPECT_EQ(0u, m_reg_count);

    free(ptr);
}

/* The region overlaps an old region with different
 * protection and memory protection does not fit one of
 * the region.
 * This should trigger an error during merge.
 *
 * Test flow:
 * +---------------------+
 * |       r+w           |  1. memory allocated with R+W prot
 * +---------+-----------+
 * | region1 |           |  2. region1 is created in part of the memory
 * +-----+---+-----------+
 * | r                   |  3. region1 is freed, all memory changed to R
 * +-----+---------------+
 * |     |    region2(w) |  4. region2 is created. region1 must be invalidated and
 * +-----+---------------+     kicked out of pagetable. Creation of region2
 *                             must fail.
 */
UCS_MT_TEST_F(test_rcache_no_register, merge_invalid_prot_slow, 5)
{
    static const size_t size1 = 10 * ucs_get_page_size();
    static const size_t size2 =  8 * ucs_get_page_size();
    int ret;

    void *mem = alloc_pages(size1+size2, PROT_READ|PROT_WRITE);
    void *ptr1 = mem;

    m_fail_reg = false;
    region *region1 = get(ptr1, size1, PROT_READ|PROT_WRITE);
    EXPECT_EQ(PROT_READ|PROT_WRITE, region1->super.prot);
    put(region1);

    void *ptr2 = (char*)mem+size1 - 1024 ;
    ret = mprotect(mem, size1, PROT_READ);
    ASSERT_EQ(0, ret) << strerror(errno);


    ucs_status_t status;
    ucs_rcache_region_t *r;

    status = ucs_rcache_get(m_rcache, ptr2, size2, PROT_WRITE, NULL, &r);
    EXPECT_EQ(UCS_ERR_IO_ERROR, status);

    barrier();
    EXPECT_EQ(0u, m_reg_count);

    munmap(mem, size1+size2);
}
