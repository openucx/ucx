/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/mem_buffer.h>
#include <common/test.h>
extern "C" {
#include <ucs/arch/atomic.h>
#include <ucs/sys/math.h>
#include <ucs/stats/stats.h>
#include <ucs/memory/rcache.h>
#include <ucs/memory/rcache_int.h>
#include <ucs/sys/sys.h>
#include <ucm/api/ucm.h>
}
#include <set>


class alloc_table
{
public:

    typedef struct alloc_region {
        ucs_pgt_region_t super;
        uint32_t         id;
        uint32_t         ref_count;
    } alloc_region_t;

    alloc_table() : m_next_alloc_id(0) {}

    bool init()
    {
        ucs_status_t status;
        status = ucs_pgtable_init(&m_pgtable, pgt_dir_alloc, pgt_dir_release);
        if (status != UCS_OK) {
            return false;
        }
        if (pthread_rwlock_init(&m_pgtable_lock, NULL) ||
            pthread_mutex_init(&m_mutex, NULL)) {
            return false;
        }
        return true;
    }

    bool cleanup()
    {
        ucs_pgtable_cleanup(&m_pgtable);
        if (pthread_rwlock_destroy(&m_pgtable_lock) ||
            pthread_mutex_destroy(&m_mutex)) {
            return false;
        }
        return true;
    }

    alloc_region_t *find(void* ptr)
    {
        return find((uintptr_t)ptr);
    }

    alloc_region_t *find(ucs_pgt_addr_t pgt_addr)
    {
        pthread_rwlock_rdlock(&m_pgtable_lock);
        ucs_pgt_region_t *pgt_region = ucs_pgtable_lookup(&m_pgtable, pgt_addr);
        pthread_rwlock_unlock(&m_pgtable_lock);
        return ucs_derived_of(pgt_region, alloc_region_t);
    }

    bool insert(void* ptr, size_t size)
    {
        bool ret = true;
        ucs_status_t status;
        alloc_region_t *alloc_region;
        ucs_pgt_addr_t alloc_start = ucs_align_down_pow2((uintptr_t)ptr, UCS_PGT_ADDR_ALIGN);
        ucs_pgt_addr_t alloc_end   = ucs_align_up_pow2((uintptr_t)ptr + size, UCS_PGT_ADDR_ALIGN);

        pthread_mutex_lock(&m_mutex);
        alloc_region = find(alloc_start);
        if (!alloc_region) {
            int err = ucs_posix_memalign((void **)&alloc_region,
                                         ucs_max(sizeof(void *), UCS_PGT_ENTRY_MIN_ALIGN),
                                         sizeof(*alloc_region),
                                         "test_rcache_valid_allocs_region");
            if (err) {
                ret = false;
                goto out_unlock;
            }

            alloc_region->super.start = alloc_start;
            alloc_region->super.end   = alloc_end;
            alloc_region->id          = m_next_alloc_id++;
            alloc_region->ref_count   = 0;
            pthread_rwlock_wrlock(&m_pgtable_lock);
            status = ucs_pgtable_insert(&m_pgtable, &alloc_region->super);
            pthread_rwlock_unlock(&m_pgtable_lock);
            if (status != UCS_OK) {
                ret = false;
                goto out_unlock;
            }
        } else if (alloc_region->super.end < alloc_end) {
            alloc_region->super.end = alloc_end;
        }

        alloc_region->ref_count++;

out_unlock:
        pthread_mutex_unlock(&m_mutex);
        return ret;
    }

    bool remove(void *ptr)
    {
        bool ret = true;
        ucs_status_t status;
        alloc_region_t *alloc_region;

        pthread_mutex_lock(&m_mutex);
        alloc_region = find(ptr);

        if (!alloc_region) {
            ret = false;
            goto out_unlock;
        }

        if(alloc_region->ref_count-- == 1) {
            pthread_rwlock_wrlock(&m_pgtable_lock);
            status = ucs_pgtable_remove(&m_pgtable, &alloc_region->super);
            pthread_rwlock_unlock(&m_pgtable_lock);
            free(alloc_region);
            if (status != UCS_OK) {
                ret = false;
                goto out_unlock;
            }
        }

out_unlock:
        pthread_mutex_unlock(&m_mutex);
        return ret;
    }

private:

    static ucs_pgt_dir_t *pgt_dir_alloc(const ucs_pgtable_t *pgtable)
    {
        ucs_pgt_dir_t *dir = NULL;
        int ret = ucs_posix_memalign((void **)&dir,
                                     ucs_max(sizeof(void *), UCS_PGT_ENTRY_MIN_ALIGN),
                                     sizeof(ucs_pgt_dir_t),
                                     "test_rcache_valid_allocs");
        EXPECT_EQ(0, ret);
        return dir;
    }

    static void pgt_dir_release(const ucs_pgtable_t *pgtable,
                                ucs_pgt_dir_t *dir)
    {
        free(dir);
    }

    ucs_pgtable_t m_pgtable;
    volatile uint32_t m_next_alloc_id;
    pthread_rwlock_t m_pgtable_lock;
    pthread_mutex_t m_mutex;
};

static ucs_rcache_params_t
get_default_rcache_params(void *context, const ucs_rcache_ops_t *ops)
{
    ucs_rcache_params_t params = {sizeof(ucs_rcache_region_t),
                                  UCS_PGT_ADDR_ALIGN,
                                  ucs_get_page_size(),
                                  UCM_EVENT_VM_UNMAPPED | UCM_EVENT_MEM_TYPE_FREE,
                                  1000,
                                  ops,
                                  context,
                                  0,
                                  ULONG_MAX,
                                  SIZE_MAX};

    return params;
}

class test_rcache_basic : public ucs::test {
};

UCS_TEST_F(test_rcache_basic, create_fail) {
    static const ucs_rcache_ops_t ops = {
        NULL, NULL, NULL
    };
    ucs_rcache_params_t params        = get_default_rcache_params(this, &ops);
    params.ucm_events                 = UCS_BIT(30); /* non-existing event */
    params.context                    = NULL;

    ucs_rcache_t *rcache;
    ucs_status_t status = ucs_rcache_create(&params, "test",
                                            ucs_stats_get_root(), &rcache);
    EXPECT_NE(UCS_OK, status); /* should fail */
    if (status == UCS_OK) {
        ucs_rcache_destroy(rcache);
    }
}


class test_rcache : public ucs::test_with_param<ucs_memory_type_t>
{
protected:

    struct region {
        ucs_rcache_region_t super;
        uint32_t            magic;
        uint32_t            id;
        uint32_t            alloc_id;
        ucs_memory_type_t   mem_type;
    };

    test_rcache() : m_reg_count(0), m_ptr(NULL) {
    }

    virtual void init()
    {
        ucs::test_with_param<ucs_memory_type_t>::init();
        ucs_rcache_params params = rcache_params();
        UCS_TEST_CREATE_HANDLE_IF_SUPPORTED(ucs_rcache_t*, m_rcache, ucs_rcache_destroy,
                                            ucs_rcache_create, &params, "test", ucs_stats_get_root());
        ASSERT_TRUE(m_valid_allocs.init());
        ASSERT_EQ(0, pthread_mutex_init(&m_mutex, NULL));
    }

    virtual void cleanup()
    {
        m_rcache.reset();
        EXPECT_EQ(0u, m_reg_count);
        ASSERT_TRUE(m_valid_allocs.cleanup());
        ASSERT_EQ(0, pthread_mutex_destroy(&m_mutex));
        ucs::test_with_param<ucs_memory_type_t>::cleanup();
    }

    virtual ucs_rcache_params_t rcache_params()
    {
        static const ucs_rcache_ops_t ops = {
            mem_reg_cb,
            mem_dereg_cb,
            mem_reg_ext_validate_cb,
            mem_dereg_ext_validate_cb,
            mem_ext_validate_cb,
            dump_region_cb
        };
        ucs_rcache_params_t params        = get_default_rcache_params(this, &ops);
        params.region_struct_size         = sizeof(region);
        return params;
    }

    region *get(void *address, size_t length,
                ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_HOST,
                int prot = PROT_READ|PROT_WRITE)
    {
        ucs_status_t status;
        ucs_rcache_region_t *r;
        status = ucs_rcache_get(m_rcache, address, length, prot,
                                reinterpret_cast<void*>(&mem_type), &r);
        ASSERT_UCS_OK(status);
        EXPECT_TRUE(r != NULL);
        struct region *region = ucs_derived_of(r, struct region);
        EXPECT_EQ(uint32_t(MAGIC), region->magic);
        EXPECT_TRUE(ucs_test_all_flags(region->super.prot, prot));
        return region;
    }

    void put(region *r)
    {
        ucs_rcache_region_put(m_rcache, &r->super);
    }

    virtual ucs_status_t mem_reg(region *region, const void *arg)
    {
        ucs_memory_type_t mem_type = *reinterpret_cast<const ucs_memory_type_t*>(arg);
        int mem_prot = ucs_get_mem_prot(region->super.super.start, region->super.super.end);
        if (mem_type == UCS_MEMORY_TYPE_HOST &&
            !ucs_test_all_flags(mem_prot, region->super.prot)) {
            ucs_debug("protection error mem_prot " UCS_RCACHE_PROT_FMT " wanted " UCS_RCACHE_PROT_FMT,
                      UCS_RCACHE_PROT_ARG(mem_prot),
                      UCS_RCACHE_PROT_ARG(region->super.prot));
            return UCS_ERR_IO_ERROR;
        }

        mlock((const void*)region->super.super.start,
              region->super.super.end - region->super.super.start);
        EXPECT_NE(uint32_t(MAGIC), region->magic);
        region->magic    = MAGIC;
        region->id       = ucs_atomic_fadd32(&next_id, 1);
        region->mem_type = mem_type;

        ucs_atomic_add32(&m_reg_count, +1);
        return UCS_OK;
    }

    virtual void mem_dereg(region *region)
    {
        munlock((const void*)region->super.super.start,
                region->super.super.end - region->super.super.start);
        EXPECT_EQ(uint32_t(MAGIC), region->magic);
        region->magic = 0;
        uint32_t prev = ucs_atomic_fsub32(&m_reg_count, 1);
        EXPECT_GT(prev, 0u);
    }

    virtual ucs_status_t mem_reg_ext_validate(region *region, const void *arg)
    {
        alloc_table::alloc_region_t *alloc_region;

        ucs_status_t status = mem_reg(region, arg);

        if (region->mem_type != UCS_MEMORY_TYPE_HOST) {
            /* Save in rcache region the id of the corresponding allocation */
            alloc_region = m_valid_allocs.find(region->super.super.start);
            EXPECT_NE((void*)NULL, alloc_region);
            region->alloc_id = alloc_region->id;
        }

        return status;
    }

    virtual int mem_ext_validate(region *region)
    {
        alloc_table::alloc_region_t *alloc_region;

        if (region->mem_type != UCS_MEMORY_TYPE_HOST) {
            alloc_region = m_valid_allocs.find(region->super.super.start);
            EXPECT_NE((void*)NULL, alloc_region);
            EXPECT_LE(region->super.super.end, alloc_region->super.end);
            return region->alloc_id == alloc_region->id;
        }

        /* HOST memory is always valid */
        return 1;
    }

    virtual void mem_dereg_ext_validate(region *region)
    {
        mem_dereg(region);
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
    alloc_table m_valid_allocs;
    pthread_mutex_t m_mutex;

private:

    static ucs_status_t mem_reg_cb(void *context, ucs_rcache_t *rcache,
                                   void *arg, ucs_rcache_region_t *r,
                                   uint16_t rcache_mem_reg_flags)
    {
        return reinterpret_cast<test_rcache*>(context)->mem_reg(
                        ucs_derived_of(r, struct region), arg);
    }

    static void mem_dereg_cb(void *context, ucs_rcache_t *rcache,
                             ucs_rcache_region_t *r)
    {
        reinterpret_cast<test_rcache*>(context)->mem_dereg(
                        ucs_derived_of(r, struct region));
    }

    static ucs_status_t
    mem_reg_ext_validate_cb(void *context, ucs_rcache_t *rcache,
                            void *arg, ucs_rcache_region_t *r,
                            uint16_t rcache_mem_reg_flags)
    {
        return reinterpret_cast<test_rcache*>(context)->mem_reg_ext_validate(
                        ucs_derived_of(r, struct region), arg);
    }

    static void mem_dereg_ext_validate_cb(void *context, ucs_rcache_t *rcache,
                                          ucs_rcache_region_t *r)
    {
        reinterpret_cast<test_rcache*>(context)->mem_dereg_ext_validate(
                        ucs_derived_of(r, struct region));
    }

    static int mem_ext_validate_cb(void *context, ucs_rcache_t *rcache,
                                   ucs_rcache_region_t *r)
    {
        return reinterpret_cast<test_rcache*>(context)->mem_ext_validate(
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
        pa = std::numeric_limits<uintptr_t>::max();
        goto out;
    }

    offset = (address / page_size) * sizeof(entry);
    ret = lseek(fd, offset, SEEK_SET);
    if (ret != offset) {
        ucs_error("failed to seek in %s to offset %zu: %m", pagemap_file, offset);
        pa = std::numeric_limits<uintptr_t>::max();
        goto out_close;
    }

    ret = read(fd, &entry, sizeof(entry));
    if (ret != sizeof(entry)) {
        ucs_error("read from %s at offset %zu returned %ld: %m", pagemap_file,
                  offset, ret);
        pa = std::numeric_limits<uintptr_t>::max();
        goto out_close;
    }

    if (entry & (1ULL << 63)) {
        pfn = entry & ((1ULL << 54) - 1);
        pa = (pfn * page_size) | (address & (page_size - 1));
    } else {
        pa = std::numeric_limits<uintptr_t>::max(); /* Page not present */
    }

out_close:
    close(fd);
out:
    return pa;
}

#define UCS_RCACHE_MALLOC_P(size, ptr) \
    /* coverity[missing_unlock] */ \
    do { \
        pthread_mutex_lock(&m_mutex); \
        ptr = mem_buffer::allocate(size, GetParam()); \
        ASSERT_TRUE(ptr != NULL); \
        ASSERT_TRUE(m_valid_allocs.insert(ptr, size)); \
        pthread_mutex_unlock(&m_mutex); \
    } while (0);

#define UCS_RCACHE_FREE_P(ptr) \
    /* coverity[use] */ \
    /* coverity[missing_unlock] */ \
    do { \
        pthread_mutex_lock(&m_mutex); \
        mem_buffer::release(ptr, GetParam()); \
        ASSERT_TRUE(m_valid_allocs.remove(ptr)); \
        pthread_mutex_unlock(&m_mutex); \
    } while (0);

#define UCS_RCACHE_ALLOC_PAGES_P(size, prot, ptr) \
    do { \
        if (GetParam() == UCS_MEMORY_TYPE_HOST) { \
            ptr = alloc_pages(size, prot); \
        } else { \
            UCS_RCACHE_MALLOC_P(size, ptr); \
        } \
    } while (0);

#define UCS_RCACHE_RELEASE_PAGES_P(ptr, size) \
    do { \
        if (GetParam() == UCS_MEMORY_TYPE_HOST) { \
            munmap(ptr, size); \
        } else { \
            UCS_RCACHE_FREE_P(ptr); \
        } \
    } while (0);

UCS_MT_TEST_P(test_rcache, basic, 10)
{
    static const size_t size = 1 * 1024 * 1024;
    void *ptr;
    UCS_RCACHE_MALLOC_P(size, ptr);
    region *region = get(ptr, size, GetParam());
    put(region);
    UCS_RCACHE_FREE_P(ptr);
}

UCS_MT_TEST_P(test_rcache, get_unmapped, 6)
{
    /*
     *  - allocate, get, put, get again -> should be same id
     *  - release, get again -> should be different id
     */
    static const size_t size = 1 * 1024 * 1024;
    region *region;
    uintptr_t pa, new_pa;
    uint32_t id;
    void *ptr;

    UCS_RCACHE_MALLOC_P(size, ptr);
    region = get(ptr, size, GetParam());
    id = region->id;
    pa = virt_to_phys(region->super.super.start);
    put(region);

    region = get(ptr, size, GetParam());
    put(region);
    UCS_RCACHE_FREE_P(ptr);

    UCS_RCACHE_MALLOC_P(size, ptr);
    region = get(ptr, size, GetParam());
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
    UCS_RCACHE_FREE_P(ptr);
}

UCS_TEST_SKIP_COND_P(test_rcache, non_host_get_free_get,
                     GetParam() == UCS_MEMORY_TYPE_HOST)
{
    /* All new non-host allocations must lead to a cache miss.
     * So, we must see a different region id.
     */
    static const size_t size = 1 * 1024 * 1024;
    region *region;
    uint32_t prev_id = 0;
    void *ptr;

    for (size_t i = 0; i < 10; i++) {
        UCS_RCACHE_MALLOC_P(size, ptr);
        region = get(ptr, size, GetParam());
        EXPECT_EQ(uint32_t(MAGIC), region->magic);
        EXPECT_NE(prev_id, region->id);
        prev_id = region->id;
        put(region);
        UCS_RCACHE_FREE_P(ptr);
    }
}

UCS_MT_TEST_P(test_rcache, merge, 6)
{
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

    UCS_RCACHE_ALLOC_PAGES_P(size1 + pad + size2, PROT_READ | PROT_WRITE, mem);

    /* Create region1 */
    ptr1 = (char*)mem;
    region1 = get(ptr1, size1, GetParam());

    /* Get same region as region1 - should be same one */
    region1_2 = get(ptr1, size1, GetParam());
    EXPECT_EQ(region1, region1_2);
    put(region1_2);

    /* Create region2 */
    ptr2 = (char*)mem + pad + size1;
    region2 = get(ptr2, size2, GetParam());

    /* Create region3 which should merge region1 and region2 */
    ptr3 = (char*)mem + pad;
    size3 = size1 + size2 - pad;
    region3 = get(ptr3, size3, GetParam());

    /* Get the same area as region1 - should be a different region now */
    region1_2 = get(ptr1, size1, GetParam());
    EXPECT_NE(region1, region1_2); /* should be different region because was merged */
    EXPECT_EQ(region3, region1_2); /* it should be the merged region */
    put(region1_2);

    put(region1);
    put(region2);
    put(region3);

    UCS_RCACHE_RELEASE_PAGES_P(mem, size1 + pad + size2);
}

UCS_MT_TEST_P(test_rcache, merge_inv, 6)
{
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

    UCS_RCACHE_ALLOC_PAGES_P(pad + size2, PROT_READ | PROT_WRITE, mem);

    /* Create region1 */
    ptr1 = (char*)mem;
    region1 = get(ptr1, size1, GetParam());
    id1 = region1->id;
    put(region1);

    /* Create overlapping region - should destroy region1 */
    ptr2 = (char*)mem + pad;
    region2 = get(ptr2, size2, GetParam());
    EXPECT_NE(id1, region2->id);
    put(region2);

    UCS_RCACHE_RELEASE_PAGES_P(mem, pad + size2);
}

UCS_MT_TEST_P(test_rcache, release_inuse, 6)
{
    static const size_t size = 1 * 1024 * 1024;

    void *ptr1;
    UCS_RCACHE_MALLOC_P(size, ptr1);
    region *region1 = get(ptr1, size, GetParam());
    UCS_RCACHE_FREE_P(ptr1);

    void *ptr2;
    UCS_RCACHE_MALLOC_P(size, ptr2);
    region *region2 = get(ptr2, size, GetParam());
    put(region2);
    UCS_RCACHE_FREE_P(ptr2);

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
UCS_MT_TEST_F(test_rcache, merge_with_unwritable, 6)
{
    static const size_t size1 = 10 * ucs_get_page_size();
    static const size_t size2 =  8 * ucs_get_page_size();

    void *mem = alloc_pages(size1 + size2, PROT_READ);
    void *ptr1 = mem;

    /* Set region1 to map all of 1-st part of the 2-nd */
    region *region1 = get(ptr1, size1 + size2 / 2,
                          UCS_MEMORY_TYPE_HOST, PROT_READ);
    EXPECT_EQ(PROT_READ, region1->super.prot);

    /* Set 2-nd part as write-only */
    void *ptr2 = (char*)mem + size1;
    int ret = mprotect(ptr2, size2, PROT_WRITE);
    ASSERT_EQ(0, ret) << strerror(errno);

    /* Get 2-nd part - should not merge with region1 */
    region *region2 = get(ptr2, size2, UCS_MEMORY_TYPE_HOST, PROT_WRITE);
    EXPECT_GE(region2->super.super.start, (uintptr_t)ptr2);
    EXPECT_EQ(PROT_WRITE, region2->super.prot);

    EXPECT_TRUE(!(region1->super.flags & UCS_RCACHE_REGION_FLAG_PGTABLE));
    put(region1);

    put(region2);
    munmap(mem, size1 + size2);
}

/* don't expand prot of our region if our pages cant support it */
UCS_MT_TEST_F(test_rcache, merge_merge_unwritable, 6)
{
    static const size_t size1 = 10 * ucs_get_page_size();
    static const size_t size2 =  8 * ucs_get_page_size();

    void *mem = alloc_pages(size1 + size2, PROT_READ|PROT_WRITE);
    ASSERT_NE(MAP_FAILED, mem) << strerror(errno);

    void *ptr1 = mem;

    /* Set region1 to map all of 1-st part of the 2-nd */
    region *region1 = get(ptr1, size1 + size2 / 2,
                          UCS_MEMORY_TYPE_HOST, PROT_READ|PROT_WRITE);
    EXPECT_EQ(PROT_READ|PROT_WRITE, region1->super.prot);

    /* Set 2-nd part as read-only */
    void *ptr2 = (char*)mem + size1;
    int ret = mprotect(ptr2, size2, PROT_READ);
    ASSERT_EQ(0, ret) << strerror(errno);

    /* Get 2-nd part - should not merge because we are read-only */
    region *region2 = get(ptr2, size2, UCS_MEMORY_TYPE_HOST, PROT_READ);
    EXPECT_GE(region2->super.super.start, (uintptr_t)ptr2);
    EXPECT_EQ(PROT_READ, region2->super.prot);

    put(region1);
    put(region2);
    munmap(mem, size1 + size2);
}

/* expand prot of new region to support existing regions */
UCS_MT_TEST_F(test_rcache, merge_expand_prot, 6)
{
    static const size_t size1 = 10 * ucs_get_page_size();
    static const size_t size2 =  8 * ucs_get_page_size();

    void *mem = alloc_pages(size1 + size2, PROT_READ|PROT_WRITE);
    ASSERT_NE(MAP_FAILED, mem) << strerror(errno);

    void *ptr1 = mem;

    /* Set region1 to map all of 1-st part of the 2-nd */
    region *region1 = get(ptr1, size1 + size2 / 2,
                          UCS_MEMORY_TYPE_HOST, PROT_READ);
    EXPECT_EQ(PROT_READ, region1->super.prot);

    /* Get 2-nd part - should merge with region1 with full protection */
    void *ptr2 = (char*)mem + size1;
    region *region2 = get(ptr2, size2, UCS_MEMORY_TYPE_HOST, PROT_WRITE);
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

    region *region1 = get(ptr1, size1, UCS_MEMORY_TYPE_HOST, PROT_READ|PROT_WRITE);
    EXPECT_EQ(PROT_READ|PROT_WRITE, region1->super.prot);
    put(region1);

    ret = mprotect(ptr1, ucs_get_page_size(), PROT_READ);
    ASSERT_EQ(0, ret) << strerror(errno);

    void *ptr2 = (char*)mem+size1 - 1024 ;
    region *region2 = get(ptr2, size2, UCS_MEMORY_TYPE_HOST, PROT_READ|PROT_WRITE);

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
    virtual ucs_status_t mem_reg(region *region, const void *arg) {
        if (m_fail_reg) {
            return UCS_ERR_IO_ERROR;
        }
        return test_rcache::mem_reg(region, arg);
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
                ucs_log_level_t level,
                const ucs_log_component_config_t *comp_conf,
                const char *message, va_list ap)
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
    ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_HOST;
    status = ucs_rcache_get(m_rcache, ptr, size,
                            PROT_READ|PROT_WRITE, &mem_type, &r);
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
    region *region1 = get(ptr1, size1,
                          UCS_MEMORY_TYPE_HOST, PROT_READ|PROT_WRITE);
    EXPECT_EQ(PROT_READ|PROT_WRITE, region1->super.prot);
    put(region1);

    void *ptr2 = (char*)mem+size1 - 1024 ;
    ret = mprotect(mem, size1, PROT_READ);
    ASSERT_EQ(0, ret) << strerror(errno);


    ucs_status_t status;
    ucs_rcache_region_t *r;

    ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_HOST;
    status = ucs_rcache_get(m_rcache, ptr2, size2, PROT_WRITE, &mem_type, &r);
    EXPECT_EQ(UCS_ERR_IO_ERROR, status);

    barrier();
    EXPECT_EQ(0u, m_reg_count);

    munmap(mem, size1+size2);
}

class test_rcache_with_limit : public test_rcache {
protected:
    virtual ucs_rcache_params_t rcache_params()
    {
        ucs_rcache_params_t params = test_rcache::rcache_params();
        params.max_regions         = 2;
        params.max_size            = 1000;
        params.alignment           = 16;
        return params;
    }

    uint32_t get_put(void *ptr, size_t size)
    {
        region *region = get(ptr, size);
        uint32_t id    = region->id;
        put(region);
        return id;
    }
};

UCS_TEST_F(test_rcache_with_limit, by_count) {
    static const size_t size = 32;

    /* First region will be added */
    void *ptr1          = malloc(size);
    uint32_t region1_id = get_put(ptr1, size);
    EXPECT_EQ(1, m_rcache.get()->num_regions);

    /* Second region will be added as well */
    void *ptr2          = malloc(size);
    uint32_t region2_id = get_put(ptr2, size);
    EXPECT_EQ(2, m_rcache.get()->num_regions);

    /* This time, something must be removed */
    void *ptr3          = malloc(size);
    uint32_t region3_id = get_put(ptr3, size);
    EXPECT_EQ(2, m_rcache.get()->num_regions);

    /* Second region should be kept by lru policy */
    uint32_t region2_new_id = get_put(ptr2, size);
    EXPECT_EQ(region2_id, region2_new_id);
    EXPECT_EQ(2, m_rcache.get()->num_regions);

    /* Third region should be also kept limit policy */
    uint32_t region3_new_id = get_put(ptr3, size);
    EXPECT_EQ(region3_new_id, region3_id);
    EXPECT_EQ(2, m_rcache.get()->num_regions);

    /* First region should be removed by lru policy */
    uint32_t region1_new_id = get_put(ptr1, size);
    EXPECT_NE(region1_new_id, region1_id);
    EXPECT_EQ(2, m_rcache.get()->num_regions);

    free(ptr3);
    free(ptr2);
    free(ptr1);
}

UCS_TEST_F(test_rcache_with_limit, by_size) {
    static const size_t size = 600;

    /* First region will be added */
    void *ptr1 = malloc(size);
    get_put(ptr1, size);
    EXPECT_EQ(1, m_rcache.get()->num_regions);

    /* Second region will cause removing of first region */
    void *ptr2 = malloc(size);
    get_put(ptr2, size);
    EXPECT_EQ(1, m_rcache.get()->num_regions);

    free(ptr2);
    free(ptr1);
}

UCS_TEST_F(test_rcache_with_limit, by_size_inuse) {
    static const size_t size = 600;

    /* First region will be added */
    void *ptr1      = malloc(size);
    region *region1 = get(ptr1, size);
    EXPECT_EQ(1, m_rcache.get()->num_regions);

    /* Second region will NOT cause removing of first region since it's still in
     * use */
    void *ptr2 = malloc(size);
    get_put(ptr2, size);
    EXPECT_EQ(2, m_rcache.get()->num_regions);

    put(region1);

    free(ptr2);
    free(ptr1);
}

#ifdef ENABLE_STATS
class test_rcache_stats : public test_rcache {
protected:

    virtual void init() {
        ucs_stats_cleanup();
        push_config();
        modify_config("STATS_DEST",    "file:/dev/null");
        modify_config("STATS_TRIGGER", "exit");
        ucs_stats_init();
        ASSERT_TRUE(ucs_stats_is_active());
        test_rcache::init();
    }

    virtual void cleanup() {
        test_rcache::cleanup();
        ucs_stats_cleanup();
        pop_config();
        ucs_stats_init();
    }

    int get_counter(int stat) {
        return (int)UCS_STATS_GET_COUNTER(m_rcache->stats, stat);
    }

    /* a helper function for stats tests debugging */
    void dump_stats() {
        printf("gets %d hf %d hs %d misses %d merges %d unmaps %d"
               " unmaps_inv %d puts %d regs %d deregs %d\n",
               get_counter(UCS_RCACHE_GETS),
               get_counter(UCS_RCACHE_HITS_FAST),
               get_counter(UCS_RCACHE_HITS_SLOW),
               get_counter(UCS_RCACHE_MISSES),
               get_counter(UCS_RCACHE_MERGES),
               get_counter(UCS_RCACHE_UNMAPS),
               get_counter(UCS_RCACHE_UNMAP_INVALIDATES),
               get_counter(UCS_RCACHE_PUTS),
               get_counter(UCS_RCACHE_REGS),
               get_counter(UCS_RCACHE_DEREGS));
    }
};

UCS_TEST_F(test_rcache_stats, basic) {
    static const size_t size = 4096;
    void *ptr = malloc(size);
    region *r1, *r2;

    r1 = get(ptr, size);
    EXPECT_EQ(1, get_counter(UCS_RCACHE_GETS));
    EXPECT_EQ(1, get_counter(UCS_RCACHE_MISSES));
    EXPECT_EQ(1, get_counter(UCS_RCACHE_REGS));

    r2 = get(ptr, size);
    EXPECT_EQ(2, get_counter(UCS_RCACHE_GETS));
    EXPECT_EQ(1, get_counter(UCS_RCACHE_HITS_FAST));
    EXPECT_EQ(1, get_counter(UCS_RCACHE_MISSES));

    put(r1);
    EXPECT_EQ(2, get_counter(UCS_RCACHE_GETS));
    EXPECT_EQ(1, get_counter(UCS_RCACHE_PUTS));

    put(r2);
    EXPECT_EQ(2, get_counter(UCS_RCACHE_GETS));
    EXPECT_EQ(2, get_counter(UCS_RCACHE_PUTS));

    free(ptr);
    EXPECT_EQ(2, get_counter(UCS_RCACHE_GETS));
    EXPECT_EQ(2, get_counter(UCS_RCACHE_PUTS));
    EXPECT_EQ(0, get_counter(UCS_RCACHE_DEREGS));
    EXPECT_EQ(0, get_counter(UCS_RCACHE_UNMAPS));
}

UCS_TEST_F(test_rcache_stats, unmap_dereg) {
    static const size_t size1 = 1024 * 1024;
    void *mem = alloc_pages(size1, PROT_READ|PROT_WRITE);
    region *r1;

    r1 = get(mem, size1);
    put(r1);

    /* Should generate umap event and invalidate the memory */
    munmap(mem, size1);
    EXPECT_EQ(1, get_counter(UCS_RCACHE_UNMAP_INVALIDATES));

    /* when doing another rcache operation, the region is actually destroyed */
    mem = alloc_pages(size1, PROT_READ|PROT_WRITE);
    r1 = get(mem, size1);
    put(r1);
    EXPECT_GE(get_counter(UCS_RCACHE_UNMAPS), 1);
    EXPECT_EQ(1, get_counter(UCS_RCACHE_DEREGS));

    /* cleanup */
    munmap(mem, size1);
}

UCS_TEST_F(test_rcache_stats, unmap_dereg_with_lock) {
    static const size_t size1 = 1024 * 1024;
    void *mem = alloc_pages(size1, PROT_READ|PROT_WRITE);
    region *r1;

    r1 = get(mem, size1);
    put(r1);

    /* Should generate umap event but no dereg or unmap invalidation.
     * We can have more unmap events if releasing the region structure triggers
     * releasing memory back to the OS.
     */
    pthread_rwlock_wrlock(&m_rcache->pgt_lock);
    munmap(mem, size1);
    pthread_rwlock_unlock(&m_rcache->pgt_lock);

    EXPECT_GE(get_counter(UCS_RCACHE_UNMAPS), 1);
    EXPECT_EQ(0, get_counter(UCS_RCACHE_UNMAP_INVALIDATES));
    EXPECT_EQ(0, get_counter(UCS_RCACHE_DEREGS));

    mem = alloc_pages(size1, PROT_READ|PROT_WRITE);

    /*
     * Adding a new region shall force a processing of invalidation queue and dereg
     */
    r1 = get(mem, size1);
    EXPECT_GE(get_counter(UCS_RCACHE_UNMAPS), 1);
    EXPECT_EQ(1, get_counter(UCS_RCACHE_UNMAP_INVALIDATES));
    EXPECT_EQ(1, get_counter(UCS_RCACHE_DEREGS));

    /* cleanup */
    put(r1);
    munmap(mem, size1);
}

UCS_TEST_F(test_rcache_stats, merge) {
    static const size_t size1 = 1024 * 1024;
    void *mem = alloc_pages(size1, PROT_READ|PROT_WRITE);
    region *r1, *r2;

    r1 = get(mem, 8192);
    /* should trigger merge of the two regions */
    r2 = get((char *)mem + 4096, 8192);
    EXPECT_EQ(1, get_counter(UCS_RCACHE_MERGES));

    EXPECT_EQ(2, get_counter(UCS_RCACHE_GETS));
    EXPECT_EQ(2, get_counter(UCS_RCACHE_MISSES));

    put(r1);
    put(r2);
    munmap(mem, size1);
}

UCS_TEST_F(test_rcache_stats, hits_slow) {
    static const size_t size1 = 1024 * 1024;
    region *r1, *r2;
    void *mem1, *mem2;

    mem1 = alloc_pages(size1, PROT_READ|PROT_WRITE);
    r1 = get(mem1, size1);
    put(r1);

    mem2 = alloc_pages(size1, PROT_READ|PROT_WRITE);
    r1 = get(mem2, size1);

    /* generate unmap event under lock, to roce using invalidation queue */
    pthread_rwlock_rdlock(&m_rcache->pgt_lock);
    munmap(mem1, size1);
    pthread_rwlock_unlock(&m_rcache->pgt_lock);

    EXPECT_EQ(1, get_counter(UCS_RCACHE_UNMAPS));

    EXPECT_EQ(2, get_counter(UCS_RCACHE_GETS));
    EXPECT_EQ(1, get_counter(UCS_RCACHE_PUTS));
    EXPECT_EQ(2, get_counter(UCS_RCACHE_MISSES));
    EXPECT_EQ(0, get_counter(UCS_RCACHE_UNMAP_INVALIDATES));
    EXPECT_EQ(0, get_counter(UCS_RCACHE_DEREGS));
    /* it should produce a slow hit because there is
     * a pending unmap event
     */
    r2 = get(mem2, size1);
    EXPECT_EQ(1, get_counter(UCS_RCACHE_HITS_SLOW));

    EXPECT_EQ(3, get_counter(UCS_RCACHE_GETS));
    EXPECT_EQ(1, get_counter(UCS_RCACHE_PUTS));
    EXPECT_EQ(2, get_counter(UCS_RCACHE_MISSES));
    EXPECT_EQ(1, get_counter(UCS_RCACHE_UNMAPS));
    /* unmap event processed */
    EXPECT_EQ(1, get_counter(UCS_RCACHE_UNMAP_INVALIDATES));
    EXPECT_EQ(1, get_counter(UCS_RCACHE_DEREGS));

    put(r1);
    put(r2);
    munmap(mem2, size1);
}
#endif


class test_rcache_pfn : public ucs::test {
public:
    void test_pfn(void *address, unsigned page_num)
    {
        pfn_enum_t ctx;
        ucs_status_t status;

        ctx.page_num = page_num;
        status       = ucs_sys_enum_pfn((uintptr_t)address,
                                        page_num, enum_pfn_cb, &ctx);
        ASSERT_UCS_OK(status);
        /* we expect that we got exact page_num PFN calls */
        ASSERT_EQ(page_num, ctx.page.size());
        ASSERT_EQ(page_num, ctx.pfn.size());
    }

protected:
    typedef std::set<unsigned> page_set_t;
    typedef std::set<unsigned long> pfn_set_t;
    typedef struct {
        unsigned   page_num;
        page_set_t page;
        pfn_set_t  pfn;
    } pfn_enum_t;

    static void enum_pfn_cb(unsigned page_num, unsigned long pfn, void *ctx)
    {
        pfn_enum_t *data = (pfn_enum_t*)ctx;

        EXPECT_LT(page_num, data->page_num);
        /* we expect that every page will have a unique page_num and a
         * unique PFN */
        EXPECT_EQ(data->pfn.end(), data->pfn.find(pfn));
        EXPECT_EQ(data->page.end(), data->page.find(page_num));
        data->pfn.insert(pfn);
        data->page.insert(page_num);
    }
};

UCS_TEST_F(test_rcache_pfn, enum_pfn) {
    const int MAX_PAGE_NUM = 1024 * 100; /* 400Mb max buffer */
    size_t page_size       = ucs_get_page_size();
    void *region;
    unsigned i;
    size_t len;
    unsigned long pfn;
    ucs_status_t status;

    /* stack page could not be mapped into zero region, if we get 0 here it
     * means the kernel does not provide PFNs */
    status = ucs_sys_get_pfn((uintptr_t)&pfn, 1, &pfn);
    ASSERT_UCS_OK(status);
    if (pfn == 0) {
        /* stack page could not be mapped into zero region */
        UCS_TEST_SKIP_R("PFN is not supported");
    }

    /* initialize stream here to avoid incorrect debug output */
    ucs::detail::message_stream ms("PAGES");

    for (i = 1; i < MAX_PAGE_NUM; i *= 2) {
        len = page_size * i;
        ms << i << " ";
        region = mmap(NULL, len, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        ASSERT_TRUE(region != MAP_FAILED);
        memset(region, 0, len); /* ensure that pages are mapped */
        /* test region aligned by page size */
        test_pfn(region, i);
        if (i > 1) { /* test pfn on mid-of-page address */
            test_pfn(UCS_PTR_BYTE_OFFSET(region, page_size / 2), i - 1);
        }

        munmap(region, len);
    }
}

INSTANTIATE_TEST_CASE_P(mem_type, test_rcache,
                        ::testing::ValuesIn(mem_buffer::supported_mem_types()));
