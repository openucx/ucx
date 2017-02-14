/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
extern "C" {
#include <ucs/datastruct/pgtable.h>
#include <ucs/time/time.h>
}
#include <algorithm>
#include <limits.h>
#include <vector>
#include <set>


class test_pgtable : public ucs::test {
protected:

    typedef std::vector<ucs_pgt_region_t*> search_result_t;

    virtual void init() {
        ucs::test::init();
        ucs_status_t status = ucs_pgtable_init(&m_pgtable, pgd_alloc, pgd_free);
        ASSERT_UCS_OK(status);
    }

    virtual void cleanup() {
        ucs_pgtable_cleanup(&m_pgtable);
        ucs::test::cleanup();
    }

    void insert(ucs_pgt_region_t *region, ucs_status_t exp_status = UCS_OK,
                const std::string& message = "")
    {
        ucs_status_t status = ucs_pgtable_insert(&m_pgtable, region);
        if (exp_status == UCS_OK) {
            ASSERT_UCS_OK(status, << " inserting 0x" << std::hex <<
                                     region->start << "..0x" <<region->end);
        } else {
            EXPECT_EQ(exp_status, status) << message;
        }
    }

    void remove(ucs_pgt_region_t *region, ucs_status_t exp_status = UCS_OK,
                const std::string& message = "")
    {
        ucs_status_t status = ucs_pgtable_remove(&m_pgtable, region);
        if (exp_status == UCS_OK) {
            ASSERT_UCS_OK(status);
        } else {
            EXPECT_EQ(exp_status, status) << message;
        }
    }

    ucs_pgt_region_t *lookup(ucs_pgt_addr_t address) {
        return ucs_pgtable_lookup(&m_pgtable, address);
    }

    unsigned num_regions() {
        return ucs_pgtable_num_regions(&m_pgtable);
    }

    void dump() {
        ucs_pgtable_dump(&m_pgtable, UCS_LOG_LEVEL_DEBUG);
    }

    void purge() {
        ucs_pgtable_purge(&m_pgtable, pgd_purge_cb, reinterpret_cast<void*>(this));
    }

    search_result_t search(ucs_pgt_addr_t from, ucs_pgt_addr_t to)
    {
        search_result_t result;
        ucs_pgtable_search_range(&m_pgtable, from, to, pgd_search_cb,
                                 reinterpret_cast<void*>(&result));
        return result;
    }

    static ucs_pgt_region_t* make_region(ucs_pgt_addr_t start, ucs_pgt_addr_t end) {
        ucs_pgt_region_t r = {start, end};
        return new ucs_pgt_region_t(r);
    }

    static bool is_overlap(const ucs_pgt_region_t *region, ucs_pgt_addr_t from,
                           ucs_pgt_addr_t to)
    {
        return ucs_max(from, region->start) <= ucs_min(to, region->end);
    }

    static unsigned count_overlap(const ucs::ptr_vector<ucs_pgt_region_t>& regions,
                                  ucs_pgt_addr_t from, ucs_pgt_addr_t to)
    {
        unsigned count = 0;
        for (ucs::ptr_vector<ucs_pgt_region_t>::const_iterator iter = regions.begin();
                        iter != regions.end(); ++iter)
        {
            if (is_overlap(*iter, from, to)) {
                ++count;
            }
        }
        return count;
    }

private:
    static ucs_pgt_dir_t *pgd_alloc(const ucs_pgtable_t *pgtable) {
        return new ucs_pgt_dir_t;
    }

    static void pgd_free(const ucs_pgtable_t *pgtable, ucs_pgt_dir_t *pgdir) {
        delete pgdir;
    }

    static void pgd_purge_cb(const ucs_pgtable_t *pgtable,
                             ucs_pgt_region_t *region, void *arg) {
    }

    static void pgd_search_cb(const ucs_pgtable_t *pgtable,
                              ucs_pgt_region_t *region, void *arg)
    {
        search_result_t *result = reinterpret_cast<search_result_t*>(arg);
        result->push_back(region);
    }

protected:
    ucs_pgtable_t m_pgtable;
};


UCS_TEST_F(test_pgtable, basic) {
    ucs_pgt_region_t region;

    region.start = 0x400800;
    region.end   = 0x403400;
    insert(&region);

    dump();

    EXPECT_EQ(&region,  lookup(0x400800));
    EXPECT_EQ(&region,  lookup(0x402020));
    EXPECT_EQ(&region,  lookup(0x4033ff));
    EXPECT_TRUE(NULL == lookup(0x403400));
    EXPECT_TRUE(NULL == lookup(0x0));
    EXPECT_TRUE(NULL == lookup(-1));
    EXPECT_EQ(1u,       num_regions());

    remove(&region);

    insert(&region);

    dump();

    purge(); /* region goes out of scope so we must remove it */
}

UCS_TEST_F(test_pgtable, lookup_adjacent) {
    ucs_pgt_region_t region1 = {0xc500000, 0xc500400};
    ucs_pgt_region_t region2 = {0xc500400, 0xc500800};
    insert(&region1);
    insert(&region2);
    dump();
    EXPECT_EQ(&region2, lookup(0xc500400));
    EXPECT_EQ(&region1, lookup(0xc500000));
    purge();
}

UCS_TEST_F(test_pgtable, multi_search) {
    for (int count = 0; count < 10; ++count) {
        ucs::ptr_vector<ucs_pgt_region_t> regions;
        ucs_pgt_addr_t min = ULONG_MAX;
        ucs_pgt_addr_t max = 0;

        /* generate random regions */
        unsigned num_regions = 0;
        for (int i = 0; i < 200 / ucs::test_time_multiplier(); ++i) {
            ucs_pgt_addr_t start = (ucs::rand() & 0x7fffffff) << 24;
            size_t         size  = ucs_min((size_t)ucs::rand(), ULONG_MAX - start);
            ucs_pgt_addr_t end   = start + ucs_align_down(size, UCS_PGT_ADDR_ALIGN);
            if (count_overlap(regions, start, end)) {
                /* Make sure regions do not overlap */
                continue;
            }

            min = ucs_min(start, min);
            max = ucs_max(start, max);
            regions.push_back(make_region(start, end));
            ++num_regions;
        }

        /* Insert regions */
        for (ucs::ptr_vector<ucs_pgt_region_t>::const_iterator iter = regions.begin();
             iter != regions.end(); ++iter)
        {
            insert(*iter);
        }

        /* Count how many fall in the [1/4, 3/4] range */
        ucs_pgt_addr_t from = ((min * 90) + (max * 10)) / 100;
        ucs_pgt_addr_t to   = ((min * 10) + (max * 90)) / 100;
        unsigned num_in_range = count_overlap(regions, from, to);

        /* Search in page table */
        search_result_t result = search(from, to);
        UCS_TEST_MESSAGE << "found " << result.size() << "/" << num_in_range <<
                            " regions in the range 0x" << std::hex << from <<
                            "..0x" << to << std::dec;
        EXPECT_EQ(num_in_range, result.size());

        purge();
    }
}

UCS_TEST_F(test_pgtable, invalid_param) {
    if (UCS_PGT_ADDR_ALIGN == 1) {
        UCS_TEST_SKIP;
    }

    ucs_pgt_region_t region1 = {0x4000, 0x4001};
    insert(&region1, UCS_ERR_INVALID_PARAM);

    ucs_pgt_region_t region2 = {0x4001, 0x5000};
    insert(&region2, UCS_ERR_INVALID_PARAM);

    ucs_pgt_region_t region3 = {0x5000, 0x4000};
    insert(&region3, UCS_ERR_INVALID_PARAM);
}

UCS_TEST_F(test_pgtable, overlap_insert) {
    ucs_pgt_region_t region1 = {0x4000, 0x6000};
    insert(&region1);

    ucs_pgt_region_t region2 = {0x5000, 0x7000};
    insert(&region2, UCS_ERR_ALREADY_EXISTS, "overlap");

    ucs_pgt_region_t region3 = {0x3000, 0x5000};
    insert(&region3, UCS_ERR_ALREADY_EXISTS, "overlap");

    remove(&region1);
}

UCS_TEST_F(test_pgtable, nonexist_remove) {
    ucs_pgt_region_t region1 = {0x4000, 0x6000};
    remove(&region1, UCS_ERR_NO_ELEM);

    ucs_pgt_region_t region2 = {0x5000, 0x7000};
    insert(&region2);

    remove(&region1, UCS_ERR_NO_ELEM);

    region1.start = 0x5000;
    region1.end   = 0x5000;
    remove(&region1, UCS_ERR_NO_ELEM);

    region1 = region2;
    remove(&region1, UCS_ERR_NO_ELEM); /* Fail - should be pointer-equal */

    remove(&region2);
}

class test_pgtable_perf : public test_pgtable {
protected:

    void insert(ucs_pgt_region_t *region) {
        /* Insert to both */
        test_pgtable::insert(region);
        m_stl_pgt.insert(region);
    }

    void purge() {
        test_pgtable::purge();
        m_stl_pgt.clear();
    }

    ucs_pgt_region_t* lookup_in_stl(ucs_pgt_addr_t address) {
        ucs_pgt_region_t search = {address, address + 1};
        stl_pgtable_t::iterator iter = m_stl_pgt.lower_bound(&search);
        if (iter == m_stl_pgt.end()) {
            return NULL;
        } else {
            ucs_pgt_region_t *region = *iter;
            ucs_assertv(address < region->end,
                        "address=0x%lx region 0x%lx..0x%lx", address,
                        region->start, region->end);
            return (address >= region->start) ? region : NULL;
        }
    }

    ucs_pgt_region_t* lookup_in_pgt(ucs_pgt_addr_t address) {
        return test_pgtable::lookup(address);
    }

    void measure_workload(ucs_pgt_addr_t max_addr,
                          size_t block_size,   /* Basic block size */
                          unsigned blocks_per_superblock, /* Number of consecutive basic blocks per big block */
                          unsigned num_superblocks, /* Number of big blocks */
                          unsigned num_lookups, /* How many lookups to generate */
                          bool random_access, /* Whether access pattern is random or ordered */
                          double hit_ratio) /* Probability of lookup hit */
    {
        block_size = ucs_align_up_pow2(block_size, UCS_PGT_ADDR_ALIGN);

        const size_t superblock_size = block_size * blocks_per_superblock;
        const size_t max_start = max_addr - superblock_size;
        ucs::ptr_vector<ucs_pgt_region_t> regions;
        std::vector<ucs_pgt_addr_t> lookups;
        lookups.clear();

        /* Generate random superblocks */
        ucs_pgt_addr_t start = 0;
        std::vector<ucs_pgt_addr_t> superblocks;
        for (unsigned i = 0; i < num_superblocks; ++i) {
            ucs_pgt_addr_t addr = random_address(start, max_start);
            superblocks.push_back(addr);
            start = addr + superblock_size * 2; /* minimal gap */
            if (start >= max_start) {
                break;
            }
        }

        num_superblocks = superblocks.size();

        /* Insert them */
        for (unsigned i = 0; i < num_superblocks; ++i) {
            for (unsigned j = 0; j < blocks_per_superblock; ++j) {
                ucs_pgt_region_t *region = new ucs_pgt_region_t;
                region->start = superblocks[i] + (j * block_size);
                region->end =   region->start + block_size;
                regions.push_back(region);
                insert(region);
            }
        }

        EXPECT_EQ(num_superblocks * blocks_per_superblock, num_regions());

        /* Create workload */
        unsigned sb_idx = 0;
        unsigned block_idx = 0;
        for (unsigned n = 0; n < num_lookups; ++n) {
            ucs_pgt_addr_t addr = superblocks[sb_idx] + block_idx * block_size;
            if (ucs::rand() > (RAND_MAX * hit_ratio)) {
                addr += superblock_size; /* make it miss by falling to inter-block gap */
            }
            lookups.push_back(addr);
            if (random_access) {
                sb_idx    = ucs::rand() % num_superblocks;
                block_idx = ucs::rand() % blocks_per_superblock;
            } else {
                block_idx = (block_idx + 1) % blocks_per_superblock;
                if (block_idx == 0)
                    sb_idx = (sb_idx + 1) % num_superblocks;
            }
        }

        invalidate_cache();

        std::pair<ucs_time_t, unsigned> result_stl =
                        measure(lookups, true);

        invalidate_cache();

        std::pair<ucs_time_t, unsigned> result_pgt =
                        measure(lookups, false);

        EXPECT_EQ(result_stl.second, result_pgt.second);

        UCS_TEST_MESSAGE << std::dec << num_superblocks << " areas of " <<
                        blocks_per_superblock << "x" << block_size << " bytes, " <<
                        (random_access ? "random" : "ordered") << ": " <<
                        "stl: " << (ucs_time_to_nsec(result_stl.first) / num_lookups) << " ns, "
                        "ucs: " << (ucs_time_to_nsec(result_pgt.first) / num_lookups) << " ns " <<
                        (result_pgt.second * 100) / lookups.size() << "% hit"
                        ;
        purge();
    }

private:
    struct region_comparator {
        bool operator()(ucs_pgt_region_t* region1, ucs_pgt_region_t* region2) {
            return region1->end <= region2->start;
        }
    };

    typedef std::set<ucs_pgt_region_t*, region_comparator> stl_pgtable_t;

    std::pair<ucs_time_t, unsigned>
    inline measure(const std::vector<ucs_pgt_addr_t>& lookups, bool use_stl)
    {
        unsigned hit_count = 0;

        ucs_time_t start_time = ucs_get_time();
        ucs_compiler_fence();
        for (std::vector<ucs_pgt_addr_t>::const_iterator iter = lookups.begin();
                       iter != lookups.end(); ++iter)
        {
            ucs_pgt_region_t *region =
                            use_stl ? lookup_in_stl(*iter) : lookup_in_pgt(*iter);
            if (region != NULL) {
               ++hit_count;
            }
        }
        ucs_compiler_fence();
        return std::make_pair(ucs_get_time() - start_time, hit_count);
    }

    ucs_pgt_addr_t random_address(ucs_pgt_addr_t start, ucs_pgt_addr_t max) {
        ucs_pgt_addr_t r = (ucs_pgt_addr_t)ucs::rand() * (max / 1000) / RAND_MAX;
        return ucs_align_up_pow2((r % (max - start)) + start,
                                 UCS_PGT_ADDR_ALIGN);
    }

    void invalidate_cache() {
        size_t size = 30 * 1024 * 1024;
        void *ptr = malloc(size);
        memset(ptr, 0xbb, size);
        free(ptr);
    }

    stl_pgtable_t m_stl_pgt;
};

/*
 * Compare out lookup performance to STL's
 */
UCS_TEST_F(test_pgtable_perf, basic) {
    ucs_pgt_region_t region = {0x4000, 0x5000};
    insert(&region);
    EXPECT_EQ(&region, lookup_in_stl(0x4500));
    EXPECT_EQ(&region, lookup_in_stl(0x4000));
    EXPECT_EQ(&region, lookup_in_pgt(0x4500));
    EXPECT_TRUE(NULL == lookup_in_stl(0x5000));
    purge();
}

UCS_TEST_F(test_pgtable_perf, workloads) {
    if (ucs::test_time_multiplier() != 1) {
        UCS_TEST_SKIP;
    }

    measure_workload(UCS_MASK(28),
                     1024,
                     10000,
                     20,
                     5000000,
                     false,
                     0.8);
    measure_workload(UCS_MASK(28),
                     1024,
                     10000,
                     20,
                     500000,
                     true,
                     0.8);
    measure_workload(UCS_MASK(28),
                     1024,
                     10000,
                     2,
                     10000000,
                     false,
                     0.8);
    measure_workload(UCS_MASK(28),
                     1024 * 256,
                     1,
                     4,
                     10000000,
                     false,
                     0.8);
}

