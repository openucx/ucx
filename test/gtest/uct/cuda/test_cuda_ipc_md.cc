/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024-2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <thread>
#include <vector>

#include <uct/test_md.h>
#include <cuda.h>

extern "C" {
#include <uct/cuda/cuda_ipc/cuda_ipc_md.h>
#include <uct/cuda/cuda_ipc/cuda_ipc_cache.h>
#include <uct/cuda/base/cuda_iface.h>
#include <uct/cuda/base/cuda_util.h>
#include <ucs/datastruct/pgtable.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/ptr_arith.h>
}

class test_cuda_ipc_md : public test_md {
protected:
    static uct_cuda_ipc_extended_rkey_t
    unpack_common(uct_md_h md, int64_t uuid, CUdeviceptr ptr, size_t size)
    {
        uct_cuda_ipc_extended_rkey_t rkey;
        uct_mem_h memh;
        EXPECT_UCS_OK(md->ops->mem_reg(md, (void *)ptr, size, NULL, &memh));
        EXPECT_UCS_OK(md->ops->mkey_pack(md, memh, (void *)ptr, size, NULL,
                                         &rkey));

        auto uuid64     = reinterpret_cast<int64_t*>(rkey.super.uuid.bytes);
        uuid64[0]       = uuid;
        uuid64[1]       = uuid;

        /* cuIpcOpenMemHandle used by cuda_ipc_cache does not allow to open
         * handle that was created by the same process */
        uct_rkey_unpack_params_t unpack_params = { 0 };
        EXPECT_EQ(UCS_ERR_UNREACHABLE,
                  uct_rkey_unpack_v2(md->component, &rkey, &unpack_params,
                                     NULL));

        uct_md_mem_dereg_params_t params;
        params.field_mask = UCT_MD_MEM_DEREG_FIELD_MEMH;
        params.memh       = memh;
        EXPECT_UCS_OK(md->ops->mem_dereg(md, &params));
        return rkey;
    }

    static uct_cuda_ipc_extended_rkey_t unpack(uct_md_h md, int64_t uuid)
    {
        CUdeviceptr ptr;
        EXPECT_EQ(CUDA_SUCCESS, cuMemAlloc(&ptr, 64));
        const uct_cuda_ipc_extended_rkey_t rkey = unpack_common(md, uuid, ptr,
                                                                64);
        EXPECT_EQ(CUDA_SUCCESS, cuMemFree(ptr));
        return rkey;
    }

#if HAVE_CUDA_FABRIC
    static void alloc_mempool(CUdeviceptr *ptr, CUmemoryPool *mpool,
                              CUstream *cu_stream, size_t size)
    {
        CUmemPoolProps pool_props = {};
        CUmemAccessDesc map_desc;
        CUdevice cu_device;

        EXPECT_EQ(CUDA_SUCCESS, cuCtxGetDevice(&cu_device));

        pool_props.allocType     = CU_MEM_ALLOCATION_TYPE_PINNED;
        pool_props.location.id   = (int)cu_device;
        pool_props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        pool_props.handleTypes   = CU_MEM_HANDLE_TYPE_FABRIC;
        pool_props.maxSize       = size;
        map_desc.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        map_desc.location        = pool_props.location;

        EXPECT_EQ(CUDA_SUCCESS,
                  cuStreamCreate(cu_stream, CU_STREAM_NON_BLOCKING));
        EXPECT_EQ(CUDA_SUCCESS, cuMemPoolCreate(mpool, &pool_props));
        EXPECT_EQ(CUDA_SUCCESS, cuMemPoolSetAccess(*mpool, &map_desc, 1));
        EXPECT_EQ(CUDA_SUCCESS,
                  cuMemAllocFromPoolAsync(ptr, size, *mpool, *cu_stream));
        EXPECT_EQ(CUDA_SUCCESS, cuStreamSynchronize(*cu_stream));
    }

    static void
    free_mempool(CUdeviceptr *ptr, CUmemoryPool *mpool, CUstream *cu_stream)
    {
        EXPECT_EQ(CUDA_SUCCESS, cuMemFree(*ptr));
        EXPECT_EQ(CUDA_SUCCESS, cuMemPoolDestroy(*mpool));
        EXPECT_EQ(CUDA_SUCCESS, cuStreamDestroy(*cu_stream));
    }

    static uct_cuda_ipc_extended_rkey_t unpack_masync(uct_md_h md, int64_t uuid)
    {
        size_t size = 4 * UCS_MBYTE;
        CUdeviceptr ptr;
        CUmemoryPool mpool;
        CUstream cu_stream;

        alloc_mempool(&ptr, &mpool, &cu_stream, size);
        const uct_cuda_ipc_extended_rkey_t rkey = unpack_common(md, uuid, ptr,
                                                                size);
        free_mempool(&ptr, &mpool, &cu_stream);
        return rkey;
    }

    struct vmm_multi_alloc_t {
        CUdeviceptr                  va_base;
        size_t                       total_size;
        size_t                       granularity;
        CUmemGenericAllocationHandle handles[8];
        unsigned                     num_chunks;
    };

    static CUresult
    alloc_vmm_multi(vmm_multi_alloc_t *alloc, unsigned num_chunks)
    {
        CUmemAllocationProp prop = {};
        CUmemAccessDesc access;
        CUdevice cu_dev;
        CUresult result;
        unsigned created_chunks;

        result = cuCtxGetDevice(&cu_dev);
        if (result != CUDA_SUCCESS) {
            return result;
        }

        prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id          = cu_dev;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

        result = cuMemGetAllocationGranularity(
                &alloc->granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (result != CUDA_SUCCESS) {
            return result;
        }

        alloc->num_chunks = num_chunks;
        alloc->total_size = num_chunks * alloc->granularity;

        result = cuMemAddressReserve(&alloc->va_base, alloc->total_size, 0, 0,
                                     0);
        if (result != CUDA_SUCCESS) {
            return result;
        }

        for (created_chunks = 0; created_chunks < num_chunks;
             created_chunks++) {
            result = cuMemCreate(&alloc->handles[created_chunks],
                                 alloc->granularity, &prop, 0);
            if (result != CUDA_SUCCESS) {
                goto err_cleanup;
            }

            result = cuMemMap(alloc->va_base +
                                      created_chunks * alloc->granularity,
                              alloc->granularity, 0,
                              alloc->handles[created_chunks], 0);
            if (result != CUDA_SUCCESS) {
                cuMemRelease(alloc->handles[created_chunks]);
                goto err_cleanup;
            }
        }

        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id   = cu_dev;
        access.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        result = cuMemSetAccess(alloc->va_base, alloc->total_size, &access, 1);
        if (result == CUDA_SUCCESS) {
            return CUDA_SUCCESS;
        }

    err_cleanup:
        for (unsigned i = 0; i < created_chunks; i++) {
            cuMemUnmap(alloc->va_base + i * alloc->granularity,
                       alloc->granularity);
            cuMemRelease(alloc->handles[i]);
        }
        cuMemAddressFree(alloc->va_base, alloc->total_size);
        return result;
    }

    static void free_vmm_multi(vmm_multi_alloc_t *alloc)
    {
        for (unsigned i = 0; i < alloc->num_chunks; i++) {
            EXPECT_EQ(CUDA_SUCCESS,
                      cuMemUnmap(alloc->va_base + i * alloc->granularity,
                                 alloc->granularity));
            EXPECT_EQ(CUDA_SUCCESS, cuMemRelease(alloc->handles[i]));
        }
        EXPECT_EQ(CUDA_SUCCESS,
                  cuMemAddressFree(alloc->va_base, alloc->total_size));
    }
#endif

    void test_mkey_pack_on_thread(void *ptr, size_t size)
    {
       uct_md_mem_reg_params_t reg_params  = {};
       uct_rkey_bundle_t       rkey_bundle = {};
       uct_mem_h memh;
       ASSERT_UCS_OK(uct_md_mem_reg_v2(md(), ptr, size, &reg_params, &memh));

       std::exception_ptr thread_exception;
       std::thread([&]() {
           try {
               uct_md_mkey_pack_params_t pack_params = {};
               std::vector<uint8_t> rkey(md_attr().rkey_packed_size);
               ASSERT_UCS_OK(uct_md_mkey_pack_v2(md(), memh, ptr, size,
                                                 &pack_params, rkey.data()));

               auto unpack_rkey = [&](const uct_rkey_unpack_params_t &unpack_params) {
                    ucs_status_t status = uct_rkey_unpack_v2(
                                             md()->component, rkey.data(),
                                             &unpack_params, &rkey_bundle);
                    ASSERT_TRUE((status == UCS_OK) ||
                                (status == UCS_ERR_UNREACHABLE));
                    if (status == UCS_OK) {
                        uct_rkey_release(md()->component, &rkey_bundle);
                    }
               };

               // No context and sys_dev is not provided
               // Reachable, because active CUDA context exists in main thread
               uct_rkey_unpack_params_t unpack_params = {};
               unpack_rkey(unpack_params);

               // No context and unknown sys_dev is provided
               // Reachable, because active CUDA context exists for some valid GPU
               unpack_params.field_mask = UCT_RKEY_UNPACK_FIELD_SYS_DEVICE;
               unpack_params.sys_device = UCS_SYS_DEVICE_ID_UNKNOWN;
               unpack_rkey(unpack_params);

               // No context and some valid sys_dev is provided
               ucs_sys_device_t sys_dev = uct_cuda_get_sys_dev(0);
               unpack_params.sys_device = sys_dev;
               unpack_rkey(unpack_params);
           } catch (...) {
               thread_exception = std::current_exception();
           }
       }).join();

       if (thread_exception) {
           std::rethrow_exception(thread_exception);
       }

       uct_md_mem_dereg_params_t dereg_params;
       dereg_params.field_mask = UCT_MD_MEM_DEREG_FIELD_MEMH;
       dereg_params.memh       = memh;
       EXPECT_UCS_OK(uct_md_mem_dereg_v2(md(), &dereg_params));
    }
};

UCS_TEST_P(test_cuda_ipc_md, mpack_legacy)
{
    constexpr size_t size = 4096;
    ucs::handle<uct_md_h> md;
    uct_mem_h memh;
    uct_cuda_ipc_extended_rkey_t rkey;
    CUdeviceptr ptr;

    UCS_TEST_CREATE_HANDLE(uct_md_h, md, uct_md_close, uct_md_open,
                           GetParam().component, GetParam().md_name.c_str(),
                           m_md_config);
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&ptr, size));
    EXPECT_UCS_OK(md->ops->mem_reg(md, (void *)ptr, size, NULL, &memh));
    EXPECT_UCS_OK(md->ops->mkey_pack(md, memh, (void *)ptr, size, NULL,
                                     &rkey));

    EXPECT_EQ(UCT_CUDA_IPC_KEY_HANDLE_TYPE_LEGACY, rkey.super.ph.handle_type);

    uct_md_mem_dereg_params_t params;
    params.field_mask = UCT_MD_MEM_DEREG_FIELD_MEMH;
    params.memh       = memh;
    EXPECT_UCS_OK(md->ops->mem_dereg(md, &params));
    cuMemFree(ptr);
}

UCS_TEST_P(test_cuda_ipc_md, mkey_pack_legacy)
{
    size_t size = 4 * UCS_MBYTE;
    CUdeviceptr ptr;

    EXPECT_EQ(CUDA_SUCCESS, cuMemAlloc(&ptr, size));
    test_mkey_pack_on_thread((void*)ptr, size);
    EXPECT_EQ(CUDA_SUCCESS, cuMemFree(ptr));
}

UCS_TEST_P(test_cuda_ipc_md, mkey_pack_mempool)
{
#if HAVE_CUDA_FABRIC
    int driver_version;
    EXPECT_EQ(CUDA_SUCCESS, cuDriverGetVersion(&driver_version));
    if (driver_version == 13000) {
        UCS_TEST_SKIP_R("in CUDA 13.0, calling cuMemPoolDestroy results in a "
                        "segmentation fault if "
                        "cuMemPoolExportToShareableHandle returned an error "
                        "before that");
    }

    size_t size = 4 * UCS_MBYTE;
    CUdeviceptr ptr;
    CUmemoryPool mpool;
    CUstream cu_stream;

    alloc_mempool(&ptr, &mpool, &cu_stream, size);
    test_mkey_pack_on_thread((void*)ptr, size);
    free_mempool(&ptr, &mpool, &cu_stream);
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}

UCS_TEST_P(test_cuda_ipc_md, mnnvl_disabled)
{
    /* Currently MNNVL is always disabled in CI */
    uct_cuda_ipc_md_t *cuda_ipc_md = ucs_derived_of(md(), uct_cuda_ipc_md_t);
    EXPECT_FALSE(cuda_ipc_md->enable_mnnvl);
}

UCS_TEST_P(test_cuda_ipc_md, mpack_vmm_multi)
{
#if HAVE_CUDA_FABRIC
    vmm_multi_alloc_t alloc = {};
    if (alloc_vmm_multi(&alloc, 4) != CUDA_SUCCESS) {
        UCS_TEST_SKIP_R("VMM allocation with fabric handle not supported");
    }

    uct_mem_h memh;
    uct_cuda_ipc_extended_rkey_t rkey;
    EXPECT_UCS_OK(md()->ops->mem_reg(md(), (void*)alloc.va_base,
                                     alloc.total_size, NULL, &memh));
    EXPECT_UCS_OK(md()->ops->mkey_pack(md(), memh, (void*)alloc.va_base,
                                       alloc.total_size, NULL, &rkey));

    EXPECT_EQ(UCT_CUDA_IPC_KEY_HANDLE_TYPE_VMM_MULTI,
              rkey.super.ph.handle_type);

    uct_md_mem_dereg_params_t params;
    params.field_mask = UCT_MD_MEM_DEREG_FIELD_MEMH;
    params.memh       = memh;
    EXPECT_UCS_OK(md()->ops->mem_dereg(md(), &params));

    free_vmm_multi(&alloc);
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}

UCS_TEST_P(test_cuda_ipc_md, mkey_pack_vmm_multi)
{
#if HAVE_CUDA_FABRIC
    vmm_multi_alloc_t alloc = {};
    if (alloc_vmm_multi(&alloc, 4) != CUDA_SUCCESS) {
        UCS_TEST_SKIP_R("VMM allocation with fabric handle not supported");
    }

    test_mkey_pack_on_thread((void*)alloc.va_base, alloc.total_size);

    free_vmm_multi(&alloc);
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}

_UCT_MD_INSTANTIATE_TEST_CASE(test_cuda_ipc_md, cuda_ipc);


class test_cuda_ipc_cache_lru : public ucs::test {
protected:
    static const size_t REGION_SIZE = UCS_MBYTE * 2;
    static const uintptr_t BASE_ADDR = 0x7f0000000000UL;

    virtual void init() {
        ucs::test::init();
        m_cache = NULL;
        /* Reset global limits to unlimited before each test */
        uct_cuda_ipc_cache_set_global_limits(ULONG_MAX, SIZE_MAX);
    }

    virtual void cleanup() {
        if (m_cache != NULL) {
            drain_cache();
            uct_cuda_ipc_destroy_cache(m_cache);
        }
        uct_cuda_ipc_cache_set_global_limits(ULONG_MAX, SIZE_MAX);
        ucs::test::cleanup();
    }

    void create_cache(unsigned long max_regions, size_t max_size) {
        uct_cuda_ipc_cache_set_global_limits(max_regions, max_size);
        m_max_regions = max_regions;
        m_max_size    = max_size;
        ASSERT_EQ(UCS_OK, uct_cuda_ipc_create_cache(&m_cache, "test_lru"));
    }

    uct_cuda_ipc_cache_region_t *insert_region(size_t index) {
        uct_cuda_ipc_cache_region_t *region;
        uintptr_t addr = BASE_ADDR + (index * REGION_SIZE * 2);
        int ret;

        ret = ucs_posix_memalign((void **)&region,
                                 ucs_max(sizeof(void *), UCS_PGT_ENTRY_MIN_ALIGN),
                                 sizeof(uct_cuda_ipc_cache_region_t),
                                 "test_cuda_ipc_cache_region");
        EXPECT_EQ(0, ret);

        region->super.start = ucs_align_down_pow2(addr, UCS_PGT_ADDR_ALIGN);
        region->super.end   = ucs_align_up_pow2(addr + REGION_SIZE,
                                                UCS_PGT_ADDR_ALIGN);
        memset(&region->key, 0, sizeof(region->key));
        region->key.b_len      = REGION_SIZE;
        region->key.d_bptr     = addr;
        region->key.ph.buffer_id = index;
        region->mapped_addr    = (void *)addr;
        region->refcount       = 1;
        region->cu_dev         = 0;
        region->in_lru         = 0;

        ucs_status_t status = ucs_pgtable_insert(&m_cache->pgtable,
                                                  &region->super);
        EXPECT_EQ(UCS_OK, status);

        m_cache->num_regions++;
        m_cache->total_size += REGION_SIZE;
        ucs_list_add_tail(&m_cache->lru_list, &region->lru_list);
        region->in_lru = 1;

        return region;
    }

    void release_region(uct_cuda_ipc_cache_region_t *region) {
        ASSERT_GE(region->refcount, 1UL);
        region->refcount--;
        if (!region->in_lru) {
            ucs_list_add_tail(&m_cache->lru_list, &region->lru_list);
            region->in_lru = 1;
        }
    }

    void reacquire_region(uct_cuda_ipc_cache_region_t *region) {
        /* Move to LRU tail (most recently used) */
        if (region->in_lru) {
            ucs_list_del(&region->lru_list);
        }
        ucs_list_add_tail(&m_cache->lru_list, &region->lru_list);
        region->in_lru = 1;
        region->refcount++;
    }

    void evict_lru() {
        uct_cuda_ipc_cache_region_t *region, *tmp;

        ucs_list_for_each_safe(region, tmp, &m_cache->lru_list, lru_list) {
            if ((m_cache->num_regions <= m_max_regions) &&
                (m_cache->total_size <= m_max_size)) {
                break;
            }

            if (region->refcount > 0) {
                /* In-use -- pull off LRU, will be re-added on release */
                ucs_list_del(&region->lru_list);
                region->in_lru = 0;
                continue;
            }

            ASSERT_EQ(UCS_OK, ucs_pgtable_remove(&m_cache->pgtable,
                                                  &region->super));
            ucs_list_del(&region->lru_list);
            region->in_lru = 0;
            m_cache->num_regions--;
            m_cache->total_size -= region->key.b_len;

            ucs_free(region);
        }
    }

    static void collect_region_cb(const ucs_pgtable_t *pgtable,
                                  ucs_pgt_region_t *pgt_region, void *arg) {
        ucs_list_link_t *list = (ucs_list_link_t *)arg;
        uct_cuda_ipc_cache_region_t *region =
                ucs_derived_of(pgt_region, uct_cuda_ipc_cache_region_t);
        ucs_list_add_tail(list, &region->list);
    }

    void drain_cache() {
        ucs_list_link_t region_list;
        uct_cuda_ipc_cache_region_t *region, *tmp;

        ucs_list_head_init(&region_list);
        ucs_pgtable_purge(&m_cache->pgtable, collect_region_cb, &region_list);
        ucs_list_for_each_safe(region, tmp, &region_list, list) {
            ucs_free(region);
        }

        ucs_list_head_init(&m_cache->lru_list);
        m_cache->num_regions = 0;
        m_cache->total_size  = 0;
    }

    bool pgtable_has(size_t index) {
        uintptr_t addr = BASE_ADDR + (index * REGION_SIZE * 2);
        return ucs_pgtable_lookup(&m_cache->pgtable, addr) != NULL;
    }

    uct_cuda_ipc_cache_t *m_cache;
    unsigned long         m_max_regions;
    size_t                m_max_size;
};

const size_t    test_cuda_ipc_cache_lru::REGION_SIZE;
const uintptr_t test_cuda_ipc_cache_lru::BASE_ADDR;

UCS_TEST_F(test_cuda_ipc_cache_lru, evict_by_count) {
    const unsigned long max_regions = 128;
    const size_t num_insert         = 192;

    create_cache(max_regions, SIZE_MAX);

    std::vector<uct_cuda_ipc_cache_region_t *> regions(num_insert);
    for (size_t i = 0; i < num_insert; i++) {
        regions[i] = insert_region(i);
    }

    for (size_t i = 0; i < num_insert; i++) {
        release_region(regions[i]);
    }

    EXPECT_EQ(num_insert, m_cache->num_regions);

    evict_lru();

    EXPECT_EQ(max_regions, m_cache->num_regions);
    EXPECT_EQ(max_regions * REGION_SIZE, m_cache->total_size);

    for (size_t i = 0; i < num_insert - max_regions; i++) {
        EXPECT_FALSE(pgtable_has(i));
    }
    for (size_t i = num_insert - max_regions; i < num_insert; i++) {
        EXPECT_TRUE(pgtable_has(i));
    }
}

UCS_TEST_F(test_cuda_ipc_cache_lru, evict_by_size) {
    const size_t max_size   = REGION_SIZE * 64;
    const size_t num_insert = 100;

    create_cache(ULONG_MAX, max_size);

    std::vector<uct_cuda_ipc_cache_region_t *> regions(num_insert);
    for (size_t i = 0; i < num_insert; i++) {
        regions[i] = insert_region(i);
    }

    for (size_t i = 0; i < num_insert; i++) {
        release_region(regions[i]);
    }

    EXPECT_EQ(num_insert, m_cache->num_regions);
    EXPECT_EQ(num_insert * REGION_SIZE, m_cache->total_size);

    evict_lru();

    size_t expected_regions = max_size / REGION_SIZE;
    EXPECT_EQ(expected_regions, m_cache->num_regions);
    EXPECT_EQ(expected_regions * REGION_SIZE, m_cache->total_size);

    for (size_t i = 0; i < num_insert - expected_regions; i++) {
        EXPECT_FALSE(pgtable_has(i));
    }
    for (size_t i = num_insert - expected_regions; i < num_insert; i++) {
        EXPECT_TRUE(pgtable_has(i));
    }
}

UCS_TEST_F(test_cuda_ipc_cache_lru, no_evict_in_use) {
    const unsigned long max_regions = 64;
    const size_t num_insert         = 128;

    create_cache(max_regions, SIZE_MAX);

    std::vector<uct_cuda_ipc_cache_region_t *> regions(num_insert);
    for (size_t i = 0; i < num_insert; i++) {
        regions[i] = insert_region(i);
    }

    /* Release only the first half -- the second half stays in-use */
    for (size_t i = 0; i < num_insert / 2; i++) {
        release_region(regions[i]);
    }

    evict_lru();

    /* Only released regions can be evicted; in-use ones remain */
    EXPECT_EQ(max_regions, m_cache->num_regions);

    for (size_t i = 0; i < num_insert / 2; i++) {
        EXPECT_FALSE(pgtable_has(i));
    }
    for (size_t i = num_insert / 2; i < num_insert; i++) {
        EXPECT_TRUE(pgtable_has(i));
    }
}

UCS_TEST_F(test_cuda_ipc_cache_lru, lru_order) {
    const unsigned long max_regions = 128;
    const size_t num_insert         = 256;

    create_cache(max_regions, SIZE_MAX);

    std::vector<uct_cuda_ipc_cache_region_t *> regions(num_insert);
    for (size_t i = 0; i < num_insert; i++) {
        regions[i] = insert_region(i);
    }

    for (size_t i = 0; i < num_insert; i++) {
        release_region(regions[i]);
    }

    /*
     * Reacquire then release the first 64 regions, moving them to
     * the tail of the LRU (most recently used). The eviction should
     * then remove regions [64..191] and keep [0..63] + [192..255].
     */
    for (size_t i = 0; i < 64; i++) {
        reacquire_region(regions[i]);
        release_region(regions[i]);
    }

    evict_lru();

    EXPECT_EQ(max_regions, m_cache->num_regions);

    for (size_t i = 0; i < 64; i++) {
        EXPECT_TRUE(pgtable_has(i));
    }
    for (size_t i = 64; i < num_insert - 64; i++) {
        EXPECT_FALSE(pgtable_has(i));
    }
    for (size_t i = num_insert - 64; i < num_insert; i++) {
        EXPECT_TRUE(pgtable_has(i));
    }
}

UCS_TEST_F(test_cuda_ipc_cache_lru, unlimited) {
    const size_t num_insert = 512;

    create_cache(ULONG_MAX, SIZE_MAX);

    for (size_t i = 0; i < num_insert; i++) {
        uct_cuda_ipc_cache_region_t *r = insert_region(i);
        release_region(r);
    }

    evict_lru();

    EXPECT_EQ(num_insert, m_cache->num_regions);
    EXPECT_EQ(num_insert * REGION_SIZE, m_cache->total_size);
    for (size_t i = 0; i < num_insert; i++) {
        EXPECT_TRUE(pgtable_has(i));
    }
}
