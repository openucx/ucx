/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2019. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <infiniband/verbs.h>
extern "C" {
#include <uct/ib/mlx5/ib_mlx5.h>
}
#include <uct/api/uct.h>
#include <uct/uct_test.h>
#include <common/test.h>

class test_devx : public uct_test {
public:
    entity* m_e;

    void init() {
        uct_test::init();

        m_e = create_entity(0);
        m_entities.push_back(m_e);

        if (!(md()->super.dev.flags & UCT_IB_DEVICE_FLAG_MLX5_PRM &&
              md()->flags & UCT_IB_MLX5_MD_FLAG_DEVX)) {
            std::stringstream ss;
            ss << "DEVX is not supported by " << GetParam();
            UCS_TEST_SKIP_R(ss.str());
        }
    }

    uct_ib_mlx5_md_t *md() const {
        return ucs_derived_of(m_e->md(), uct_ib_mlx5_md_t);
    }

    uct_priv_worker_t *worker() const {
        return ucs_derived_of(m_e->worker(), uct_priv_worker_t);
    }
};

UCS_TEST_P(test_devx, dbrec)
{
    uct_ib_mlx5_dbrec_t *dbrec;

    dbrec = (uct_ib_mlx5_dbrec_t *)ucs_mpool_get_inline(&md()->dbrec_pool);
    ASSERT_FALSE(dbrec == NULL);
    ucs_mpool_put_inline(dbrec);
}

UCT_INSTANTIATE_IB_TEST_CASE(test_devx);


class test_devx_umr_mkey : public test_devx {
public:
    void init() {
        test_devx::init();
    }

    bool check_xgvmi() const {
        if (md()->flags & UCT_IB_MLX5_MD_FLAG_INDIRECT_XGVMI) {
            uct_ib_mlx5_devx_mem_t *memh = create_memh(1);
            ASSERT_UCS_OK(uct_ib_mlx5_devx_reg_exported_key(md(), memh));
            /* XGVMI capability might be dropped by this point! */
            destroy_memh(memh);
        }

        return md()->flags & UCT_IB_MLX5_MD_FLAG_INDIRECT_XGVMI;
    }

    void skip_no_xgvmi() const {
        if (!check_xgvmi()) {
            UCS_TEST_SKIP_R("XGVMI capability is not supported");
        }
    }

    uct_ib_mlx5_devx_mem_t *create_memh(size_t length) const {
        void *addr                     = ucs_malloc(length, "test");
        uct_md_mem_reg_params_t params = {};
        uct_mem_h memh;

        EXPECT_NE(nullptr, addr);
        ASSERT_UCS_OK(uct_ib_mlx5_devx_mem_reg(m_e->md(), addr, length, &params,
                                               &memh));
        return ucs_derived_of(memh, uct_ib_mlx5_devx_mem_t);
    }

    uct_ib_mlx5_devx_mem_t *
    import_memh(const uct_ib_mlx5_devx_mem_t *exported_memh) const {
        uct_ib_md_packed_mkey_t packed_mkey = {
            .lkey    = exported_memh->exported_lkey,
            .vhca_id = md()->super.vhca_id
        };

        uct_md_mem_attach_params_t params = {};
        uct_ib_mlx5_devx_mem_t *imported_memh;
        ASSERT_UCS_OK(uct_ib_mlx5_devx_mem_attach(m_e->md(), &packed_mkey,
                                                  &params,
                                                  (uct_mem_h*)&imported_memh));
        return imported_memh;
    }

    void destroy_memh(uct_ib_mlx5_devx_mem_t *memh) const {
        void *addr = memh->address;
        uct_md_mem_dereg_params_t params;

        params.field_mask = UCT_MD_MEM_DEREG_FIELD_MEMH;
        params.memh       = (uct_mem_h)memh;

        ASSERT_UCS_OK(uct_ib_mlx5_devx_mem_dereg(m_e->md(), &params));
        ucs_free(addr);
    }

    void check_umr_init(bool init_export, bool init_import) const {
        EXPECT_EQ(init_export, nullptr != md()->umr.qp);
        EXPECT_EQ(init_export, nullptr != md()->umr.cq);
        if (!init_export) {
            EXPECT_TRUE(ucs_list_is_empty(&md()->umr.mkey_pool));
        }

        EXPECT_EQ(init_import, nullptr != md()->umr.mkey_hash);
    }

    static void check_memh_export(uct_ib_mlx5_devx_mem_t *memh, bool is_umr) {
        EXPECT_EQ(is_umr,
                  UCT_IB_MLX5_MKEY_TAG_UMR == (memh->exported_lkey & 0xff));
        EXPECT_EQ(is_umr, nullptr != memh->exported_umr_mkey);
        EXPECT_EQ(is_umr, nullptr == memh->cross_mr);
    }

    void perf_export_import_mkey(size_t memh_count, size_t memh_size,
                                 ucs_time_t &export_time,
                                 ucs_time_t &import_time) {
        std::vector<uct_ib_mlx5_devx_mem_t*> exported_memh, imported_memh;
        exported_memh.reserve(memh_count);
        imported_memh.reserve(memh_count);

        for (size_t i = 0; i < memh_count; ++i) {
            exported_memh.push_back(create_memh(memh_size));
        }

        ucs_time_t start_time = ucs_get_time();
        for (auto it : exported_memh) {
            ASSERT_UCS_OK(uct_ib_mlx5_devx_reg_exported_key(md(), it));
            check_memh_export(it, true);
        }
        export_time = ucs_get_time() - start_time;

        start_time = ucs_get_time();
        for (auto it : exported_memh) {
            imported_memh.push_back(import_memh(it));
        }
        import_time = ucs_get_time() - start_time;

        for (size_t i = 0; i < memh_count; ++i) {
            destroy_memh(imported_memh[i]);
            destroy_memh(exported_memh[i]);
        }
    }
};

UCS_TEST_P(test_devx_umr_mkey, lazy_init)
{
    check_umr_init(false, false);
}

UCS_TEST_P(test_devx_umr_mkey, export_mkey_no_xgvmi, "IB_XGVMI_UMR_ENABLE=n")
{
    skip_no_xgvmi();

    uct_ib_mlx5_devx_mem_t *memh = create_memh(1024);

    ASSERT_UCS_OK(uct_ib_mlx5_devx_reg_exported_key(md(), memh));

    /* UMR objects remains uninitialized, exported key is not UMR one */
    check_umr_init(false, false);
    check_memh_export(memh, false);
    destroy_memh(memh);
}

UCS_TEST_P(test_devx_umr_mkey, export_mkey_xgvmi)
{
    skip_no_xgvmi();

    uct_ib_mlx5_devx_mem_t *memh = create_memh(1024);
    ASSERT_UCS_OK(uct_ib_mlx5_devx_reg_exported_key(md(), memh));
    /* XGVMI capability might be dropped by this point! */

    /* UMR export transport objects must be initialized */
    check_umr_init(true, false);
    /* UMR mkey pool must be empty */
    EXPECT_TRUE(ucs_list_is_empty(&md()->umr.mkey_pool));
    /* Whether mkey is UMR depends on XGVMI capability */
    check_memh_export(memh, true);

    destroy_memh(memh);

    /* After release of memh, associated UMR mkey is moved back to mkey pool */
    EXPECT_EQ(1, ucs_list_length(&md()->umr.mkey_pool));
}

UCS_TEST_P(test_devx_umr_mkey, import_mkey_no_umr, "IB_XGVMI_UMR_ENABLE=n")
{
    skip_no_xgvmi();

    /* Export any mkey */
    uct_ib_mlx5_devx_mem_t *exported_memh = create_memh(1024);
    ASSERT_UCS_OK(uct_ib_mlx5_devx_reg_exported_key(md(), exported_memh));

    /* Import as non-UMR mkey */
    uct_ib_mlx5_devx_mem_t *imported_memh = import_memh(exported_memh);
    EXPECT_NE(nullptr, imported_memh);

    /* UMR hash map remains uninitialized */
    EXPECT_EQ(nullptr, md()->umr.mkey_hash);

    destroy_memh(imported_memh);
    destroy_memh(exported_memh);
}

UCS_TEST_P(test_devx_umr_mkey, import_mkey_umr)
{
    skip_no_xgvmi();

    /* Export UMR mkey */
    uct_ib_mlx5_devx_mem_t *exported_memh = create_memh(1024);
    ASSERT_UCS_OK(uct_ib_mlx5_devx_reg_exported_key(md(), exported_memh));
    check_memh_export(exported_memh, true);

    /* Import first UMR mkey */
    uct_ib_mlx5_devx_mem_t *imported_memh = import_memh(exported_memh);

    check_umr_init(true, true);
    EXPECT_EQ(1, kh_size(md()->umr.mkey_hash));

    /* Second import with the same packed mkey, should reuse alias */
    uct_ib_mlx5_devx_mem_t *imported_memh2 = import_memh(exported_memh);
    EXPECT_NE(imported_memh, imported_memh2);
    EXPECT_EQ(imported_memh->super.lkey, imported_memh2->super.lkey);
    EXPECT_EQ(1, kh_size(md()->umr.mkey_hash));

    destroy_memh(imported_memh);
    destroy_memh(imported_memh2);
    destroy_memh(exported_memh);
}

UCS_TEST_P(test_devx_umr_mkey, perf_export_import_mkey_xgvmi)
{
    skip_no_xgvmi();

    const size_t MEMH_COUNT = RUNNING_ON_VALGRIND ? 100 : 1000;
    ucs_time_t export_create_time, import_create_time;
    perf_export_import_mkey(MEMH_COUNT, 1024, export_create_time,
                            import_create_time);

    /* After release all allocated UMR keys go back to the mkey pool */
    EXPECT_EQ(MEMH_COUNT, ucs_list_length(&md()->umr.mkey_pool));

    /* Now reuse the keys for different handles of different size */
    ucs_time_t export_reuse_time, import_reuse_time;
    perf_export_import_mkey(MEMH_COUNT, 2048, export_reuse_time,
                            import_reuse_time);

    /* Reusing existing UMR mkeys is normally 200x faster than creating new
     * ones, given jitter we check that it's at least 10 times faster */
    EXPECT_LT(export_reuse_time, export_create_time / 10);
    EXPECT_LT(export_reuse_time, export_create_time / 10);

    char buf[1024] = {};
    snprintf(buf, sizeof(buf), "Create %zu UMR mkeys, export: %.3f ms, import: "
             "%.3f ms", MEMH_COUNT, ucs_time_to_msec(export_create_time),
             ucs_time_to_msec(import_create_time));
    UCS_TEST_MESSAGE << buf;
    snprintf(buf, sizeof(buf), "Reuse %zu UMR mkeys, export: %.3f ms, import: "
             "%.3f ms", MEMH_COUNT, ucs_time_to_msec(export_reuse_time),
            ucs_time_to_msec(import_reuse_time));
    UCS_TEST_MESSAGE << buf;

    EXPECT_EQ(MEMH_COUNT, ucs_list_length(&md()->umr.mkey_pool));
}

UCT_INSTANTIATE_IB_TEST_CASE(test_devx_umr_mkey);
