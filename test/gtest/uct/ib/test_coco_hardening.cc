/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>

#include <algorithm>
#include <stdint.h>
#include <vector>

extern "C" {
#include <uct/ib/base/ib_md.h>
#include <uct/ib/mlx5/ib_mlx5.h>
#include <uct/ib/mlx5/ib_mlx5_coco.h>
#include <uct/ib/mlx5/rc/rc_mlx5_common.h>
}

namespace {

void init_coco_rc_mlx5_config(uct_rc_iface_common_config_t *rc_config,
                              uct_rc_mlx5_iface_common_config_t *mlx5_config)
{
    static char cyclic[] = "cyclic";
    static char *cyclic_topo[] = {cyclic};

    *rc_config                      = {};
    *mlx5_config                    = {};
    mlx5_config->srq_topo.types     = cyclic_topo;
    mlx5_config->srq_topo.count     = 1;
    mlx5_config->tm.mp_enable       = UCS_TRY;
    mlx5_config->ddp_enable         = UCS_TRY;
    rc_config->super.inl[UCT_IB_DIR_TX] = 64;
}

uct_ib_mlx5_md_t make_mlx5_md(int coco_hardened)
{
    uct_ib_mlx5_md_t md = {};

    md.super.cc_dma_bounce = coco_hardened;
    return md;
}

struct fake_shared_backend {
    std::vector<uint8_t> storage;
    struct mlx5dv_devx_umem umem;
    size_t alloc_size;
    unsigned alloc_count;
    unsigned umem_reg_count;
    unsigned umem_dereg_count;
    unsigned unmap_count;
    unsigned close_count;
    int scrubbed_before_unmap;
    int fd;

    fake_shared_backend() : alloc_size(0), alloc_count(0), umem_reg_count(0),
                            umem_dereg_count(0), unmap_count(0),
                            close_count(0), scrubbed_before_unmap(0), fd(17)
    {
        memset(&umem, 0, sizeof(umem));
        umem.umem_id = 0xabcdu;
    }
};

static ucs_status_t fake_shared_alloc(size_t size, void **addr_p, int *fd_p,
                                      void *arg)
{
    fake_shared_backend *backend = static_cast<fake_shared_backend*>(arg);

    backend->storage.assign(size, 0x5a);
    backend->alloc_size = size;
    backend->alloc_count++;
    *addr_p = backend->storage.data();
    *fd_p   = backend->fd;
    return UCS_OK;
}

static ucs_status_t
fake_shared_umem_reg(uct_ib_mlx5_md_t *md,
                     const uct_ib_mlx5_coco_shared_alloc_t *alloc,
                     int access_mode, struct mlx5dv_devx_umem **umem_p,
                     void *arg)
{
    fake_shared_backend *backend = static_cast<fake_shared_backend*>(arg);

    EXPECT_TRUE(uct_ib_md_is_coco_hardened(&md->super));
    EXPECT_EQ(0, access_mode);
    EXPECT_EQ(backend->storage.data(), alloc->addr);
    EXPECT_EQ(backend->alloc_size, alloc->exposed_size);
    EXPECT_TRUE(std::all_of(backend->storage.begin(), backend->storage.end(),
                            [](uint8_t value) { return value == 0; }));
    backend->umem_reg_count++;
    *umem_p = &backend->umem;
    return UCS_OK;
}

static ucs_status_t fake_shared_umem_dereg(struct mlx5dv_devx_umem *umem,
                                           void *arg)
{
    fake_shared_backend *backend = static_cast<fake_shared_backend*>(arg);

    EXPECT_EQ(&backend->umem, umem);
    backend->umem_dereg_count++;
    return UCS_OK;
}

static ucs_status_t fake_shared_unmap(void *addr, size_t size, void *arg)
{
    fake_shared_backend *backend = static_cast<fake_shared_backend*>(arg);

    EXPECT_EQ(backend->storage.data(), addr);
    EXPECT_EQ(backend->storage.size(), size);
    backend->scrubbed_before_unmap = std::all_of(
        backend->storage.begin(), backend->storage.end(),
        [](uint8_t value) { return value == 0; });
    backend->unmap_count++;
    return UCS_OK;
}

static ucs_status_t fake_shared_close(int fd, void *arg)
{
    fake_shared_backend *backend = static_cast<fake_shared_backend*>(arg);

    EXPECT_EQ(backend->fd, fd);
    backend->close_count++;
    return UCS_OK;
}

class fake_shared_ops_scope {
public:
    explicit fake_shared_ops_scope(fake_shared_backend *backend)
    {
        ops.alloc      = fake_shared_alloc;
        ops.umem_reg   = fake_shared_umem_reg;
        ops.umem_dereg = fake_shared_umem_dereg;
        ops.unmap      = fake_shared_unmap;
        ops.close_fd   = fake_shared_close;
        uct_ib_mlx5_coco_set_shared_alloc_ops(&ops, backend);
    }

    ~fake_shared_ops_scope()
    {
        uct_ib_mlx5_coco_set_shared_alloc_ops(NULL, NULL);
    }

private:
    uct_ib_mlx5_coco_shared_alloc_ops_t ops;
};

}


class test_coco_hardening : public ucs::test {
};

UCS_TEST_F(test_coco_hardening, policy_helper_requires_cc_dma_bounce)
{
    uct_ib_md_t md = {};

    md.cc_dma_bounce = 0;
    EXPECT_FALSE(uct_ib_md_is_coco_hardened(&md));

    md.cc_dma_bounce = 1;
    EXPECT_TRUE(uct_ib_md_is_coco_hardened(&md));
}

UCS_TEST_F(test_coco_hardening, policy_transport_rc_mlx5_only)
{
    uct_ib_md_t md = {};

    md.cc_dma_bounce = 0;
    EXPECT_TRUE(uct_ib_md_coco_transport_allowed(&md, "rc_mlx5"));
    EXPECT_TRUE(uct_ib_md_coco_transport_allowed(&md, "dc_mlx5"));
    EXPECT_TRUE(uct_ib_md_coco_transport_allowed(&md, "ud_mlx5"));
    EXPECT_TRUE(uct_ib_md_coco_transport_allowed(&md, "rc_verbs"));
    EXPECT_TRUE(uct_ib_md_coco_transport_allowed(&md, "ud_verbs"));

    md.cc_dma_bounce = 1;
    EXPECT_TRUE(uct_ib_md_coco_transport_allowed(&md, "rc_mlx5"));
    EXPECT_FALSE(uct_ib_md_coco_transport_allowed(&md, "dc_mlx5"));
    EXPECT_FALSE(uct_ib_md_coco_transport_allowed(&md, "ud_mlx5"));
    EXPECT_FALSE(uct_ib_md_coco_transport_allowed(&md, "rc_verbs"));
    EXPECT_FALSE(uct_ib_md_coco_transport_allowed(&md, "ud_verbs"));
}

UCS_TEST_F(test_coco_hardening, config_rejects_tm)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uct_rc_iface_common_config_t rc_config;
    uct_rc_mlx5_iface_common_config_t mlx5_config;

    init_coco_rc_mlx5_config(&rc_config, &mlx5_config);
    mlx5_config.tm.enable = 1;

    scoped_log_handler slh(hide_errors_logger);
    EXPECT_NE(UCS_OK, uct_rc_mlx5_coco_check_config(&md, &rc_config,
                                                    &mlx5_config));
}

UCS_TEST_F(test_coco_hardening, config_rejects_mp)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uct_rc_iface_common_config_t rc_config;
    uct_rc_mlx5_iface_common_config_t mlx5_config;

    init_coco_rc_mlx5_config(&rc_config, &mlx5_config);
    mlx5_config.tm.mp_enable = UCS_YES;

    scoped_log_handler slh(hide_errors_logger);
    EXPECT_NE(UCS_OK, uct_rc_mlx5_coco_check_config(&md, &rc_config,
                                                    &mlx5_config));
}

UCS_TEST_F(test_coco_hardening, config_rejects_cqe_zip)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uct_rc_iface_common_config_t rc_config;
    uct_rc_mlx5_iface_common_config_t mlx5_config;

    init_coco_rc_mlx5_config(&rc_config, &mlx5_config);
    mlx5_config.super.cqe_zip_enable[UCT_IB_DIR_TX] = 1;
    scoped_log_handler slh(hide_errors_logger);
    EXPECT_NE(UCS_OK, uct_rc_mlx5_coco_check_config(&md, &rc_config,
                                                    &mlx5_config));

    mlx5_config.super.cqe_zip_enable[UCT_IB_DIR_TX] = 0;
    mlx5_config.super.cqe_zip_enable[UCT_IB_DIR_RX] = 1;
    EXPECT_NE(UCS_OK, uct_rc_mlx5_coco_check_config(&md, &rc_config,
                                                    &mlx5_config));
}

UCS_TEST_F(test_coco_hardening, config_rejects_ddp_yes)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uct_rc_iface_common_config_t rc_config;
    uct_rc_mlx5_iface_common_config_t mlx5_config;

    init_coco_rc_mlx5_config(&rc_config, &mlx5_config);
    mlx5_config.ddp_enable = UCS_YES;

    scoped_log_handler slh(hide_errors_logger);
    EXPECT_NE(UCS_OK, uct_rc_mlx5_coco_check_config(&md, &rc_config,
                                                    &mlx5_config));
}

UCS_TEST_F(test_coco_hardening, config_requires_cyclic_srq)
{
    static char list[] = "list";
    static char *list_topo[] = {list};
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uct_rc_iface_common_config_t rc_config;
    uct_rc_mlx5_iface_common_config_t mlx5_config;

    init_coco_rc_mlx5_config(&rc_config, &mlx5_config);
    mlx5_config.srq_topo.types = list_topo;
    mlx5_config.srq_topo.count = 1;

    scoped_log_handler slh(hide_errors_logger);
    EXPECT_NE(UCS_OK, uct_rc_mlx5_coco_check_config(&md, &rc_config,
                                                    &mlx5_config));
}

UCS_TEST_F(test_coco_hardening, config_effective_disables_unsafe_defaults)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uct_rc_iface_common_config_t rc_config;
    uct_rc_mlx5_iface_common_config_t mlx5_config;
    uct_ib_iface_init_attr_t init_attr = {};

    init_coco_rc_mlx5_config(&rc_config, &mlx5_config);
    mlx5_config.tm.enable = 0;
    mlx5_config.tm.mp_enable = UCS_TRY;
    mlx5_config.ddp_enable = UCS_TRY;
    init_attr.flags = UCT_IB_DDP_SUPPORTED | UCT_IB_TM_SUPPORTED;
    init_attr.cqe_zip_sizes[UCT_IB_DIR_TX] = 64;
    init_attr.cqe_zip_sizes[UCT_IB_DIR_RX] = 64;

    uct_rc_mlx5_coco_apply_effective_config(&md, &rc_config, &mlx5_config,
                                            &init_attr);

    EXPECT_EQ(0, mlx5_config.tm.enable);
    EXPECT_EQ(UCS_NO, mlx5_config.tm.mp_enable);
    EXPECT_EQ(UCS_NO, mlx5_config.ddp_enable);
    EXPECT_EQ(0ul, rc_config.super.inl[UCT_IB_DIR_TX]);
    EXPECT_EQ(0, init_attr.flags & UCT_IB_DDP_SUPPORTED);
    EXPECT_EQ(0, init_attr.flags & UCT_IB_TM_SUPPORTED);
    EXPECT_EQ(0, init_attr.cqe_zip_sizes[UCT_IB_DIR_TX]);
    EXPECT_EQ(0, init_attr.cqe_zip_sizes[UCT_IB_DIR_RX]);
}

UCS_TEST_F(test_coco_hardening, config_masks_atomics_initially)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uct_iface_attr_t attr = {};

    attr.cap.flags = UCT_IFACE_FLAG_AM_BCOPY |
                     UCT_IFACE_FLAG_ATOMIC_CPU |
                     UCT_IFACE_FLAG_ATOMIC_DEVICE;
    attr.cap.atomic64.op_flags  = UCS_BIT(UCT_ATOMIC_OP_ADD);
    attr.cap.atomic64.fop_flags = UCS_BIT(UCT_ATOMIC_OP_CSWAP);
    attr.cap.atomic32.op_flags  = UCS_BIT(UCT_ATOMIC_OP_ADD);
    attr.cap.atomic32.fop_flags = UCS_BIT(UCT_ATOMIC_OP_CSWAP);

    uct_rc_mlx5_coco_mask_capabilities(&md, &attr);

    EXPECT_EQ(0, attr.cap.flags & UCT_IFACE_FLAG_ATOMIC_CPU);
    EXPECT_EQ(0, attr.cap.flags & UCT_IFACE_FLAG_ATOMIC_DEVICE);
    EXPECT_EQ(0ul, attr.cap.atomic64.op_flags);
    EXPECT_EQ(0ul, attr.cap.atomic64.fop_flags);
    EXPECT_EQ(0ul, attr.cap.atomic32.op_flags);
    EXPECT_EQ(0ul, attr.cap.atomic32.fop_flags);
    EXPECT_NE(0, attr.cap.flags & UCT_IFACE_FLAG_AM_BCOPY);
}

UCS_TEST_F(test_coco_hardening, config_masks_rma_before_mr_mkey)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uct_iface_attr_t attr = {};

    attr.cap.flags = UCT_IFACE_FLAG_AM_BCOPY |
                     UCT_IFACE_FLAG_PUT_SHORT |
                     UCT_IFACE_FLAG_PUT_BCOPY |
                     UCT_IFACE_FLAG_PUT_ZCOPY |
                     UCT_IFACE_FLAG_GET_SHORT |
                     UCT_IFACE_FLAG_GET_BCOPY |
                     UCT_IFACE_FLAG_GET_ZCOPY;

    uct_rc_mlx5_coco_mask_capabilities(&md, &attr);

    EXPECT_EQ(0, attr.cap.flags & UCT_IFACE_FLAG_PUT_SHORT);
    EXPECT_EQ(0, attr.cap.flags & UCT_IFACE_FLAG_PUT_BCOPY);
    EXPECT_EQ(0, attr.cap.flags & UCT_IFACE_FLAG_PUT_ZCOPY);
    EXPECT_EQ(0, attr.cap.flags & UCT_IFACE_FLAG_GET_SHORT);
    EXPECT_EQ(0, attr.cap.flags & UCT_IFACE_FLAG_GET_BCOPY);
    EXPECT_EQ(0, attr.cap.flags & UCT_IFACE_FLAG_GET_ZCOPY);
    EXPECT_NE(0, attr.cap.flags & UCT_IFACE_FLAG_AM_BCOPY);
}

UCS_TEST_F(test_coco_hardening, config_non_coco_unchanged)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(0);
    uct_rc_iface_common_config_t rc_config;
    uct_rc_mlx5_iface_common_config_t mlx5_config;
    uct_ib_iface_init_attr_t init_attr = {};
    uct_iface_attr_t attr = {};

    init_coco_rc_mlx5_config(&rc_config, &mlx5_config);
    mlx5_config.tm.enable = 1;
    mlx5_config.tm.mp_enable = UCS_YES;
    mlx5_config.ddp_enable = UCS_YES;
    mlx5_config.super.cqe_zip_enable[UCT_IB_DIR_TX] = 1;
    mlx5_config.super.cqe_zip_enable[UCT_IB_DIR_RX] = 1;
    init_attr.flags = UCT_IB_DDP_SUPPORTED;
    init_attr.cqe_zip_sizes[UCT_IB_DIR_TX] = 64;
    init_attr.cqe_zip_sizes[UCT_IB_DIR_RX] = 128;
    attr.cap.flags = UCT_IFACE_FLAG_AM_BCOPY |
                     UCT_IFACE_FLAG_PUT_BCOPY |
                     UCT_IFACE_FLAG_GET_BCOPY |
                     UCT_IFACE_FLAG_ATOMIC_CPU;

    EXPECT_EQ(UCS_OK, uct_rc_mlx5_coco_check_config(&md, &rc_config,
                                                    &mlx5_config));
    uct_rc_mlx5_coco_apply_effective_config(&md, &rc_config, &mlx5_config,
                                            &init_attr);
    uct_rc_mlx5_coco_mask_capabilities(&md, &attr);

    EXPECT_EQ(1, mlx5_config.tm.enable);
    EXPECT_EQ(UCS_YES, mlx5_config.tm.mp_enable);
    EXPECT_EQ(UCS_YES, mlx5_config.ddp_enable);
    EXPECT_EQ(1, mlx5_config.super.cqe_zip_enable[UCT_IB_DIR_TX]);
    EXPECT_EQ(1, mlx5_config.super.cqe_zip_enable[UCT_IB_DIR_RX]);
    EXPECT_EQ(64ul, rc_config.super.inl[UCT_IB_DIR_TX]);
    EXPECT_NE(0, init_attr.flags & UCT_IB_DDP_SUPPORTED);
    EXPECT_EQ(64, init_attr.cqe_zip_sizes[UCT_IB_DIR_TX]);
    EXPECT_EQ(128, init_attr.cqe_zip_sizes[UCT_IB_DIR_RX]);
    EXPECT_NE(0, attr.cap.flags & UCT_IFACE_FLAG_AM_BCOPY);
    EXPECT_NE(0, attr.cap.flags & UCT_IFACE_FLAG_PUT_BCOPY);
    EXPECT_NE(0, attr.cap.flags & UCT_IFACE_FLAG_GET_BCOPY);
    EXPECT_NE(0, attr.cap.flags & UCT_IFACE_FLAG_ATOMIC_CPU);
}

UCS_TEST_F(test_coco_hardening, memory_shared_rounds_to_page)
{
    size_t page_size = ucs_get_page_size();
    size_t exposed_size;

    EXPECT_EQ(UCS_OK, uct_ib_mlx5_coco_exposed_size(page_size + 1,
                                                    &exposed_size));
    EXPECT_EQ(2 * page_size, exposed_size);
}

UCS_TEST_F(test_coco_hardening, memory_shared_zeroes_full_exposed_size)
{
    fake_shared_backend backend;
    fake_shared_ops_scope scope(&backend);
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uct_ib_mlx5_devx_umem_t mem;
    void *buf;

    EXPECT_EQ(UCS_OK, uct_ib_mlx5_coco_state_init(&md));
    EXPECT_EQ(UCS_OK, uct_ib_mlx5_coco_md_buf_alloc_shared(
                         &md, ucs_get_page_size() + 37, 0, &buf, &mem, 0,
                         (char*)"fake shared"));

    EXPECT_EQ(1u, backend.alloc_count);
    EXPECT_EQ(1u, backend.umem_reg_count);
    EXPECT_EQ(ucs_get_page_size() + 37, mem.size);
    EXPECT_EQ(2 * ucs_get_page_size(), mem.mmap_size);
    EXPECT_EQ(backend.storage.data(), buf);
    EXPECT_TRUE(std::all_of(backend.storage.begin(), backend.storage.end(),
                            [](uint8_t value) { return value == 0; }));

    uct_ib_mlx5_coco_md_buf_free_shared(&md, buf, &mem);
    uct_ib_mlx5_coco_state_cleanup(&md);
}

UCS_TEST_F(test_coco_hardening, memory_shared_scrubs_before_release)
{
    fake_shared_backend backend;
    fake_shared_ops_scope scope(&backend);
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uct_ib_mlx5_devx_umem_t mem;
    void *buf;

    EXPECT_EQ(UCS_OK, uct_ib_mlx5_coco_state_init(&md));
    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_md_buf_alloc_shared(
                          &md, 128, 0, &buf, &mem, 0, (char*)"fake shared"));
    std::fill(backend.storage.begin(), backend.storage.end(), 0x7du);

    uct_ib_mlx5_coco_md_buf_free_shared(&md, buf, &mem);

    EXPECT_EQ(1u, backend.umem_dereg_count);
    EXPECT_EQ(1u, backend.unmap_count);
    EXPECT_EQ(1u, backend.close_count);
    EXPECT_TRUE(backend.scrubbed_before_unmap);
    EXPECT_EQ(NULL, mem.mem);
    EXPECT_EQ(UCT_IB_MLX5_INVALID_DMABUF_FD, mem.dmabuf_fd);
    uct_ib_mlx5_coco_state_cleanup(&md);
}

UCS_TEST_F(test_coco_hardening, memory_shared_rejects_zero_size)
{
    fake_shared_backend backend;
    fake_shared_ops_scope scope(&backend);
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uct_ib_mlx5_devx_umem_t mem;
    void *buf = reinterpret_cast<void*>(0x1);

    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              uct_ib_mlx5_coco_md_buf_alloc_shared(&md, 0, 1, &buf, &mem, 0,
                                                   (char*)"zero shared"));
    EXPECT_EQ(0u, backend.alloc_count);
    EXPECT_EQ(NULL, buf);
}

UCS_TEST_F(test_coco_hardening, memory_umem_rejects_duplicate_id)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uint8_t buffer[64];

    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_state_init(&md));
    EXPECT_EQ(UCS_OK, uct_ib_mlx5_coco_umem_record_add(
                         md.coco, 7, buffer, sizeof(buffer),
                         ucs_get_page_size(), 0));
    EXPECT_EQ(UCS_ERR_ALREADY_EXISTS, uct_ib_mlx5_coco_umem_record_add(
                         md.coco, 7, buffer, sizeof(buffer),
                         ucs_get_page_size(), 0));
    uct_ib_mlx5_coco_state_cleanup(&md);
}

UCS_TEST_F(test_coco_hardening, memory_umem_rejects_widened_length)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uint8_t buffer[64];

    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_state_init(&md));
    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_umem_record_add(
                          md.coco, 8, buffer, sizeof(buffer),
                          ucs_get_page_size(), 0x3));
    EXPECT_EQ(UCS_OK, uct_ib_mlx5_coco_umem_record_validate(
                         md.coco, 8, buffer, sizeof(buffer),
                         ucs_get_page_size(), 0x3));
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              uct_ib_mlx5_coco_umem_record_validate(
                  md.coco, 8, buffer, sizeof(buffer),
                  2 * ucs_get_page_size(), 0x3));
    uct_ib_mlx5_coco_state_cleanup(&md);
}

UCS_TEST_F(test_coco_hardening, memory_umem_destroy_removes_record)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uint8_t buffer[64];

    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_state_init(&md));
    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_umem_record_add(
                          md.coco, 9, buffer, sizeof(buffer),
                          ucs_get_page_size(), 0));
    ASSERT_NE(static_cast<const uct_ib_mlx5_coco_umem_record_t*>(NULL),
              uct_ib_mlx5_coco_umem_record_find(md.coco, 9));
    EXPECT_EQ(UCS_OK, uct_ib_mlx5_coco_umem_record_remove(md.coco, 9));
    EXPECT_EQ(static_cast<const uct_ib_mlx5_coco_umem_record_t*>(NULL),
              uct_ib_mlx5_coco_umem_record_find(md.coco, 9));
    uct_ib_mlx5_coco_state_cleanup(&md);
}

UCS_TEST_F(test_coco_hardening, memory_non_coco_no_umem_registry)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(0);

    EXPECT_EQ(UCS_OK, uct_ib_mlx5_coco_state_init(&md));
    EXPECT_EQ(NULL, md.coco);
}

UCS_TEST_F(test_coco_hardening, mr_mkey_registers_bounds)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uint8_t buffer[128];

    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_state_init(&md));
    EXPECT_EQ(UCS_OK, uct_ib_mlx5_coco_mkey_record_add(
                         md.coco, 0x101, 0x202, buffer, sizeof(buffer),
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ));
    const uct_ib_mlx5_coco_mkey_record_t *record =
        uct_ib_mlx5_coco_mkey_record_find_lkey(md.coco, 0x101);
    ASSERT_NE(static_cast<const uct_ib_mlx5_coco_mkey_record_t*>(NULL), record);
    EXPECT_EQ(buffer, record->base);
    EXPECT_EQ(sizeof(buffer), record->length);
    EXPECT_EQ(0x202u, record->rkey);
    uct_ib_mlx5_coco_state_cleanup(&md);
}

UCS_TEST_F(test_coco_hardening, mr_mkey_rejects_duplicate_lkey)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uint8_t buffer[128];

    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_state_init(&md));
    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_mkey_record_add(
                          md.coco, 0x101, 0x202, buffer, 64,
                          IBV_ACCESS_LOCAL_WRITE));
    EXPECT_EQ(UCS_ERR_ALREADY_EXISTS, uct_ib_mlx5_coco_mkey_record_add(
                         md.coco, 0x101, 0x303, buffer, 64,
                         IBV_ACCESS_LOCAL_WRITE));
    uct_ib_mlx5_coco_state_cleanup(&md);
}

UCS_TEST_F(test_coco_hardening, mr_mkey_rejects_duplicate_rkey)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uint8_t buffer[128];

    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_state_init(&md));
    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_mkey_record_add(
                          md.coco, 0x101, 0x202, buffer, 64,
                          IBV_ACCESS_LOCAL_WRITE));
    EXPECT_EQ(UCS_ERR_ALREADY_EXISTS, uct_ib_mlx5_coco_mkey_record_add(
                         md.coco, 0x303, 0x202, buffer, 64,
                         IBV_ACCESS_LOCAL_WRITE));
    uct_ib_mlx5_coco_state_cleanup(&md);
}

UCS_TEST_F(test_coco_hardening, mr_mkey_rejects_widened_range)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uint8_t buffer[128];

    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_state_init(&md));
    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_mkey_record_add(
                          md.coco, 0x101, 0x202, buffer + 16, 64,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ));
    EXPECT_EQ(UCS_OK, uct_ib_mlx5_coco_mkey_record_validate(
                         md.coco, 0x101, 0x202, buffer + 16, 64,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ));
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              uct_ib_mlx5_coco_mkey_record_validate(
                  md.coco, 0x101, 0x202, buffer + 15, 65,
                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ));
    uct_ib_mlx5_coco_state_cleanup(&md);
}

UCS_TEST_F(test_coco_hardening, mr_mkey_rejects_permission_widening)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uint8_t buffer[128];

    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_state_init(&md));
    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_mkey_record_add(
                          md.coco, 0x101, 0x202, buffer, 64,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ));
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              uct_ib_mlx5_coco_mkey_record_validate(
                  md.coco, 0x101, 0x202, buffer, 64,
                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                  IBV_ACCESS_REMOTE_WRITE));
    uct_ib_mlx5_coco_state_cleanup(&md);
}

UCS_TEST_F(test_coco_hardening, mr_mkey_masks_atomics_until_enabled)
{
    uint64_t access = uct_ib_mlx5_coco_mkey_sanitize_access(
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_ATOMIC);

    EXPECT_NE(0ul, access & IBV_ACCESS_LOCAL_WRITE);
    EXPECT_NE(0ul, access & IBV_ACCESS_REMOTE_READ);
    EXPECT_EQ(0ul, access & IBV_ACCESS_REMOTE_ATOMIC);
}

UCS_TEST_F(test_coco_hardening, mr_mkey_unmasks_rma_after_bounds_ready)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uct_iface_attr_t attr = {};

    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_state_init(&md));
    attr.cap.flags = UCT_IFACE_FLAG_AM_BCOPY |
                     UCT_IFACE_FLAG_PUT_BCOPY |
                     UCT_IFACE_FLAG_GET_BCOPY |
                     UCT_IFACE_FLAG_ATOMIC_CPU;

    uct_rc_mlx5_coco_mask_capabilities(&md, &attr);

    EXPECT_NE(0, attr.cap.flags & UCT_IFACE_FLAG_PUT_BCOPY);
    EXPECT_NE(0, attr.cap.flags & UCT_IFACE_FLAG_GET_BCOPY);
    EXPECT_EQ(0, attr.cap.flags & UCT_IFACE_FLAG_ATOMIC_CPU);
    uct_ib_mlx5_coco_state_cleanup(&md);
}

UCS_TEST_F(test_coco_hardening, mr_mkey_non_coco_no_registry)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(0);

    EXPECT_EQ(UCS_OK, uct_ib_mlx5_coco_state_init(&md));
    EXPECT_EQ(NULL, md.coco);
}
