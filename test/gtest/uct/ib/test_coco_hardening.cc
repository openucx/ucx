/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>

extern "C" {
#include <uct/ib/base/ib_md.h>
#include <uct/ib/mlx5/ib_mlx5.h>
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
