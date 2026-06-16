/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>

#include <fstream>
#include <sstream>
#include <string>

extern "C" {
#include <uct/ib/base/ib_md.h>
#include <uct/ib/mlx5/ib_mlx5.h>
#include <uct/ib/mlx5/rc/rc_mlx5_common.h>
}

namespace {

std::string read_source_file(const char *relative_path)
{
    std::string path = std::string(TOP_SRCDIR) + "/" + relative_path;
    std::ifstream input(path.c_str());
    std::stringstream contents;

    EXPECT_TRUE(input.is_open()) << path;
    if (!input.is_open()) {
        return std::string();
    }

    contents << input.rdbuf();
    return contents.str();
}

void expect_query_uses_transport_helper(const char *relative_path,
                                        const char *function_name,
                                        const char *tl_name)
{
    std::string source = read_source_file(relative_path);
    size_t function_pos = source.find(function_name);

    ASSERT_NE(std::string::npos, function_pos) << function_name;

    size_t entry_pos = source.find("UCT_TL_DEFINE_ENTRY", function_pos);
    ASSERT_NE(std::string::npos, entry_pos) << function_name;

    std::string function_source = source.substr(function_pos,
                                                entry_pos - function_pos);
    EXPECT_NE(std::string::npos,
              function_source.find("uct_ib_md_coco_transport_allowed"))
        << function_name;
    EXPECT_NE(std::string::npos, function_source.find(tl_name))
        << function_name;
}

std::string get_function_source(const char *relative_path,
                                const char *function_name,
                                const char *end_marker)
{
    std::string source = read_source_file(relative_path);
    size_t function_pos = source.find(function_name);

    EXPECT_NE(std::string::npos, function_pos) << function_name;
    if (function_pos == std::string::npos) {
        return std::string();
    }

    size_t end_pos = source.find(end_marker, function_pos);
    EXPECT_NE(std::string::npos, end_pos) << end_marker;
    if (end_pos == std::string::npos) {
        return std::string();
    }

    return source.substr(function_pos, end_pos - function_pos);
}

void expect_init_backstop_uses_hardened_predicate(const char *relative_path,
                                                  const char *function_name,
                                                  const char *feature_name)
{
    std::string function_source = get_function_source(
        relative_path, function_name, "UCS_CLASS_CLEANUP_FUNC");

    EXPECT_NE(std::string::npos, function_source.find(feature_name))
        << function_name;
    EXPECT_NE(std::string::npos,
              function_source.find("uct_ib_md_is_coco_hardened"))
        << function_name;
    EXPECT_EQ(std::string::npos,
              function_source.find("uct_ib_md_is_cc_dma_bounce"))
        << function_name;
}

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

UCS_TEST_F(test_coco_hardening, policy_query_functions_use_transport_helper)
{
    expect_query_uses_transport_helper(
        "src/uct/ib/mlx5/rc/rc_mlx5_iface.c",
        "uct_rc_mlx5_query_tl_devices", "rc_mlx5");
    expect_query_uses_transport_helper(
        "src/uct/ib/mlx5/dc/dc_mlx5.c",
        "uct_dc_mlx5_query_tl_devices", "dc_mlx5");
    expect_query_uses_transport_helper(
        "src/uct/ib/mlx5/ud/ud_mlx5.c",
        "uct_ud_mlx5_query_tl_devices", "ud_mlx5");
    expect_query_uses_transport_helper(
        "src/uct/ib/rc/verbs/rc_verbs_iface.c",
        "uct_rc_verbs_query_tl_devices", "rc_verbs");
    expect_query_uses_transport_helper(
        "src/uct/ib/ud/verbs/ud_verbs.c",
        "uct_ud_verbs_query_tl_devices", "ud_verbs");
}

UCS_TEST_F(test_coco_hardening, policy_observable_log_uses_hardened_predicate)
{
    std::string source = read_source_file(
        "src/uct/ib/mlx5/rc/rc_mlx5_iface.c");

    EXPECT_NE(std::string::npos,
              source.find("uct_rc_mlx5_coco_log_policy"));
    EXPECT_NE(std::string::npos, source.find("cc_dma_bounce"));
    EXPECT_NE(std::string::npos,
              source.find("uct_ib_md_is_coco_hardened"));
    EXPECT_NE(std::string::npos, source.find("\"rc_mlx5\""));
}

UCS_TEST_F(test_coco_hardening, policy_query_logs_observable_policy)
{
    std::string function_source = get_function_source(
        "src/uct/ib/mlx5/rc/rc_mlx5_iface.c",
        "uct_rc_mlx5_query_tl_devices", "UCT_TL_DEFINE_ENTRY");
    size_t allowed_pos = function_source.find(
        "uct_ib_md_coco_transport_allowed");
    size_t log_pos = function_source.find("uct_rc_mlx5_coco_log_policy");

    ASSERT_NE(std::string::npos, allowed_pos);
    ASSERT_NE(std::string::npos, log_pos);
    EXPECT_LT(allowed_pos, log_pos);
}

UCS_TEST_F(test_coco_hardening, policy_init_backstops_use_hardened_predicate)
{
    expect_init_backstop_uses_hardened_predicate(
        "src/uct/ib/mlx5/rc/rc_mlx5_iface.c",
        "uct_rc_mlx5_iface_init_rx", "tag-matching DEVX");
    expect_init_backstop_uses_hardened_predicate(
        "src/uct/ib/mlx5/dc/dc_mlx5.c",
        "UCS_CLASS_INIT_FUNC(uct_dc_mlx5_iface_t", "dc_mlx5");
    expect_init_backstop_uses_hardened_predicate(
        "src/uct/ib/mlx5/ud/ud_mlx5.c",
        "UCS_CLASS_INIT_FUNC(uct_ud_mlx5_iface_t", "ud_mlx5");
}

UCS_TEST_F(test_coco_hardening, config_coco_gate_precedes_ddp_init)
{
    std::string constructor = get_function_source(
        "src/uct/ib/mlx5/rc/rc_mlx5_iface.c",
        "UCS_CLASS_INIT_FUNC(uct_rc_mlx5_iface_t",
        "UCS_CLASS_CLEANUP_FUNC(uct_rc_mlx5_iface_t");
    size_t check_pos = constructor.find("uct_rc_mlx5_coco_check_config");
    size_t ddp_flag_pos = constructor.find("UCT_IB_DDP_SUPPORTED");
    size_t apply_pos = constructor.find("uct_rc_mlx5_coco_apply_effective_config");
    size_t ooo_init_pos = constructor.find("uct_rc_mlx5_dp_ordering_ooo_init");

    ASSERT_NE(std::string::npos, check_pos);
    ASSERT_NE(std::string::npos, ddp_flag_pos);
    ASSERT_NE(std::string::npos, apply_pos);
    ASSERT_NE(std::string::npos, ooo_init_pos);
    EXPECT_LT(check_pos, ddp_flag_pos);
    EXPECT_LT(apply_pos, ooo_init_pos);
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
