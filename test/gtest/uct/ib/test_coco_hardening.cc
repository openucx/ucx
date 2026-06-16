/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>

#include <algorithm>
#include <stdint.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <vector>

extern "C" {
#include <uct/ib/base/ib_md.h>
#include <uct/ib/mlx5/ib_mlx5.h>
#include <uct/ib/mlx5/ib_mlx5_coco.h>
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

class guarded_output {
public:
    explicit guarded_output(size_t readable_size) : m_base(MAP_FAILED),
                                                    m_ptr(NULL),
                                                    m_readable_size(readable_size),
                                                    m_mapping_size(0)
    {
        const size_t page_size = ucs_get_page_size();

        m_mapping_size = 2 * page_size;
        m_base = mmap(NULL, m_mapping_size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        EXPECT_NE(MAP_FAILED, m_base);
        if (m_base == MAP_FAILED) {
            return;
        }

        EXPECT_EQ(0, mprotect(UCS_PTR_BYTE_OFFSET(m_base, page_size),
                              page_size, PROT_NONE));
        m_ptr = UCS_PTR_BYTE_OFFSET(m_base, page_size - readable_size);
        memset(m_ptr, 0, readable_size);
    }

    ~guarded_output()
    {
        if (m_base != MAP_FAILED) {
            munmap(m_base, m_mapping_size);
        }
    }

    const void *ptr() const
    {
        return m_ptr;
    }

    size_t size() const
    {
        return m_readable_size;
    }

private:
    void   *m_base;
    void   *m_ptr;
    size_t m_readable_size;
    size_t m_mapping_size;
};

enum {
    TEST_COCO_CQ_UMEM_ID  = 0x11,
    TEST_COCO_DBR_UMEM_ID = 0x12,
    TEST_COCO_WQ_UMEM_ID  = 0x13
};

void add_lifecycle_umems(uct_ib_mlx5_md_t *md, void *cq_buf, void *dbr_buf,
                         void *wq_buf)
{
    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_umem_record_add(
                          md->coco, TEST_COCO_CQ_UMEM_ID, cq_buf, 128,
                          ucs_get_page_size(), IBV_ACCESS_LOCAL_WRITE));
    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_umem_record_add(
                          md->coco, TEST_COCO_DBR_UMEM_ID, dbr_buf, 64,
                          ucs_get_page_size(), 0));
    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_umem_record_add(
                          md->coco, TEST_COCO_WQ_UMEM_ID, wq_buf, 256,
                          ucs_get_page_size(), 0));
}

uct_ib_mlx5_coco_cq_req_t make_valid_cq_req()
{
    uct_ib_mlx5_coco_cq_req_t req = {};

    req.cq_len         = 16;
    req.cqe_size       = 64;
    req.cq_umem_id     = TEST_COCO_CQ_UMEM_ID;
    req.cq_umem_offset = 0;
    req.dbr_umem_id    = TEST_COCO_DBR_UMEM_ID;
    req.dbr_offset     = 0;
    req.eqn            = 1;
    req.uar_page       = 2;
    return req;
}

uct_ib_mlx5_coco_qp_req_t make_valid_qp_req(uint32_t send_cqn,
                                            uint32_t recv_cqn)
{
    uct_ib_mlx5_coco_qp_req_t req = {};

    req.qp_type       = IBV_QPT_RC;
    req.send_cqn      = send_cqn;
    req.recv_cqn      = recv_cqn;
    req.rmpn          = 0;
    req.wq_umem_id    = TEST_COCO_WQ_UMEM_ID;
    req.dbr_umem_id   = TEST_COCO_DBR_UMEM_ID;
    req.sq_wqe_count  = 64;
    req.rq_wqe_count  = 0;
    return req;
}

uct_ib_mlx5_coco_rmp_req_t make_valid_rmp_req()
{
    uct_ib_mlx5_coco_rmp_req_t req = {};

    req.wq_umem_id  = TEST_COCO_WQ_UMEM_ID;
    req.dbr_umem_id = TEST_COCO_DBR_UMEM_ID;
    req.wq_size     = 64;
    req.stride      = 64;
    req.cyclic      = 1;
    req.mp_enabled  = 0;
    return req;
}

void set_cq_output(void *out, uint32_t cqn)
{
    memset(out, 0, UCT_IB_MLX5DV_ST_SZ_BYTES(create_cq_out));
    UCT_IB_MLX5DV_SET(create_cq_out, static_cast<char*>(out), cqn, cqn);
}

void set_qp_output(void *out, uint32_t qpn)
{
    memset(out, 0, UCT_IB_MLX5DV_ST_SZ_BYTES(create_qp_out));
    UCT_IB_MLX5DV_SET(create_qp_out, static_cast<char*>(out), qpn, qpn);
}

void set_rmp_output(void *out, uint32_t rmpn)
{
    memset(out, 0, UCT_IB_MLX5DV_ST_SZ_BYTES(create_rmp_out));
    UCT_IB_MLX5DV_SET(create_rmp_out, static_cast<char*>(out), rmpn, rmpn);
}

void add_valid_cq_objects(uct_ib_mlx5_md_t *md, uint32_t send_cqn,
                          uint32_t recv_cqn)
{
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(create_cq_out)];
    uct_ib_mlx5_coco_cq_req_t req = make_valid_cq_req();
    uint32_t cqn;

    set_cq_output(out, send_cqn);
    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_validate_cq_output(
                          md, &req, out, sizeof(out), &cqn));
    EXPECT_EQ(send_cqn, cqn);

    set_cq_output(out, recv_cqn);
    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_validate_cq_output(
                          md, &req, out, sizeof(out), &cqn));
    EXPECT_EQ(recv_cqn, cqn);
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

UCS_TEST_F(test_coco_hardening, memory_non_coco_allocator_path_unchanged)
{
    std::string function_source = get_function_source(
        "src/uct/ib/mlx5/ib_mlx5.h", "uct_ib_mlx5_md_buf_alloc(",
        "static inline void\nuct_ib_mlx5_md_buf_free");

    EXPECT_NE(std::string::npos,
              function_source.find("uct_ib_md_is_coco_hardened"));
    EXPECT_NE(std::string::npos,
              function_source.find("uct_ib_mlx5_coco_md_buf_alloc_shared"));
    EXPECT_NE(std::string::npos, function_source.find("ucs_posix_memalign"));
}

UCS_TEST_F(test_coco_hardening, memory_dbrec_metadata_is_private)
{
    std::string source = read_source_file(
        "src/uct/ib/mlx5/dv/ib_mlx5dv_md.c");

    EXPECT_NE(std::string::npos, source.find("uct_ib_mlx5_find_dbrec_page"));
    EXPECT_NE(std::string::npos, source.find("md->dbrec_pages"));
    EXPECT_EQ(std::string::npos, source.find("page->mem = mem"));
    EXPECT_EQ(std::string::npos,
              source.find("(uct_ib_mlx5_dbrec_page_t*)chunk - 1"));
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

UCS_TEST_F(test_coco_hardening, devx_output_cq_rejects_short_buffer_guarded)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uint8_t cq_buf[128], dbr_buf[64], wq_buf[256];
    guarded_output out(1);
    uct_ib_mlx5_coco_cq_req_t req = make_valid_cq_req();
    uint32_t cqn = 0xdeadbeefu;

    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_state_init(&md));
    add_lifecycle_umems(&md, cq_buf, dbr_buf, wq_buf);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              uct_ib_mlx5_coco_validate_cq_output(&md, &req, out.ptr(),
                                                  out.size(), &cqn));
    EXPECT_EQ(0xdeadbeefu, cqn);
    uct_ib_mlx5_coco_state_cleanup(&md);
}

UCS_TEST_F(test_coco_hardening, devx_output_cq_rejects_duplicate_cqn)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uint8_t cq_buf[128], dbr_buf[64], wq_buf[256];
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(create_cq_out)];
    uct_ib_mlx5_coco_cq_req_t req = make_valid_cq_req();
    uint32_t cqn;

    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_state_init(&md));
    add_lifecycle_umems(&md, cq_buf, dbr_buf, wq_buf);
    set_cq_output(out, 0x31);

    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_validate_cq_output(
                          &md, &req, out, sizeof(out), &cqn));
    EXPECT_EQ(0x31u, cqn);
    EXPECT_EQ(UCS_ERR_ALREADY_EXISTS,
              uct_ib_mlx5_coco_validate_cq_output(&md, &req, out,
                                                  sizeof(out), &cqn));
    uct_ib_mlx5_coco_state_cleanup(&md);
}

UCS_TEST_F(test_coco_hardening, devx_output_qp_rejects_duplicate_qpn)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uint8_t cq_buf[128], dbr_buf[64], wq_buf[256];
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(create_qp_out)];
    uct_ib_mlx5_coco_qp_req_t req = make_valid_qp_req(0x41, 0x42);
    uint32_t qpn;

    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_state_init(&md));
    add_lifecycle_umems(&md, cq_buf, dbr_buf, wq_buf);
    add_valid_cq_objects(&md, 0x41, 0x42);
    set_qp_output(out, 0x51);

    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_validate_qp_output(
                          &md, &req, out, sizeof(out), &qpn));
    EXPECT_EQ(0x51u, qpn);
    EXPECT_EQ(UCS_ERR_ALREADY_EXISTS,
              uct_ib_mlx5_coco_validate_qp_output(&md, &req, out,
                                                  sizeof(out), &qpn));
    uct_ib_mlx5_coco_state_cleanup(&md);
}

UCS_TEST_F(test_coco_hardening, devx_output_rmp_rejects_duplicate_rmpn)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uint8_t cq_buf[128], dbr_buf[64], wq_buf[256];
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(create_rmp_out)];
    uct_ib_mlx5_coco_rmp_req_t req = make_valid_rmp_req();
    uint32_t rmpn;

    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_state_init(&md));
    add_lifecycle_umems(&md, cq_buf, dbr_buf, wq_buf);
    set_rmp_output(out, 0x61);

    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_validate_rmp_output(
                          &md, &req, out, sizeof(out), &rmpn));
    EXPECT_EQ(0x61u, rmpn);
    EXPECT_EQ(UCS_ERR_ALREADY_EXISTS,
              uct_ib_mlx5_coco_validate_rmp_output(&md, &req, out,
                                                   sizeof(out), &rmpn));
    uct_ib_mlx5_coco_state_cleanup(&md);
}

UCS_TEST_F(test_coco_hardening, devx_input_qp_rejects_non_rc)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uint8_t cq_buf[128], dbr_buf[64], wq_buf[256];
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(create_qp_out)];
    uct_ib_mlx5_coco_qp_req_t req = make_valid_qp_req(0x41, 0x42);
    uint32_t qpn = 0;

    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_state_init(&md));
    add_lifecycle_umems(&md, cq_buf, dbr_buf, wq_buf);
    add_valid_cq_objects(&md, 0x41, 0x42);
    set_qp_output(out, 0x51);
    req.qp_type = UCT_IB_QPT_DCI;

    EXPECT_EQ(UCS_ERR_UNSUPPORTED,
              uct_ib_mlx5_coco_validate_qp_output(&md, &req, out,
                                                  sizeof(out), &qpn));
    uct_ib_mlx5_coco_state_cleanup(&md);
}

UCS_TEST_F(test_coco_hardening, devx_input_rmp_rejects_mp)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uint8_t cq_buf[128], dbr_buf[64], wq_buf[256];
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(create_rmp_out)];
    uct_ib_mlx5_coco_rmp_req_t req = make_valid_rmp_req();
    uint32_t rmpn = 0;

    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_state_init(&md));
    add_lifecycle_umems(&md, cq_buf, dbr_buf, wq_buf);
    set_rmp_output(out, 0x61);
    req.mp_enabled = 1;

    EXPECT_EQ(UCS_ERR_UNSUPPORTED,
              uct_ib_mlx5_coco_validate_rmp_output(&md, &req, out,
                                                   sizeof(out), &rmpn));
    uct_ib_mlx5_coco_state_cleanup(&md);
}

UCS_TEST_F(test_coco_hardening, devx_input_rmp_rejects_non_cyclic)
{
    uct_ib_mlx5_md_t md = make_mlx5_md(1);
    uint8_t cq_buf[128], dbr_buf[64], wq_buf[256];
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(create_rmp_out)];
    uct_ib_mlx5_coco_rmp_req_t req = make_valid_rmp_req();
    uint32_t rmpn = 0;

    ASSERT_EQ(UCS_OK, uct_ib_mlx5_coco_state_init(&md));
    add_lifecycle_umems(&md, cq_buf, dbr_buf, wq_buf);
    set_rmp_output(out, 0x61);
    req.cyclic = 0;

    EXPECT_EQ(UCS_ERR_UNSUPPORTED,
              uct_ib_mlx5_coco_validate_rmp_output(&md, &req, out,
                                                   sizeof(out), &rmpn));
    uct_ib_mlx5_coco_state_cleanup(&md);
}

UCS_TEST_F(test_coco_hardening,
           devx_wrapper_order_cq_validates_before_cqn_consume)
{
    std::string function_source = get_function_source(
        "src/uct/ib/mlx5/dv/ib_mlx5_dv.c",
        "uct_ib_mlx5_devx_create_cq_common(",
        "ucs_status_t\nuct_ib_mlx5_devx_create_cq(");
    size_t validate_pos = function_source.find(
        "uct_ib_mlx5_coco_validate_cq_output");
    size_t consume_pos = function_source.find("uct_ib_mlx5_init_cq_common");

    ASSERT_NE(std::string::npos, validate_pos);
    ASSERT_NE(std::string::npos, consume_pos);
    EXPECT_LT(validate_pos, consume_pos);
}

UCS_TEST_F(test_coco_hardening,
           devx_wrapper_order_qp_validates_before_qpn_publish)
{
    std::string function_source = get_function_source(
        "src/uct/ib/mlx5/dv/ib_mlx5_dv.c",
        "uct_ib_mlx5_devx_create_qp_common(",
        "ucs_status_t uct_ib_mlx5_devx_create_qp(");
    size_t validate_pos = function_source.find(
        "uct_ib_mlx5_coco_validate_qp_output");
    size_t publish_pos = function_source.find("qp->qp_num =");

    ASSERT_NE(std::string::npos, validate_pos);
    ASSERT_NE(std::string::npos, publish_pos);
    EXPECT_LT(validate_pos, publish_pos);
}

UCS_TEST_F(test_coco_hardening,
           devx_wrapper_order_rmp_validates_before_rmpn_publish)
{
    std::string function_source = get_function_source(
        "src/uct/ib/mlx5/rc/rc_mlx5_devx.c",
        "ucs_status_t uct_rc_mlx5_devx_init_rx(",
        "void uct_rc_mlx5_devx_cleanup_srq(");
    size_t validate_pos = function_source.find(
        "uct_ib_mlx5_coco_validate_rmp_output");
    size_t publish_pos = function_source.find("iface->rx.srq.srq_num =");

    ASSERT_NE(std::string::npos, validate_pos);
    ASSERT_NE(std::string::npos, publish_pos);
    EXPECT_LT(validate_pos, publish_pos);
}

UCS_TEST_F(test_coco_hardening,
           devx_wrapper_order_non_coco_validation_not_called)
{
    std::string cq_source = get_function_source(
        "src/uct/ib/mlx5/dv/ib_mlx5_dv.c",
        "uct_ib_mlx5_devx_create_cq_common(",
        "ucs_status_t\nuct_ib_mlx5_devx_create_cq(");
    std::string qp_source = get_function_source(
        "src/uct/ib/mlx5/dv/ib_mlx5_dv.c",
        "uct_ib_mlx5_devx_create_qp_common(",
        "ucs_status_t uct_ib_mlx5_devx_create_qp(");
    std::string rmp_source = get_function_source(
        "src/uct/ib/mlx5/rc/rc_mlx5_devx.c",
        "ucs_status_t uct_rc_mlx5_devx_init_rx(",
        "void uct_rc_mlx5_devx_cleanup_srq(");

    EXPECT_NE(std::string::npos,
              cq_source.find("if (uct_ib_md_is_coco_hardened(&md->super))"));
    EXPECT_NE(std::string::npos,
              qp_source.find("if (uct_ib_md_is_coco_hardened(&md->super))"));
    EXPECT_NE(std::string::npos,
              rmp_source.find("if (uct_ib_md_is_coco_hardened(&md->super))"));
}
