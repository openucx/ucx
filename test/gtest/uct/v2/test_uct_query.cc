/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2021. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
#include <gtest/uct/uct_p2p_test.h>

extern "C" {
#include <ucs/sys/topo/base/topo.h>
#include <uct/api/uct.h>
#include <uct/api/v2/uct_v2.h>
#if HAVE_CUDA
#include <uct/cuda/cuda_copy/cuda_copy_iface.h>
#endif
#include <uct/base/uct_iface.h>
}

#include <dirent.h>
#include <limits.h>
#include <unistd.h>
#include <string>


#define IB_SEND_OVERHEAD_BCOPY     1
#define IB_SEND_OVERHEAD_CQE       2
#define IB_SEND_OVERHEAD_DB        3
#define IB_SEND_OVERHEAD_WQE_FETCH 4
#define IB_SEND_OVERHEAD_WQE_POST  5
#define MM_SEND_OVERHEAD_AM_SHORT  6
#define MM_SEND_OVERHEAD_AM_BCOPY  7
#define MM_RECV_OVERHEAD_AM_SHORT  8
#define MM_RECV_OVERHEAD_AM_BCOPY  9


class test_uct_query : public uct_test {
public:
    void init() override;
    ucs_status_t iface_estimate_perf(uct_perf_attr_t *perf_attr) const;
    const uct_iface_attr &get_iface_attr() const;
    static uct_perf_attr_t init_perf_attr();

private:
    entity *m_e = nullptr;
};

void test_uct_query::init()
{
    m_e = create_entity(0);
    m_entities.push_back(m_e);
}

ucs_status_t
test_uct_query::iface_estimate_perf(uct_perf_attr_t *perf_attr) const
{
    return uct_iface_estimate_perf(m_e->iface(), perf_attr);
}

const uct_iface_attr &test_uct_query::get_iface_attr() const
{
    return m_e->iface_attr();
}

uct_perf_attr_t test_uct_query::init_perf_attr()
{
    uct_perf_attr_t perf_attr = {
        .field_mask         = UCT_PERF_ATTR_FIELD_OPERATION |
                              UCT_PERF_ATTR_FIELD_LOCAL_MEMORY_TYPE |
                              UCT_PERF_ATTR_FIELD_REMOTE_MEMORY_TYPE |
                              UCT_PERF_ATTR_FIELD_LOCAL_SYS_DEVICE |
                              UCT_PERF_ATTR_FIELD_REMOTE_SYS_DEVICE,
        .operation          = UCT_EP_OP_AM_SHORT,
        .local_memory_type  = UCS_MEMORY_TYPE_HOST,
        .remote_memory_type = UCS_MEMORY_TYPE_HOST,
        .local_sys_device   = UCS_SYS_DEVICE_ID_UNKNOWN,
        .remote_sys_device  = UCS_SYS_DEVICE_ID_UNKNOWN
    };

    return perf_attr;
}

UCS_TEST_P(test_uct_query, query_perf)
{
    auto perf_attr        = init_perf_attr();
    perf_attr.field_mask |= UCT_PERF_ATTR_FIELD_SEND_PRE_OVERHEAD |
                            UCT_PERF_ATTR_FIELD_SEND_POST_OVERHEAD |
                            UCT_PERF_ATTR_FIELD_RECV_OVERHEAD |
                            UCT_PERF_ATTR_FIELD_BANDWIDTH;
    EXPECT_EQ(iface_estimate_perf(&perf_attr),
              has_transport("cuda_copy") ? UCS_ERR_UNSUPPORTED : UCS_OK);

    perf_attr.remote_memory_type = UCS_MEMORY_TYPE_CUDA;
    perf_attr.operation          = UCT_EP_OP_PUT_SHORT;
    EXPECT_EQ(iface_estimate_perf(&perf_attr), UCS_OK);

    /* At least one type of bandwidth must be non-zero */
    EXPECT_NE(0, perf_attr.bandwidth.shared + perf_attr.bandwidth.dedicated);

    if (has_transport("cuda_copy") ||
        has_transport("gdr_copy")  ||
        has_transport("rocm_copy")) {
        uct_perf_attr_t perf_attr_get;
        perf_attr_get.field_mask = UCT_PERF_ATTR_FIELD_OPERATION |
                                   UCT_PERF_ATTR_FIELD_BANDWIDTH;
        perf_attr_get.operation  = UCT_EP_OP_GET_SHORT;
        EXPECT_EQ(iface_estimate_perf(&perf_attr_get), UCS_OK);

        /* Put and get operations have different bandwidth in cuda_copy
           and gdr_copy transports */
        EXPECT_NE(perf_attr.bandwidth.shared, perf_attr_get.bandwidth.shared);
    }
}

UCT_INSTANTIATE_TEST_CASE(test_uct_query)

#if HAVE_CUDA

class test_uct_cuda_copy_bw : public ucs::test {
protected:
    static constexpr double REG_HOST_BW = 30000.0 * UCS_MBYTE;
};

UCS_TEST_F(test_uct_cuda_copy_bw, registered_host_bw_from_pci)
{
    EXPECT_DOUBLE_EQ(REG_HOST_BW,
                     uct_cuda_copy_registered_host_bw(REG_HOST_BW, 0.0));
    EXPECT_DOUBLE_EQ(REG_HOST_BW,
                     uct_cuda_copy_registered_host_bw(REG_HOST_BW,
                                                     UCS_INFINITY));
    EXPECT_DOUBLE_EQ(REG_HOST_BW,
                     uct_cuda_copy_registered_host_bw(REG_HOST_BW,
                                                     30000.0 * UCS_MBYTE));
    EXPECT_DOUBLE_EQ(36000.0 * UCS_MBYTE,
                     uct_cuda_copy_registered_host_bw(REG_HOST_BW,
                                                     40000.0 * UCS_MBYTE));
}

class test_uct_query_cuda_copy : public test_uct_query {
protected:
    static constexpr double LEGACY_H2D_BW = 8300.0 * UCS_MBYTE;
    static constexpr double LEGACY_D2H_BW = 11660.0 * UCS_MBYTE;
    static constexpr double REG_HOST_BW   = 30000.0 * UCS_MBYTE;

    static double sys_dev_pci_bw(ucs_sys_device_t sys_dev)
    {
        ucs_sys_bus_id_t bus_id;
        char sysfs_path[PATH_MAX];
        char bdf_name[UCS_SYS_BDF_NAME_MAX];
        double pci_bw;

        if (sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) {
            return 0.0;
        }

        if (ucs_topo_get_device_bus_id(sys_dev, &bus_id) != UCS_OK) {
            return 0.0;
        }

        ucs_topo_sys_device_bdf_name(sys_dev, bdf_name, sizeof(bdf_name));
        snprintf(sysfs_path, sizeof(sysfs_path), "/sys/bus/pci/devices/%s",
                 bdf_name);
        pci_bw = ucs_topo_get_pci_bw(bdf_name, sysfs_path);
        return ((pci_bw <= 0.0) || (pci_bw == UCS_INFINITY)) ? 0.0 : pci_bw;
    }

    static double registered_host_bw(ucs_sys_device_t cuda_sys_dev)
    {
        return uct_cuda_copy_registered_host_bw(REG_HOST_BW,
                                                sys_dev_pci_bw(cuda_sys_dev));
    }

    static ucs_status_t find_pci_sys_device(double min_pci_bw,
                                            ucs_sys_device_t *sys_dev_p,
                                            double *pci_bw_p)
    {
        static const char *root = "/sys/bus/pci/devices";
        struct dirent *entry;
        DIR *dir;

        dir = opendir(root);
        if (dir == nullptr) {
            return UCS_ERR_NO_ELEM;
        }

        while ((entry = readdir(dir)) != nullptr) {
            if (entry->d_name[0] == '.') {
                continue;
            }

            const std::string sysfs_path =
                    std::string(root) + "/" + entry->d_name;
            if ((access((sysfs_path + "/current_link_width").c_str(), R_OK) !=
                 0) ||
                (access((sysfs_path + "/current_link_speed").c_str(), R_OK) !=
                 0)) {
                continue;
            }

            const double pci_bw =
                    ucs_topo_get_pci_bw(entry->d_name, sysfs_path.c_str());
            if ((pci_bw == UCS_INFINITY) || (pci_bw <= min_pci_bw)) {
                continue;
            }

            if (ucs_topo_find_device_by_bdf_name(entry->d_name, sys_dev_p) ==
                UCS_OK) {
                *pci_bw_p = pci_bw;
                closedir(dir);
                return UCS_OK;
            }
        }

        closedir(dir);
        return UCS_ERR_NO_ELEM;
    }
};

UCS_TEST_P(test_uct_query_cuda_copy, auto_host_device_bw_and_scope)
{
    auto h2d_attr = init_perf_attr();
    auto d2h_attr = init_perf_attr();

    h2d_attr.field_mask |= UCT_PERF_ATTR_FIELD_BANDWIDTH |
                           UCT_PERF_ATTR_FIELD_PATH_BANDWIDTH |
                           UCT_PERF_ATTR_FIELD_BANDWIDTH_SHARED_SCOPE |
                           UCT_PERF_ATTR_FIELD_BANDWIDTH_SHARED_SYS_DEVICE;
    h2d_attr.operation          = UCT_EP_OP_PUT_ZCOPY;
    h2d_attr.local_memory_type  = UCS_MEMORY_TYPE_HOST;
    h2d_attr.remote_memory_type = UCS_MEMORY_TYPE_CUDA;
    h2d_attr.remote_sys_device  = 2;
    EXPECT_EQ(iface_estimate_perf(&h2d_attr), UCS_OK);
    EXPECT_DOUBLE_EQ(LEGACY_H2D_BW, h2d_attr.bandwidth.shared);
    EXPECT_DOUBLE_EQ(LEGACY_H2D_BW, h2d_attr.path_bandwidth.shared);
    EXPECT_EQ(0, h2d_attr.bandwidth.dedicated);
    EXPECT_EQ(UCT_PERF_ATTR_BANDWIDTH_SHARED_SCOPE_SYS_DEVICE,
              h2d_attr.bandwidth_shared_scope);
    EXPECT_EQ(2, h2d_attr.bandwidth_shared_sys_device);

    d2h_attr.field_mask |= UCT_PERF_ATTR_FIELD_BANDWIDTH |
                           UCT_PERF_ATTR_FIELD_PATH_BANDWIDTH |
                           UCT_PERF_ATTR_FIELD_BANDWIDTH_SHARED_SCOPE |
                           UCT_PERF_ATTR_FIELD_BANDWIDTH_SHARED_SYS_DEVICE;
    d2h_attr.operation          = UCT_EP_OP_GET_ZCOPY;
    d2h_attr.local_memory_type  = UCS_MEMORY_TYPE_HOST;
    d2h_attr.remote_memory_type = UCS_MEMORY_TYPE_CUDA;
    d2h_attr.local_sys_device   = UCS_SYS_DEVICE_ID_UNKNOWN;
    d2h_attr.remote_sys_device  = UCS_SYS_DEVICE_ID_UNKNOWN;
    EXPECT_EQ(iface_estimate_perf(&d2h_attr), UCS_OK);
    EXPECT_DOUBLE_EQ(LEGACY_D2H_BW, d2h_attr.bandwidth.shared);
    EXPECT_DOUBLE_EQ(LEGACY_D2H_BW, d2h_attr.path_bandwidth.shared);
    EXPECT_EQ(0, d2h_attr.bandwidth.dedicated);
    EXPECT_EQ(UCT_PERF_ATTR_BANDWIDTH_SHARED_SCOPE_UNKNOWN,
              d2h_attr.bandwidth_shared_scope);
    EXPECT_EQ(UCS_SYS_DEVICE_ID_UNKNOWN,
              d2h_attr.bandwidth_shared_sys_device);
}

UCS_TEST_P(test_uct_query_cuda_copy, auto_bw_uses_host_memory_class,
           "CUDA_COPY_BW=default:10000MBs,h2d:auto,d2h:auto,d2d:320GBs")
{
    auto h2d_unknown            = init_perf_attr();
    auto h2d_reg                = init_perf_attr();
    auto h2d_short              = init_perf_attr();
    auto h2d_reg_unknown_sysdev = init_perf_attr();
    auto d2h_unknown            = init_perf_attr();
    auto d2h_reg                = init_perf_attr();
    auto d2h_remote             = init_perf_attr();
    auto d2h_reg_unknown_sysdev = init_perf_attr();

    h2d_unknown.field_mask |= UCT_PERF_ATTR_FIELD_BANDWIDTH |
                              UCT_PERF_ATTR_FIELD_PATH_BANDWIDTH;
    h2d_unknown.operation          = UCT_EP_OP_PUT_ZCOPY;
    h2d_unknown.local_memory_type  = UCS_MEMORY_TYPE_HOST;
    h2d_unknown.remote_memory_type = UCS_MEMORY_TYPE_CUDA;
    h2d_unknown.remote_sys_device  = 2;
    EXPECT_EQ(iface_estimate_perf(&h2d_unknown), UCS_OK);
    EXPECT_DOUBLE_EQ(LEGACY_H2D_BW, h2d_unknown.bandwidth.shared);
    EXPECT_DOUBLE_EQ(LEGACY_H2D_BW, h2d_unknown.path_bandwidth.shared);
    EXPECT_EQ(0, h2d_unknown.bandwidth.dedicated);

    h2d_reg.field_mask |= UCT_PERF_ATTR_FIELD_BANDWIDTH |
                          UCT_PERF_ATTR_FIELD_PATH_BANDWIDTH |
                          UCT_PERF_ATTR_FIELD_LOCAL_HOST_MEMORY_CLASS;
    h2d_reg.operation                = UCT_EP_OP_PUT_ZCOPY;
    h2d_reg.local_memory_type        = UCS_MEMORY_TYPE_HOST;
    h2d_reg.remote_memory_type       = UCS_MEMORY_TYPE_CUDA;
    h2d_reg.remote_sys_device        = 2;
    h2d_reg.local_host_memory_class  =
            UCT_PERF_ATTR_HOST_MEMORY_CLASS_REGISTERED_LOCKED;
    EXPECT_EQ(iface_estimate_perf(&h2d_reg), UCS_OK);
    const double h2d_reg_bw = registered_host_bw(h2d_reg.remote_sys_device);
    EXPECT_DOUBLE_EQ(h2d_reg_bw, h2d_reg.bandwidth.shared);
    EXPECT_DOUBLE_EQ(h2d_reg.bandwidth.shared,
                     h2d_reg.path_bandwidth.shared);
    EXPECT_EQ(0, h2d_reg.bandwidth.dedicated);

    h2d_short.field_mask |= UCT_PERF_ATTR_FIELD_BANDWIDTH |
                            UCT_PERF_ATTR_FIELD_LOCAL_HOST_MEMORY_CLASS;
    h2d_short.operation                = UCT_EP_OP_PUT_SHORT;
    h2d_short.local_memory_type        = UCS_MEMORY_TYPE_HOST;
    h2d_short.remote_memory_type       = UCS_MEMORY_TYPE_CUDA;
    h2d_short.remote_sys_device        = 2;
    h2d_short.local_host_memory_class  =
            UCT_PERF_ATTR_HOST_MEMORY_CLASS_REGISTERED_LOCKED;
    EXPECT_EQ(iface_estimate_perf(&h2d_short), UCS_OK);
    EXPECT_DOUBLE_EQ(0.95 * LEGACY_H2D_BW, h2d_short.bandwidth.shared);

    h2d_reg_unknown_sysdev.field_mask |=
            UCT_PERF_ATTR_FIELD_BANDWIDTH |
            UCT_PERF_ATTR_FIELD_LOCAL_HOST_MEMORY_CLASS;
    h2d_reg_unknown_sysdev.operation                = UCT_EP_OP_PUT_ZCOPY;
    h2d_reg_unknown_sysdev.local_memory_type        = UCS_MEMORY_TYPE_HOST;
    h2d_reg_unknown_sysdev.remote_memory_type       = UCS_MEMORY_TYPE_CUDA;
    h2d_reg_unknown_sysdev.remote_sys_device        =
            UCS_SYS_DEVICE_ID_UNKNOWN;
    h2d_reg_unknown_sysdev.local_host_memory_class  =
            UCT_PERF_ATTR_HOST_MEMORY_CLASS_REGISTERED_LOCKED;
    EXPECT_EQ(iface_estimate_perf(&h2d_reg_unknown_sysdev), UCS_OK);
    EXPECT_DOUBLE_EQ(REG_HOST_BW,
                     h2d_reg_unknown_sysdev.bandwidth.shared);

    d2h_unknown.field_mask |= UCT_PERF_ATTR_FIELD_BANDWIDTH |
                              UCT_PERF_ATTR_FIELD_PATH_BANDWIDTH;
    d2h_unknown.operation          = UCT_EP_OP_GET_ZCOPY;
    d2h_unknown.local_memory_type  = UCS_MEMORY_TYPE_HOST;
    d2h_unknown.remote_memory_type = UCS_MEMORY_TYPE_CUDA;
    d2h_unknown.remote_sys_device  = 3;
    EXPECT_EQ(iface_estimate_perf(&d2h_unknown), UCS_OK);
    EXPECT_DOUBLE_EQ(LEGACY_D2H_BW, d2h_unknown.bandwidth.shared);
    EXPECT_DOUBLE_EQ(LEGACY_D2H_BW, d2h_unknown.path_bandwidth.shared);
    EXPECT_EQ(0, d2h_unknown.bandwidth.dedicated);

    d2h_reg.field_mask |= UCT_PERF_ATTR_FIELD_BANDWIDTH |
                          UCT_PERF_ATTR_FIELD_PATH_BANDWIDTH |
                          UCT_PERF_ATTR_FIELD_LOCAL_HOST_MEMORY_CLASS;
    d2h_reg.operation                = UCT_EP_OP_GET_ZCOPY;
    d2h_reg.local_memory_type        = UCS_MEMORY_TYPE_HOST;
    d2h_reg.remote_memory_type       = UCS_MEMORY_TYPE_CUDA;
    d2h_reg.remote_sys_device        = 3;
    d2h_reg.local_host_memory_class  =
            UCT_PERF_ATTR_HOST_MEMORY_CLASS_REGISTERED_LOCKED;
    EXPECT_EQ(iface_estimate_perf(&d2h_reg), UCS_OK);
    const double d2h_reg_bw = registered_host_bw(d2h_reg.remote_sys_device);
    EXPECT_DOUBLE_EQ(d2h_reg_bw, d2h_reg.bandwidth.shared);
    EXPECT_DOUBLE_EQ(d2h_reg.bandwidth.shared,
                     d2h_reg.path_bandwidth.shared);
    EXPECT_EQ(0, d2h_reg.bandwidth.dedicated);

    d2h_reg_unknown_sysdev.field_mask |=
            UCT_PERF_ATTR_FIELD_BANDWIDTH |
            UCT_PERF_ATTR_FIELD_LOCAL_HOST_MEMORY_CLASS;
    d2h_reg_unknown_sysdev.operation                = UCT_EP_OP_GET_ZCOPY;
    d2h_reg_unknown_sysdev.local_memory_type        = UCS_MEMORY_TYPE_HOST;
    d2h_reg_unknown_sysdev.remote_memory_type       = UCS_MEMORY_TYPE_CUDA;
    d2h_reg_unknown_sysdev.remote_sys_device        =
            UCS_SYS_DEVICE_ID_UNKNOWN;
    d2h_reg_unknown_sysdev.local_host_memory_class  =
            UCT_PERF_ATTR_HOST_MEMORY_CLASS_REGISTERED_LOCKED;
    EXPECT_EQ(iface_estimate_perf(&d2h_reg_unknown_sysdev), UCS_OK);
    EXPECT_DOUBLE_EQ(REG_HOST_BW,
                     d2h_reg_unknown_sysdev.bandwidth.shared);

    d2h_remote.field_mask |= UCT_PERF_ATTR_FIELD_BANDWIDTH |
                             UCT_PERF_ATTR_FIELD_REMOTE_HOST_MEMORY_CLASS;
    d2h_remote.operation                 = UCT_EP_OP_PUT_ZCOPY;
    d2h_remote.local_memory_type         = UCS_MEMORY_TYPE_CUDA;
    d2h_remote.remote_memory_type        = UCS_MEMORY_TYPE_HOST;
    d2h_remote.local_sys_device          = 3;
    d2h_remote.remote_host_memory_class  =
            UCT_PERF_ATTR_HOST_MEMORY_CLASS_REGISTERED_LOCKED;
    EXPECT_EQ(iface_estimate_perf(&d2h_remote), UCS_OK);
    EXPECT_DOUBLE_EQ(registered_host_bw(d2h_remote.local_sys_device),
                     d2h_remote.bandwidth.shared);
}

UCS_TEST_P(test_uct_query_cuda_copy, auto_registered_host_bw_uses_pci_link,
           "CUDA_COPY_BW=default:10000MBs,h2d:auto,d2h:auto,d2d:320GBs")
{
    const double min_pci_bw = REG_HOST_BW /
                              UCT_CUDA_COPY_REG_HOST_PCI_BW_FACTOR;
    ucs_sys_device_t sys_dev;
    double pci_bw;

    if (find_pci_sys_device(min_pci_bw, &sys_dev, &pci_bw) != UCS_OK) {
        UCS_TEST_SKIP_R("no PCIe sysfs device above registered-host fallback");
    }

    const double expected_bw = registered_host_bw(sys_dev);
    auto h2d_reg            = init_perf_attr();
    auto d2h_reg            = init_perf_attr();

    h2d_reg.field_mask |= UCT_PERF_ATTR_FIELD_BANDWIDTH |
                          UCT_PERF_ATTR_FIELD_PATH_BANDWIDTH |
                          UCT_PERF_ATTR_FIELD_LOCAL_HOST_MEMORY_CLASS;
    h2d_reg.operation               = UCT_EP_OP_PUT_ZCOPY;
    h2d_reg.local_memory_type       = UCS_MEMORY_TYPE_HOST;
    h2d_reg.remote_memory_type      = UCS_MEMORY_TYPE_CUDA;
    h2d_reg.remote_sys_device       = sys_dev;
    h2d_reg.local_host_memory_class =
            UCT_PERF_ATTR_HOST_MEMORY_CLASS_REGISTERED_LOCKED;
    EXPECT_EQ(iface_estimate_perf(&h2d_reg), UCS_OK);
    EXPECT_DOUBLE_EQ(expected_bw, h2d_reg.bandwidth.shared);
    EXPECT_DOUBLE_EQ(h2d_reg.bandwidth.shared,
                     h2d_reg.path_bandwidth.shared);
    EXPECT_EQ(0, h2d_reg.bandwidth.dedicated);

    d2h_reg.field_mask |= UCT_PERF_ATTR_FIELD_BANDWIDTH |
                          UCT_PERF_ATTR_FIELD_PATH_BANDWIDTH |
                          UCT_PERF_ATTR_FIELD_REMOTE_HOST_MEMORY_CLASS;
    d2h_reg.operation                = UCT_EP_OP_PUT_ZCOPY;
    d2h_reg.local_memory_type        = UCS_MEMORY_TYPE_CUDA;
    d2h_reg.remote_memory_type       = UCS_MEMORY_TYPE_HOST;
    d2h_reg.local_sys_device         = sys_dev;
    d2h_reg.remote_host_memory_class =
            UCT_PERF_ATTR_HOST_MEMORY_CLASS_REGISTERED_LOCKED;
    EXPECT_EQ(iface_estimate_perf(&d2h_reg), UCS_OK);
    EXPECT_DOUBLE_EQ(expected_bw, d2h_reg.bandwidth.shared);
    EXPECT_DOUBLE_EQ(d2h_reg.bandwidth.shared,
                     d2h_reg.path_bandwidth.shared);
    EXPECT_EQ(0, d2h_reg.bandwidth.dedicated);
}

UCS_TEST_P(test_uct_query_cuda_copy, explicit_bw_overrides_host_memory_class,
           "CUDA_COPY_BW=default:10000MBs,h2d:12345MBs,d2h:23456MBs,d2d:320GBs")
{
    auto h2d_reg = init_perf_attr();
    auto d2h_reg = init_perf_attr();

    h2d_reg.field_mask |= UCT_PERF_ATTR_FIELD_BANDWIDTH |
                          UCT_PERF_ATTR_FIELD_LOCAL_HOST_MEMORY_CLASS;
    h2d_reg.operation               = UCT_EP_OP_PUT_ZCOPY;
    h2d_reg.local_memory_type       = UCS_MEMORY_TYPE_HOST;
    h2d_reg.remote_memory_type      = UCS_MEMORY_TYPE_CUDA;
    h2d_reg.remote_sys_device       = 2;
    h2d_reg.local_host_memory_class =
            UCT_PERF_ATTR_HOST_MEMORY_CLASS_REGISTERED_LOCKED;
    EXPECT_EQ(iface_estimate_perf(&h2d_reg), UCS_OK);
    EXPECT_DOUBLE_EQ(12345.0 * UCS_MBYTE, h2d_reg.bandwidth.shared);

    d2h_reg.field_mask |= UCT_PERF_ATTR_FIELD_BANDWIDTH |
                          UCT_PERF_ATTR_FIELD_REMOTE_HOST_MEMORY_CLASS;
    d2h_reg.operation                = UCT_EP_OP_PUT_ZCOPY;
    d2h_reg.local_memory_type        = UCS_MEMORY_TYPE_CUDA;
    d2h_reg.remote_memory_type       = UCS_MEMORY_TYPE_HOST;
    d2h_reg.local_sys_device         = 3;
    d2h_reg.remote_host_memory_class =
            UCT_PERF_ATTR_HOST_MEMORY_CLASS_REGISTERED_LOCKED;
    EXPECT_EQ(iface_estimate_perf(&d2h_reg), UCS_OK);
    EXPECT_DOUBLE_EQ(23456.0 * UCS_MBYTE, d2h_reg.bandwidth.shared);
}

_UCT_INSTANTIATE_TEST_CASE(test_uct_query_cuda_copy, cuda_copy)

#endif

class test_uct_query_ib : public test_uct_query {
public:
    double get_attr_latency_c() const;
};

double test_uct_query_ib::get_attr_latency_c() const
{
    return get_iface_attr().latency.c;
}

UCS_TEST_P(test_uct_query_ib, send_overhead,
           "IB_SEND_OVERHEAD=bcopy:" UCS_PP_MAKE_STRING(IB_SEND_OVERHEAD_BCOPY)
           ",cqe:" UCS_PP_MAKE_STRING(IB_SEND_OVERHEAD_CQE) ",db:"
           UCS_PP_MAKE_STRING(IB_SEND_OVERHEAD_DB) ",wqe_fetch:"
           UCS_PP_MAKE_STRING(IB_SEND_OVERHEAD_WQE_FETCH) ",wqe_post:"
           UCS_PP_MAKE_STRING(IB_SEND_OVERHEAD_WQE_POST))
{
    auto perf_attr        = init_perf_attr();
    perf_attr.field_mask |= UCT_PERF_ATTR_FIELD_SEND_PRE_OVERHEAD |
                            UCT_PERF_ATTR_FIELD_SEND_POST_OVERHEAD |
                            UCT_PERF_ATTR_FIELD_LATENCY;

    for (auto i = int(UCT_EP_OP_AM_SHORT); i < int(UCT_EP_OP_LAST); ++i) {
        auto op             = uct_ep_operation_t(i);
        perf_attr.operation = op;
        EXPECT_EQ(iface_estimate_perf(&perf_attr), UCS_OK);

        const float post_overhead = uct_ep_op_is_zcopy(op) ?
                IB_SEND_OVERHEAD_DB + IB_SEND_OVERHEAD_CQE :
                IB_SEND_OVERHEAD_DB;
        const float pre_overhead  = uct_ep_op_is_bcopy(op) ?
                IB_SEND_OVERHEAD_WQE_POST + IB_SEND_OVERHEAD_BCOPY :
                IB_SEND_OVERHEAD_WQE_POST;
        const float latency_c     = (uct_ep_op_is_bcopy(op) ||
                                     uct_ep_op_is_zcopy(op)) ?
                get_attr_latency_c() + IB_SEND_OVERHEAD_WQE_FETCH :
                get_attr_latency_c();

        EXPECT_FLOAT_EQ(perf_attr.send_post_overhead, post_overhead);
        EXPECT_FLOAT_EQ(perf_attr.send_pre_overhead, pre_overhead);
        EXPECT_FLOAT_EQ(perf_attr.latency.c, latency_c);
    }
}

UCT_INSTANTIATE_IB_TEST_CASE(test_uct_query_ib);

class test_uct_query_mm : public test_uct_query {
};

UCS_TEST_P(test_uct_query_mm, send_recv_overhead,
           "MM_SEND_OVERHEAD=am_short:"
           UCS_PP_MAKE_STRING(MM_SEND_OVERHEAD_AM_SHORT) ",am_bcopy:"
           UCS_PP_MAKE_STRING(MM_SEND_OVERHEAD_AM_BCOPY),
           "MM_RECV_OVERHEAD=am_short:"
           UCS_PP_MAKE_STRING(MM_RECV_OVERHEAD_AM_SHORT) ",am_bcopy:"
           UCS_PP_MAKE_STRING(MM_RECV_OVERHEAD_AM_BCOPY))
{
    auto perf_attr        = init_perf_attr();
    perf_attr.field_mask |= UCT_PERF_ATTR_FIELD_SEND_PRE_OVERHEAD |
                            UCT_PERF_ATTR_FIELD_RECV_OVERHEAD;

    EXPECT_EQ(iface_estimate_perf(&perf_attr), UCS_OK);
    EXPECT_FLOAT_EQ(perf_attr.send_pre_overhead, MM_SEND_OVERHEAD_AM_SHORT);
    EXPECT_FLOAT_EQ(perf_attr.recv_overhead, MM_RECV_OVERHEAD_AM_SHORT);

    perf_attr.operation = UCT_EP_OP_AM_BCOPY;
    EXPECT_EQ(iface_estimate_perf(&perf_attr), UCS_OK);
    EXPECT_FLOAT_EQ(perf_attr.send_pre_overhead, MM_SEND_OVERHEAD_AM_BCOPY);
    EXPECT_FLOAT_EQ(perf_attr.recv_overhead, MM_RECV_OVERHEAD_AM_BCOPY);
}

UCS_TEST_P(test_uct_query_mm, default_shared_bandwidth_scope)
{
    auto perf_attr = init_perf_attr();

    perf_attr.bandwidth_shared_scope =
            UCT_PERF_ATTR_BANDWIDTH_SHARED_SCOPE_UNKNOWN;
    perf_attr.bandwidth_shared_sys_device = 2;
    perf_attr.field_mask |= UCT_PERF_ATTR_FIELD_BANDWIDTH_SHARED_SCOPE |
                            UCT_PERF_ATTR_FIELD_BANDWIDTH_SHARED_SYS_DEVICE;

    EXPECT_EQ(iface_estimate_perf(&perf_attr), UCS_OK);
    EXPECT_EQ(UCT_PERF_ATTR_BANDWIDTH_SHARED_SCOPE_NODE,
              perf_attr.bandwidth_shared_scope);
    EXPECT_EQ(UCS_SYS_DEVICE_ID_UNKNOWN,
              perf_attr.bandwidth_shared_sys_device);
}

UCT_INSTANTIATE_MM_TEST_CASE(test_uct_query_mm)
