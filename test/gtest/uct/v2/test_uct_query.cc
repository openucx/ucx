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
#include <uct/base/uct_iface.h>
}


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
    EXPECT_EQ(iface_estimate_perf(&perf_attr), UCS_OK);

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

UCT_INSTANTIATE_MM_TEST_CASE(test_uct_query_mm)
