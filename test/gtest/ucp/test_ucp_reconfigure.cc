/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"
#include "common/test.h"

extern "C" {
#include <ucp/core/ucp_ep.inl>
#include <ucp/wireup/address.h>
#include <ucp/wireup/wireup_ep.h>
}

class test_ucp_reconfigure : public ucp_test {
protected:
    using address_t = std::pair<void*, ucp_unpacked_address_t>;
    using address_p = std::unique_ptr<address_t, void (*)(address_t*)>;

    class entity {
    public:
        entity(const ucp_test_base::entity *e, unsigned init_flags,
               bool reuse_lanes, bool asymmetric_scale, bool single_transport) :
            m_worker_entity(e),
            m_init_flags(init_flags),
            m_reuse_lanes(reuse_lanes),
            m_asymmetric_scale(asymmetric_scale),
            m_single_transport(single_transport)
        {
        }

        void connect(const entity &other, bool exclude_ifaces);

        void verify_configuration(const entity &other) const;

        bool is_reconfigured() const
        {
            return m_cfg_index != ep()->cfg_index;
        }

        ucp_ep_h ep() const
        {
            return m_ep;
        }

        ucp_worker_h worker() const
        {
            return m_worker_entity->worker();
        }

        void cleanup()
        {
            if (m_ep != nullptr) {
                ucp_ep_destroy(m_ep);
            }
        }

    private:
        void store_config();
        ucp_tl_bitmap_t ep_tl_bitmap(bool non_reused_only = false) const;
        address_p get_address(bool ep_only) const;
        bool has_matching_lane(ucp_ep_h ep, ucp_lane_index_t lane_idx,
                               const entity &other) const;
        unsigned num_reused() const;

        ucp_worker_cfg_index_t       m_cfg_index = UCP_WORKER_CFG_INDEX_NULL;
        ucp_ep_h                     m_ep        = nullptr;
        std::vector<uct_ep_h>        m_uct_eps;
        const ucp_test_base::entity *m_worker_entity;
        unsigned                     m_init_flags;
        bool                         m_reuse_lanes;
        bool                         m_asymmetric_scale;
        bool                         m_single_transport;
    };

    typedef enum {
        EXCLUDE_IFACES   = 0,
        ASYMMETRIC_SCALE = 1,
        RANK_UPDATE      = 2
    } method_t;

    bool is_single_transport() const
    {
        return GetParam().transports.size() == 1;
    }

    void create_entities_and_connect(unsigned init_flags, method_t method,
                                     bool reuse_lanes);

    std::unique_ptr<entity> m_sender;
    std::unique_ptr<entity> m_receiver;

public:
    void init()
    {
        ucp_test::init();

        /* num_tls = single device + UD */
        if (sender().ucph()->num_tls <= 2) {
            UCS_TEST_SKIP_R("test requires at least 2 ifaces to work");
        }

        if (!is_proto_enabled()) {
            UCS_TEST_SKIP_R("protov1 is not supported (has no protocol reset)");
        }
    }

    void cleanup()
    {
        if (m_sender) {
            m_sender->cleanup();
        }

        if (m_receiver) {
            m_receiver->cleanup();
        }

        ucp_test::cleanup();
    }

    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant_with_value(variants, UCP_FEATURE_TAG, 0, "");
    }

    void run(method_t method, unsigned init_flags = UCP_EP_INIT_CREATE_AM_LANE,
             bool bidirectional = false, bool reuse_lanes = false);

    void send_message(const entity &e1, const entity &e2,
                      const std::string &sbuf, const std::string &rbuf,
                      std::vector<void*> &reqs)
    {
        const ucp_request_param_t param = {
            .op_attr_mask = UCP_OP_ATTR_FLAG_NO_IMM_CMPL
        };

        void *sreq = ucp_tag_send_nbx(e1.ep(), sbuf.c_str(), msg_size, 0,
                                      &param);
        void *rreq = ucp_tag_recv_nbx(e2.worker(), (void*)rbuf.c_str(),
                                      msg_size, 0, 0, &param);
        reqs.push_back(rreq);
        reqs.push_back(sreq);
    }

    void init_buffers(std::vector<std::string> &sbufs,
                      std::vector<std::string> &rbufs)
    {
        for (unsigned i = 0; i < num_iterations; ++i) {
            sbufs[i].resize(msg_size);
            rbufs[i].resize(msg_size);
            std::fill(sbufs[i].begin(), sbufs[i].end(), 'a');
            std::fill(rbufs[i].begin(), rbufs[i].end(), 'b');
        }
    }

    void send_recv(bool bidirectional = false)
    {
        std::vector<std::string> sbufs(num_iterations), rbufs(num_iterations);
        /* Buffers for the opposite direction */
        std::vector<std::string> o_sbufs(num_iterations),
                o_rbufs(num_iterations);
        std::vector<void*> reqs;

        init_buffers(sbufs, rbufs);
        init_buffers(o_sbufs, o_rbufs);

        for (unsigned i = 0; i < num_iterations; ++i) {
            send_message(*m_sender.get(), *m_receiver.get(), sbufs[i], rbufs[i],
                         reqs);

            if (bidirectional) {
                send_message(*m_receiver.get(), *m_sender.get(), o_sbufs[i],
                             o_rbufs[i], reqs);
            }
        }

        requests_wait(reqs);
        EXPECT_EQ(sbufs, rbufs);

        if (bidirectional) {
            EXPECT_EQ(o_sbufs, o_rbufs);
        }
    }

    static constexpr size_t msg_size         = 16 * UCS_KBYTE;
    static constexpr unsigned num_iterations = 1000;
};

unsigned test_ucp_reconfigure::entity::num_reused() const
{
    if (m_reuse_lanes) {
        return ucp_ep_num_lanes(ep()) / 2;
    }

    unsigned num_shm = 0;
    auto context     = m_worker_entity->ucph();

    for (auto lane = 0; lane < ucp_ep_num_lanes(ep()); ++lane) {
        auto rsc_idx = ucp_ep_get_rsc_index(ep(), lane);
        num_shm     += (context->tl_rscs[rsc_idx].tl_rsc.dev_type ==
                        UCT_DEVICE_TYPE_SHM);
    }

    return m_asymmetric_scale ? num_shm : 0;
}

void test_ucp_reconfigure::entity::store_config()
{
    for (ucp_lane_index_t lane = 0; lane < ucp_ep_num_lanes(ep()); ++lane) {
        uct_ep_h uct_ep = ucp_ep_get_lane(ep(), lane);

        /* Store UCT endpoints in order to compare with next configuration */
        if (ucp_wireup_ep_test(uct_ep)) {
            m_uct_eps.push_back(ucp_wireup_ep(uct_ep)->super.uct_ep);
        } else {
            m_uct_eps.push_back(uct_ep);
        }
    }

    m_cfg_index = ep()->cfg_index;
}

ucp_tl_bitmap_t
test_ucp_reconfigure::entity::ep_tl_bitmap(bool non_reused_only) const
{
    ucp_tl_bitmap_t tl_bitmap = UCS_STATIC_BITMAP_ZERO_INITIALIZER;
    size_t num_tls            = 0;
    auto context              = m_worker_entity->ucph();
    ucp_rsc_index_t rsc_idx;

    if ((ep() == NULL) && m_single_transport) {
        UCS_STATIC_BITMAP_FOR_EACH_BIT(rsc_idx, &context->tl_bitmap) {
            if (++num_tls > (context->num_tls / 2)) {
                UCS_STATIC_BITMAP_SET(&tl_bitmap, rsc_idx);
            }
        }
    } else if (ep() != NULL) {
        auto first_lane = non_reused_only ? num_reused() : 0;
        for (auto lane = first_lane; lane < ucp_ep_num_lanes(ep()); ++lane) {
            UCS_STATIC_BITMAP_SET(&tl_bitmap, ucp_ep_get_rsc_index(ep(), lane));
        }
    }

    return tl_bitmap;
}

void test_ucp_reconfigure::entity::connect(const entity &other,
                                           bool is_exclude_iface)
{
    auto worker_addr = other.get_address(false);
    ucp_tl_bitmap_t tl_bitmap;
    unsigned addr_indices[UCP_MAX_LANES];

    tl_bitmap = is_exclude_iface ? UCS_STATIC_BITMAP_NOT(
                                           other.ep_tl_bitmap(m_reuse_lanes)) :
                                   ucp_tl_bitmap_max;

    UCS_ASYNC_BLOCK(&worker()->async);
    ASSERT_UCS_OK(ucp_ep_create_to_worker_addr(worker(), &tl_bitmap,
                                               &worker_addr->second,
                                               m_init_flags, "reconfigure test",
                                               addr_indices, &m_ep));

    m_ep->conn_sn = 0;
    ASSERT_TRUE(ucp_ep_match_insert(worker(), m_ep, worker_addr->second.uuid,
                                    m_ep->conn_sn, UCS_CONN_MATCH_QUEUE_EXP));

    if (!(m_ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED)) {
        ASSERT_UCS_OK(ucp_wireup_send_request(m_ep));
    }

    UCS_ASYNC_UNBLOCK(&worker()->async);
    store_config();
}

bool test_ucp_reconfigure::entity::has_matching_lane(ucp_ep_h ep,
                                                     ucp_lane_index_t lane_idx,
                                                     const entity &other) const
{
    const auto lane     = &ucp_ep_config(ep)->key.lanes[lane_idx];
    auto context        = m_worker_entity->ucph();
    const auto resource = &context->tl_rscs[ucp_ep_get_rsc_index(ep, lane_idx)];
    auto addr           = other.get_address(true);
    const ucp_address_entry_t *ae;

    ucs_carray_for_each(ae, addr->second.address_list,
                        addr->second.address_count) {
        if ((resource->tl_name_csum == ae->tl_name_csum) &&
            ucp_wireup_is_lane_connected(ep, lane_idx, ae)) {
            EXPECT_EQ(ae->sys_dev, lane->dst_sys_dev);
            EXPECT_EQ(ae->md_index, lane->dst_md_index);
            return true;
        }
    }

    return false;
}

void test_ucp_reconfigure::entity::verify_configuration(
        const entity &other) const
{
    unsigned reused                  = 0;
    const ucp_lane_index_t num_lanes = ucp_ep_num_lanes(ep());

    EXPECT_EQ(num_lanes, ucp_ep_num_lanes(other.ep()));

    for (ucp_lane_index_t lane = 0; lane < num_lanes; ++lane) {
        /* Verify local and remote lanes are identical */
        EXPECT_TRUE(has_matching_lane(ep(), lane, other));

        /* Verify correct number of reused lanes */
        auto uct_ep = ucp_ep_get_lane(ep(), lane);
        auto it     = std::find(m_uct_eps.begin(), m_uct_eps.end(), uct_ep);
        reused     += (it != m_uct_eps.end());
    }

    EXPECT_EQ(reused, is_reconfigured() ? num_reused() : num_lanes);
}

test_ucp_reconfigure::address_p
test_ucp_reconfigure::entity::get_address(bool ep_only) const
{
    const ucp_tl_bitmap_t tl_bitmap = ep_only ? ep_tl_bitmap() :
                                                ucp_tl_bitmap_max;
    size_t addr_len;

    address_p address(new address_t(nullptr, {0}), [](address_t *addr) {
        ucs_free(addr->first);
        ucs_free(addr->second.address_list);
        delete addr;
    });

    unsigned flags = ucp_worker_default_address_pack_flags(worker());
    auto context   = m_worker_entity->ucph();

    ASSERT_UCS_OK(ucp_address_pack(worker(), ep(), &tl_bitmap, flags,
                                   context->config.ext.worker_addr_version,
                                   NULL, UINT_MAX, &addr_len, &address->first));

    ASSERT_UCS_OK(ucp_address_unpack(worker(), address->first, flags,
                                     &address->second));
    return address;
}

void test_ucp_reconfigure::create_entities_and_connect(unsigned init_flags,
                                                       method_t method,
                                                       bool reuse_lanes)
{
    bool is_asymmetric_scale = (method == ASYMMETRIC_SCALE);

    m_sender.reset(new entity(create_entity(true), init_flags, reuse_lanes,
                              is_asymmetric_scale, is_single_transport()));

    if (is_asymmetric_scale) {
        modify_config("NUM_EPS", "200");
    }

    m_receiver.reset(new entity(create_entity(false), init_flags, reuse_lanes,
                                is_asymmetric_scale, is_single_transport()));

    m_sender->connect(*m_receiver.get(), method == EXCLUDE_IFACES);
    m_receiver->connect(*m_sender.get(), method == EXCLUDE_IFACES);
}

void test_ucp_reconfigure::run(method_t method, unsigned init_flags,
                               bool bidirectional, bool reuse_lanes)
{
    create_entities_and_connect(init_flags, method, reuse_lanes);
    send_recv(bidirectional);

    EXPECT_NE(m_sender->is_reconfigured(), m_receiver->is_reconfigured());
    m_sender->verify_configuration(*m_receiver.get());
    m_receiver->verify_configuration(*m_sender.get());
}

class test_reconfigure_exclude_ifaces : public test_ucp_reconfigure {
public:
    void init()
    {
        test_ucp_reconfigure::init();

        if (has_transport("ib") && !has_resource(sender(), "rc_verbs")) {
            UCS_TEST_SKIP_R("wireup is not triggered without p2p ifaces");
        }
    }
};

UCS_TEST_P(test_reconfigure_exclude_ifaces, basic)
{
    run(EXCLUDE_IFACES);
}

UCS_TEST_P(test_reconfigure_exclude_ifaces, request_reset,
           "PROTO_REQUEST_RESET=y")
{
    run(EXCLUDE_IFACES);
}

UCS_TEST_P(test_reconfigure_exclude_ifaces, reuse_lanes)
{
    /* Use single path so that num_lanes/2 will only exclude some lanes and
     * not all. */
    modify_config("IB_NUM_PATHS", "1", SETENV_IF_NOT_EXIST);
    run(EXCLUDE_IFACES, UCP_EP_INIT_CREATE_AM_LANE, false, true);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_reconfigure_exclude_ifaces, rc, "rc_v")
UCP_INSTANTIATE_TEST_CASE_TLS(test_reconfigure_exclude_ifaces, rcx, "rc_x")
UCP_INSTANTIATE_TEST_CASE_TLS(test_reconfigure_exclude_ifaces, shm_ib, "shm,ib")

class test_reconfigure_resolve_remote : public test_ucp_reconfigure {
};

UCS_TEST_P(test_reconfigure_resolve_remote, resolve_id, "RNDV_THRESH=0")
{
    if (has_transport("tcp")) {
        UCS_TEST_SKIP_R("asymmetric setup is not supported for this transport "
                        "due to reachability bug in TCP (ib0 connects to ib1 "
                        "and fails)");
    }

    if (has_transport("dc_x")) {
        UCS_TEST_SKIP_R("extra path added for AM_LANE when using DC");
    }

    run(EXCLUDE_IFACES, UCP_EP_INIT_CREATE_AM_LANE, true);
}

UCP_INSTANTIATE_TEST_CASE(test_reconfigure_resolve_remote)

class test_reconfigure_asym_scale : public test_ucp_reconfigure {
};

UCS_TEST_P(test_reconfigure_asym_scale, basic)
{
    run(ASYMMETRIC_SCALE);
}

UCS_TEST_P(test_reconfigure_asym_scale, request_reset, "PROTO_REQUEST_RESET=y")
{
    run(ASYMMETRIC_SCALE);
}

UCS_TEST_P(test_reconfigure_asym_scale, resolve_id, "RNDV_THRESH=0")
{
    run(ASYMMETRIC_SCALE, UCP_EP_INIT_CREATE_AM_LANE, true);
}

UCS_TEST_P(test_reconfigure_asym_scale, reuse_lanes)
{
    modify_config("IB_NUM_PATHS", "1", SETENV_IF_NOT_EXIST);
    run(ASYMMETRIC_SCALE, UCP_EP_INIT_CREATE_AM_LANE, false, true);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_reconfigure_asym_scale, shm_ib, "shm,ib");

class test_reconfigure_update_rank : public test_ucp_reconfigure {
};

UCS_TEST_P(test_reconfigure_update_rank, promote,
           "DYNAMIC_TL_SWITCH_INTERVAL=3s", "NUM_EPS=200", "RNDV_THRESH=inf")
{
    create_entities_and_connect(UCP_EP_INIT_CREATE_AM_LANE, RANK_UPDATE, false);
    send_recv();

    ucs_usage_tracker_set_min_score(m_sender->worker()->usage_tracker.handle,
                                    m_sender->ep(), 0.7);
    ucp_wireup_send_promotion_request(m_sender->ep(), NULL, 1);
    send_recv();
}

UCS_TEST_P(test_reconfigure_update_rank, demote,
           "DYNAMIC_TL_SWITCH_INTERVAL=3s", "RNDV_THRESH=inf")
{
    create_entities_and_connect(UCP_EP_INIT_CREATE_AM_LANE, RANK_UPDATE, false);
    send_recv();

    m_sender->worker()->context->config.est_num_eps   = 200;
    m_receiver->worker()->context->config.est_num_eps = 200;
    ucp_wireup_send_demotion_request(m_sender->ep(), NULL, 1);
    send_recv();
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_reconfigure_update_rank, shm_ib, "shm,ib");
