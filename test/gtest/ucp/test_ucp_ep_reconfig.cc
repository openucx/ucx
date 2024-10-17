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

class test_ucp_ep_reconfig : public ucp_test {
protected:
    using address_t = std::pair<void*, ucp_unpacked_address_t>;
    using address_p = std::unique_ptr<address_t, void (*)(address_t*)>;

    class entity : public ucp_test_base::entity {
    public:
        entity(const ucp_test_param &test_param, ucp_config_t *ucp_config,
               const ucp_worker_params_t &worker_params,
               const ucp_test_base *test_owner, bool exclude_ifaces) :
            ucp_test_base::entity(test_param, ucp_config, worker_params,
                                  test_owner),
            m_exclude_ifaces(exclude_ifaces)
        {
        }

        void connect(const ucp_test_base::entity *other,
                     const ucp_ep_params_t &ep_params, int ep_idx = 0,
                     int do_set_ep = 1) override;

        void verify_configuration(const entity &other) const;

        bool is_reconfigured() const
        {
            return m_cfg_index != ep()->cfg_index;
        }

    private:
        void store_config();
        ucp_tl_bitmap_t ep_tl_bitmap() const;
        address_p get_address(const ucp_tl_bitmap_t &tl_bitmap) const;
        bool is_lane_connected(ucp_ep_h ep, ucp_lane_index_t lane_idx,
                               const entity &other) const;

        ucp_worker_cfg_index_t m_cfg_index = UCP_WORKER_CFG_INDEX_NULL;
        std::vector<uct_ep_h>  m_uct_eps;
        bool                   m_exclude_ifaces;
    };

    bool is_single_transport()
    {
        return GetParam().transports.size() == 1;
    }

    bool has_p2p_transport()
    {
        return has_resource(sender(), "rc_verbs") ||
               has_resource(sender(), "rc_mlx5");
    }

    void create_entity(bool push_front, bool exclude_ifaces)
    {
        auto e = new entity(GetParam(), m_ucp_config, get_worker_params(), this,
                            exclude_ifaces);

        if (push_front) {
            m_entities.push_front(e);
        } else {
            m_entities.push_back(e);
        }
    }

    virtual void create_entities_and_connect()
    {
        create_entity(true, true);
        create_entity(false, true);
        sender().connect(&receiver(), get_ep_params());
        receiver().connect(&sender(), get_ep_params());
    }

public:
    void init()
    {
        ucp_test::init();

        /* num_tls = single device + UD */
        if (sender().ucph()->num_tls <= 2) {
            UCS_TEST_SKIP_R("test requires at least 2 ifaces to work");
        }
    }

    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant_with_value(variants, UCP_FEATURE_TAG, 0, "");
    }

    void run(bool bidirectional = false);
    void skip_non_p2p();

    void send_message(const ucp_test_base::entity &e1,
                      const ucp_test_base::entity &e2, const mem_buffer &sbuf,
                      const mem_buffer &rbuf, size_t msg_size,
                      std::vector<void*> &reqs)
    {
        const ucp_request_param_t param = {
            .op_attr_mask = UCP_OP_ATTR_FLAG_NO_IMM_CMPL
        };

        void *sreq = ucp_tag_send_nbx(e1.ep(), sbuf.ptr(), msg_size, 0, &param);
        void *sreq_sync = ucp_tag_send_sync_nbx(e1.ep(), sbuf.ptr(), msg_size,
                                                0, &param);
        reqs.insert(reqs.end(), {sreq, sreq_sync});

        for (unsigned iter = 0; iter < 2; iter++) {
            void *rreq = ucp_tag_recv_nbx(e2.worker(), rbuf.ptr(), msg_size, 0,
                                          0, &param);
            reqs.push_back(rreq);
        }
    }

    void send_recv(bool bidirectional)
    {
        static const unsigned num_iterations = 1000;
/* TODO: remove this when 100MB asan bug is solved */
#ifdef __SANITIZE_ADDRESS__
        static const size_t msg_sizes[] = {8, 1024, 16384, 65536};
#else
        static const size_t msg_sizes[] = {8, 1024, 16384, UCS_MBYTE};
#endif

        for (auto msg_size : msg_sizes) {
            for (unsigned i = 0; i < num_iterations; ++i) {
                mem_buffer sbuf(msg_size, UCS_MEMORY_TYPE_HOST, i);
                mem_buffer rbuf(msg_size, UCS_MEMORY_TYPE_HOST, ucs::rand());
                mem_buffer o_sbuf(msg_size, UCS_MEMORY_TYPE_HOST, i);
                mem_buffer o_rbuf(msg_size, UCS_MEMORY_TYPE_HOST, ucs::rand());

                std::vector<void*> reqs;
                send_message(sender(), receiver(), sbuf, rbuf, msg_size, reqs);

                if (bidirectional) {
                    send_message(receiver(), sender(), o_sbuf, o_rbuf, msg_size,
                                 reqs);
                }

                requests_wait(reqs);
                rbuf.pattern_check(i);

                if (bidirectional) {
                    o_rbuf.pattern_check(i);
                }
            }
        }
    }
};

void test_ucp_ep_reconfig::entity::store_config()
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

ucp_tl_bitmap_t test_ucp_ep_reconfig::entity::ep_tl_bitmap() const
{
    if (ep() == NULL) {
        return UCS_STATIC_BITMAP_ZERO_INITIALIZER;
    }

    ucp_tl_bitmap_t tl_bitmap = UCS_STATIC_BITMAP_ZERO_INITIALIZER;
    for (ucp_lane_index_t lane = 0; lane < ucp_ep_num_lanes(ep()); ++lane) {
        UCS_STATIC_BITMAP_SET(&tl_bitmap, ucp_ep_get_rsc_index(ep(), lane));
    }

    return tl_bitmap;
}

void test_ucp_ep_reconfig::entity::connect(const ucp_test_base::entity *other,
                                           const ucp_ep_params_t &ep_params,
                                           int ep_idx, int do_set_ep)
{
    auto r_other     = static_cast<const entity*>(other);
    auto worker_addr = r_other->get_address(ucp_tl_bitmap_max);
    ucp_tl_bitmap_t tl_bitmap;
    unsigned addr_indices[UCP_MAX_LANES];
    ucp_ep_h ucp_ep;

    tl_bitmap = m_exclude_ifaces ?
                        UCS_STATIC_BITMAP_NOT(r_other->ep_tl_bitmap()) :
                        ucp_tl_bitmap_max;

    UCS_ASYNC_BLOCK(&worker()->async);
    ASSERT_UCS_OK(ucp_ep_create_to_worker_addr(worker(), &tl_bitmap,
                                               &worker_addr->second,
                                               UCP_EP_INIT_CREATE_AM_LANE,
                                               "reconfigure test", addr_indices,
                                               &ucp_ep));

    ucp_ep->conn_sn = 0;
    ASSERT_TRUE(ucp_ep_match_insert(worker(), ucp_ep, worker_addr->second.uuid,
                                    ucp_ep->conn_sn, UCS_CONN_MATCH_QUEUE_EXP));
    m_workers[0].second.push_back(
            ucs::handle<ucp_ep_h, ucp_test_base::entity*>(ucp_ep,
                                                          ucp_ep_destroy));

    if (!(ucp_ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED)) {
        ASSERT_UCS_OK(ucp_wireup_send_request(ucp_ep));
    }

    store_config();
    UCS_ASYNC_UNBLOCK(&worker()->async);
}

bool test_ucp_ep_reconfig::entity::is_lane_connected(ucp_ep_h ep,
                                                     ucp_lane_index_t lane_idx,
                                                     const entity &other) const
{
    const auto lane     = &ucp_ep_config(ep)->key.lanes[lane_idx];
    const auto resource = &ucph()->tl_rscs[ucp_ep_get_rsc_index(ep, lane_idx)];
    auto addr           = other.get_address(other.ep_tl_bitmap());
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

void test_ucp_ep_reconfig::entity::verify_configuration(
        const entity &other) const
{
    unsigned reused                  = 0;
    const ucp_lane_index_t num_lanes = ucp_ep_num_lanes(ep());

    for (ucp_lane_index_t lane = 0; lane < num_lanes; ++lane) {
        /* Verify local and remote lanes are identical */
        EXPECT_TRUE(is_lane_connected(ep(), lane, other));

        /* Verify correct number of reused lanes is configured */
        auto uct_ep = ucp_ep_get_lane(ep(), lane);
        auto it     = std::find(m_uct_eps.begin(), m_uct_eps.end(), uct_ep);
        reused     += (it != m_uct_eps.end());
    }

    EXPECT_EQ(reused, is_reconfigured() ? 0 : num_lanes);
}

test_ucp_ep_reconfig::address_p
test_ucp_ep_reconfig::entity::get_address(const ucp_tl_bitmap_t &tl_bitmap) const
{
    size_t addr_len;

    /* Initialize std::unique_ptr with custom deleter */
    address_p address(new address_t(nullptr, {0}), [](address_t *addr) {
        ucs_free(addr->first);
        ucs_free(addr->second.address_list);
        delete addr;
    });

    unsigned flags = ucp_worker_default_address_pack_flags(worker());
    ASSERT_UCS_OK(ucp_address_pack(worker(), ep(), &tl_bitmap, flags,
                                   ucph()->config.ext.worker_addr_version, NULL,
                                   UINT_MAX, &addr_len, &address->first));

    ASSERT_UCS_OK(ucp_address_unpack(worker(), address->first, flags,
                                     &address->second));
    return address;
}

void test_ucp_ep_reconfig::run(bool bidirectional)
{
    create_entities_and_connect();
    send_recv(bidirectional);

    auto r_sender   = static_cast<const entity*>(&sender());
    auto r_receiver = static_cast<const entity*>(&receiver());

    EXPECT_NE(r_sender->is_reconfigured(), r_receiver->is_reconfigured());
    r_sender->verify_configuration(*r_receiver);
    r_receiver->verify_configuration(*r_sender);
}

void test_ucp_ep_reconfig::skip_non_p2p()
{
    if (!has_p2p_transport()) {
        UCS_TEST_SKIP_R("No p2p TLs available, config will be non-wireup");
    }
}

/* TODO: Remove skip condition after next PRs are merged. */
UCS_TEST_SKIP_COND_P(test_ucp_ep_reconfig, basic,
                     !has_transport("rc_x") || !has_transport("rc_v"))
{
    run();
}

UCS_TEST_P(test_ucp_ep_reconfig, request_reset, "PROTO_REQUEST_RESET=y")
{
    if (is_single_transport()) {
        /* One side will consume all ifaces and the other side will have no ifaces left to use */
        UCS_TEST_SKIP_R("exclude_iface requires at least 2 transports to work "
                        "(for example DC + SHM)");
    }

    skip_non_p2p();
    run();
}

/* SHM causes lane reuse, disable for now. */
UCS_TEST_SKIP_COND_P(test_ucp_ep_reconfig, resolve_remote_id,
                     has_transport("shm") || is_self(), "MAX_RNDV_LANES=0")
{
    if (has_transport("tcp")) {
        UCS_TEST_SKIP_R("lanes are reused (not supported yet)");
    }

    run(true);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_ep_reconfig);
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_ep_reconfig, rc_x_v, "rc");

class test_reconfig_asymmetric : public test_ucp_ep_reconfig {
protected:
    void create_entities_and_connect() override
    {
        create_entity(true, false);

        modify_config("NUM_EPS", "200");
        create_entity(false, false);

        sender().connect(&receiver(), get_ep_params());
        receiver().connect(&sender(), get_ep_params());
    }
};

/* Will be relevant when reuse + non-wireup is supported */
UCS_TEST_SKIP_COND_P(test_reconfig_asymmetric, basic, has_transport("shm"))
{
    run();
}

/* Will be relevant when reuse + non-wireup is supported */
UCS_TEST_SKIP_COND_P(test_reconfig_asymmetric, request_reset,
                     has_transport("shm"), "PROTO_REQUEST_RESET=y")
{
    skip_non_p2p();
    run();
}

/* SHM causes lane reuse, disable for now. */
UCS_TEST_SKIP_COND_P(test_reconfig_asymmetric, resolve_remote_id,
                     has_transport("shm") || is_self(), "MAX_RNDV_LANES=0")
{
    run(true);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_reconfig_asymmetric, shm_ib, "shm,ib");
