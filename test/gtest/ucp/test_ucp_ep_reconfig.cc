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
    using address_t        = std::pair<void*, ucp_unpacked_address_t>;
    using address_p        = std::unique_ptr<address_t, void (*)(address_t*)>;
    using limits           = std::numeric_limits<unsigned>;
    using mem_buffer_p     = std::unique_ptr<mem_buffer>;
    using mem_buffer_vec_t = std::vector<mem_buffer_p>;

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

        void
        verify_configuration(const entity &other, unsigned reused_rscs) const;

        bool is_reconfigured() const
        {
            return m_cfg_index != ep()->cfg_index;
        }

        unsigned num_reused_rscs() const
        {
            return m_num_reused_rscs;
        }

    private:
        void store_config();
        ucp_tl_bitmap_t
        ep_tl_bitmap(unsigned max_num_rscs = limits::max()) const;
        address_p get_address(const ucp_tl_bitmap_t &tl_bitmap) const;
        bool is_lane_connected(ucp_ep_h ep, ucp_lane_index_t lane_idx,
                               const entity &other) const;
        ucp_tl_bitmap_t reduced_tl_bitmap() const;
        unsigned num_shm_rscs() const;

        ucp_worker_cfg_index_t m_cfg_index       = UCP_WORKER_CFG_INDEX_NULL;
        unsigned               m_num_reused_rscs = 0;
        std::vector<uct_ep_h>  m_uct_eps;
        bool                   m_exclude_ifaces;
    };

    void init_buffers(mem_buffer_vec_t &sbufs, mem_buffer_vec_t &rbufs,
                      size_t size, bool bidirectional);

    void pattern_check(const mem_buffer_vec_t &rbufs) const;

    bool is_single_transport() const
    {
        return GetParam().transports.size() == 1;
    }

    virtual bool should_reconfigure()
    {
        return true;
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

        UCS_TEST_SKIP_R("Test is skipped due to unresolved failure");

        /* num_tls = single device + UD */
        /* coverity[unreachable] */
        if (sender().ucph()->num_tls <= 2) {
            UCS_TEST_SKIP_R("test requires at least 2 ifaces to work");
        }

        if (has_transport("tcp")) {
            UCS_TEST_SKIP_R("TODO: fix lane matching functionality in case "
                            "there's matching remote MDs and different "
                            "sys_devs");
        }

        if (has_transport("gga") && !reuse_lanes()) {
            UCS_TEST_SKIP_R("TODO: revert this after replacing "
                            "'is_lane_connected' with protocols check");
        }

        /* TODO: replace with more specific 'fence mode' check after Michal's
         * PR is merged */
        if (!is_proto_enabled()) {
            UCS_TEST_SKIP_R("proto v1 use weak fence by default");
        }
    }

    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant_with_value(variants, UCP_FEATURE_TAG, 0, "");
        add_variant_with_value(variants, UCP_FEATURE_TAG, 1, "reuse");
    }

    void run(bool bidirectional = false);
    virtual ucp_tl_bitmap_t tl_bitmap();
    void check_single_flush();

    bool reuse_lanes() const
    {
        return get_variant_value();
    }

    void send_message(const ucp_test_base::entity &e1,
                      const ucp_test_base::entity &e2, const mem_buffer *sbuf,
                      const mem_buffer *rbuf, std::vector<void*> &reqs)
    {
        const ucp_request_param_t param = {
            .op_attr_mask = UCP_OP_ATTR_FLAG_NO_IMM_CMPL
        };

        void *sreq = ucp_tag_send_nbx(e1.ep(), sbuf->ptr(), sbuf->size(), 0,
                                      &param);
        void *sreq_sync = ucp_tag_send_sync_nbx(e1.ep(), sbuf->ptr(),
                                                sbuf->size(), 0, &param);
        reqs.insert(reqs.end(), {sreq, sreq_sync});

        for (unsigned iter = 0; iter < 2; iter++) {
            void *rreq = ucp_tag_recv_nbx(e2.worker(), rbuf->ptr(),
                                          rbuf->size(), 0, 0, &param);
            reqs.push_back(rreq);
        }
    }

    void send_recv(bool bidirectional)
    {
/* TODO: remove this when large messages asan bug is solved (size > ~70MB) */
#ifdef __SANITIZE_ADDRESS__
        static const size_t msg_sizes[] = {8, 1024, 16384, 32768};
#else
        static const size_t msg_sizes[] = {8, 1024, 16384, UCS_MBYTE};
#endif

        for (auto msg_size : msg_sizes) {
            std::vector<void*> reqs;
            mem_buffer_vec_t sbufs, rbufs;

            init_buffers(sbufs, rbufs, msg_size, bidirectional);

            for (unsigned i = 0; i < num_iterations; ++i) {
                send_message(sender(), receiver(), sbufs[i].get(),
                             rbufs[i].get(), reqs);

                if (bidirectional) {
                    send_message(receiver(), sender(),
                                 sbufs[i + num_iterations].get(),
                                 rbufs[i + num_iterations].get(), reqs);
                }
            }

            requests_wait(reqs);
            pattern_check(rbufs);
        }
    }

    static constexpr unsigned num_iterations = 1000;
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

    /* Calculate number of reused resources by:
     * 1) Count number of resources used in EP configuration.
     * 2) Take half of total resources to be reused.
     * 3) For asymmetric mode, only SHM resources are reused. */
    auto num_reused   = UCS_STATIC_BITMAP_POPCOUNT(ep_tl_bitmap()) / 2;
    auto test         = static_cast<const test_ucp_ep_reconfig*>(m_test);
    m_num_reused_rscs = m_exclude_ifaces ?
                                (test->reuse_lanes() ? num_reused : 0) :
                                num_shm_rscs();
}

unsigned test_ucp_ep_reconfig::entity::num_shm_rscs() const
{
    unsigned num_shm = 0;
    auto tl_bitmap   = ep_tl_bitmap();
    ucp_rsc_index_t rsc_idx;

    UCS_STATIC_BITMAP_FOR_EACH_BIT(rsc_idx, &tl_bitmap) {
        num_shm += (ucph()->tl_rscs[rsc_idx].tl_rsc.dev_type ==
                    UCT_DEVICE_TYPE_SHM);
    }

    return num_shm;
}

ucp_tl_bitmap_t
test_ucp_ep_reconfig::entity::ep_tl_bitmap(unsigned max_num_rscs) const
{
    ucp_tl_bitmap_t tl_bitmap = UCS_STATIC_BITMAP_ZERO_INITIALIZER;
    unsigned rsc_count        = 0;

    for (auto lane = 0; lane < ucp_ep_num_lanes(ep()); ++lane) {
        auto rsc_index = ucp_ep_get_rsc_index(ep(), lane);
        if (UCS_STATIC_BITMAP_GET(tl_bitmap, rsc_index)) {
            continue;
        }

        if (rsc_count++ >= max_num_rscs) {
            break;
        }

        UCS_STATIC_BITMAP_SET(&tl_bitmap, rsc_index);
    }

    return tl_bitmap;
}

ucp_tl_bitmap_t test_ucp_ep_reconfig::entity::reduced_tl_bitmap() const
{
    if ((ep() == NULL) || !m_exclude_ifaces) {
        /* Take bitmap from test */
        return ((test_ucp_ep_reconfig*)m_test)->tl_bitmap();
    }

    /* Use only resources not already in use, or part of reuse bitmap */
    auto reused_bitmap = ep_tl_bitmap(num_reused_rscs());
    return UCS_STATIC_BITMAP_OR(UCS_STATIC_BITMAP_NOT(ep_tl_bitmap()),
                                reused_bitmap);
}

void test_ucp_ep_reconfig::entity::connect(const ucp_test_base::entity *other,
                                           const ucp_ep_params_t &ep_params,
                                           int ep_idx, int do_set_ep)
{
    auto r_other                    = static_cast<const entity*>(other);
    auto worker_addr                = r_other->get_address(ucp_tl_bitmap_max);
    const ucp_tl_bitmap_t tl_bitmap = r_other->reduced_tl_bitmap();
    unsigned addr_indices[UCP_MAX_LANES];
    ucp_ep_h ucp_ep;

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
        const entity &other, unsigned expected_reused_rscs) const
{
    unsigned reused_lanes            = 0;
    const ucp_lane_index_t num_lanes = ucp_ep_num_lanes(ep());
    ucp_tl_bitmap_t reused_rscs      = UCS_STATIC_BITMAP_ZERO_INITIALIZER;

    for (ucp_lane_index_t lane = 0; lane < num_lanes; ++lane) {
        /* Verify local and remote lanes are identical */
        EXPECT_TRUE(is_lane_connected(ep(), lane, other));

        /* Verify correct number of reused lanes */
        auto uct_ep = ucp_ep_get_lane(ep(), lane);
        auto it     = std::find(m_uct_eps.begin(), m_uct_eps.end(), uct_ep);
        if (it == m_uct_eps.end()) {
            continue;
        }

        reused_lanes++;
        UCS_STATIC_BITMAP_SET(&reused_rscs, ucp_ep_get_rsc_index(ep(), lane));
    }

    auto test = static_cast<const test_ucp_ep_reconfig*>(m_test);

    if (!is_reconfigured()) {
        EXPECT_EQ(num_lanes, reused_lanes);
    } else if (test->reuse_lanes() && (expected_reused_rscs > 0)) {
        EXPECT_EQ(expected_reused_rscs,
                           UCS_STATIC_BITMAP_POPCOUNT(reused_rscs));
    }
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

void test_ucp_ep_reconfig::init_buffers(mem_buffer_vec_t &sbufs,
                                        mem_buffer_vec_t &rbufs, size_t size,
                                        bool bidirectional)
{
    auto num_bufs = bidirectional ? (2 * num_iterations) : num_iterations;

    for (unsigned i = 0; i < num_bufs; ++i) {
        sbufs.push_back(
                mem_buffer_p(new mem_buffer(size, UCS_MEMORY_TYPE_HOST, i)));
        rbufs.push_back(mem_buffer_p(
                new mem_buffer(size, UCS_MEMORY_TYPE_HOST, ucs::rand())));
    }
}

void test_ucp_ep_reconfig::pattern_check(const mem_buffer_vec_t &rbufs) const
{
    for (unsigned i = 0; i < rbufs.size(); ++i) {
        rbufs[i]->pattern_check(i);
    }
}

ucp_tl_bitmap_t test_ucp_ep_reconfig::tl_bitmap()
{
    if (!is_single_transport()) {
        return ucp_tl_bitmap_max;
    }

    /* For single transport, half of the resources should be reserved for
     * receiver side to use */
    ucp_tl_bitmap_t tl_bitmap = UCS_STATIC_BITMAP_ZERO_INITIALIZER;
    size_t num_tls            = 0;
    ucp_rsc_index_t rsc_idx;

    UCS_STATIC_BITMAP_FOR_EACH_BIT(rsc_idx, &sender().ucph()->tl_bitmap) {
        if (++num_tls > (sender().ucph()->num_tls / 2)) {
            UCS_STATIC_BITMAP_SET(&tl_bitmap, rsc_idx);
        }
    }

    return tl_bitmap;
}

void test_ucp_ep_reconfig::run(bool bidirectional)
{
    create_entities_and_connect();
    send_recv(bidirectional);

    auto r_sender   = static_cast<const entity*>(&sender());
    auto r_receiver = static_cast<const entity*>(&receiver());

    if (should_reconfigure()) {
        EXPECT_NE(r_sender->is_reconfigured(), r_receiver->is_reconfigured());
    } else {
        EXPECT_FALSE(r_sender->is_reconfigured());
        EXPECT_FALSE(r_receiver->is_reconfigured());
    }

    r_sender->verify_configuration(*r_receiver, r_sender->num_reused_rscs());
    r_receiver->verify_configuration(*r_sender, r_sender->num_reused_rscs());
}

void test_ucp_ep_reconfig::check_single_flush()
{
    if (has_transport("shm")) {
        /* TODO: add support for reconfiguration of separate wireup and AM
         * lanes */
        modify_config("WIREUP_VIA_AM_LANE", "y");
    }
}

UCS_TEST_P(test_ucp_ep_reconfig, basic)
{
    check_single_flush();
    run();
}

UCS_TEST_P(test_ucp_ep_reconfig, request_reset, "PROTO_REQUEST_RESET=y")
{
    check_single_flush();
    run();
}

UCS_TEST_P(test_ucp_ep_reconfig, resolve_remote_id)
{
    check_single_flush();
    run(true);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_ep_reconfig);

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

    ucp_tl_bitmap_t tl_bitmap() override
    {
        return ucp_tl_bitmap_max;
    }

    bool should_reconfigure() override
    {
        static const std::vector<std::string> ib_tls = {"rc_mlx5", "dc_mlx5",
                                                        "rc_verbs", "ud_verbs",
                                                        "ud_mlx5"};

        /* In case there's no IB devices, new config will be identical to
         * old config (thus no reconfiguration will be triggered). */
        return std::any_of(ib_tls.begin(), ib_tls.end(),
                           [&](const std::string &tl_name) {
                               return has_resource(sender(), tl_name);
                           });
    }
};

UCS_TEST_P(test_reconfig_asymmetric, basic)
{
    run();
}

UCS_TEST_P(test_reconfig_asymmetric, request_reset, "PROTO_REQUEST_RESET=y")
{
    run();
}

UCS_TEST_P(test_reconfig_asymmetric, resolve_remote_id)
{
    run(true);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_reconfig_asymmetric, shm_ib, "shm,ib");
