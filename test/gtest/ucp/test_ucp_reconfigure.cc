/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"
#include "common/test.h"

extern "C" {
#include <ucp/wireup/address.h>
#include <ucp/wireup/wireup_ep.h>
#include <ucp/core/ucp_ep.inl>
}

class test_ucp_reconfigure : public ucp_test {
protected:
    using address_pair_t = std::pair<void*, ucp_unpacked_address_t>;
    using address_pair_p =
            std::unique_ptr<address_pair_t, void (*)(address_pair_t*)>;

    class entity : public ucp_test_base::entity {
    public:
        entity(const ucp_test_param &test_params, ucp_config_t *ucp_config,
               const ucp_worker_params_t &worker_params,
               const ucp_test *test_owner, unsigned init_flags, bool reuse_lanes) :
            ucp_test_base::entity(test_params, ucp_config, worker_params,
                                  test_owner),
            m_cfg_index(UCP_WORKER_CFG_INDEX_NULL),
            m_init_flags(init_flags),
            m_reuse_lanes(reuse_lanes)
        {
        }

        void connect(const ucp_test_base::entity *other,
                     const ucp_ep_params_t &ep_params, int ep_idx = 0,
                     int do_set_ep = 1) override;

        void verify(const entity &other) const;

        bool is_reconfigured() const
        {
            return m_cfg_index != ep()->cfg_index;
        }

        static const entity &to_reconfigured(const ucp_test_base::entity &e)
        {
            return *static_cast<const entity*>(&e);
        }

    private:
        void store_config();
        ucp_tl_bitmap_t ep_tl_bitmap(bool non_reused_only = false) const;
        address_pair_p get_address(bool ep_only) const;
        bool has_matching_lane(ucp_ep_h ep, ucp_lane_index_t lane_idx,
                               const entity &other) const;

        ucp_lane_index_t num_reused() const
        {
            return m_reuse_lanes ? (ucp_ep_num_lanes(ep()) / 2) : 0;
        }

        ucp_worker_cfg_index_t m_cfg_index;
        std::vector<uct_ep_h>  m_uct_eps;
        unsigned               m_init_flags;
        bool                   m_reuse_lanes;
    };

    typedef enum {
        EXCLUDE_IFACES   = 0,
        ASYMMETRIC_SCALE = 1
    } asymmetric_mode_t;

    void init() override
    {
        ucp_test::init();

        if (asymmetric_scale() && has_transport("rc")) {
            UCS_TEST_SKIP_R("asymmetric scale mode does not work without DC");
        }
    }

    bool is_single_transport()
    {
        return (GetParam().transports.size() == 1) && !has_transport("rc");
    }

public:
    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant_values(variants, get_test_variants_feature, EXCLUDE_IFACES,
                           "excl_if");
        add_variant_values(variants, get_test_variants_feature, ASYMMETRIC_SCALE,
                           "asym_scale");
    }

    static void
    get_test_variants_feature(std::vector<ucp_test_variant> &variants)
    {
        add_variant_with_value(variants, UCP_FEATURE_TAG, 0, "");
    }

    void run(unsigned init_flags = UCP_EP_INIT_CREATE_AM_LANE,
             bool bidirectional = false, bool reuse_lanes = false);

    bool exclude_iface() const
    {
        return get_variant_value(1) == EXCLUDE_IFACES;
    }

    bool asymmetric_scale() const
    {
        return get_variant_value(1) == ASYMMETRIC_SCALE;
    }

    void create_entity(bool push_front, unsigned init_flags, bool reuse_lanes)
    {
        entity *e = new entity(GetParam(), m_ucp_config, get_worker_params(),
                               this, init_flags, reuse_lanes);
        if (push_front) {
            m_entities.push_front(e);
        } else {
            m_entities.push_back(e);
        }
    }

    void send_message(const ucp_test_base::entity &e1,
                      const ucp_test_base::entity &e2, const std::string &sbuf,
                      const std::string &rbuf, std::vector<void*> &reqs)
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

    void send_recv(bool bidirectional)
    {
        static constexpr unsigned num_iterations = 1000;
        std::string sbuf(msg_size, 'a'), rbuf(msg_size, 'b');
        /* Buffers for the opposite direction */
        std::string o_sbuf(msg_size, 'c'), o_rbuf(msg_size, 'd');

        sender().connect(&receiver(), get_ep_params());
        receiver().connect(&sender(), get_ep_params());

        for (unsigned i = 0; i < num_iterations; ++i) {
            std::vector<void*> reqs;
            send_message(sender(), receiver(), sbuf, rbuf, reqs);

            if (bidirectional) {
                send_message(receiver(), sender(), o_sbuf, o_rbuf, reqs);
            }

            requests_wait(reqs);
            EXPECT_EQ(sbuf, rbuf);

            if (bidirectional) {
                EXPECT_EQ(o_sbuf, o_rbuf);
            }
        }
    }

    static constexpr size_t msg_size = 16 * UCS_KBYTE;
};

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

ucp_tl_bitmap_t test_ucp_reconfigure::entity::ep_tl_bitmap(bool non_reused_only) const
{
    ucp_tl_bitmap_t tl_bitmap = UCS_STATIC_BITMAP_ZERO_INITIALIZER;

    if (ep() == NULL) {
        return tl_bitmap;
    }

    auto num_lanes  = ucp_ep_num_lanes(ep());

    /* Reuse lanes according to priority */
    auto first_lane = non_reused_only ? num_lanes - num_reused() - 1 : 0;

    for (auto lane = first_lane; lane < num_lanes; ++lane) {
        UCS_STATIC_BITMAP_SET(&tl_bitmap, ucp_ep_get_rsc_index(ep(), lane));
    }

    return tl_bitmap;
}

void test_ucp_reconfigure::entity::connect(const ucp_test_base::entity *other,
                                           const ucp_ep_params_t &ep_params,
                                           int ep_idx, int do_set_ep)
{
    auto r_test      = static_cast<const test_ucp_reconfigure*>(m_test);
    auto &r_other    = to_reconfigured(*other);
    auto worker_addr = r_other.get_address(false);
    ucp_tl_bitmap_t tl_bitmap;
    ucp_ep_h ucp_ep;
    unsigned addr_indices[UCP_MAX_LANES];

    tl_bitmap = r_test->exclude_iface() ?
                        UCS_STATIC_BITMAP_NOT(r_other.ep_tl_bitmap(m_reuse_lanes)) :
                        ucp_tl_bitmap_max;

    UCS_ASYNC_BLOCK(&worker()->async);
    ASSERT_UCS_OK(ucp_ep_create_to_worker_addr(worker(), &tl_bitmap,
                                               &worker_addr->second,
                                               m_init_flags, "reconfigure test",
                                               addr_indices, &ucp_ep));
    m_workers[0].second.push_back({ucp_ep, ucp_ep_destroy});

    ucp_ep->conn_sn = 0;
    ASSERT_TRUE(ucp_ep_match_insert(worker(), ucp_ep, worker_addr->second.uuid,
                                    ucp_ep->conn_sn, UCS_CONN_MATCH_QUEUE_EXP));

    if (!(ucp_ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED)) {
        ASSERT_UCS_OK(ucp_wireup_send_request(ucp_ep));
    }

    UCS_ASYNC_UNBLOCK(&worker()->async);
    store_config();
}

bool test_ucp_reconfigure::entity::has_matching_lane(ucp_ep_h ep,
                                                     ucp_lane_index_t lane_idx,
                                                     const entity &other) const
{
    const auto lane     = &ucp_ep_config(ep)->key.lanes[lane_idx];
    const auto resource = &ucph()->tl_rscs[ucp_ep_get_rsc_index(ep, lane_idx)];
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

void test_ucp_reconfigure::entity::verify(const entity &other) const
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

test_ucp_reconfigure::address_pair_p
test_ucp_reconfigure::entity::get_address(bool ep_only) const
{
    const ucp_tl_bitmap_t tl_bitmap = ep_only ? ep_tl_bitmap() :
                                                ucp_tl_bitmap_max;
    size_t addr_len;

    address_pair_p address(new address_pair_t(nullptr, {0}),
                           [](address_pair_t *pair) {
        ucs_free(pair->first);
        ucs_free(pair->second.address_list);
        delete pair;
    });

    unsigned flags = ucp_worker_default_address_pack_flags(worker());
    ASSERT_UCS_OK(ucp_address_pack(worker(), ep(), &tl_bitmap, flags,
                                   ucph()->config.ext.worker_addr_version, NULL,
                                   UINT_MAX, &addr_len, &address->first));

    ASSERT_UCS_OK(ucp_address_unpack(worker(), address->first, flags,
                                     &address->second));
    return address;
}

void test_ucp_reconfigure::run(unsigned init_flags, bool bidirectional,
                               bool reuse_lanes)
{
    static const std::string num_eps_scaled("200");

    create_entity(true, init_flags, reuse_lanes);

    if (asymmetric_scale()) {
        modify_config("NUM_EPS", num_eps_scaled);
    }

    create_entity(false, init_flags, reuse_lanes);
    send_recv(bidirectional);

    auto &e1 = entity::to_reconfigured(sender());
    auto &e2 = entity::to_reconfigured(receiver());

    EXPECT_NE(e1.is_reconfigured(), e2.is_reconfigured());
    e1.verify(e2);
    e2.verify(e1);
}

/* TODO: Remove skip condition after next PRs are merged. */
UCS_TEST_SKIP_COND_P(test_ucp_reconfigure, basic,
                     !has_transport("rc_x") || !has_transport("rc_v"))
{
    run();
}

/* asymmetric_scale causes one side to be not wired-up, disable for now. */
UCS_TEST_SKIP_COND_P(test_ucp_reconfigure, request_reset, asymmetric_scale(),
                     "PROTO_REQUEST_RESET=y")
{
    if (exclude_iface() && is_single_transport()) {
        /* One side will consume all ifaces and the other side will have no ifaces left to use */
        UCS_TEST_SKIP_R("exclude_iface requires at least 2 transports to work "
                        "(for example DC + SHM)");
    }

    if (has_transport("ib") && !has_resource(sender(), "rc_verbs")) {
        UCS_TEST_SKIP_R("No IB devs available, config will be non-wireup");
    }

    run();
}

UCS_TEST_SKIP_COND_P(test_ucp_reconfigure, resolve_remote_id, is_self(), "RNDV_THRESH=0")
{
    /* There's no support for multiple paths + AM_ONLY in UCX */
    modify_config("IB_NUM_PATHS", "1", SETENV_IF_NOT_EXIST);

    if (asymmetric_scale()) {
        /* Either test has a single transport or it has SHM transport:
         * 1. Single transport causes the same configuration in both EPs (thus no reconf).
         * 2. The same happens with SHM as it's preferred over IB. */
        UCS_TEST_SKIP_R("asymmetric_scale + single transport causes same lane to "
                        "be selected in both sides");
    }

    /* num_tls = single device + UD */
    if (exclude_iface() && (sender().ucph()->num_tls <= 2)) {
        UCS_TEST_SKIP_R("exclude_iface requires at least 2 ifaces to work");
    }

    if (exclude_iface() && has_transport("dc_x")) {
        UCS_TEST_SKIP_R("extra path added for AM_LANE when using DC");
    }

    /* Create only AM_LANE to ensure we have only wireup EPs in
     * configuration. */
    run(UCP_EP_INIT_CREATE_AM_LANE_ONLY, true);
}

UCS_TEST_SKIP_COND_P(test_ucp_reconfigure, reuse_lanes, is_self())
{
    /* Use single path so that num_lanes/2 will only exclude some lanes and
     * not all. */
    modify_config("IB_NUM_PATHS", "1", SETENV_IF_NOT_EXIST);

    if (has_transport("ud_v") || has_transport("ud_x")) {
        UCS_TEST_SKIP_R("UD configuration has only a single lane, reuse test "
                        "is not relevant for this case");
    }

    if (exclude_iface() && has_transport("dc_x")) {
        UCS_TEST_SKIP_R("extra path added for AM_LANE when using DC");
    }

    if (asymmetric_scale() && is_single_transport()) {
        UCS_TEST_SKIP_R("asymmetric_scale + single transport causes same lane to "
                        "be selected in both sides");
    }

    if (has_transport("shm")) {
        UCS_TEST_SKIP_R("non wired-up lanes are not supported yet");
    }

    if (exclude_iface() && is_single_transport()) {
        UCS_TEST_SKIP_R("reduce num lanes + all reused case is not supported yet");
    }

    run(UCP_EP_INIT_CREATE_AM_LANE, false, true);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_reconfigure, rc_x_v, "rc");
UCP_INSTANTIATE_TEST_CASE(test_ucp_reconfigure);
