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

    class entity : public ucp_test_base::entity {
    public:
        entity(const ucp_test_param &test_params, ucp_config_t *ucp_config,
               const ucp_worker_params_t &worker_params,
               const ucp_test *test_owner) :
            ucp_test_base::entity(test_params, ucp_config, worker_params,
                                  test_owner),
            m_cfg_index(UCP_WORKER_CFG_INDEX_NULL)
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

        static const entity &to_reconfigurable(const ucp_test_base::entity &e)
        {
            return *static_cast<const entity*>(&e);
        }

    private:
        void store_config();
        ucp_tl_bitmap_t ep_tl_bitmap() const;
        address_p get_address(bool ep_only) const;
        bool has_matching_lane(ucp_ep_h ep, ucp_lane_index_t lane_idx,
                               const entity &other) const;

        ucp_worker_cfg_index_t m_cfg_index;
        std::vector<uct_ep_h>  m_uct_eps;
    };

    void init() override
    {
        ucp_test::init();

        /* Check presence of IB devices using rc_verbs, as all devices must
         * support it. */
        if (!has_resource(sender(), "rc_verbs")) {
            UCS_TEST_SKIP_R("IB transport is not present");
        }

        m_entities.clear();
    }

    void create_entities_and_connect()
    {
        create_entity();
        create_entity();
        sender().connect(&receiver(), get_ep_params());
        receiver().connect(&sender(), get_ep_params());
    }

public:
    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant_with_value(variants, UCP_FEATURE_TAG, 0, "");
        add_variant_with_value(variants, UCP_FEATURE_TAG, 1, "excl_if");
    }

    bool is_exclude_iface() const
    {
        return get_variant_value();
    }

    void create_entity()
    {
        m_entities.push_back(new entity(GetParam(), m_ucp_config,
                                        get_worker_params(), this));
    }

    void send_recv()
    {
        static constexpr unsigned num_iterations = 100;
        static constexpr size_t msg_size         = 16 * UCS_KBYTE;
        const ucp_request_param_t param          = {
            .op_attr_mask = UCP_OP_ATTR_FLAG_NO_IMM_CMPL
        };
        std::string sbuf(msg_size, 'a'), rbuf(msg_size, 'b');

        for (unsigned i = 0; i < num_iterations; ++i) {
            void *sreq = ucp_tag_send_nbx(sender().ep(), sbuf.c_str(), msg_size,
                                          0, &param);
            void *rreq = ucp_tag_recv_nbx(receiver().worker(),
                                          (void*)rbuf.c_str(), msg_size, 0, 0,
                                          &param);
            request_wait(rreq);
            request_wait(sreq);
            EXPECT_EQ(sbuf, rbuf);
        }
    }
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

ucp_tl_bitmap_t test_ucp_reconfigure::entity::ep_tl_bitmap() const
{
    ucp_tl_bitmap_t tl_bitmap = UCS_STATIC_BITMAP_ZERO_INITIALIZER;

    if (ep() == NULL) {
        return tl_bitmap;
    }

    for (ucp_lane_index_t lane = 0; lane < ucp_ep_num_lanes(ep()); ++lane) {
        UCS_STATIC_BITMAP_SET(&tl_bitmap, ucp_ep_get_rsc_index(ep(), lane));
    }

    return tl_bitmap;
}

void test_ucp_reconfigure::entity::connect(const ucp_test_base::entity *other,
                                           const ucp_ep_params_t &ep_params,
                                           int ep_idx, int do_set_ep)
{
    auto self        = static_cast<const test_ucp_reconfigure*>(m_test);
    auto &r_other    = to_reconfigurable(*other);
    auto worker_addr = r_other.get_address(false);
    ucp_tl_bitmap_t tl_bitmap;
    ucp_ep_h ucp_ep;
    unsigned addr_indices[UCP_MAX_LANES];

    tl_bitmap = self->is_exclude_iface() ?
                        UCS_STATIC_BITMAP_NOT(r_other.ep_tl_bitmap()) :
                        ucp_tl_bitmap_max;

    UCS_ASYNC_BLOCK(&worker()->async);
    ASSERT_UCS_OK(ucp_ep_create_to_worker_addr(worker(), &tl_bitmap,
                                               &worker_addr->second,
                                               UCP_EP_INIT_CREATE_AM_LANE,
                                               "reconfigure test", addr_indices,
                                               &ucp_ep));
    m_workers[0].second.push_back({ucp_ep, ucp_ep_destroy});

    ucp_ep->conn_sn = 0;
    ASSERT_TRUE(ucp_ep_match_insert(worker(), ucp_ep, worker_addr->second.uuid,
                                    ucp_ep->conn_sn, UCS_CONN_MATCH_QUEUE_EXP));

    ASSERT_UCS_OK(ucp_wireup_send_request(ucp_ep));
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

void test_ucp_reconfigure::entity::verify_configuration(
        const entity &other) const
{
    unsigned reused                  = 0;
    const ucp_lane_index_t num_lanes = ucp_ep_num_lanes(ep());

    EXPECT_EQ(num_lanes, ucp_ep_num_lanes(other.ep()));

    for (ucp_lane_index_t lane = 0; lane < num_lanes; ++lane) {
        /* Verify local and remote lanes are identical */
        EXPECT_TRUE(has_matching_lane(ep(), lane, other));

        /* Verify correct number of reused lanes is configured */
        auto uct_ep = ucp_ep_get_lane(ep(), lane);
        auto it     = std::find(m_uct_eps.begin(), m_uct_eps.end(), uct_ep);
        reused     += (it != m_uct_eps.end());
    }

    EXPECT_EQ(reused, is_reconfigured() ? 0 : num_lanes);
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
    ASSERT_UCS_OK(ucp_address_pack(worker(), ep(), &tl_bitmap, flags,
                                   ucph()->config.ext.worker_addr_version, NULL,
                                   UINT_MAX, &addr_len, &address->first));

    ASSERT_UCS_OK(ucp_address_unpack(worker(), address->first, flags,
                                     &address->second));
    return address;
}

UCS_TEST_P(test_ucp_reconfigure, basic)
{
    create_entities_and_connect();
    send_recv();

    auto &e1 = entity::to_reconfigurable(sender());
    auto &e2 = entity::to_reconfigurable(receiver());

    if (is_exclude_iface()) {
        EXPECT_NE(e1.is_reconfigured(), e2.is_reconfigured());
    } else {
        EXPECT_FALSE(e1.is_reconfigured());
        EXPECT_FALSE(e2.is_reconfigured());
    }

    e1.verify_configuration(e2);
    e2.verify_configuration(e1);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_reconfigure, rc, "rc");
