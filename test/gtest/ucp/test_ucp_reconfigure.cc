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
        void connect(const entity &other, bool is_exclude_iface);

        void verify_configuration(const entity &other) const;

        bool is_reconfigured() const
        {
            return m_cfg_index != ep()->cfg_index;
        }

        void set_worker_entity(const ucp_test_base::entity *e)
        {
            m_worker_entity = e;
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
        ucp_tl_bitmap_t ep_tl_bitmap() const;
        address_p get_address(bool ep_only) const;
        bool has_matching_lane(ucp_ep_h ep, ucp_lane_index_t lane_idx,
                               const entity &other) const;

        ucp_worker_cfg_index_t       m_cfg_index     = UCP_WORKER_CFG_INDEX_NULL;
        std::vector<uct_ep_h>        m_uct_eps;
        const ucp_test_base::entity *m_worker_entity = nullptr;
        ucp_ep_h                     m_ep            = nullptr;
    };

    void create_entities_and_connect()
    {
        m_sender.set_worker_entity(create_entity(true));
        m_receiver.set_worker_entity(create_entity(false));

        m_sender.connect(m_receiver, is_exclude_iface());
        m_receiver.connect(m_sender, is_exclude_iface());
    }

    void cleanup()
    {
        m_sender.cleanup();
        m_receiver.cleanup();
        ucp_test::cleanup();
    }

    entity m_sender;
    entity m_receiver;

public:
    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant_with_value(variants, UCP_FEATURE_TAG, 0, "");
        add_variant_with_value(variants, UCP_FEATURE_TAG, 1, "exclude_iface");
    }

    bool is_exclude_iface() const
    {
        return get_variant_value();
    }

    void send_recv()
    {
        static constexpr unsigned num_iterations = 100;
        static constexpr size_t msg_size         = 16 * UCS_KBYTE;
        const ucp_request_param_t param          = {
            .op_attr_mask = UCP_OP_ATTR_FLAG_NO_IMM_CMPL
        };

        for (unsigned i = 0; i < num_iterations; ++i) {
            std::string sbuf(msg_size, 'a'), rbuf(msg_size, 'b');

            void *sreq = ucp_tag_send_nbx(m_sender.ep(), sbuf.c_str(), msg_size,
                                          0, &param);
            void *rreq = ucp_tag_recv_nbx(m_receiver.worker(),
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

void test_ucp_reconfigure::entity::connect(const entity &other,
                                           bool is_exclude_iface)
{
    auto worker_addr = other.get_address(false);
    ucp_tl_bitmap_t tl_bitmap;
    unsigned addr_indices[UCP_MAX_LANES];

    tl_bitmap = is_exclude_iface ? UCS_STATIC_BITMAP_NOT(other.ep_tl_bitmap()) :
                                   ucp_tl_bitmap_max;

    UCS_ASYNC_BLOCK(&worker()->async);
    ASSERT_UCS_OK(ucp_ep_create_to_worker_addr(worker(), &tl_bitmap,
                                               &worker_addr->second,
                                               UCP_EP_INIT_CREATE_AM_LANE,
                                               "reconfigure test", addr_indices,
                                               &m_ep));

    m_ep->conn_sn = 0;
    ASSERT_TRUE(ucp_ep_match_insert(worker(), m_ep, worker_addr->second.uuid,
                                    m_ep->conn_sn, UCS_CONN_MATCH_QUEUE_EXP));

    ASSERT_UCS_OK(ucp_wireup_send_request(m_ep));
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
    auto context   = m_worker_entity->ucph();

    ASSERT_UCS_OK(ucp_address_pack(worker(), ep(), &tl_bitmap, flags,
                                   context->config.ext.worker_addr_version,
                                   NULL, UINT_MAX, &addr_len, &address->first));

    ASSERT_UCS_OK(ucp_address_unpack(worker(), address->first, flags,
                                     &address->second));
    return address;
}

UCS_TEST_P(test_ucp_reconfigure, basic)
{
    create_entities_and_connect();
    send_recv();

    if (is_exclude_iface()) {
        EXPECT_NE(m_sender.is_reconfigured(), m_receiver.is_reconfigured());
    } else {
        EXPECT_FALSE(m_sender.is_reconfigured());
        EXPECT_FALSE(m_receiver.is_reconfigured());
    }

    m_sender.verify_configuration(m_receiver);
    m_receiver.verify_configuration(m_sender);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_reconfigure, rc, "rc");
