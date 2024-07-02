/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>

#include "ucp_test.h"
#include "common/test.h"
#include "common/test_helpers.h"
#include "ucp/ucp_test.h"

#include <algorithm>
#include <set>

extern "C" {
#include <ucp/wireup/address.h>
#include <ucp/wireup/wireup_ep.h>
#include <ucp/core/ucp_ep.inl>
#include <ucs/sys/math.h>
#include <uct/base/uct_iface.h>
}

class test_ucp_reconfigure : public ucp_test {
protected:
    class address {
    public:
        address(ucp_address_t *packed, const ucp_unpacked_address_t &unpacked) :
            m_packed(packed)
        {
            memcpy(&m_unpacked, &unpacked, sizeof(unpacked));
        }

        ~address()
        {
            ucs_free(m_packed);
            ucs_free(m_unpacked.address_list);
        }

        const ucp_unpacked_address_t *get() const
        {
            return &m_unpacked;
        }

    private:
        ucp_address_t         *m_packed;
        ucp_unpacked_address_t m_unpacked;
    };

    class entity : public ucp_test_base::entity {
    public:
        entity(const ucp_test_param &test_params, ucp_config_t* ucp_config,
               const ucp_worker_params_t& worker_params,
               const ucp_test *test_owner) :
           ucp_test_base::entity(test_params, ucp_config, worker_params, test_owner) {
            m_worker_addr = get_address();
        }

        void connect(const ucp_test_base::entity* other,
                     const ucp_ep_params_t& ep_params, int ep_idx = 0,
                     int do_set_ep = 1) override;
        void verify(const entity &other) const;

        bool is_reconfigured() const
        {
            return m_cfg_index != ep()->cfg_index;
        }

        static const entity& to_reconfigured(const ucp_test_base::entity &e)
        {
            return *static_cast<const entity*>(&e);
        }

    private:
        void store_config()
        {
            for (ucp_lane_index_t lane = 0; lane < ucp_ep_num_lanes(ep()); ++lane) {
                uct_ep_h uct_ep = ucp_ep_get_lane(ep(), lane);

                if (ucp_wireup_ep_test(uct_ep)) {
                    m_uct_eps.push_back(ucp_wireup_ep(uct_ep)->super.uct_ep);
                } else {
                    m_uct_eps.push_back(uct_ep);
                }
            }

            m_cfg_index = ep()->cfg_index;
        }

        ucp_tl_bitmap_t get_tl_bitmap() const;
        std::unique_ptr<address> get_address() const;
        bool has_matching_lane(ucp_ep_h ep, ucp_lane_index_t lane_idx, const entity &other) const;

        ucp_worker_cfg_index_t   m_cfg_index;
        std::vector<uct_ep_h>    m_uct_eps;
        std::unique_ptr<address> m_worker_addr;
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

public:
    static void
    get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant_values(variants, get_test_variants_feature, 0);
        add_variant_values(variants, get_test_variants_feature, 1, "reused");
    }

    static void
    get_test_variants_feature(std::vector<ucp_test_variant> &variants)
    {
        add_variant_with_value(variants, UCP_FEATURE_TAG, 0, "");
    }

    bool reuse_lanes() const
    {
        return get_variant_value(1);
    }

    void create_entity()
    {
        m_entities.push_back(new entity(GetParam(), m_ucp_config,
                             get_worker_params(), this));
    }

    void send_recv()
    {
        sender().connect(&receiver(), get_ep_params());
        receiver().connect(&sender(), get_ep_params());

        const ucp_request_param_t param = {
            .op_attr_mask = UCP_OP_ATTR_FLAG_NO_IMM_CMPL
        };
        std::string m_sbuf, m_rbuf;

        for (int i = 0; i < 100; ++ i) {
            m_sbuf.resize(msg_size);
            m_rbuf.resize(msg_size);
            std::fill(m_sbuf.begin(), m_sbuf.end(), 'a');

            void *sreq = ucp_tag_send_nbx(sender().ep(), m_sbuf.c_str(),
                                          msg_size, 0, &param);
            void *rreq = ucp_tag_recv_nbx(receiver().worker(), (void*)m_rbuf.c_str(),
                                          msg_size, 0, 0, &param);
            request_wait(rreq);
            request_wait(sreq);
        }
    }

    void verify()
    {
        auto& e1 = entity::to_reconfigured(sender());
        auto& e2 = entity::to_reconfigured(receiver());

        if (reuse_lanes()) {
            EXPECT_FALSE(e1.is_reconfigured());
            EXPECT_FALSE(e2.is_reconfigured());
        } else {
            EXPECT_NE(e1.is_reconfigured(), e2.is_reconfigured());
        }

        e1.verify(e2);
        e2.verify(e1);
    }

    static constexpr size_t msg_size = 16 * UCS_KBYTE;
};

ucp_tl_bitmap_t
test_ucp_reconfigure::entity::get_tl_bitmap() const
{
    ucp_tl_bitmap_t tl_bitmap = UCS_STATIC_BITMAP_ZERO_INITIALIZER;

    for (ucp_lane_index_t lane = 0; lane < ucp_ep_num_lanes(ep()); ++lane) {
        UCS_STATIC_BITMAP_SET(&tl_bitmap, ucp_ep_get_rsc_index(ep(), lane));
    }

    return tl_bitmap;
}

void
test_ucp_reconfigure::entity::connect(const ucp_test_base::entity* other,
                                      const ucp_ep_params_t& ep_params,
                                      int ep_idx, int do_set_ep)
{
    auto rtest = static_cast<const test_ucp_reconfigure*>(m_test);
    ucp_tl_bitmap_t tl_bitmap;
    ucp_ep_h ucp_ep;
    unsigned addr_indices[UCP_MAX_LANES];
    auto worker_addr = to_reconfigured(*other).m_worker_addr->get();

    tl_bitmap = (rtest->reuse_lanes() || (other->ep() == NULL)) ? ucp_tl_bitmap_max :
            UCS_STATIC_BITMAP_NOT(to_reconfigured(*other).get_tl_bitmap());

    UCS_ASYNC_BLOCK(&worker()->async);
    ASSERT_UCS_OK(ucp_ep_create_to_worker_addr(worker(), &tl_bitmap,
                                               worker_addr, UCP_EP_INIT_CREATE_AM_LANE,
                                               "reconfigure test", addr_indices, &ucp_ep));
    ucs::handle<ucp_ep_h,ucp_test_base::entity*> ep_h(ucp_ep, ucp_ep_destroy);
    m_workers[0].second.push_back(ep_h);

    ucp_ep->conn_sn = ucp_ep_match_get_sn(worker(), worker_addr->uuid);
    ASSERT_TRUE(ucp_ep_match_insert(worker(), ucp_ep, worker_addr->uuid, ucp_ep->conn_sn,
                                    UCS_CONN_MATCH_QUEUE_EXP));
    ASSERT_UCS_OK(ucp_wireup_send_request(ucp_ep));

    UCS_ASYNC_UNBLOCK(&worker()->async);
    store_config();
}

bool
test_ucp_reconfigure::entity::has_matching_lane(ucp_ep_h ep, ucp_lane_index_t lane_idx,
                                                const entity &other) const
{
    auto lane     = &ucp_ep_config(ep)->key.lanes[lane_idx];
    auto resource = &ucph()->tl_rscs[ucp_ep_get_rsc_index(ep, lane_idx)];
    auto addr     = other.get_address();
    ucp_address_entry_t *ae;

    ucs_carray_for_each(ae, addr->get()->address_list, addr->get()->address_count) {
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
    auto reused_lanes = std::count_if(m_uct_eps.begin(), m_uct_eps.end(),
                                [this](uct_ep_h uct_ep) {
        for (ucp_lane_index_t lane = 0; lane < ucp_ep_num_lanes(ep()); ++lane) {
            if (ucp_ep_get_lane(ep(), lane) == uct_ep) {
                return true;
            }
        }

        return false;
    });

    EXPECT_EQ(reused_lanes, is_reconfigured() ? 0 : ucp_ep_num_lanes(ep()));
    EXPECT_EQ(ucp_ep_num_lanes(ep()), ucp_ep_num_lanes(other.ep()));

    for (ucp_lane_index_t lane = 0; lane < ucp_ep_num_lanes(ep()); ++lane) {
        EXPECT_TRUE(has_matching_lane(ep(), lane, other));
    }
}

std::unique_ptr<test_ucp_reconfigure::address> test_ucp_reconfigure::entity::get_address() const
{
    unsigned flags           = ucp_worker_default_address_pack_flags(worker());
    ucp_object_version_t ver = ucph()->config.ext.worker_addr_version;
    size_t addr_len;
    ucp_address_t *packed_addr;
    ucp_unpacked_address_t unpacked_addr;

    const ucp_tl_bitmap_t tl_bitmap = (ep() == NULL) ? ucp_tl_bitmap_max : get_tl_bitmap();

    ASSERT_UCS_OK(ucp_address_pack(worker(), ep(), &tl_bitmap, flags,
                            ver, NULL, UINT_MAX, &addr_len,
                            (void**)&packed_addr));

    ASSERT_UCS_OK(ucp_address_unpack(worker(), packed_addr, flags, &unpacked_addr));
    //todo: release packed if unpack fails
    return std::unique_ptr<address>(new address(packed_addr, unpacked_addr));
}

UCS_TEST_P(test_ucp_reconfigure, basic)
{
    create_entity();
    create_entity();
    send_recv();
    verify();
}

UCS_TEST_P(test_ucp_reconfigure, num_lanes_diff)
{
    {
        ucs::scoped_setenv num_paths("UCX_IB_NUM_PATHS", "1");
        create_entity();
    }
    {
        ucs::scoped_setenv num_paths("UCX_IB_NUM_PATHS", "2");
        create_entity();
    }

    send_recv();
    verify();
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_reconfigure, rc, "rc_v,rc_x");
