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
    class reconf_ep_t {
    public:
        reconf_ep_t(const entity &e) :
            m_ep(e.ep()), m_cfg_index(e.ep()->cfg_index)
        {
            for (ucp_lane_index_t lane = 0; lane < num_lanes(); ++lane) {
                uct_ep_h uct_ep = ucp_ep_get_lane(e.ep(), lane);

                if (ucp_wireup_ep_test(uct_ep)) {
                    m_uct_eps.push_back(ucp_wireup_ep(uct_ep)->super.uct_ep);
                } else {
                    m_uct_eps.push_back(uct_ep);
                }
            }

            m_transport = ucp_ep_get_tl_rsc(m_ep, m_ep->am_lane)->tl_name;
        }

        static bool is_scale_mode(ucp_ep_h ep)
        {
            return ep->worker->context->config.est_num_eps ==
                   m_num_eps_scale_mode;
        }

        bool is_tl_scalable(const std::string &tl_name)
        {
            static const std::string scalable_tls[2] = {"dc_mlx5", "ud_mlx5"};

            for (const auto &tl : scalable_tls) {
                if (tl_name == tl) {
                    return true;
                }
            }

            return false;
        }

        ucp_lane_index_t num_lanes()
        {
            return ucp_ep_config(m_ep)->key.num_lanes;
        }

        unsigned count_dc_resources()
        {
            unsigned dc_count     = 0;
            ucp_context_h context = m_ep->worker->context;
            ucp_rsc_index_t tl_id;

            UCS_BITMAP_FOR_EACH_BIT(context->tl_bitmap, tl_id) {
                std::string tl_name = context->tl_rscs[tl_id].tl_rsc.tl_name;

                if (tl_name == "dc_mlx5") {
                    dc_count++;
                }
            }

            return dc_count;
        }

        void verify(const std::vector<ucp_rsc_index_t> &dc_disabled_devs,
                    bool reconfigured = true)
        {
            bool is_reconfigured = (m_ep->cfg_index != m_cfg_index);
            EXPECT_EQ(is_reconfigured, reconfigured);

            static const unsigned num_paths = 2;
            unsigned dc_count               = count_dc_resources();
            unsigned common_num_lanes       = std::min((size_t)num_lanes(),
                                                       m_uct_eps.size());
            unsigned expected_reused        = is_reconfigured ?
                                                             common_num_lanes -
                                                              dc_count * num_paths :
                                                             common_num_lanes;

            EXPECT_LE(reused_count(), expected_reused);
            if (expected_reused > 0) {
                EXPECT_GE(reused_count(), expected_reused - 1);
            }

            const auto config = ucp_ep_config(m_ep);

            for (int i = 0; i < num_lanes(); ++i) {
                const auto lane = config->key.rma_bw_lanes[i];
                if (lane == UCP_NULL_LANE) {
                    break;
                }

                auto transport    = ucp_ep_get_tl_rsc(m_ep, lane)->tl_name;
                bool expect_scale = is_reconfigured ?
                                            (m_transport == "rc_mlx5") :
                                            is_tl_scalable(m_transport);
                auto resource =
                        &m_ep->worker->context
                                 ->tl_rscs[ucp_ep_get_rsc_index(m_ep, lane)];
                bool dc_disabled = std::find(dc_disabled_devs.begin(),
                                             dc_disabled_devs.end(),
                                             resource->dev_index) !=
                                   dc_disabled_devs.end();

                if (expect_scale && !dc_disabled) {
                    EXPECT_TRUE(is_tl_scalable(transport));
                } else {
                    EXPECT_STREQ("rc_mlx5", transport);
                }
            }
        }

    private:
        bool uct_ep_reused(uct_ep_h uct_ep)
        {
            for (ucp_lane_index_t lane = 0; lane < num_lanes(); ++lane) {
                if (ucp_ep_get_lane(m_ep, lane) == uct_ep) {
                    return true;
                }
            }

            return false;
        }

        unsigned reused_count()
        {
            unsigned reused_count = 0;

            for (auto &uct_ep : m_uct_eps) {
                if (uct_ep_reused(uct_ep)) {
                    reused_count++;
                }
            }

            return reused_count;
        }

        ucp_ep_h m_ep;
        ucp_worker_cfg_index_t m_cfg_index;
        std::vector<uct_ep_h> m_uct_eps;
        std::string m_transport;
    };

    enum {
        MSG_SIZE_SMALL  = 64,
        MSG_SIZE_MEDIUM = 4096,
        MSG_SIZE_LARGE  = 262144
    };

    void init()
    {
        ucp_test::init();

        if (!has_resource(sender(), "rc_mlx5") &&
            !has_resource(sender(), "dc_mlx5")) {
            UCS_TEST_SKIP_R("IB transport is not present");
        }

        m_reconfigure = false;
    }

    virtual bool start_scaled()
    {
        return get_variant_value(1);
    }

    virtual unsigned msg_size()
    {
        return get_variant_value(2);
    }

    void set_scale(bool enable, bool race = false)
    {
        if (enable) {
            sender().ucph()->config.est_num_eps   = m_num_eps_scale_mode;
            receiver().ucph()->config.est_num_eps = race ? 1 :
                                                           m_num_eps_scale_mode;
        } else {
            sender().ucph()->config.est_num_eps = 1;
            receiver().ucph()->config.est_num_eps =
                    race ? m_num_eps_scale_mode : 1;
        }
    }

    void connect()
    {
        set_scale(start_scaled());
        sender().connect(&receiver(), get_ep_params());
        receiver().connect(&sender(), get_ep_params());
        wireup_wait();
    }

    void *send_nb(std::string &buffer, uint8_t data)
    {
        ucp_request_param_t param = {
            .op_attr_mask = UCP_OP_ATTR_FLAG_NO_IMM_CMPL
        };

        buffer.resize(msg_size());
        std::fill(buffer.begin(), buffer.end(), data);

        return ucp_tag_send_nbx(sender().ep(), buffer.c_str(), msg_size(), 0,
                                &param);
    }

    void send_recv(unsigned count)
    {
        std::vector<void*> sreqs;

        for (int i = 0; i < count; ++i) {
            sreqs.push_back(send_nb(m_sbuf, i + 1));
            request_wait(recv_nb());
        }

        requests_wait(sreqs);
    }

    void reconfigure_nb(bool scale)
    {
        set_scale(scale);
        m_reconfigure = true;
        UCS_ASYNC_BLOCK(&sender().worker()->async);
        ucp_wireup_send_pre_request(sender().ep());
        UCS_ASYNC_UNBLOCK(&sender().worker()->async);
    }

    bool is_connected(ucp_ep_h ep)
    {
        ucp_lane_index_t lane;
        auto finish_state = (reconf_ep_t::is_scale_mode(sender().ep()) &&
                             !m_reconfigure) ?
                                    UCP_EP_FLAG_LOCAL_CONNECTED :
                                    UCP_EP_FLAG_REMOTE_CONNECTED;

        for (lane = 0; lane < ucp_ep_config(ep)->key.num_lanes; ++lane) {
            if (ucp_wireup_ep_test(ucp_ep_get_lane(ep, lane))) {
                return false;
            }
        }

        return ep->flags & finish_state;
    }

    void wireup_wait()
    {
        while (m_reconfigure &&
               !(receiver().ep()->flags & UCP_EP_FLAG_CONNECT_REQ_QUEUED)) {
            progress();
        }

        while (!is_connected(sender().ep()) || !is_connected(receiver().ep())) {
            progress();
        }
    }

    void *recv_nb()
    {
        ucp_request_param_t param = {0};
        param.op_attr_mask        = UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
        m_rbuf.resize(msg_size());

        return ucp_tag_recv_nbx(receiver().worker(), (void*)m_rbuf.c_str(),
                                msg_size(), 0, 0, &param);
    }

    static const unsigned m_num_eps_scale_mode = 128;
    std::string m_sbuf;
    std::string m_rbuf;
    bool m_reconfigure;
    std::vector<ucp_rsc_index_t> m_dc_disabled_devs;

public:
    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant_values(variants, get_test_variants_scaled, MSG_SIZE_SMALL,
                           "small");
        add_variant_values(variants, get_test_variants_scaled, MSG_SIZE_MEDIUM,
                           "medium");
        add_variant_values(variants, get_test_variants_scaled, MSG_SIZE_LARGE,
                           "large");
    }

    static void
    get_test_variants_scaled(std::vector<ucp_test_variant> &variants)
    {
        add_variant_values(variants, get_test_variants_feature, 0);
        add_variant_values(variants, get_test_variants_feature, 1, "scaled");
    }

    static void
    get_test_variants_feature(std::vector<ucp_test_variant> &variants)
    {
        add_variant_with_value(variants, UCP_FEATURE_TAG, 0, "");
    }

    bool is_reconf_ep(const entity *e)
    {
        const entity *other;
        other = (e == &sender()) ? &receiver() : &sender();

        return !(e->ep()->flags & UCP_EP_FLAG_CONNECT_REQ_QUEUED) &&
               (other->ep()->flags & UCP_EP_FLAG_CONNECT_REQ_QUEUED);
    }

    void reconfigure_during_traffic(bool scaled)
    {
        const size_t msg_count = 1000;
        std::vector<std::string> buffers(msg_count);
        std::vector<void*> reqs;

        connect();
        send_recv(100);

        if (start_scaled() && (msg_size() == MSG_SIZE_LARGE)) {
            while (sender().ep()->ext->remote_ep_id ==
                   UCS_PTR_MAP_KEY_INVALID) {
                progress();
            }
        }

        for (int i = 0; i < msg_count; ++i) {
            reqs.push_back(send_nb(buffers[i], i + 1));
        }

        reconfigure_nb(scaled);

        for (int i = 0; i < msg_count; ++i) {
            request_wait(recv_nb());
            EXPECT_EQ(buffers[i], m_rbuf);
        }

        requests_wait(reqs);
    }

    reconf_ep_t race_connect()
    {
        sender().connect(&receiver(), get_ep_params());
        receiver().connect(&sender(), get_ep_params());

        reconf_ep_t reconf_sender(sender()), reconf_receiver(receiver());

        while (!is_connected(sender().ep()) || !is_connected(receiver().ep())) {
            progress();
        }

        send_recv(100);

        return (is_reconf_ep(&sender()) ||
                (!is_reconf_ep(&receiver()) &&
                 sender().worker()->uuid > receiver().worker()->uuid)) ?
                       reconf_sender :
                       reconf_receiver;
    }

    void disable_dc_dev(unsigned dev_index)
    {
        if (count_resources(sender(), "dc_mlx5") == 0) {
            UCS_TEST_SKIP_R("no DC ifaces found");
        }

        int dc_count              = 0;
        ucp_tl_bitmap_t tl_bitmap = sender().ucph()->tl_bitmap;
        ucp_rsc_index_t tl_id;

        UCS_BITMAP_FOR_EACH_BIT(sender().ucph()->tl_bitmap, tl_id) {
            auto resource = &sender().ucph()->tl_rscs[tl_id];

            if (std::string(resource->tl_rsc.tl_name) == "dc_mlx5") {
                dc_count++;
            }

            if ((dc_count - 1) == dev_index) {
                UCS_BITMAP_UNSET(tl_bitmap, tl_id);
                m_dc_disabled_devs.push_back(resource->dev_index);
                break;
            }
        }

        create_entity(true, &tl_bitmap);
        create_entity(false, &tl_bitmap);
    }
};

UCS_TEST_SKIP_COND_P(test_ucp_reconfigure, race_all_reuse, is_self())
{
    set_scale(start_scaled());
    auto reconf_ep = race_connect();
    reconf_ep.verify(m_dc_disabled_devs, false);
}

UCS_TEST_SKIP_COND_P(test_ucp_reconfigure, race_all_reuse_part_scale, is_self())
{
    disable_dc_dev(0);
    set_scale(start_scaled());
    auto reconf_ep = race_connect();
    reconf_ep.verify(m_dc_disabled_devs, false);
}

UCS_TEST_SKIP_COND_P(test_ucp_reconfigure, race_no_reuse,
                     is_self() || ucs::has_roce_devices())
{
    set_scale(start_scaled(), true);
    auto reconf_ep = race_connect();
    reconf_ep.verify(m_dc_disabled_devs);
}

UCS_TEST_SKIP_COND_P(test_ucp_reconfigure, race_no_reuse_switch_wireup_lane,
                     is_self() || ucs::has_roce_devices(),
                     "RESOLVE_REMOTE_EP_ID=y")
{
    set_scale(start_scaled(), true);
    auto reconf_ep = race_connect();
    reconf_ep.verify(m_dc_disabled_devs);
}

UCS_TEST_SKIP_COND_P(test_ucp_reconfigure, race_part_reuse,
                     is_self() || ucs::has_roce_devices())
{
    if (!start_scaled()) {
        /* Sender must prefer DC in order for the receiver to reconfigure itself to DC as well. */
        UCS_TEST_SKIP_R("sender must be scaled");
    }

    disable_dc_dev(0);
    set_scale(start_scaled(), true);
    auto reconf_ep = race_connect();
    reconf_ep.verify(m_dc_disabled_devs);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_reconfigure, ib, "ib")
