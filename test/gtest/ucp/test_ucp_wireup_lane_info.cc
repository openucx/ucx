/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>

#include <string>
#include <vector>

extern "C" {
#include <ucp/wireup/wireup_lane_info.h>
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_ep.h>
#include <ucp/proto/lane_type.h>
#include <ucs/datastruct/string_buffer.h>
#include <ucs/sys/string.h>
#include <ucs/sys/topo/base/topo.h>
}


class test_ucp_wireup_lane_info : public ucs::test {
public:
    /* Isolate the global topology subsystem so that system devices registered
     * by a test are deterministic (and removed afterwards). */
    virtual void init()
    {
        ucs::test::init();
        m_topo_state = ucs_topo_extract_state();
    }

    virtual void cleanup()
    {
        ucs_topo_restore_state(m_topo_state);
        ucs::test::cleanup();
    }

protected:
    ucs_global_state_t *m_topo_state;

    struct rsc_desc {
        const char       *tl_name;
        const char       *dev_name;
        ucs_sys_device_t sys_device;
    };

    /* Owns a ucp_context_t plus the backing tl_rscs table built from a list of
     * transport resources, keeping tl_rscs valid for the object's lifetime. */
    class dummy_context {
    public:
        dummy_context(const std::vector<rsc_desc> &rscs,
                      const std::string &name = "") :
            m_rscs(rscs.size())
        {
            memset(&m_ctx, 0, sizeof(m_ctx));
            for (size_t i = 0; i < rscs.size(); ++i) {
                uct_tl_resource_desc_t *tl = &m_rscs[i].tl_rsc;
                ucs_strncpy_safe(tl->tl_name, rscs[i].tl_name,
                                 sizeof(tl->tl_name));
                ucs_strncpy_safe(tl->dev_name, rscs[i].dev_name,
                                 sizeof(tl->dev_name));
                tl->sys_device = rscs[i].sys_device;
            }
            m_ctx.tl_rscs = m_rscs.data();
            m_ctx.num_tls = static_cast<ucp_rsc_index_t>(rscs.size());
            ucs_strncpy_safe(m_ctx.name, name.c_str(), sizeof(m_ctx.name));
        }

        ucp_context_h get()
        {
            return &m_ctx;
        }

    private:
        ucp_context_t m_ctx;
        std::vector<ucp_tl_resource_desc_t> m_rscs;
    };

    static ucp_lane_type_mask_t
    lane_mask(std::initializer_list<ucp_lane_type_t> types)
    {
        ucp_lane_type_mask_t mask = 0;

        for (ucp_lane_type_t type : types) {
            mask |= UCS_BIT(type);
        }
        return mask;
    }

    /* Register a system device with a known name and return its id. */
    ucs_sys_device_t add_sys_device(uint8_t bus, const char *name)
    {
        ucs_sys_bus_id_t bus_id = {};
        ucs_sys_device_t sys_dev;

        bus_id.bus = bus;
        EXPECT_EQ(UCS_OK, ucs_topo_find_device_by_bus_id(&bus_id, &sys_dev));
        EXPECT_EQ(UCS_OK, ucs_topo_sys_device_set_name(sys_dev, name, 1));
        return sys_dev;
    }

    /* Search the context's transport resources for the (tl, dev) pair. */
    ucp_rsc_index_t find_rsc_index(ucp_context_h context, const char *tl_name,
                                   const char *dev_name)
    {
        for (ucp_rsc_index_t i = 0; i < context->num_tls; ++i) {
            const uct_tl_resource_desc_t *tl = &context->tl_rscs[i].tl_rsc;
            if ((strcmp(tl->tl_name, tl_name) == 0) &&
                (strcmp(tl->dev_name, dev_name) == 0)) {
                return i;
            }
        }

        ADD_FAILURE() << "unknown resource '" << tl_name << "/" << dev_name
                      << "'";
        return 0;
    }

    ucp_ep_config_key_t make_key(unsigned flags)
    {
        ucp_ep_config_key_t key;

        memset(&key, 0, sizeof(key));
        key.num_lanes = 0;
        key.cm_lane   = UCP_NULL_LANE;
        key.flags     = flags;
        return key;
    }

    void add_lane(ucp_context_h context, ucp_ep_config_key_t &key,
                  const char *tl_name, const char *dev_name,
                  std::initializer_list<ucp_lane_type_t> types)
    {
        ucp_ep_config_key_lane_t *lane = &key.lanes[key.num_lanes];

        memset(lane, 0, sizeof(*lane));
        lane->rsc_index  = find_rsc_index(context, tl_name, dev_name);
        lane->lane_types = lane_mask(types);
        ++key.num_lanes;
    }

    void add_cm_lane(ucp_ep_config_key_t &key,
                     std::initializer_list<ucp_lane_type_t> types)
    {
        ucp_ep_config_key_lane_t *lane = &key.lanes[key.num_lanes];

        memset(lane, 0, sizeof(*lane));
        lane->lane_types = lane_mask(types);
        key.cm_lane      = key.num_lanes;
        ++key.num_lanes;
    }

    std::string render(ucp_context_h context, const ucp_ep_config_key_t &key,
                       ucp_worker_cfg_index_t cfg_index)
    {
        ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
        ucs_status_t status;

        status = ucp_wireup_render_ep_lanes(context, &key, cfg_index, &strb);
        EXPECT_EQ(UCS_OK, status);

        std::string out(ucs_string_buffer_cstr(&strb));
        ucs_string_buffer_cleanup(&strb);
        return out;
    }
};


/* A single transport spanning several devices: only the first device shows the
 * transport name, the rest continue the same block (example output #2). */
UCS_TEST_F(test_ucp_wireup_lane_info, single_transport_multi_device) {
    ucs_sys_device_t sd_mlx5_0  = add_sys_device(0x01, "mlx5_0");
    ucs_sys_device_t sd_ens10f0 = add_sys_device(0x02, "ens10f0");
    ucs_sys_device_t sd_mlx5_1  = add_sys_device(0x03, "mlx5_1");
    dummy_context context(
        {
            {"tcp", "ibs2", sd_mlx5_0},
            {"tcp", "ens10f0", sd_ens10f0},
            {"tcp", "ibs7f0", sd_mlx5_1},
        },
        "perftest");
    ucp_ep_config_key_t key = make_key(UCP_EP_CONFIG_KEY_FLAG_SELF);

    add_lane(context.get(), key, "tcp", "ibs2",
             {UCP_LANE_TYPE_AM, UCP_LANE_TYPE_RMA_BW});
    add_lane(context.get(), key, "tcp", "ens10f0", {UCP_LANE_TYPE_RMA_BW});
    add_lane(context.get(), key, "tcp", "ibs7f0", {UCP_LANE_TYPE_RMA_BW});

    EXPECT_EQ("+-----------+--------------------+---------+------------+\n"
              "| Endpoint Config #0 (ctx: perftest, type: self)        |\n"
              "+-----------+--------------------+---------+------------+\n"
              "| Transport | Device (Sys. dev.) | # Lanes | Lane Types |\n"
              "+-----------+--------------------+---------+------------+\n"
              "| tcp       | ibs2 (mlx5_0)      |       1 | am, rma_bw |\n"
              "|           | ens10f0 (ens10f0)  |       1 | rma_bw     |\n"
              "|           | ibs7f0 (mlx5_1)    |       1 | rma_bw     |\n"
              "+-----------+--------------------+---------+------------+\n",
              render(context.get(), key, 0));
}

/* Several transports, some spanning multiple devices and some sharing a device
 * name across transports, with system-device annotations (example output #1). */
UCS_TEST_F(test_ucp_wireup_lane_info, multi_transport) {
    ucs_sys_device_t sd_mlx5_2 = add_sys_device(0x01, "mlx5_2");
    ucs_sys_device_t sd_mlx5_1 = add_sys_device(0x02, "mlx5_1");
    ucs_sys_device_t sd_mlx5_0 = add_sys_device(0x03, "mlx5_0");
    ucs_sys_device_t sd_gpu0   = add_sys_device(0x04, "GPU0");
    dummy_context context(
        {
            {"self", "memory", UCS_SYS_DEVICE_ID_UNKNOWN},
            {"rc_mlx5", "mlx5_2:1", sd_mlx5_2},
            {"rc_mlx5", "mlx5_1:1", sd_mlx5_1},
            {"cma", "memory", UCS_SYS_DEVICE_ID_UNKNOWN},
            {"cuda_copy", "cuda", sd_gpu0},
            {"cuda_copy", "mlx5_0:1", sd_mlx5_0},
            {"cuda_ipc", "cuda", sd_gpu0},
        },
        "perftest");
    ucp_ep_config_key_t key = make_key(UCP_EP_CONFIG_KEY_FLAG_SELF);

    add_lane(context.get(), key, "self", "memory",
             {UCP_LANE_TYPE_AM, UCP_LANE_TYPE_RMA, UCP_LANE_TYPE_RKEY_PTR});
    add_lane(context.get(), key, "rc_mlx5", "mlx5_2:1", {UCP_LANE_TYPE_RMA});
    add_lane(context.get(), key, "rc_mlx5", "mlx5_2:1", {UCP_LANE_TYPE_RMA_BW});
    add_lane(context.get(), key, "rc_mlx5", "mlx5_1:1", {UCP_LANE_TYPE_RMA_BW});
    add_lane(context.get(), key, "rc_mlx5", "mlx5_1:1", {UCP_LANE_TYPE_RMA_BW});
    add_lane(context.get(), key, "cma", "memory", {UCP_LANE_TYPE_RMA_BW});
    add_lane(context.get(), key, "cuda_copy", "cuda", {UCP_LANE_TYPE_RMA_BW});
    add_lane(context.get(), key, "cuda_copy", "mlx5_0:1",
             {UCP_LANE_TYPE_RMA_BW});
    add_lane(context.get(), key, "cuda_copy", "mlx5_0:1",
             {UCP_LANE_TYPE_RMA_BW});
    add_lane(context.get(), key, "cuda_ipc", "cuda", {UCP_LANE_TYPE_RMA_BW});

    EXPECT_EQ(
        "+-----------+--------------------+---------+-------------------+\n"
        "| Endpoint Config #1 (ctx: perftest, type: self)               |\n"
        "+-----------+--------------------+---------+-------------------+\n"
        "| Transport | Device (Sys. dev.) | # Lanes | Lane Types        |\n"
        "+-----------+--------------------+---------+-------------------+\n"
        "| self      | memory             |       1 | am, rma, rkey_ptr |\n"
        "+-----------+--------------------+---------+-------------------+\n"
        "| rc_mlx5   | mlx5_2:1 (mlx5_2)  |       2 | rma, rma_bw       |\n"
        "|           | mlx5_1:1 (mlx5_1)  |       2 | rma_bw            |\n"
        "+-----------+--------------------+---------+-------------------+\n"
        "| cma       | memory             |       1 | rma_bw            |\n"
        "+-----------+--------------------+---------+-------------------+\n"
        "| cuda_copy | cuda (GPU0)        |       1 | rma_bw            |\n"
        "|           | mlx5_0:1 (mlx5_0)  |       2 | rma_bw            |\n"
        "+-----------+--------------------+---------+-------------------+\n"
        "| cuda_ipc  | cuda (GPU0)        |       1 | rma_bw            |\n"
        "+-----------+--------------------+---------+-------------------+\n",
        render(context.get(), key, 1));
}

/* The "# Lanes" count is the number of lanes on the device, and "Lane Types"
 * is the union of their types in enum order. */
UCS_TEST_F(test_ucp_wireup_lane_info, lane_types_and_count) {
    dummy_context context({{"tcp", "eth0", UCS_SYS_DEVICE_ID_UNKNOWN}});
    ucp_ep_config_key_t key = make_key(UCP_EP_CONFIG_KEY_FLAG_SELF);

    add_lane(context.get(), key, "tcp", "eth0", {UCP_LANE_TYPE_AM});
    add_lane(context.get(), key, "tcp", "eth0", {UCP_LANE_TYPE_RMA});
    add_lane(context.get(), key, "tcp", "eth0", {UCP_LANE_TYPE_AMO});

    EXPECT_EQ("+-----------+--------------------+---------+--------------+\n"
              "| Endpoint Config #0 (type: self)                         |\n"
              "+-----------+--------------------+---------+--------------+\n"
              "| Transport | Device (Sys. dev.) | # Lanes | Lane Types   |\n"
              "+-----------+--------------------+---------+--------------+\n"
              "| tcp       | eth0               |       3 | am, rma, amo |\n"
              "+-----------+--------------------+---------+--------------+\n",
              render(context.get(), key, 0));
}

/* The CM lane is rendered with "cm" as both transport and device. */
UCS_TEST_F(test_ucp_wireup_lane_info, cm_lane) {
    dummy_context context({{"tcp", "eth0", UCS_SYS_DEVICE_ID_UNKNOWN}});
    ucp_ep_config_key_t key = make_key(UCP_EP_CONFIG_KEY_FLAG_SELF);

    add_lane(context.get(), key, "tcp", "eth0", {UCP_LANE_TYPE_AM});
    add_cm_lane(key, {UCP_LANE_TYPE_CM});

    EXPECT_EQ("+-----------+--------------------+---------+------------+\n"
              "| Endpoint Config #0 (type: self)                       |\n"
              "+-----------+--------------------+---------+------------+\n"
              "| Transport | Device (Sys. dev.) | # Lanes | Lane Types |\n"
              "+-----------+--------------------+---------+------------+\n"
              "| tcp       | eth0               |       1 | am         |\n"
              "+-----------+--------------------+---------+------------+\n"
              "| cm        | cm                 |       1 | cm         |\n"
              "+-----------+--------------------+---------+------------+\n",
              render(context.get(), key, 0));
}

/* The title reflects the endpoint type and only includes the context name when
 * it is set. */
UCS_TEST_F(test_ucp_wireup_lane_info, title_variants) {
    dummy_context named_ctx({{"tcp", "eth0", UCS_SYS_DEVICE_ID_UNKNOWN}},
                            "perftest");
    dummy_context anon_ctx({{"tcp", "eth0", UCS_SYS_DEVICE_ID_UNKNOWN}});
    ucp_ep_config_key_t key;

    key = make_key(UCP_EP_CONFIG_KEY_FLAG_INTRA_NODE);
    add_lane(named_ctx.get(), key, "tcp", "eth0", {UCP_LANE_TYPE_AM});
    EXPECT_NE(std::string::npos,
              render(named_ctx.get(), key, 2)
                      .find("Endpoint Config #2 (ctx: perftest, "
                            "type: intra-node)"));

    key = make_key(0);
    add_lane(anon_ctx.get(), key, "tcp", "eth0", {UCP_LANE_TYPE_AM});
    EXPECT_NE(std::string::npos,
              render(anon_ctx.get(), key, 3)
                      .find("Endpoint Config #3 (type: inter-node)"));
}
