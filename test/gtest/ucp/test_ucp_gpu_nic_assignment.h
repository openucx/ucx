/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef TEST_UCP_GPU_NIC_ASSIGNMENT_H_
#define TEST_UCP_GPU_NIC_ASSIGNMENT_H_

#include "ucp_test.h"

#include <common/mem_buffer.h>

extern "C" {
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_gpu_nic_assignment.h>
#include <ucp/core/ucp_worker.inl>
#include <ucp/proto/proto_select.h>
}

#include <string>


/* UCX_GPU_NIC_ASSIGNMENT_MODE values that set explicit assignment modes */
static const char *gpu_nic_assignment_modes[] = {
    "flip",
    "round_robin",
    "shared",
};

/* Helpers for tests that move GPU memory with a gpu-nic assignment */
class gpu_nic_assignment_checks {
protected:
    gpu_nic_assignment_checks() :
        m_proto_used(false), m_has_unassigned_bw_lane(false)
    {
    }

    /* Check that the GPU the test buffers are allocated on has assigned NICs.
     * With 'flip' and 'round_robin', a GPU may be left without any. */
    static bool is_buffer_gpu_assigned(ucp_context_h context)
    {
        const ucs_sys_device_bitmap_t *bitmap;
        ucp_memory_info_t mem_info;

        if (context->gpu_nic_assignment == nullptr) {
            return false;
        }

        mem_buffer buffer(64, UCS_MEMORY_TYPE_CUDA);
        ucp_memory_detect(context, buffer.ptr(), buffer.size(), &mem_info);

        bitmap = ucp_gpu_nic_assignment_lookup(context->gpu_nic_assignment,
                                               mem_info.sys_dev);
        return (bitmap != nullptr) && !UCS_STATIC_BITMAP_IS_ZERO(*bitmap);
    }

    /* Check that the protocol moved data, and only over the NICs assigned to
     * the GPU of the buffer it moved */
    void
    expect_assigned_lanes(const ucs::ptr_vector<ucp_test_base::entity> &entities,
                          const std::string &proto_name)
    {
        ucp_rkey_config_t **rkey_config_p;
        ucp_ep_config_t *ep_config;
        ucp_worker_h worker;

        m_proto_used             = false;
        m_has_unassigned_bw_lane = false;

        for (auto iter = entities.begin(); iter != entities.end(); ++iter) {
            worker = (*iter)->worker();
            ucs_array_for_each(ep_config, &worker->ep_config) {
                check_proto_select(worker, &ep_config->proto_select,
                                   proto_name);
            }

            ucs_array_for_each(rkey_config_p, &worker->rkey_config) {
                check_proto_select(worker, &(*rkey_config_p)->proto_select,
                                   proto_name);
            }
        }

        ASSERT_TRUE(m_proto_used) << proto_name << " was not used";
        if (!m_has_unassigned_bw_lane) {
            UCS_TEST_SKIP_R("the assignment keeps every bandwidth nic of the "
                            "endpoints");
        }
    }

private:
    void check_proto_select(ucp_worker_h worker,
                            const ucp_proto_select_t *proto_select,
                            const std::string &proto_name)
    {
        const ucp_proto_threshold_elem_t *thresh;
        size_t range_start;
        khiter_t khiter;

        for (khiter = kh_begin(proto_select->hash);
             khiter != kh_end(proto_select->hash); ++khiter) {
            if (!kh_exist(proto_select->hash, khiter)) {
                continue;
            }

            thresh      = kh_val(proto_select->hash, khiter).thresholds;
            range_start = 0;
            do {
                if ((thresh->proto_config.selections > 0) &&
                    (proto_name == thresh->proto_config.proto->name)) {
                    check_proto_config(worker, &thresh->proto_config,
                                       range_start);
                    m_proto_used = true;
                }
                range_start = thresh->max_msg_length + 1;
            } while ((thresh++)->max_msg_length < SIZE_MAX);
        }
    }

    void check_proto_config(ucp_worker_h worker,
                            const ucp_proto_config_t *proto_config,
                            size_t msg_length)
    {
        const ucp_context_h context = worker->context;
        const ucp_ep_config_t *ep_config =
                ucp_worker_ep_config(worker, proto_config->ep_cfg_index);
        const ucs_sys_device_t gpu_sys_dev = proto_config->select_param.sys_dev;
        const ucs_sys_device_bitmap_t *bitmap;
        const uct_tl_resource_desc_t *tl_rsc;
        ucp_proto_query_attr_t attr;
        ucp_lane_index_t lane;

        bitmap = ucp_gpu_nic_assignment_lookup(context->gpu_nic_assignment,
                                               gpu_sys_dev);
        ASSERT_NE(nullptr, bitmap)
                << proto_config->proto->name << " buffer sys_dev "
                << static_cast<int>(gpu_sys_dev);

        ucp_proto_config_query(worker, proto_config, msg_length, &attr);
        ucs_for_each_bit(lane, attr.lane_map) {
            tl_rsc = lane_tl_rsc(context, ep_config, lane);
            EXPECT_FALSE(is_unassigned_nic(tl_rsc, bitmap))
                    << proto_config->proto->name << " lane "
                    << static_cast<int>(lane) << " on unassigned "
                    << tl_rsc->dev_name;
        }

        if (!m_has_unassigned_bw_lane &&
            has_unassigned_bw_lane(context, ep_config, bitmap)) {
            m_has_unassigned_bw_lane = true;
        }
    }

    /* Without a bandwidth lane on an unassigned NIC, the lane check passes
     * even if the protocol ignores the assignment */
    static bool has_unassigned_bw_lane(ucp_context_h context,
                                       const ucp_ep_config_t *ep_config,
                                       const ucs_sys_device_bitmap_t *bitmap)
    {
        ucp_lane_index_t lane;

        for (lane = 0; lane < ep_config->key.num_lanes; ++lane) {
            if ((ep_config->key.lanes[lane].lane_types &
                 UCS_BIT(UCP_LANE_TYPE_RMA_BW)) &&
                is_unassigned_nic(lane_tl_rsc(context, ep_config, lane),
                                  bitmap)) {
                return true;
            }
        }

        return false;
    }

    static const uct_tl_resource_desc_t *
    lane_tl_rsc(ucp_context_h context, const ucp_ep_config_t *ep_config,
                ucp_lane_index_t lane)
    {
        return &context->tl_rscs[ep_config->key.lanes[lane].rsc_index].tl_rsc;
    }

    static bool is_unassigned_nic(const uct_tl_resource_desc_t *tl_rsc,
                                  const ucs_sys_device_bitmap_t *bitmap)
    {
        return (tl_rsc->dev_type == UCT_DEVICE_TYPE_NET) &&
               !ucp_gpu_nic_bitmap_get(bitmap, tl_rsc->sys_device);
    }

    bool m_proto_used;
    bool m_has_unassigned_bw_lane;
};

#endif
