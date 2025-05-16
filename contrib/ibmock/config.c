/**
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"

struct efadv_device_attr efa_dev_attr = {
    .comp_mask       = 0,
    .max_sq_wr       = 4096,
    .max_rq_wr       = 32768,
    .max_sq_sge      = 2,
    .max_rq_sge      = 3,
    .inline_buf_size = 32,
    .device_caps     = 15,
    .max_rdma_size   = 1073741824
};

struct ibv_device_attr efa_ibv_dev_attr = {
    .fw_ver                    = "0.0.0.0",
    .node_guid                 = 0,
    .sys_image_guid            = 0,
    .max_mr_size               = 103079215104,
    .page_size_cap             = 4294963200,
    .vendor_id                 = 7439,
    .vendor_part_id            = 61345,
    .hw_ver                    = 61345,
    .max_qp                    = 256,
    .max_qp_wr                 = 4096,
    .device_cap_flags          = 0,
    .max_sge                   = 2,
    .max_sge_rd                = 1,
    .max_cq                    = 512,
    .max_cqe                   = 32768,
    .max_mr                    = 262144,
    .max_pd                    = 256,
    .max_qp_rd_atom            = 0,
    .max_ee_rd_atom            = 0,
    .max_res_rd_atom           = 0,
    .max_qp_init_rd_atom       = 0,
    .max_ee_init_rd_atom       = 0,
    .atomic_cap                = IBV_ATOMIC_NONE,
    .max_ee                    = 0,
    .max_rdd                   = 0,
    .max_mw                    = 0,
    .max_raw_ipv6_qp           = 0,
    .max_raw_ethy_qp           = 0,
    .max_mcast_grp             = 0,
    .max_mcast_qp_attach       = 0,
    .max_total_mcast_qp_attach = 0,
    .max_ah                    = 1024,
    .max_fmr                   = 0,
    .max_map_per_fmr           = 0,
    .max_srq                   = 0,
    .max_srq_wr                = 0,
    .max_srq_sge               = 0,
    .max_pkeys                 = 1,
    .local_ca_ack_delay        = 0,
    .phys_port_cnt             = 1
};

struct ibv_port_attr efa_ib_port_attr = {
    .state           = IBV_PORT_ACTIVE,
    .max_mtu         = IBV_MTU_4096,
    .active_mtu      = IBV_MTU_4096,
    .gid_tbl_len     = 1,
    .port_cap_flags  = 0,
    .max_msg_sz      = 8928,
    .bad_pkey_cntr   = 0,
    .qkey_viol_cntr  = 0,
    .pkey_tbl_len    = 1,
    .lid             = 0,
    .sm_lid          = 0,
    .lmc             = 1,
    .max_vl_num      = 1,
    .sm_sl           = 0,
    .subnet_timeout  = 0,
    .init_type_reply = 0,
    .active_width    = 2,
    .active_speed    = 32,
    .phys_state      = 5,
    .link_layer      = 0,
    .flags           = 0,
    .port_cap_flags2 = 0,
    /* .active_speed_ex = 0 */
};

struct ibv_qp_attr efa_ib_qp_attr = {
    .cap = {
        .max_send_wr     = 256,
        .max_recv_wr     = 4096,
        .max_send_sge    = 2,
        .max_recv_sge    = 1,
        .max_inline_data = 32
    }
};

struct ibv_qp_init_attr efa_ib_qp_init_attr = {
    .srq        = 0x0,
    .cap        = {
        .max_send_wr     = 256,
        .max_recv_wr     = 4096,
        .max_send_sge    = 2,
        .max_recv_sge    = 1,
        .max_inline_data = 32
    },
    .sq_sig_all = 1
};
