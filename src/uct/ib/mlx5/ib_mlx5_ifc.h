/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_IB_MLX5_IFC_H_
#define UCT_IB_MLX5_IFC_H_

#include <ucs/sys/compiler_def.h>

#include <endian.h>
#include <linux/types.h>

#define __uct_nullp(_typ) ((struct uct_ib_mlx5_##_typ##_bits *)0)
#define __uct_bit_sz(_typ, _fld) sizeof(__uct_nullp(_typ)->_fld)
#define __uct_bit_off(_typ, _fld) (offsetof(struct uct_ib_mlx5_##_typ##_bits, _fld))
#define __uct_dw_off(_typ, _fld) (__uct_bit_off(_typ, _fld) / 32)
#define __uct_64_off(_typ, _fld) (__uct_bit_off(_typ, _fld) / 64)
#define __uct_dw_bit_off(_typ, _fld) (32 - __uct_bit_sz(_typ, _fld) - (__uct_bit_off(_typ, _fld) & 0x1f))
#define __uct_mask(_typ, _fld) ((uint32_t)((1ull << __uct_bit_sz(_typ, _fld)) - 1))
#define __uct_dw_mask(_typ, _fld) (__uct_mask(_typ, _fld) << __uct_dw_bit_off(_typ, _fld))
#define __uct_st_sz_bits(_typ) sizeof(struct uct_ib_mlx5_##_typ##_bits)

#define UCT_IB_MLX5DV_FLD_SZ_BYTES(_typ, _fld) (__uct_bit_sz(_typ, _fld) / 8)
#define UCT_IB_MLX5DV_ST_SZ_BYTES(_typ) (sizeof(struct uct_ib_mlx5_##_typ##_bits) / 8)
#define UCT_IB_MLX5DV_ST_SZ_DW(_typ) (sizeof(struct uct_ib_mlx5_##_typ##_bits) / 32)
#define UCT_IB_MLX5DV_ST_SZ_QW(_typ) (sizeof(struct uct_ib_mlx5_##_typ##_bits) / 64)
#define UCT_IB_MLX5DV_UN_SZ_BYTES(_typ) (sizeof(union uct_ib_mlx5_##_typ##_bits) / 8)
#define UCT_IB_MLX5DV_UN_SZ_DW(_typ) (sizeof(union uct_ib_mlx5_##_typ##_bits) / 32)
#define UCT_IB_MLX5DV_BYTE_OFF(_typ, _fld) (__uct_bit_off(_typ, _fld) / 8)
#define UCT_IB_MLX5DV_ADDR_OF(_typ, _p, _fld) ((char *)(_p) + UCT_IB_MLX5DV_BYTE_OFF(_typ, _fld))

/* insert a value to a struct */
#define UCT_IB_MLX5DV_SET(_typ, _p, _fld, _v) \
    do { \
        uint32_t ___v = _v; \
        UCS_STATIC_ASSERT(__uct_st_sz_bits(_typ) % 32 == 0); \
        *((__be32 *)(_p) + __uct_dw_off(_typ, _fld)) = \
            htobe32((be32toh(*((__be32 *)(_p) + __uct_dw_off(_typ, _fld))) & \
            (~__uct_dw_mask(_typ, _fld))) | (((___v) & __uct_mask(_typ, _fld)) \
            << __uct_dw_bit_off(_typ, _fld))); \
    } while (0)

#define UCT_IB_MLX5DV_GET(_typ, _p, _fld) \
    ((be32toh(*((__be32 *)(_p) + \
        __uct_dw_off(_typ, _fld))) >> __uct_dw_bit_off(_typ, _fld)) & \
        __uct_mask(_typ, _fld))

#define UCT_IB_MLX5DV_SET64(_typ, _p, _fld, _v) \
    do { \
        UCS_STATIC_ASSERT(__uct_st_sz_bits(_typ) % 64 == 0); \
        UCS_STATIC_ASSERT(__uct_bit_sz(_typ, _fld) == 64); \
        *((__be64 *)(_p) + __uct_64_off(_typ, _fld)) = htobe64(_v); \
    } while (0)

#define UCT_IB_MLX5DV_GET64(_typ, _p, _fld) \
    be64toh(*((__be64 *)(_p) + __uct_64_off(_typ, _fld)))

enum {
    UCT_IB_MLX5_CMD_OP_QUERY_HCA_CAP        = 0x100
};

enum {
    UCT_IB_MLX5_HCA_CAP_OPMOD_GET_MAX	= 0,
    UCT_IB_MLX5_HCA_CAP_OPMOD_GET_CUR	= 1
};

enum {
    UCT_IB_MLX5_CAP_GENERAL = 0,
    UCT_IB_MLX5_CAP_ATOMIC  = 3
};

struct uct_ib_mlx5_cmd_hca_cap_bits {
    uint8_t    reserved_at_0[0x30];
    uint8_t    vhca_id[0x10];

    uint8_t    reserved_at_40[0x40];

    uint8_t    log_max_srq_sz[0x8];
    uint8_t    log_max_qp_sz[0x8];
    uint8_t    reserved_at_90[0xb];
    uint8_t    log_max_qp[0x5];

    uint8_t    reserved_at_a0[0xb];
    uint8_t    log_max_srq[0x5];
    uint8_t    reserved_at_b0[0x10];

    uint8_t    reserved_at_c0[0x8];
    uint8_t    log_max_cq_sz[0x8];
    uint8_t    reserved_at_d0[0xb];
    uint8_t    log_max_cq[0x5];

    uint8_t    log_max_eq_sz[0x8];
    uint8_t    reserved_at_e8[0x2];
    uint8_t    log_max_mkey[0x6];
    uint8_t    reserved_at_f0[0x4];
    uint8_t    cmd_on_behalf[0x1];
    uint8_t    device_emulation_manager[0x1];
    uint8_t    reserved_at_f6[0x6];
    uint8_t    log_max_eq[0x4];

    uint8_t    max_indirection[0x8];
    uint8_t    fixed_buffer_size[0x1];
    uint8_t    log_max_mrw_sz[0x7];
    uint8_t    force_teardown[0x1];
    uint8_t    reserved_at_111[0x1];
    uint8_t    log_max_bsf_list_size[0x6];
    uint8_t    umr_extended_translation_offset[0x1];
    uint8_t    null_mkey[0x1];
    uint8_t    log_max_klm_list_size[0x6];

    uint8_t    reserved_at_120[0xa];
    uint8_t    log_max_ra_req_dc[0x6];
    uint8_t    reserved_at_130[0xa];
    uint8_t    log_max_ra_res_dc[0x6];

    uint8_t    reserved_at_140[0xa];
    uint8_t    log_max_ra_req_qp[0x6];
    uint8_t    reserved_at_150[0xa];
    uint8_t    log_max_ra_res_qp[0x6];

    uint8_t    end_pad[0x1];
    uint8_t    cc_query_allowed[0x1];
    uint8_t    cc_modify_allowed[0x1];
    uint8_t    start_pad[0x1];
    uint8_t    cache_line_128byte[0x1];
    uint8_t    reserved_at_165[0xa];
    uint8_t    qcam_reg[0x1];
    uint8_t    gid_table_size[0x10];

    uint8_t    out_of_seq_cnt[0x1];
    uint8_t    vport_counters[0x1];
    uint8_t    retransmission_q_counters[0x1];
    uint8_t    debug[0x1];
    uint8_t    modify_rq_counter_set_id[0x1];
    uint8_t    rq_delay_drop[0x1];
    uint8_t    max_qp_cnt[0xa];
    uint8_t    pkey_table_size[0x10];

    uint8_t    vport_group_manager[0x1];
    uint8_t    vhca_group_manager[0x1];
    uint8_t    ib_virt[0x1];
    uint8_t    eth_virt[0x1];
    uint8_t    vnic_env_queue_counters[0x1];
    uint8_t    ets[0x1];
    uint8_t    nic_flow_table[0x1];
    uint8_t    eswitch_flow_table[0x1];
    uint8_t    device_memory[0x1];
    uint8_t    mcam_reg[0x1];
    uint8_t    pcam_reg[0x1];
    uint8_t    local_ca_ack_delay[0x5];
    uint8_t    port_module_event[0x1];
    uint8_t    enhanced_error_q_counters[0x1];
    uint8_t    ports_check[0x1];
    uint8_t    reserved_at_1b3[0x1];
    uint8_t    disable_link_up[0x1];
    uint8_t    beacon_led[0x1];
    uint8_t    port_type[0x2];
    uint8_t    num_ports[0x8];

    uint8_t    reserved_at_1c0[0x1];
    uint8_t    pps[0x1];
    uint8_t    pps_modify[0x1];
    uint8_t    log_max_msg[0x5];
    uint8_t    reserved_at_1c8[0x4];
    uint8_t    max_tc[0x4];
    uint8_t    reserved_at_1d0[0x1];
    uint8_t    dcbx[0x1];
    uint8_t    general_notification_event[0x1];
    uint8_t    reserved_at_1d3[0x2];
    uint8_t    fpga[0x1];
    uint8_t    rol_s[0x1];
    uint8_t    rol_g[0x1];
    uint8_t    reserved_at_1d8[0x1];
    uint8_t    wol_s[0x1];
    uint8_t    wol_g[0x1];
    uint8_t    wol_a[0x1];
    uint8_t    wol_b[0x1];
    uint8_t    wol_m[0x1];
    uint8_t    wol_u[0x1];
    uint8_t    wol_p[0x1];

    uint8_t    stat_rate_support[0x10];
    uint8_t    reserved_at_1f0[0xc];
    uint8_t    cqe_version[0x4];

    uint8_t    compact_address_vector[0x1];
    uint8_t    striding_rq[0x1];
    uint8_t    reserved_at_202[0x1];
    uint8_t    ipoib_enhanced_offloads[0x1];
    uint8_t    ipoib_basic_offloads[0x1];
    uint8_t    reserved_at_205[0x1];
    uint8_t    repeated_block_disabled[0x1];
    uint8_t    umr_modify_entity_size_disabled[0x1];
    uint8_t    umr_modify_atomic_disabled[0x1];
    uint8_t    umr_indirect_mkey_disabled[0x1];
    uint8_t    umr_fence[0x2];
    uint8_t    reserved_at_20c[0x3];
    uint8_t    drain_sigerr[0x1];
    uint8_t    cmdif_checksum[0x2];
    uint8_t    sigerr_cqe[0x1];
    uint8_t    reserved_at_213[0x1];
    uint8_t    wq_signature[0x1];
    uint8_t    sctr_data_cqe[0x1];
    uint8_t    reserved_at_216[0x1];
    uint8_t    sho[0x1];
    uint8_t    tph[0x1];
    uint8_t    rf[0x1];
    uint8_t    dct[0x1];
    uint8_t    qos[0x1];
    uint8_t    eth_net_offloads[0x1];
    uint8_t    roce[0x1];
    uint8_t    atomic[0x1];
    uint8_t    reserved_at_21f[0x1];

    uint8_t    cq_oi[0x1];
    uint8_t    cq_resize[0x1];
    uint8_t    cq_moderation[0x1];
    uint8_t    reserved_at_223[0x3];
    uint8_t    cq_eq_remap[0x1];
    uint8_t    pg[0x1];
    uint8_t    block_lb_mc[0x1];
    uint8_t    reserved_at_229[0x1];
    uint8_t    scqe_break_moderation[0x1];
    uint8_t    cq_period_start_from_cqe[0x1];
    uint8_t    cd[0x1];
    uint8_t    reserved_at_22d[0x1];
    uint8_t    apm[0x1];
    uint8_t    vector_calc[0x1];
    uint8_t    umr_ptr_rlky[0x1];
    uint8_t     imaicl[0x1];
    uint8_t    reserved_at_232[0x4];
    uint8_t    qkv[0x1];
    uint8_t    pkv[0x1];
    uint8_t    set_deth_sqpn[0x1];
    uint8_t    reserved_at_239[0x3];
    uint8_t    xrc[0x1];
    uint8_t    ud[0x1];
    uint8_t    uc[0x1];
    uint8_t    rc[0x1];

    uint8_t    uar_4k[0x1];
    uint8_t    reserved_at_241[0x9];
    uint8_t    uar_sz[0x6];
    uint8_t    reserved_at_250[0x8];
    uint8_t    log_pg_sz[0x8];

    uint8_t    bf[0x1];
    uint8_t    driver_version[0x1];
    uint8_t    pad_tx_eth_packet[0x1];
    uint8_t    reserved_at_263[0x8];
    uint8_t    log_bf_reg_size[0x5];

    uint8_t    reserved_at_270[0xb];
    uint8_t    lag_master[0x1];
    uint8_t    num_lag_ports[0x4];

    uint8_t    reserved_at_280[0x10];
    uint8_t    max_wqe_sz_sq[0x10];

    uint8_t    reserved_at_2a0[0x10];
    uint8_t    max_wqe_sz_rq[0x10];

    uint8_t    max_flow_counter_31_16[0x10];
    uint8_t    max_wqe_sz_sq_dc[0x10];

    uint8_t    reserved_at_2e0[0x7];
    uint8_t    max_qp_mcg[0x19];

    uint8_t    reserved_at_300[0x18];
    uint8_t    log_max_mcg[0x8];

    uint8_t    reserved_at_320[0x3];
    uint8_t    log_max_transport_domain[0x5];
    uint8_t    reserved_at_328[0x3];
    uint8_t    log_max_pd[0x5];
    uint8_t    reserved_at_330[0xb];
    uint8_t    log_max_xrcd[0x5];

    uint8_t    nic_receive_steering_discard[0x1];
    uint8_t    receive_discard_vport_down[0x1];
    uint8_t    transmit_discard_vport_down[0x1];
    uint8_t    reserved_at_343[0x5];
    uint8_t    log_max_flow_counter_bulk[0x8];
    uint8_t    max_flow_counter_15_0[0x10];


    uint8_t    reserved_at_360[0x3];
    uint8_t    log_max_rq[0x5];
    uint8_t    reserved_at_368[0x3];
    uint8_t    log_max_sq[0x5];
    uint8_t    reserved_at_370[0x3];
    uint8_t    log_max_tir[0x5];
    uint8_t    reserved_at_378[0x3];
    uint8_t    log_max_tis[0x5];

    uint8_t    basic_cyclic_rcv_wqe[0x1];
    uint8_t    reserved_at_381[0x2];
    uint8_t    log_max_rmp[0x5];
    uint8_t    reserved_at_388[0x3];
    uint8_t    log_max_rqt[0x5];
    uint8_t    reserved_at_390[0x3];
    uint8_t    log_max_rqt_size[0x5];
    uint8_t    reserved_at_398[0x3];
    uint8_t    log_max_tis_per_sq[0x5];

    uint8_t    ext_stride_num_range[0x1];
    uint8_t    reserved_at_3a1[0x2];
    uint8_t    log_max_stride_sz_rq[0x5];
    uint8_t    reserved_at_3a8[0x3];
    uint8_t    log_min_stride_sz_rq[0x5];
    uint8_t    reserved_at_3b0[0x3];
    uint8_t    log_max_stride_sz_sq[0x5];
    uint8_t    reserved_at_3b8[0x3];
    uint8_t    log_min_stride_sz_sq[0x5];

    uint8_t    hairpin[0x1];
    uint8_t    reserved_at_3c1[0x2];
    uint8_t    log_max_hairpin_queues[0x5];
    uint8_t    reserved_at_3c8[0x3];
    uint8_t    log_max_hairpin_wq_data_sz[0x5];
    uint8_t    reserved_at_3d0[0x3];
    uint8_t    log_max_hairpin_num_packets[0x5];
    uint8_t    reserved_at_3d8[0x3];
    uint8_t    log_max_wq_sz[0x5];

    uint8_t    nic_vport_change_event[0x1];
    uint8_t    disable_local_lb_uc[0x1];
    uint8_t    disable_local_lb_mc[0x1];
    uint8_t    log_min_hairpin_wq_data_sz[0x5];
    uint8_t    reserved_at_3e8[0x3];
    uint8_t    log_max_vlan_list[0x5];
    uint8_t    reserved_at_3f0[0x3];
    uint8_t    log_max_current_mc_list[0x5];
    uint8_t    reserved_at_3f8[0x3];
    uint8_t    log_max_current_uc_list[0x5];

    uint8_t    general_obj_types[0x40];

    uint8_t    reserved_at_440[0x40];

    uint8_t    reserved_at_480[0x3];
    uint8_t    log_max_l2_table[0x5];
    uint8_t    reserved_at_488[0x8];
    uint8_t    log_uar_page_sz[0x10];

    uint8_t    reserved_at_4a0[0x20];
    uint8_t    device_frequency_mhz[0x20];
    uint8_t    device_frequency_khz[0x20];

    uint8_t    reserved_at_500[0x20];
    uint8_t    num_of_uars_per_page[0x20];
    uint8_t    reserved_at_540[0x40];

    uint8_t    reserved_at_580[0x3d];
    uint8_t    cqe_128_always[0x1];
    uint8_t    cqe_compression_128[0x1];
    uint8_t    cqe_compression[0x1];

    uint8_t    cqe_compression_timeout[0x10];
    uint8_t    cqe_compression_max_num[0x10];

    uint8_t    reserved_at_5e0[0x10];
    uint8_t    tag_matching[0x1];
    uint8_t    rndv_offload_rc[0x1];
    uint8_t    rndv_offload_dc[0x1];
    uint8_t    log_tag_matching_list_sz[0x5];
    uint8_t    reserved_at_5f8[0x3];
    uint8_t    log_max_xrq[0x5];

    uint8_t    affiliate_nic_vport_criteria[0x8];
    uint8_t    native_port_num[0x8];
    uint8_t    num_vhca_ports[0x8];
    uint8_t    reserved_at_618[0x6];
    uint8_t    sw_owner_id[0x1];
    uint8_t    reserved_at_61f[0x1e1];
};

enum {
    UCT_IB_MLX5_ATOMIC_OPS_CMP_SWAP          = UCS_BIT(0),
    UCT_IB_MLX5_ATOMIC_OPS_FETCH_ADD         = UCS_BIT(1),
    UCT_IB_MLX5_ATOMIC_OPS_MASKED_CMP_SWAP   = UCS_BIT(2),
    UCT_IB_MLX5_ATOMIC_OPS_MASKED_FETCH_ADD  = UCS_BIT(3)
};

struct uct_ib_mlx5_atomic_caps_bits {
    uint8_t    reserved_at_0[0x40];

    uint8_t    atomic_req_8B_endianness_mode[0x2];
    uint8_t    reserved_at_42[0x4];
    uint8_t    supported_atomic_req_8B_endianness_mode_1[0x1];

    uint8_t    reserved_at_47[0x19];

    uint8_t    reserved_at_60[0x20];

    uint8_t    reserved_at_80[0x10];
    uint8_t    atomic_operations[0x10];

    uint8_t    reserved_at_a0[0x10];
    uint8_t    atomic_size_qp[0x10];

    uint8_t    reserved_at_c0[0x10];
    uint8_t    atomic_size_dc[0x10];

    uint8_t    reserved_at_e0[0x720];
};

union uct_ib_mlx5_hca_cap_union_bits {
    struct uct_ib_mlx5_cmd_hca_cap_bits cmd_hca_cap;
    struct uct_ib_mlx5_atomic_caps_bits atomic_caps;
    uint8_t    reserved_at_0[0x8000];
};

struct uct_ib_mlx5_query_hca_cap_out_bits {
    uint8_t    status[0x8];
    uint8_t    reserved_at_8[0x18];

    uint8_t    syndrome[0x20];

    uint8_t    reserved_at_40[0x40];

    union uct_ib_mlx5_hca_cap_union_bits capability;
};

struct uct_ib_mlx5_query_hca_cap_in_bits {
    uint8_t    opcode[0x10];
    uint8_t    uid[0x10];

    uint8_t    reserved_at_20[0x10];
    uint8_t    op_mod[0x10];

    uint8_t    reserved_at_40[0x40];
};

#endif
