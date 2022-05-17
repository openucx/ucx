/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_IB_MLX5_IFC_H_
#define UCT_IB_MLX5_IFC_H_

#include <ucs/sys/compiler_def.h>

#include <stdint.h>
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
        char *___p = _p; \
        uint32_t ___v = _v; \
        uint32_t ___h; \
        UCS_STATIC_ASSERT(__uct_st_sz_bits(_typ) % 32 == 0); \
        ___h = (be32toh(*((__be32 *)(___p) + __uct_dw_off(_typ, _fld))) & \
                (~__uct_dw_mask(_typ, _fld))) | \
               (((___v) & __uct_mask(_typ, _fld)) << \
                __uct_dw_bit_off(_typ, _fld)); \
        *((__be32 *)(___p) + __uct_dw_off(_typ, _fld)) = htobe32(___h); \
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
    UCT_IB_MLX5_CMD_OP_QUERY_HCA_CAP           = 0x100,
    UCT_IB_MLX5_CMD_OP_CREATE_MKEY             = 0x200,
    UCT_IB_MLX5_CMD_OP_CREATE_QP               = 0x500,
    UCT_IB_MLX5_CMD_OP_RST2INIT_QP             = 0x502,
    UCT_IB_MLX5_CMD_OP_INIT2RTR_QP             = 0x503,
    UCT_IB_MLX5_CMD_OP_RTR2RTS_QP              = 0x504,
    UCT_IB_MLX5_CMD_OP_2ERR_QP                 = 0x507,
    UCT_IB_MLX5_CMD_OP_2RST_QP                 = 0x50a,
    UCT_IB_MLX5_CMD_OP_QUERY_QP                = 0x50b,
    UCT_IB_MLX5_CMD_OP_CREATE_RMP              = 0x90c,
    UCT_IB_MLX5_CMD_OP_CREATE_DCT              = 0x710,
    UCT_IB_MLX5_CMD_OP_DRAIN_DCT               = 0x712,
    UCT_IB_MLX5_CMD_OP_CREATE_XRQ              = 0x717,
    UCT_IB_MLX5_CMD_OP_SET_XRQ_DC_PARAMS_ENTRY = 0x726,
    UCT_IB_MLX5_CMD_OP_QUERY_HCA_VPORT_CONTEXT = 0x762,
    UCT_IB_MLX5_CMD_OP_QUERY_LAG               = 0x842,
    UCT_IB_MLX5_CMD_OP_CREATE_GENERAL_OBJECT   = 0xa00,
    UCT_IB_MLX5_CMD_OP_MODIFY_GENERAL_OBJECT   = 0xa01,
    UCT_IB_MLX5_CMD_OP_QUERY_GENERAL_OBJECT    = 0xa02
};

enum {
    UCT_IB_MLX5_HCA_CAP_OPMOD_GET_MAX = 0,
    UCT_IB_MLX5_HCA_CAP_OPMOD_GET_CUR = 1
};

enum {
    UCT_IB_MLX5_CAP_GENERAL   = 0x0,
    UCT_IB_MLX5_CAP_ODP       = 0x2,
    UCT_IB_MLX5_CAP_ATOMIC    = 0x3,
    UCT_IB_MLX5_CAP_2_GENERAL = 0x20,
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
    uint8_t    reserved_at_130[0x8];
    uint8_t    ooo_sl_mask[0x1];
    uint8_t    reserved_at_139[0x1];
    uint8_t    log_max_ra_res_dc[0x6];

    uint8_t    reserved_at_140[0xa];
    uint8_t    log_max_ra_req_qp[0x6];
    uint8_t    reserved_at_150[0x2];
    uint8_t    rts2rts_lag_tx_port_affinity[0x1];
    uint8_t    reserved_at_153[0x7];
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
    uint8_t    reserved_at_1f0[0x8];
    uint8_t    init2_lag_tx_port_affinity[0x1];
    uint8_t    reserved_at_1f9[0x3];
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
    uint8_t    reserved_at_223[0x2];
    uint8_t    ib_striding_wq_cq_first_indication[0x1];
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
    uint8_t    dci_no_rdma_wr_optimized_performance[0x1];
    uint8_t    reserved_at_242[0x8];
    uint8_t    uar_sz[0x6];
    uint8_t    reserved_at_250[0x8];
    uint8_t    log_pg_sz[0x8];

    uint8_t    bf[0x1];
    uint8_t    driver_version[0x1];
    uint8_t    pad_tx_eth_packet[0x1];
    uint8_t    reserved_at_263[0x8];
    uint8_t    log_bf_reg_size[0x5];
    uint8_t    reserved_at_270[0x6];
    uint8_t    lag_dct[0x2];
    uint8_t    lag_tx_port_affinity[0x1];
    uint8_t    reserved_at_279[0x2];
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

    uint8_t    reserved_at_580[0x3b];
    uint8_t    enhanced_cqe_compression[0x1];
    uint8_t    mini_cqe_resp_stride_index[0x1];
    uint8_t    cqe_128_always[0x1];
    uint8_t    cqe_compression_128b[0x1];
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

    uint8_t    reserved_at_e0[0x1a0];

    uint8_t    fetch_add_pci_atomic[0x10];
    uint8_t    swap_pci_atomic[0x10];
    uint8_t    compare_swap_pci_atomic[0x10];
    uint8_t    reserved_at_2b0[0x10];

    uint8_t    reserved_at_2c0[0x540];
};

struct uct_ib_mlx5_cmd_hca_cap_2_bits {
    uint8_t    reserved_at_0[0x80];

    uint8_t    reserved_at_80[0x13];
    /* Log (base 2) of the minimum bulk granularity of
       allocated RESERVED_QPN objects */
    uint8_t    log_reserved_qpn_granularity[0x5];
    uint8_t    reserved_at_98[0x8];

    uint8_t    reserved_at_a0[0x760];
};

struct uct_ib_mlx5_odp_per_transport_service_cap_bits {
    uint8_t         send[0x1];
    uint8_t         receive[0x1];
    uint8_t         write[0x1];
    uint8_t         read[0x1];
    uint8_t         atomic[0x1];
    uint8_t         srq_receive[0x1];
    uint8_t         reserved_at_6[0x1a];
};

struct uct_ib_mlx5_odp_cap_bits {
    uint8_t         reserved_at_0[0x40];

    uint8_t         sig[0x1];
    uint8_t         reserved_at_41[0x1f];

    uint8_t         reserved_at_60[0x20];

    struct uct_ib_mlx5_odp_per_transport_service_cap_bits rc_odp_caps;

    struct uct_ib_mlx5_odp_per_transport_service_cap_bits uc_odp_caps;

    struct uct_ib_mlx5_odp_per_transport_service_cap_bits ud_odp_caps;

    struct uct_ib_mlx5_odp_per_transport_service_cap_bits xrc_odp_caps;

    struct uct_ib_mlx5_odp_per_transport_service_cap_bits dc_odp_caps;

    uint8_t         reserved_at_100[0x700];
};

union uct_ib_mlx5_hca_cap_union_bits {
    struct uct_ib_mlx5_cmd_hca_cap_bits cmd_hca_cap;
    struct uct_ib_mlx5_odp_cap_bits odp_cap;
    struct uct_ib_mlx5_atomic_caps_bits atomic_caps;
    struct uct_ib_mlx5_cmd_hca_cap_2_bits cmd_hca_cap_2;
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

struct uct_ib_mlx5_lag_context_bits {
    uint8_t    reserved_at_0[0x1d];
    uint8_t    lag_state[0x3];
    uint8_t    reserved_at_20[0x20];
};

struct uct_ib_mlx5_query_lag_out_bits {
    uint8_t    status[0x8];
    uint8_t    reserved_at_8[0x18];

    uint8_t    syndrome[0x20];

    struct uct_ib_mlx5_lag_context_bits lag_context;
};

struct uct_ib_mlx5_query_lag_in_bits {
    uint8_t    opcode[0x10];
    uint8_t    uid[0x10];

    uint8_t    reserved_at_20[0x10];
    uint8_t    op_mod[0x10];

    uint8_t    reserved_at_40[0x40];
};

struct uct_ib_mlx5_hca_vport_context_bits {
    uint8_t    field_select[0x20];

    uint8_t    reserved_at_20[0xe0];

    uint8_t    sm_virt_aware[0x1];
    uint8_t    has_smi[0x1];
    uint8_t    has_raw[0x1];
    uint8_t    grh_required[0x1];
    uint8_t    reserved_at_104[0xc];
    uint8_t    port_physical_state[0x4];
    uint8_t    vport_state_policy[0x4];
    uint8_t    port_state[0x4];
    uint8_t    vport_state[0x4];

    uint8_t    reserved_at_120[0x20];

    uint8_t    system_image_guid[0x40];

    uint8_t    port_guid[0x40];

    uint8_t    node_guid[0x40];

    uint8_t    cap_mask1[0x20];

    uint8_t    cap_mask1_field_select[0x20];

    uint8_t    cap_mask2[0x20];

    uint8_t    cap_mask2_field_select[0x20];

    uint8_t    reserved_at_280[0x10];

    uint8_t    ooo_sl_mask[0x10];

    uint8_t    reserved_at_296[0x40];

    uint8_t    lid[0x10];
    uint8_t    reserved_at_310[0x4];
    uint8_t    init_type_reply[0x4];
    uint8_t    lmc[0x3];
    uint8_t    subnet_timeout[0x5];

    uint8_t    sm_lid[0x10];
    uint8_t    sm_sl[0x4];
    uint8_t    reserved_at_334[0xc];

    uint8_t    qkey_violation_counter[0x10];
    uint8_t    pkey_violation_counter[0x10];

    uint8_t    reserved_at_360[0xca0];
};

struct uct_ib_mlx5_query_hca_vport_context_out_bits {
    uint8_t    status[0x8];
    uint8_t    reserved_at_8[0x18];

    uint8_t    syndrome[0x20];

    uint8_t    reserved_at_40[0x40];

    struct uct_ib_mlx5_hca_vport_context_bits hca_vport_context;
};

struct uct_ib_mlx5_query_hca_vport_context_in_bits {
    uint8_t    opcode[0x10];
    uint8_t    reserved_at_10[0x10];

    uint8_t    reserved_at_20[0x10];
    uint8_t    op_mod[0x10];

    uint8_t    other_vport[0x1];
    uint8_t    reserved_at_41[0xb];
    uint8_t    port_num[0x4];
    uint8_t    vport_number[0x10];

    uint8_t    reserved_at_60[0x20];
};

enum {
    UCT_IB_MLX5_MKC_ACCESS_MODE_PA    = 0x0,
    UCT_IB_MLX5_MKC_ACCESS_MODE_MTT   = 0x1,
    UCT_IB_MLX5_MKC_ACCESS_MODE_KLMS  = 0x2,
    UCT_IB_MLX5_MKC_ACCESS_MODE_KSM   = 0x3,
    UCT_IB_MLX5_MKC_ACCESS_MODE_MEMIC = 0x5
};

struct uct_ib_mlx5_mkc_bits {
    uint8_t    reserved_at_0[0x1];
    uint8_t    free[0x1];
    uint8_t    reserved_at_2[0x1];
    uint8_t    access_mode_4_2[0x3];
    uint8_t    reserved_at_6[0x7];
    uint8_t    relaxed_ordering_write[0x1];
    uint8_t    reserved_at_e[0x1];
    uint8_t    small_fence_on_rdma_read_response[0x1];
    uint8_t    umr_en[0x1];
    uint8_t    a[0x1];
    uint8_t    rw[0x1];
    uint8_t    rr[0x1];
    uint8_t    lw[0x1];
    uint8_t    lr[0x1];
    uint8_t    access_mode_1_0[0x2];
    uint8_t    reserved_at_18[0x8];

    uint8_t    qpn[0x18];
    uint8_t    mkey_7_0[0x8];

    uint8_t    reserved_at_40[0x20];

    uint8_t    length64[0x1];
    uint8_t    bsf_en[0x1];
    uint8_t    sync_umr[0x1];
    uint8_t    reserved_at_63[0x2];
    uint8_t    expected_sigerr_count[0x1];
    uint8_t    reserved_at_66[0x1];
    uint8_t    en_rinval[0x1];
    uint8_t    pd[0x18];

    uint8_t    start_addr[0x40];

    uint8_t    len[0x40];

    uint8_t    bsf_octword_size[0x20];

    uint8_t    reserved_at_120[0x80];

    uint8_t    translations_octword_size[0x20];

    uint8_t    reserved_at_1c0[0x1b];
    uint8_t    log_entity_size[0x5];

    uint8_t    reserved_at_1e0[0x20];
};

struct uct_ib_mlx5_create_mkey_in_bits {
    uint8_t    opcode[0x10];
    uint8_t    uid[0x10];

    uint8_t    reserved_at_20[0x10];
    uint8_t    op_mod[0x10];

    uint8_t    reserved_at_40[0x20];

    uint8_t    pg_access[0x1];
    uint8_t    mkey_umem_valid[0x1];
    uint8_t    cmd_on_behalf[0x1];
    uint8_t    reserved_at_63[0xd];
    uint8_t    function_id[0x10];

    struct uct_ib_mlx5_mkc_bits memory_key_mkey_entry;

    uint8_t    reserved_at_280[0x80];

    uint8_t    translations_octword_actual_size[0x20];

    uint8_t    mkey_umem_id[0x20];

    uint8_t    mkey_umem_offset[0x40];

    uint8_t    reserved_at_380[0x500];

    uint8_t    klm_pas_mtt[0][0x20];
};

struct uct_ib_mlx5_klm_bits {
    uint8_t    byte_count[0x20];

    uint8_t    mkey[0x20];

    uint8_t    address[0x40];
};

struct uct_ib_mlx5_create_mkey_out_bits {
    uint8_t    status[0x8];
    uint8_t    reserved_at_8[0x18];

    uint8_t    syndrome[0x20];

    uint8_t    reserved_at_40[0x8];
    uint8_t    mkey_index[0x18];

    uint8_t    reserved_at_60[0x20];
};

struct uct_ib_mlx5_set_xrq_dc_params_entry_out_bits {
    uint8_t         status[0x8];
    uint8_t         reserved_at_8[0x18];

    uint8_t         syndrome[0x20];

    uint8_t         reserved_at_40[0x40];
};

struct uct_ib_mlx5_set_xrq_dc_params_entry_in_bits {
    uint8_t         opcode[0x10];
    uint8_t         reserved_at_10[0x10];

    uint8_t         reserved_at_20[0x10];
    uint8_t         op_mod[0x10];

    uint8_t         reserved_at_40[0x8];
    uint8_t         xrqn[0x18];

    uint8_t         reserved_at_60[0x20];

    uint8_t         reserved_at_80[0x3];
    uint8_t         ack_timeout[0x5];
    uint8_t         reserved_at_88[0x4];
    uint8_t         multi_path[0x1];
    uint8_t         mtu[0x3];
    uint8_t         pkey_table_index[0x10];

    uint8_t         reserved_at_a0[0xc];
    uint8_t         cnak_reverse_sl[0x4];
    uint8_t         reserved_at_b0[0x4];
    uint8_t         reverse_sl[0x4];
    uint8_t         reserved_at_b8[0x4];
    uint8_t         sl[0x4];

    uint8_t         dc_access_key[0x40];

    uint8_t         reserved_at_100[0x80];
};

enum {
    UCT_IB_MLX5_DCTC_STATE_ACTIVE    = 0x0,
    UCT_IB_MLX5_DCTC_STATE_DRAINING  = 0x1,
    UCT_IB_MLX5_DCTC_STATE_DRAINED   = 0x2
};

enum {
    UCT_IB_MLX5_DCTC_CS_RES_DISABLE    = 0x0,
    UCT_IB_MLX5_DCTC_CS_RES_NA         = 0x1,
    UCT_IB_MLX5_DCTC_CS_RES_UP_TO_64B  = 0x2
};

enum {
    UCT_IB_MLX5_DCTC_MTU_256_BYTES  = 0x1,
    UCT_IB_MLX5_DCTC_MTU_512_BYTES  = 0x2,
    UCT_IB_MLX5_DCTC_MTU_1K_BYTES   = 0x3,
    UCT_IB_MLX5_DCTC_MTU_2K_BYTES   = 0x4,
    UCT_IB_MLX5_DCTC_MTU_4K_BYTES   = 0x5
};

struct uct_ib_mlx5_dctc_bits {
    uint8_t         reserved_at_0[0x4];
    uint8_t         state[0x4];
    uint8_t         reserved_at_8[0x10];
    uint8_t         offload_type[0x4];
    uint8_t         reserved_at_1c[0x4];

    uint8_t         reserved_at_20[0x8];
    uint8_t         user_index[0x18];

    uint8_t         reserved_at_40[0x8];
    uint8_t         cqn[0x18];

    uint8_t         counter_set_id[0x8];
    uint8_t         atomic_mode[0x4];
    uint8_t         rre[0x1];
    uint8_t         rwe[0x1];
    uint8_t         rae[0x1];
    uint8_t         atomic_like_write_en[0x1];
    uint8_t         latency_sensitive[0x1];
    uint8_t         rlky[0x1];
    uint8_t         force_full_handshake[0x1];
    uint8_t         multi_path[0x1];
    uint8_t         reserved_at_74[0xc];

    uint8_t         reserved_at_80[0x8];
    uint8_t         cs_res[0x8];
    uint8_t         reserved_at_90[0x3];
    uint8_t         min_rnr_nak[0x5];
    uint8_t         reserved_at_98[0x8];

    uint8_t         reserved_at_a0[0x8];
    uint8_t         srqn_xrqn[0x18];

    uint8_t         reserved_at_c0[0x8];
    uint8_t         pd[0x18];

    uint8_t         tclass[0x8];
    uint8_t         reserved_at_e8[0x4];
    uint8_t         flow_label[0x14];

    uint8_t         dc_access_key[0x40];

    uint8_t         reserved_at_140[0x5];
    uint8_t         mtu[0x3];
    uint8_t         port[0x8];
    uint8_t         pkey_index[0x10];

    uint8_t         reserved_at_160[0x8];
    uint8_t         my_addr_index[0x8];
    uint8_t         reserved_at_170[0x8];
    uint8_t         hop_limit[0x8];

    uint8_t         dc_access_key_violation_count[0x20];

    uint8_t         reserved_at_1a0[0x14];
    uint8_t         dei_cfi[0x1];
    uint8_t         eth_prio[0x3];
    uint8_t         ecn[0x2];
    uint8_t         dscp[0x6];

    uint8_t         reserved_at_1c0[0x40];
};

struct uct_ib_mlx5_create_dct_out_bits {
    uint8_t         status[0x8];
    uint8_t         reserved_at_8[0x18];

    uint8_t         syndrome[0x20];

    uint8_t         reserved_at_40[0x8];
    uint8_t         dctn[0x18];

    uint8_t         reserved_at_60[0x20];
};

struct uct_ib_mlx5_create_dct_in_bits {
    uint8_t         opcode[0x10];
    uint8_t         uid[0x10];

    uint8_t         reserved_at_20[0x10];
    uint8_t         op_mod[0x10];

    uint8_t         reserved_at_40[0x40];

    struct uct_ib_mlx5_dctc_bits dct_context_entry;

    uint8_t         reserved_at_280[0x180];
};

struct uct_ib_mlx5_drain_dct_out_bits {
    uint8_t         status[0x8];
    uint8_t         reserved_at_8[0x18];

    uint8_t         syndrome[0x20];

    uint8_t         reserved_at_40[0x40];
};

struct uct_ib_mlx5_drain_dct_in_bits {
    uint8_t         opcode[0x10];
    uint8_t         uid[0x10];

    uint8_t         reserved_at_20[0x10];
    uint8_t         op_mod[0x10];

    uint8_t         reserved_at_40[0x8];
    uint8_t         dctn[0x18];

    uint8_t         reserved_at_60[0x20];
};

struct uct_ib_mlx5_cmd_pas_bits {
    uint8_t         pa_h[0x20];

    uint8_t         pa_l[0x14];
    uint8_t         reserved_at_34[0xc];
};

enum {
    UCT_IB_MLX5_WQ_WQ_TYPE_WQ_LINKED_LIST  = 0x0,
    UCT_IB_MLX5_WQ_WQ_TYPE_WQ_CYCLIC       = 0x1
};

enum {
    UCT_IB_MLX5_WQ_END_PADDING_MODE_END_PAD_NONE   = 0x0,
    UCT_IB_MLX5_WQ_END_PADDING_MODE_END_PAD_ALIGN  = 0x1
};

struct uct_ib_mlx5_wq_bits {
    uint8_t         wq_type[0x4];
    uint8_t         wq_signature[0x1];
    uint8_t         end_padding_mode[0x2];
    uint8_t         cd_slave[0x1];
    uint8_t         reserved_at_8[0x18];

    uint8_t         hds_skip_first_sge[0x1];
    uint8_t         log2_hds_buf_size[0x3];
    uint8_t         reserved_at_24[0x7];
    uint8_t         page_offset[0x5];
    uint8_t         lwm[0x10];

    uint8_t         reserved_at_40[0x8];
    uint8_t         pd[0x18];

    uint8_t         reserved_at_60[0x8];
    uint8_t         uar_page[0x18];

    uint8_t         dbr_addr[0x40];

    uint8_t         hw_counter[0x20];

    uint8_t         sw_counter[0x20];

    uint8_t         reserved_at_100[0xc];
    uint8_t         log_wq_stride[0x4];
    uint8_t         reserved_at_110[0x3];
    uint8_t         log_wq_pg_sz[0x5];
    uint8_t         reserved_at_118[0x3];
    uint8_t         log_wq_sz[0x5];

    uint8_t         dbr_umem_valid[0x1];
    uint8_t         wq_umem_valid[0x1];
    uint8_t         reserved_at_122[0x1];
    uint8_t         log_hairpin_num_packets[0x5];
    uint8_t         reserved_at_128[0x3];
    uint8_t         log_hairpin_data_sz[0x5];
    uint8_t         reserved_at_130[0x4];
    uint8_t         log_wqe_num_of_strides[0x4];
    uint8_t         two_byte_shift_en[0x1];
    uint8_t         reserved_at_139[0x4];
    uint8_t         log_wqe_stride_size[0x3];

    uint8_t         dbr_umem_id[0x20];

    uint8_t         wq_umem_id[0x20];

    uint8_t         reserved_at_180[0x480];

    struct uct_ib_mlx5_cmd_pas_bits pas[0];
};

enum {
    UCT_IB_MLX5_XRQC_STATE_GOOD   = 0x0,
    UCT_IB_MLX5_XRQC_STATE_ERROR  = 0x1
};

enum {
    UCT_IB_MLX5_XRQC_TOPOLOGY_NO_SPECIAL_TOPOLOGY = 0x0,
    UCT_IB_MLX5_XRQC_TOPOLOGY_TAG_MATCHING        = 0x1
};

enum {
    UCT_IB_MLX5_XRQC_OFFLOAD_RNDV = 0x1
};

struct uct_ib_mlx5_tag_matching_topology_context_bits {
    uint8_t         log_matching_list_sz[0x4];
    uint8_t         reserved_at_4[0xc];
    uint8_t         append_next_index[0x10];

    uint8_t         sw_phase_cnt[0x10];
    uint8_t         hw_phase_cnt[0x10];

    uint8_t         reserved_at_40[0x40];
};

struct uct_ib_mlx5_xrqc_bits {
    uint8_t         state[0x4];
    uint8_t         rlkey[0x1];
    uint8_t         reserved_at_5[0xf];
    uint8_t         topology[0x4];
    uint8_t         reserved_at_18[0x4];
    uint8_t         offload[0x4];

    uint8_t         reserved_at_20[0x8];
    uint8_t         user_index[0x18];

    uint8_t         reserved_at_40[0x8];
    uint8_t         cqn[0x18];

    uint8_t         reserved_at_60[0x1f];
    uint8_t         dc[0x1];

    uint8_t         reserved_at_80[0x80];

    struct uct_ib_mlx5_tag_matching_topology_context_bits tag_matching_topology_context;

    uint8_t         reserved_at_180[0x280];

    struct uct_ib_mlx5_wq_bits wq;
};

struct uct_ib_mlx5_create_xrq_out_bits {
    uint8_t         status[0x8];
    uint8_t         reserved_at_8[0x18];

    uint8_t         syndrome[0x20];

    uint8_t         reserved_at_40[0x8];
    uint8_t         xrqn[0x18];

    uint8_t         reserved_at_60[0x20];
};

struct uct_ib_mlx5_create_xrq_in_bits {
    uint8_t         opcode[0x10];
    uint8_t         uid[0x10];

    uint8_t         reserved_at_20[0x10];
    uint8_t         op_mod[0x10];

    uint8_t         reserved_at_40[0x40];

    struct uct_ib_mlx5_xrqc_bits xrq_context;
};

enum {
    UCT_IB_MLX5_RMPC_STATE_RDY = 0x1,
    UCT_IB_MLX5_RMPC_STATE_ERR = 0x3
};

struct uct_ib_mlx5_rmpc_bits {
    uint8_t         reserved_at_0[0x8];
    uint8_t         state[0x4];
    uint8_t         reserved_at_c[0x14];

    uint8_t         basic_cyclic_rcv_wqe[0x1];
    uint8_t         reserved_at_21[0x1f];

    uint8_t         reserved_at_40[0x140];

    struct uct_ib_mlx5_wq_bits wq;
};

struct uct_ib_mlx5_create_rmp_out_bits {
    uint8_t         status[0x8];
    uint8_t         reserved_at_8[0x18];

    uint8_t         syndrome[0x20];

    uint8_t         reserved_at_40[0x8];
    uint8_t         rmpn[0x18];

    uint8_t         reserved_at_60[0x20];
};

struct uct_ib_mlx5_create_rmp_in_bits {
    uint8_t         opcode[0x10];
    uint8_t         uid[0x10];

    uint8_t         reserved_at_20[0x10];
    uint8_t         op_mod[0x10];

    uint8_t         reserved_at_40[0xc0];

    struct uct_ib_mlx5_rmpc_bits rmp_context;
};

enum {
    UCT_IB_MLX5_ADS_STAT_RATE_NO_LIMIT  = 0x0,
    UCT_IB_MLX5_ADS_STAT_RATE_2_5GBPS   = 0x7,
    UCT_IB_MLX5_ADS_STAT_RATE_10GBPS    = 0x8,
    UCT_IB_MLX5_ADS_STAT_RATE_30GBPS    = 0x9,
    UCT_IB_MLX5_ADS_STAT_RATE_5GBPS     = 0xa,
    UCT_IB_MLX5_ADS_STAT_RATE_20GBPS    = 0xb,
    UCT_IB_MLX5_ADS_STAT_RATE_40GBPS    = 0xc,
    UCT_IB_MLX5_ADS_STAT_RATE_60GBPS    = 0xd,
    UCT_IB_MLX5_ADS_STAT_RATE_80GBPS    = 0xe,
    UCT_IB_MLX5_ADS_STAT_RATE_120GBPS   = 0xf
};

struct uct_ib_mlx5_ads_bits {
    uint8_t         fl[0x1];
    uint8_t         free_ar[0x1];
    uint8_t         reserved_at_2[0xe];
    uint8_t         pkey_index[0x10];

    uint8_t         reserved_at_20[0x8];
    uint8_t         grh[0x1];
    uint8_t         mlid[0x7];
    uint8_t         rlid[0x10];

    uint8_t         ack_timeout[0x5];
    uint8_t         reserved_at_45[0x3];
    uint8_t         src_addr_index[0x8];
    uint8_t         log_rtm[0x4];
    uint8_t         stat_rate[0x4];
    uint8_t         hop_limit[0x8];

    uint8_t         reserved_at_60[0x4];
    uint8_t         tclass[0x8];
    uint8_t         flow_label[0x14];

    uint8_t         rgid_rip[16][0x8];

    uint8_t         reserved_at_100[0x4];
    uint8_t         f_dscp[0x1];
    uint8_t         f_ecn[0x1];
    uint8_t         reserved_at_106[0x1];
    uint8_t         f_eth_prio[0x1];
    uint8_t         ecn[0x2];
    uint8_t         dscp[0x6];
    uint8_t         udp_sport[0x10];

    uint8_t         dei_cfi[0x1];
    uint8_t         eth_prio[0x3];
    uint8_t         sl[0x4];
    uint8_t         vhca_port_num[0x8];
    uint8_t         rmac_47_32[0x10];

    uint8_t         rmac_31_0[0x20];
};

enum {
    UCT_IB_MLX5_QPC_STATE_RST        = 0x0,
    UCT_IB_MLX5_QPC_STATE_INIT       = 0x1,
    UCT_IB_MLX5_QPC_STATE_RTR        = 0x2,
    UCT_IB_MLX5_QPC_STATE_RTS        = 0x3,
    UCT_IB_MLX5_QPC_STATE_SQER       = 0x4,
    UCT_IB_MLX5_QPC_STATE_ERR        = 0x6,
    UCT_IB_MLX5_QPC_STATE_SQD        = 0x7,
    UCT_IB_MLX5_QPC_STATE_SUSPENDED  = 0x9
};

enum {
    UCT_IB_MLX5_QPC_ST_RC            = 0x0,
    UCT_IB_MLX5_QPC_ST_UC            = 0x1,
    UCT_IB_MLX5_QPC_ST_UD            = 0x2,
    UCT_IB_MLX5_QPC_ST_XRC           = 0x3,
    UCT_IB_MLX5_QPC_ST_DCI           = 0x5,
    UCT_IB_MLX5_QPC_ST_QP0           = 0x7,
    UCT_IB_MLX5_QPC_ST_QP1           = 0x8,
    UCT_IB_MLX5_QPC_ST_RAW_DATAGRAM  = 0x9,
    UCT_IB_MLX5_QPC_ST_REG_UMR       = 0xc
};

enum {
    UCT_IB_MLX5_QPC_PM_STATE_ARMED     = 0x0,
    UCT_IB_MLX5_QPC_PM_STATE_REARM     = 0x1,
    UCT_IB_MLX5_QPC_PM_STATE_RESERVED  = 0x2,
    UCT_IB_MLX5_QPC_PM_STATE_MIGRATED  = 0x3
};

enum {
    UCT_IB_MLX5_QPC_OFFLOAD_TYPE_RNDV  = 0x1
};

enum {
    UCT_IB_MLX5_QPC_END_PADDING_MODE_SCATTER_AS_IS                = 0x0,
    UCT_IB_MLX5_QPC_END_PADDING_MODE_PAD_TO_CACHE_LINE_ALIGNMENT  = 0x1
};

enum {
    UCT_IB_MLX5_QPC_MTU_256_BYTES        = 0x1,
    UCT_IB_MLX5_QPC_MTU_512_BYTES        = 0x2,
    UCT_IB_MLX5_QPC_MTU_1K_BYTES         = 0x3,
    UCT_IB_MLX5_QPC_MTU_2K_BYTES         = 0x4,
    UCT_IB_MLX5_QPC_MTU_4K_BYTES         = 0x5,
    UCT_IB_MLX5_QPC_MTU_RAW_ETHERNET_QP  = 0x7
};

enum {
    UCT_IB_MLX5_QPC_ATOMIC_MODE_IB_SPEC     = 0x1,
    UCT_IB_MLX5_QPC_ATOMIC_MODE_ONLY_8B     = 0x2,
    UCT_IB_MLX5_QPC_ATOMIC_MODE_UP_TO_8B    = 0x3,
    UCT_IB_MLX5_QPC_ATOMIC_MODE_UP_TO_16B   = 0x4,
    UCT_IB_MLX5_QPC_ATOMIC_MODE_UP_TO_32B   = 0x5,
    UCT_IB_MLX5_QPC_ATOMIC_MODE_UP_TO_64B   = 0x6,
    UCT_IB_MLX5_QPC_ATOMIC_MODE_UP_TO_128B  = 0x7,
    UCT_IB_MLX5_QPC_ATOMIC_MODE_UP_TO_256B  = 0x8
};

enum {
    UCT_IB_MLX5_QPC_CS_REQ_DISABLE    = 0x0,
    UCT_IB_MLX5_QPC_CS_REQ_UP_TO_32B  = 0x11,
    UCT_IB_MLX5_QPC_CS_REQ_UP_TO_64B  = 0x22
};

static inline unsigned uct_ib_mlx5_qpc_cs_req(unsigned size)
{
    return (size > 32) ? UCT_IB_MLX5_QPC_CS_REQ_UP_TO_64B :
                  size ? UCT_IB_MLX5_QPC_CS_REQ_UP_TO_32B :
                         UCT_IB_MLX5_QPC_CS_REQ_DISABLE;
}

enum {
    UCT_IB_MLX5_QPC_CS_RES_DISABLE    = 0x0,
    UCT_IB_MLX5_QPC_CS_RES_UP_TO_32B  = 0x1,
    UCT_IB_MLX5_QPC_CS_RES_UP_TO_64B  = 0x2
};

enum {
    UCT_IB_MLX5_QP_OPTPAR_RRE        = 1 << 1,
    UCT_IB_MLX5_QP_OPTPAR_RAE        = 1 << 2,
    UCT_IB_MLX5_QP_OPTPAR_RWE        = 1 << 3,
    UCT_IB_MLX5_QP_OPTPAR_LAG_TX_AFF = 1 << 15
};

static inline unsigned uct_ib_mlx5_qpc_cs_res(unsigned size, int dc)
{
    return (size > 32) ? UCT_IB_MLX5_QPC_CS_RES_UP_TO_64B :
         (size && !dc) ? UCT_IB_MLX5_QPC_CS_RES_UP_TO_32B :
                         UCT_IB_MLX5_QPC_CS_RES_DISABLE;
}

struct uct_ib_mlx5_qpc_bits {
    uint8_t         state[0x4];
    uint8_t         lag_tx_port_affinity[0x4];
    uint8_t         st[0x8];
    uint8_t         reserved_at_10[0x3];
    uint8_t         pm_state[0x2];
    uint8_t         rdma_wr_disabled[0x1];
    uint8_t         req_e2e_credit_mode[0x2];
    uint8_t         offload_type[0x4];
    uint8_t         end_padding_mode[0x2];
    uint8_t         reserved_at_1e[0x2];

    uint8_t         wq_signature[0x1];
    uint8_t         block_lb_mc[0x1];
    uint8_t         atomic_like_write_en[0x1];
    uint8_t         latency_sensitive[0x1];
    uint8_t         reserved_at_24[0x1];
    uint8_t         drain_sigerr[0x1];
    uint8_t         reserved_at_26[0x2];
    uint8_t         pd[0x18];

    uint8_t         mtu[0x3];
    uint8_t         log_msg_max[0x5];
    uint8_t         reserved_at_48[0x1];
    uint8_t         log_rq_size[0x4];
    uint8_t         log_rq_stride[0x3];
    uint8_t         no_sq[0x1];
    uint8_t         log_sq_size[0x4];
    uint8_t         reserved_at_55[0x6];
    uint8_t         rlky[0x1];
    uint8_t         ulp_stateless_offload_mode[0x4];

    uint8_t         counter_set_id[0x8];
    uint8_t         uar_page[0x18];

    uint8_t         reserved_at_80[0x3];
    uint8_t         full_handshake[0x1];
    uint8_t         cnak_reverse_sl[0x4];
    uint8_t         user_index[0x18];

    uint8_t         reserved_at_a0[0x3];
    uint8_t         log_page_size[0x5];
    uint8_t         remote_qpn[0x18];

    struct uct_ib_mlx5_ads_bits primary_address_path;

    struct uct_ib_mlx5_ads_bits secondary_address_path;

    uint8_t         log_ack_req_freq[0x4];
    uint8_t         reserved_at_384[0x4];
    uint8_t         log_sra_max[0x3];
    uint8_t         reserved_at_38b[0x2];
    uint8_t         retry_count[0x3];
    uint8_t         rnr_retry[0x3];
    uint8_t         reserved_at_393[0x1];
    uint8_t         fre[0x1];
    uint8_t         cur_rnr_retry[0x3];
    uint8_t         cur_retry_count[0x3];
    uint8_t         reserved_at_39b[0x5];

    uint8_t         reserved_at_3a0[0x20];

    uint8_t         reserved_at_3c0[0x8];
    uint8_t         next_send_psn[0x18];

    uint8_t         reserved_at_3e0[0x8];
    uint8_t         cqn_snd[0x18];

    uint8_t         reserved_at_400[0x8];
    uint8_t         deth_sqpn[0x18];

    uint8_t         reserved_at_420[0x20];

    uint8_t         reserved_at_440[0x8];
    uint8_t         last_acked_psn[0x18];

    uint8_t         reserved_at_460[0x8];
    uint8_t         ssn[0x18];

    uint8_t         reserved_at_480[0x8];
    uint8_t         log_rra_max[0x3];
    uint8_t         reserved_at_48b[0x1];
    uint8_t         atomic_mode[0x4];
    uint8_t         rre[0x1];
    uint8_t         rwe[0x1];
    uint8_t         rae[0x1];
    uint8_t         reserved_at_493[0x1];
    uint8_t         page_offset[0x6];
    uint8_t         reserved_at_49a[0x3];
    uint8_t         cd_slave_receive[0x1];
    uint8_t         cd_slave_send[0x1];
    uint8_t         cd_master[0x1];

    uint8_t         reserved_at_4a0[0x3];
    uint8_t         min_rnr_nak[0x5];
    uint8_t         next_rcv_psn[0x18];

    uint8_t         reserved_at_4c0[0x8];
    uint8_t         xrcd[0x18];

    uint8_t         reserved_at_4e0[0x8];
    uint8_t         cqn_rcv[0x18];

    uint8_t         dbr_addr[0x40];

    uint8_t         q_key[0x20];

    uint8_t         reserved_at_560[0x5];
    uint8_t         rq_type[0x3];
    uint8_t         srqn_rmpn_xrqn[0x18];

    uint8_t         reserved_at_580[0x8];
    uint8_t         rmsn[0x18];

    uint8_t         hw_sq_wqebb_counter[0x10];
    uint8_t         sw_sq_wqebb_counter[0x10];

    uint8_t         hw_rq_counter[0x20];

    uint8_t         sw_rq_counter[0x20];

    uint8_t         reserved_at_600[0x20];

    uint8_t         reserved_at_620[0xf];
    uint8_t         cgs[0x1];
    uint8_t         cs_req[0x8];
    uint8_t         cs_res[0x8];

    uint8_t         dc_access_key[0x40];

    uint8_t         reserved_at_680[0x3];
    uint8_t         dbr_umem_valid[0x1];
    uint8_t         reserved_at_684[0x1c];

    uint8_t         reserved_at_6a0[0x80];

    uint8_t         dbr_umem_id[0x20];
};

struct uct_ib_mlx5_create_qp_out_bits {
    uint8_t         status[0x8];
    uint8_t         reserved_at_8[0x18];

    uint8_t         syndrome[0x20];

    uint8_t         reserved_at_40[0x8];
    uint8_t         qpn[0x18];

    uint8_t         reserved_at_60[0x20];
};

struct uct_ib_mlx5_create_qp_in_bits {
    uint8_t         opcode[0x10];
    uint8_t         uid[0x10];

    uint8_t         reserved_at_20[0x10];
    uint8_t         op_mod[0x10];

    uint8_t         reserved_at_40[0x40];

    uint8_t         opt_param_mask[0x20];

    uint8_t         reserved_at_a0[0x20];

    struct uct_ib_mlx5_qpc_bits qpc;

    uint8_t         reserved_at_800[0x40];

    uint8_t         wq_umem_id[0x20];

    uint8_t         wq_umem_valid[0x1];
    uint8_t         reserved_at_861[0x1f];

    uint8_t         pas[0][0x40];
};

struct uct_ib_mlx5_init2rtr_qp_out_bits {
    uint8_t         status[0x8];
    uint8_t         reserved_at_8[0x18];

    uint8_t         syndrome[0x20];

    uint8_t         reserved_at_40[0x40];
};

struct uct_ib_mlx5_init2rtr_qp_in_bits {
    uint8_t         opcode[0x10];
    uint8_t         uid[0x10];

    uint8_t         reserved_at_20[0x10];
    uint8_t         op_mod[0x10];

    uint8_t         reserved_at_40[0x8];
    uint8_t         qpn[0x18];

    uint8_t         reserved_at_60[0x20];

    uint8_t         opt_param_mask[0x20];

    uint8_t         reserved_at_a0[0x20];

    struct uct_ib_mlx5_qpc_bits qpc;

    uint8_t         reserved_at_800[0x80];
};

struct uct_ib_mlx5_rtr2rts_qp_out_bits {
    uint8_t         status[0x8];
    uint8_t         reserved_at_8[0x18];

    uint8_t         syndrome[0x20];

    uint8_t         reserved_at_40[0x40];
};

struct uct_ib_mlx5_rtr2rts_qp_in_bits {
    uint8_t         opcode[0x10];
    uint8_t         uid[0x10];

    uint8_t         reserved_at_20[0x10];
    uint8_t         op_mod[0x10];

    uint8_t         reserved_at_40[0x8];
    uint8_t         qpn[0x18];

    uint8_t         reserved_at_60[0x20];

    uint8_t         opt_param_mask[0x20];

    uint8_t         reserved_at_a0[0x20];

    struct uct_ib_mlx5_qpc_bits qpc;

    uint8_t         reserved_at_800[0x80];
};

struct uct_ib_mlx5_rst2init_qp_out_bits {
    uint8_t         status[0x8];
    uint8_t         reserved_at_8[0x18];

    uint8_t         syndrome[0x20];

    uint8_t         reserved_at_40[0x40];
};

struct uct_ib_mlx5_rst2init_qp_in_bits {
    uint8_t         opcode[0x10];
    uint8_t         uid[0x10];

    uint8_t         reserved_at_20[0x10];
    uint8_t         op_mod[0x10];

    uint8_t         reserved_at_40[0x8];
    uint8_t         qpn[0x18];

    uint8_t         reserved_at_60[0x20];

    uint8_t         opt_param_mask[0x20];

    uint8_t         reserved_at_a0[0x20];

    struct uct_ib_mlx5_qpc_bits qpc;

    uint8_t         reserved_at_800[0x80];
};

struct uct_ib_mlx5_modify_qp_out_bits {
    uint8_t         status[0x8];
    uint8_t         reserved_at_8[0x18];

    uint8_t         syndrome[0x20];

    uint8_t         reserved_at_40[0x40];
};

struct uct_ib_mlx5_modify_qp_in_bits {
    uint8_t         opcode[0x10];
    uint8_t         uid[0x10];

    uint8_t         reserved_at_20[0x10];
    uint8_t         op_mod[0x10];

    uint8_t         reserved_at_40[0x8];
    uint8_t         qpn[0x18];

    uint8_t         reserved_at_60[0x20];
};

struct uct_ib_mlx5_query_qp_out_bits {
    uint8_t         status[0x8];
    uint8_t         reserved_at_8[0x18];

    uint8_t         syndrome[0x20];

    uint8_t         reserved_at_40[0x40];

    uint8_t         opt_param_mask[0x20];

    uint8_t         reserved_at_a0[0x20];

    struct uct_ib_mlx5_qpc_bits qpc;

    uint8_t         reserved_at_800[0x80];

    uint8_t         pas[0][0x40];
};

struct uct_ib_mlx5_query_qp_in_bits {
    uint8_t         opcode[0x10];
    uint8_t         reserved_at_10[0x10];

    uint8_t         reserved_at_20[0x10];
    uint8_t         op_mod[0x10];

    uint8_t         reserved_at_40[0x8];
    uint8_t         qpn[0x18];

    uint8_t         reserved_at_60[0x20];
};

enum {
    UCT_IB_MLX5_EVENT_TYPE_SRQ_LAST_WQE       = 0x13
};

struct uct_ib_mlx5_general_obj_out_cmd_hdr_bits {
    uint8_t         status[0x8];
    uint8_t         reserved_at_8[0x18];

    uint8_t         syndrome[0x20];

    uint8_t         obj_id[0x20];

    uint8_t         reserved_at_60[0x20];
};

struct uct_ib_mlx5_general_obj_in_cmd_hdr_bits {
    uint8_t         opcode[0x10];
    uint8_t         uid[0x10];

    uint8_t         reserved_at_20[0x10];
    uint8_t         obj_type[0x10];

    uint8_t         obj_id[0x20];

    uint8_t         reserved_at_60[0x3];
    uint8_t         log_obj_range[0x5];
    uint8_t         reserved_at_68[0x18];
};

struct uct_ib_mlx5_reserved_qpn_bits {
    uint8_t         reserved_at_0[0x80];
};

struct uct_ib_mlx5_create_reserved_qpn_in_bits {
    struct uct_ib_mlx5_general_obj_in_cmd_hdr_bits  hdr;
    struct uct_ib_mlx5_modify_qp_in_bits            qpns;
};

enum {
    UCT_IB_MLX5_OBJ_TYPE_RESERVED_QPN = 0x002C,
};

#endif
