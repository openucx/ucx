/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ud_mlx5_common.h"


ucs_config_field_t uct_ud_mlx5_iface_common_config_table[] = {
  {"COMPACT_AV", "yes",
   "Enable compact address-vector optimization.",
   ucs_offsetof(uct_ud_mlx5_iface_common_config_t, enable_compact_av), UCS_CONFIG_TYPE_BOOL},

  {NULL}
};

ucs_status_t uct_ud_mlx5_iface_common_init(uct_ib_iface_t *ib_iface,
                                           uct_ud_mlx5_iface_common_t *iface,
                                           uct_ud_mlx5_iface_common_config_t *config)
{
    if (config->enable_compact_av) {
        /* Check that compact AV supported by device */
        return uct_ib_mlx5_get_compact_av(ib_iface, &iface->config.compact_av);
    }

    iface->config.compact_av = 0;
    return UCS_OK;
}

ucs_status_t uct_ud_mlx5_iface_get_av(uct_ib_iface_t *iface,
                                      uct_ud_mlx5_iface_common_t *ud_common_iface,
                                      const uct_ib_address_t *ib_addr,
                                      uint8_t path_bits,
                                      uct_ib_mlx5_base_av_t *base_av,
                                      struct mlx5_grh_av *grh_av,
                                      int *is_global)
{
    ucs_status_t        status;
    struct ibv_ah      *ah;
    struct mlx5_wqe_av  mlx5_av;
    struct ibv_ah_attr  ah_attr;

    uct_ib_iface_fill_ah_attr_from_addr(iface, ib_addr, path_bits, &ah_attr);
    status = uct_ib_iface_create_ah(iface, &ah_attr, &ah);
    if (status != UCS_OK) {
        return status;
    }
    *is_global = ah_attr.is_global;

    uct_ib_mlx5_get_av(ah, &mlx5_av);
    ibv_destroy_ah(ah);

    base_av->stat_rate_sl = mlx5_av_base(&mlx5_av)->stat_rate_sl;
    base_av->fl_mlid      = mlx5_av_base(&mlx5_av)->fl_mlid;
    base_av->rlid         = mlx5_av_base(&mlx5_av)->rlid;

    base_av->dqp_dct = (ud_common_iface->config.compact_av) ? 0 :
                        UCT_IB_MLX5_EXTENDED_UD_AV;

    ucs_assertv_always((UCT_IB_MLX5_AV_FULL_SIZE > UCT_IB_MLX5_AV_BASE_SIZE) ||
                       (base_av->dqp_dct & UCT_IB_MLX5_EXTENDED_UD_AV),
                       "compact address vector not supported, and EXTENDED_AV flag is missing");

    if (*is_global) {
        ucs_assert_always(grh_av != NULL);
        memcpy(grh_av, mlx5_av_grh(&mlx5_av), sizeof(*grh_av));
    }
    return UCS_OK;
}

