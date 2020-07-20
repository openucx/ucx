/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_IB_MLX5_LOG_H_
#define UCT_IB_MLX5_LOG_H_

#include "ib_mlx5.h"

#include <uct/base/uct_log.h>


typedef struct uct_ib_log_sge {
    int            num_sge;
    uint64_t       inline_bitmap;
    struct ibv_sge sg_list[2];
} uct_ib_log_sge_t;

ucs_status_t uct_ib_mlx5_completion_with_err(uct_ib_iface_t *iface,
                                             uct_ib_mlx5_err_cqe_t *ecqe,
                                             uct_ib_mlx5_txwq_t *txwq,
                                             ucs_log_level_t log_level);


void __uct_ib_mlx5_log_tx(const char *file, int line, const char *function,
                          uct_ib_iface_t *iface, void *wqe, void *qstart,
                          void *qend, int max_sge, uct_ib_log_sge_t *log_sge,
                          uct_log_data_dump_func_t packet_dump_cb);

void __uct_ib_mlx5_log_rx(const char *file, int line, const char *function,
                          uct_ib_iface_t *iface, struct mlx5_cqe64 *cqe,
                          void *data, uct_log_data_dump_func_t packet_dump_cb);

void uct_ib_mlx5_cqe_dump(const char *file, int line, const char *function,
                          struct mlx5_cqe64 *cqe);

void uct_ib_mlx5_av_dump(char *buf, size_t max,
                         const uct_ib_mlx5_base_av_t *base_av,
                         const struct mlx5_grh_av *grh_av, int is_eth);

#define uct_ib_mlx5_log_tx(_iface, _wqe, _qstart, _qend, _max_sge, _log_sge, _dump_cb) \
    if (ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_DATA)) { \
        __uct_ib_mlx5_log_tx(__FILE__, __LINE__, __FUNCTION__, \
                             _iface, _wqe, _qstart, _qend, _max_sge, _log_sge, _dump_cb); \
    }

#define uct_ib_mlx5_log_rx(_iface, _cqe, _data, _dump_cb) \
    if (ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_DATA)) { \
        __uct_ib_mlx5_log_rx(__FILE__, __LINE__, __FUNCTION__, \
                             _iface, _cqe, _data, _dump_cb); \
    }

#define uct_ib_mlx5_log_cqe(_cqe) \
    if (ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_DATA)) { \
        uct_ib_mlx5_cqe_dump(__FILE__, __LINE__, __FUNCTION__, \
                             cqe); \
    }

#endif
