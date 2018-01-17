/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_IB_MLX5_LOG_H_
#define UCT_IB_MLX5_LOG_H_

#include "ib_mlx5.h"

#include <uct/base/uct_log.h>


ucs_status_t uct_ib_mlx5_completion_with_err(struct mlx5_err_cqe *ecqe,
                                             ucs_log_level_t log_level);


void __uct_ib_mlx5_log_tx(const char *file, int line, const char *function,
                          uct_ib_iface_t *iface, enum ibv_qp_type qp_type,
                          void *wqe, void *qstart, void *qend,
                          uct_log_data_dump_func_t packet_dump_cb);

void __uct_ib_mlx5_log_rx(const char *file, int line, const char *function,
                          uct_ib_iface_t *iface, enum ibv_qp_type qp_type,
                          struct mlx5_cqe64 *cqe, void *data,
                          uct_log_data_dump_func_t packet_dump_cb);

void uct_ib_mlx5_cqe_dump(const char *file, int line, const char *function,
                          struct mlx5_cqe64 *cqe);

#define uct_ib_mlx5_log_tx(_iface, _qpt, _wqe, _qstart, _qend, _dump_cb) \
    if (ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_DATA)) { \
        __uct_ib_mlx5_log_tx(__FILE__, __LINE__, __FUNCTION__, \
                             _iface, _qpt, _wqe, _qstart, _qend, _dump_cb); \
    }

#define uct_ib_mlx5_log_rx(_iface, _qpt, _cqe, _data, _dump_cb) \
    if (ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_DATA)) { \
        __uct_ib_mlx5_log_rx(__FILE__, __LINE__, __FUNCTION__, \
                             _iface, _qpt, _cqe, _data, _dump_cb); \
    }

#define uct_ib_mlx5_log_cqe(_cqe) \
    if (ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_DATA)) { \
        uct_ib_mlx5_cqe_dump(__FILE__, __LINE__, __FUNCTION__, \
                             cqe); \
    }

#endif
