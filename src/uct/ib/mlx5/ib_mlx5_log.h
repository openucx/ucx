/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_IB_MLX5_LOG_H_
#define UCT_IB_MLX5_LOG_H_

#include "ib_mlx5.h"

#include <uct/tl/tl_log.h>


void uct_ib_mlx5_completion_with_err(struct mlx5_err_cqe *ecqe);


void __uct_ib_mlx5_log_tx(const char *file, int line, const char *function,
                          enum ibv_qp_type qp_type, void *wqe, void *qstart,
                          void *qend, uct_log_data_dump_func_t packet_dump_cb);

void __uct_ib_mlx5_log_rx(const char *file, int line, const char *function,
                          enum ibv_qp_type qp_type, struct mlx5_cqe64 *cqe, void *data,
                          uct_log_data_dump_func_t packet_dump_cb);

#define uct_ib_mlx5_log_tx(_qpt, _wqe, _qstart, _qend, _dump_cb) \
    if (ucs_log_enabled(UCS_LOG_LEVEL_TRACE_DATA)) { \
        __uct_ib_mlx5_log_tx(__FILE__, __LINE__, __FUNCTION__, \
                             _qpt, _wqe, _qstart, _qend, _dump_cb); \
    }

#define uct_ib_mlx5_log_rx(_qpt, _cqe, _data, _dump_cb) \
    if (ucs_log_enabled(UCS_LOG_LEVEL_TRACE_DATA)) { \
        __uct_ib_mlx5_log_rx(__FILE__, __LINE__, __FUNCTION__, \
                             _qpt, _cqe, _data, _dump_cb); \
    }

#endif
