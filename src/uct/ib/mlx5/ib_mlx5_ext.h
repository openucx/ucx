#ifndef UCT_IB_MLX5_EXT_H_
#define UCT_IB_MLX5_EXT_H_

#include <stdint.h>
#include <ucs/debug/assert.h>
#include <ucs/type/status.h>
#include <ucs/sys/stubs.h>


struct ibv_qp;
struct mlx5dv_devx_obj;

/**
 * @brief Field mask bits for @ref uct_ib_mlx5_ext_qp_query_attr_t.
 * Selects which output fields to populate.
 */
enum uct_ib_mlx5_ext_qp_query_attr_field {
    UCT_IB_MLX5_EXT_QP_QUERY_ATTR_FIELD_TX_TOKEN_LEN = UCS_BIT(0),
    UCT_IB_MLX5_EXT_QP_QUERY_ATTR_FIELD_RX_TOKEN_LEN = UCS_BIT(1),
    UCT_IB_MLX5_EXT_QP_QUERY_ATTR_FIELD_TX_TOKEN     = UCS_BIT(2),
    UCT_IB_MLX5_EXT_QP_QUERY_ATTR_FIELD_RX_TOKEN     = UCS_BIT(3),
    UCT_IB_MLX5_EXT_QP_QUERY_ATTR_FIELD_QP_NUM       = UCS_BIT(4)
};

/**
 * @brief QP query output attributes.
 */
typedef struct uct_ib_mlx5_ext_qp_query_attr {
    uint64_t field_mask;
    size_t   tx_token_len;
    size_t   rx_token_len;
    void     *tx_token;
    void     *rx_token;
    uint32_t qp_num;
} uct_ib_mlx5_ext_qp_query_attr_t;


typedef ucs_status_t (*uct_ib_mlx5_ext_iface_flags_func_t)(uint64_t *flags);
typedef ucs_status_t (*uct_ib_mlx5_ext_qp_query_func_t)(
        struct ibv_qp *qp, struct mlx5dv_devx_obj *devx_obj,
        uct_ib_plugin_qp_query_attr_t *attr);

typedef struct uct_ib_mlx5_ext_ops {
    uct_ib_mlx5_ext_iface_flags_func_t iface_flags;
    uct_ib_mlx5_ext_qp_query_func_t    qp_query;
} uct_ib_mlx5_ext_ops_t;

extern uct_ib_mlx5_ext_ops_t uct_ib_mlx5_ext_ops;

extern void uct_ib_mlx5_ext_register(const uct_ib_mlx5_ext_ops_t *ops);

#endif /* UCT_IB_MLX5_EXT_H_ */
