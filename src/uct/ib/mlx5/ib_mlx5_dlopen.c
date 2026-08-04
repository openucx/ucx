
#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <uct/ib/base/ib_dlopen.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include <infiniband/mlx5dv.h>

#define UCT_IB_MLX5_LIB_NAME "libmlx5.so.1"

#define UCT_IB_MLX5_VOID_OPS(_op, _module, _ops) \
    _op(_module, _ops, mlx5dv_devx_free_uar, \
        (struct mlx5dv_devx_uar *devx_uar), (devx_uar)) \
    _op(_module, _ops, mlx5dv_devx_destroy_event_channel, \
        (struct mlx5dv_devx_event_channel *event_channel), (event_channel))

#define UCT_IB_MLX5_FWD_OPS(_op, _module, _ops) \
    _op(_module, _ops, bool, false, mlx5dv_is_supported, \
        (struct ibv_device *device), (device)) \
    _op(_module, _ops, struct ibv_context *, NULL, mlx5dv_open_device, \
        (struct ibv_device *device, struct mlx5dv_context_attr *attr), \
        (device, attr)) \
    _op(_module, _ops, int, -1, mlx5dv_query_device, \
        (struct ibv_context *ctx_in, struct mlx5dv_context *attrs_out), \
        (ctx_in, attrs_out)) \
    _op(_module, _ops, struct ibv_qp *, NULL, mlx5dv_create_qp, \
        (struct ibv_context *context, struct ibv_qp_init_attr_ex *qp_attr, \
         struct mlx5dv_qp_init_attr *mlx5_qp_attr), \
        (context, qp_attr, mlx5_qp_attr)) \
    _op(_module, _ops, struct ibv_cq_ex *, NULL, mlx5dv_create_cq, \
        (struct ibv_context *context, struct ibv_cq_init_attr_ex *cq_attr, \
         struct mlx5dv_cq_init_attr *mlx5_cq_attr), \
        (context, cq_attr, mlx5_cq_attr)) \
    _op(_module, _ops, int, -1, mlx5dv_init_obj, \
        (struct mlx5dv_obj *obj, uint64_t obj_type), (obj, obj_type)) \
    _op(_module, _ops, struct mlx5dv_mkey *, NULL, mlx5dv_create_mkey, \
        (struct mlx5dv_mkey_init_attr *mkey_init_attr), (mkey_init_attr)) \
    _op(_module, _ops, int, -1, mlx5dv_destroy_mkey, \
        (struct mlx5dv_mkey *mkey), (mkey)) \
    _op(_module, _ops, struct ibv_mr *, NULL, mlx5dv_reg_dmabuf_mr, \
        (struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, \
         int fd, int access, int mlx5_access), \
        (pd, offset, length, iova, fd, access, mlx5_access)) \
    _op(_module, _ops, int, -1, mlx5dv_get_data_direct_sysfs_path, \
        (struct ibv_context *context, char *buf, size_t buf_len), \
        (context, buf, buf_len)) \
    _op(_module, _ops, struct mlx5dv_devx_obj *, NULL, \
        mlx5dv_devx_obj_create, \
        (struct ibv_context *context, const void *in, size_t inlen, \
         void *out, size_t outlen), (context, in, inlen, out, outlen)) \
    _op(_module, _ops, int, -1, mlx5dv_devx_obj_destroy, \
        (struct mlx5dv_devx_obj *obj), (obj)) \
    _op(_module, _ops, int, -1, mlx5dv_devx_obj_modify, \
        (struct mlx5dv_devx_obj *obj, const void *in, size_t inlen, \
         void *out, size_t outlen), (obj, in, inlen, out, outlen)) \
    _op(_module, _ops, int, -1, mlx5dv_devx_obj_query, \
        (struct mlx5dv_devx_obj *obj, const void *in, size_t inlen, \
         void *out, size_t outlen), (obj, in, inlen, out, outlen)) \
    _op(_module, _ops, int, -1, mlx5dv_devx_general_cmd, \
        (struct ibv_context *context, const void *in, size_t inlen, \
         void *out, size_t outlen), (context, in, inlen, out, outlen)) \
    _op(_module, _ops, int, -1, mlx5dv_devx_qp_modify, \
        (struct ibv_qp *qp, const void *in, size_t inlen, void *out, \
         size_t outlen), (qp, in, inlen, out, outlen)) \
    _op(_module, _ops, int, -1, mlx5dv_devx_qp_query, \
        (struct ibv_qp *qp, const void *in, size_t inlen, void *out, \
         size_t outlen), (qp, in, inlen, out, outlen)) \
    _op(_module, _ops, struct mlx5dv_devx_umem *, NULL, \
        mlx5dv_devx_umem_reg, \
        (struct ibv_context *ctx, void *addr, size_t size, \
         uint32_t access), (ctx, addr, size, access)) \
    _op(_module, _ops, struct mlx5dv_devx_umem *, NULL, \
        mlx5dv_devx_umem_reg_ex, \
        (struct ibv_context *ctx, struct mlx5dv_devx_umem_in *umem_in), \
        (ctx, umem_in)) \
    _op(_module, _ops, int, -1, mlx5dv_devx_umem_dereg, \
        (struct mlx5dv_devx_umem *umem), (umem)) \
    _op(_module, _ops, struct mlx5dv_devx_uar *, NULL, \
        mlx5dv_devx_alloc_uar, \
        (struct ibv_context *context, uint32_t flags), (context, flags)) \
    _op(_module, _ops, int, -1, mlx5dv_devx_query_eqn, \
        (struct ibv_context *context, uint32_t vector, uint32_t *eqn), \
        (context, vector, eqn)) \
    _op(_module, _ops, struct mlx5dv_devx_event_channel *, NULL, \
        mlx5dv_devx_create_event_channel, \
        (struct ibv_context *context, \
         enum mlx5dv_devx_create_event_channel_flags flags), \
        (context, flags)) \
    _op(_module, _ops, int, -1, mlx5dv_devx_subscribe_devx_event, \
        (struct mlx5dv_devx_event_channel *event_channel, \
         struct mlx5dv_devx_obj *obj, uint16_t events_sz, \
         uint16_t events_num[], uint64_t cookie), \
        (event_channel, obj, events_sz, events_num, cookie)) \
    _op(_module, _ops, ssize_t, -1, mlx5dv_devx_get_event, \
        (struct mlx5dv_devx_event_channel *event_channel, \
         struct mlx5dv_devx_async_event_hdr *event_data, \
         size_t event_resp_len), (event_channel, event_data, event_resp_len))

#define UCT_IB_MLX5_OPS(_op, _void_op, _module, _ops) \
    UCT_IB_MLX5_FWD_OPS(_op, _module, _ops) \
    UCT_IB_MLX5_VOID_OPS(_void_op, _module, _ops)

typedef struct uct_ib_mlx5_ops {
    UCT_IB_MLX5_OPS(UCT_IB_DLOPEN_OP_FIELD, UCT_IB_DLOPEN_VOID_OP_FIELD,
                    uct_ib_mlx5, uct_ib_mlx5_ops)
} uct_ib_mlx5_ops_t;

UCT_IB_DLOPEN_DEFINE_MODULE(uct_ib_mlx5, UCT_IB_MLX5_LIB_NAME,
                            uct_ib_mlx5_ops_t, UCT_IB_MLX5_OPS,
                            uct_ib_mlx5_dlopen_check)

UCT_IB_MLX5_FWD_OPS(UCT_IB_DLOPEN_FWD_OP, uct_ib_mlx5, uct_ib_mlx5_ops)
UCT_IB_MLX5_VOID_OPS(UCT_IB_DLOPEN_FWD_VOID_OP, uct_ib_mlx5, uct_ib_mlx5_ops)
