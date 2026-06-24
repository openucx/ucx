
#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include <infiniband/verbs.h>
#if defined(HAVE_MLX5_DV)
#  include <infiniband/mlx5dv.h>
#endif

/* ───────────────────────── lazy handle acquisition ─────────────────────── */

static void *uct_ib_verbs_lib;   /* libibverbs.so.1, or NULL when absent */
static void *uct_ib_mlx5_lib;    /* libmlx5.so.1,    or NULL when absent */

static void uct_ib_dlopen_once(void)
{
    uct_ib_verbs_lib = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_LOCAL);
    uct_ib_mlx5_lib  = dlopen("libmlx5.so.1",    RTLD_NOW | RTLD_LOCAL);
}

static void uct_ib_dlopen(void)
{
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, uct_ib_dlopen_once);
}

static void *uct_ib_verbs_sym(const char *name)
{
    uct_ib_dlopen();
    return (uct_ib_verbs_lib != NULL) ? dlsym(uct_ib_verbs_lib, name) : NULL;
}

#if defined(HAVE_MLX5_DV)
static void *uct_ib_mlx5_sym(const char *name)
{
    uct_ib_dlopen();
    return (uct_ib_mlx5_lib != NULL) ? dlsym(uct_ib_mlx5_lib, name) : NULL;
}
#endif

/* ───────────────────────── forwarding macros ───────────────────────────── */

#define IBV_FWD(_ret, _name, _proto, _call)                     \
    _ret _name _proto                                           \
    {                                                           \
        typedef _ret (*_fn_t) _proto;                           \
        static _fn_t _fn;                                       \
        if (_fn == NULL) {                                      \
            _fn = (_fn_t)uct_ib_verbs_sym(#_name);              \
        }                                                       \
        if (_fn == NULL) {                                      \
            return (_ret)0;                                     \
        }                                                       \
        return _fn _call;                                       \
    }

#define IBV_FWD_VOID(_name, _proto, _call)                      \
    void _name _proto                                           \
    {                                                           \
        typedef void (*_fn_t) _proto;                           \
        static _fn_t _fn;                                       \
        if (_fn == NULL) {                                      \
            _fn = (_fn_t)uct_ib_verbs_sym(#_name);              \
        }                                                       \
        if (_fn != NULL) {                                      \
            _fn _call;                                          \
        }                                                       \
    }

#if defined(HAVE_MLX5_DV)
#define DV_FWD(_ret, _name, _proto, _call)                      \
    _ret _name _proto                                           \
    {                                                           \
        typedef _ret (*_fn_t) _proto;                           \
        static _fn_t _fn;                                       \
        if (_fn == NULL) {                                      \
            _fn = (_fn_t)uct_ib_mlx5_sym(#_name);               \
        }                                                       \
        if (_fn == NULL) {                                      \
            return (_ret)0;                                     \
        }                                                       \
        return _fn _call;                                       \
    }

#define DV_FWD_VOID(_name, _proto, _call)                       \
    void _name _proto                                           \
    {                                                           \
        typedef void (*_fn_t) _proto;                           \
        static _fn_t _fn;                                       \
        if (_fn == NULL) {                                      \
            _fn = (_fn_t)uct_ib_mlx5_sym(#_name);               \
        }                                                       \
        if (_fn != NULL) {                                      \
            _fn _call;                                          \
        }                                                       \
    }
#endif

/* ───────────────────────── verbs: device / context ─────────────────────── */

/* Special-cased so a missing library reports zero devices (clean degrade). */
struct ibv_device **ibv_get_device_list(int *num_devices)
{
    static struct ibv_device **(*fn)(int *);
    if (fn == NULL) {
        fn = (struct ibv_device **(*)(int *))uct_ib_verbs_sym("ibv_get_device_list");
    }
    if (fn == NULL) {
        if (num_devices != NULL) {
            *num_devices = 0;
        }
        return NULL;
    }
    return fn(num_devices);
}

IBV_FWD_VOID(ibv_free_device_list, (struct ibv_device **list), (list))
IBV_FWD(const char *, ibv_get_device_name, (struct ibv_device *device), (device))
IBV_FWD(__be64, ibv_get_device_guid, (struct ibv_device *device), (device))
IBV_FWD(int, ibv_get_device_index, (struct ibv_device *device), (device))
IBV_FWD(struct ibv_context *, ibv_open_device, (struct ibv_device *device), (device))
IBV_FWD(int, ibv_close_device, (struct ibv_context *context), (context))
IBV_FWD(int, ibv_fork_init, (void), ())
IBV_FWD(int, ibv_get_async_event,
        (struct ibv_context *context, struct ibv_async_event *event),
        (context, event))
IBV_FWD_VOID(ibv_ack_async_event, (struct ibv_async_event *event), (event))
IBV_FWD(struct ibv_comp_channel *, ibv_create_comp_channel,
        (struct ibv_context *context), (context))
IBV_FWD(int, ibv_destroy_comp_channel,
        (struct ibv_comp_channel *channel), (channel))

/* ───────────────────────── verbs: PD ───────────────────────────────────── */

IBV_FWD(struct ibv_pd *, ibv_alloc_pd, (struct ibv_context *context), (context))
IBV_FWD(int, ibv_dealloc_pd, (struct ibv_pd *pd), (pd))
/* ibv_alloc_td, ibv_dealloc_td, and ibv_alloc_parent_domain are static inline
 * wrappers in verbs.h that dispatch through the verbs_context ops table; they
 * must not be forwarded (forwarding redefines the header's inline symbol). */

/* ───────────────────────── verbs: MR / DM ──────────────────────────────── */

/* ibv_reg_mr is a function-like macro in modern verbs.h that expands to the
 * static inline __ibv_reg_mr, which in turn calls the real exported symbols
 * ibv_reg_mr / ibv_reg_mr_iova2. Undefine the macro and forward those real
 * symbols (the header's __ibv_reg_mr inline keeps working on top of them). */
#undef ibv_reg_mr
IBV_FWD(struct ibv_mr *, ibv_reg_mr,
        (struct ibv_pd *pd, void *addr, size_t length, int access),
        (pd, addr, length, access))
IBV_FWD(struct ibv_mr *, ibv_reg_mr_iova2,
        (struct ibv_pd *pd, void *addr, size_t length, uint64_t iova,
         unsigned int access),
        (pd, addr, length, iova, access))

IBV_FWD(int, ibv_dereg_mr, (struct ibv_mr *mr), (mr))
IBV_FWD(struct ibv_mr *, ibv_reg_dmabuf_mr,
        (struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova,
         int fd, int access),
        (pd, offset, length, iova, fd, access))
/* ibv_advise_mr, ibv_reg_dm_mr, ibv_alloc_dm, and ibv_free_dm are static inline
 * in verbs.h (dispatch via verbs_context ops) - do not forward. */

/* ───────────────────────── verbs: CQ ───────────────────────────────────── */

IBV_FWD(struct ibv_cq *, ibv_create_cq,
        (struct ibv_context *context, int cqe, void *cq_context,
         struct ibv_comp_channel *channel, int comp_vector),
        (context, cqe, cq_context, channel, comp_vector))
IBV_FWD(int, ibv_destroy_cq, (struct ibv_cq *cq), (cq))

/* ───────────────────────── verbs: QP ───────────────────────────────────── */

IBV_FWD(struct ibv_qp *, ibv_create_qp,
        (struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr),
        (pd, qp_init_attr))
IBV_FWD(int, ibv_destroy_qp, (struct ibv_qp *qp), (qp))
IBV_FWD(int, ibv_modify_qp,
        (struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask),
        (qp, attr, attr_mask))
IBV_FWD(int, ibv_query_qp,
        (struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask,
         struct ibv_qp_init_attr *init_attr),
        (qp, attr, attr_mask, init_attr))
IBV_FWD(int, ibv_query_ece, (struct ibv_qp *qp, struct ibv_ece *ece), (qp, ece))
IBV_FWD(int, ibv_set_ece,   (struct ibv_qp *qp, struct ibv_ece *ece), (qp, ece))

/* ───────────────────────── verbs: SRQ ──────────────────────────────────── */

IBV_FWD(struct ibv_srq *, ibv_create_srq,
        (struct ibv_pd *pd, struct ibv_srq_init_attr *srq_init_attr),
        (pd, srq_init_attr))
/* ibv_create_srq_ex is static inline (dispatch via verbs_context ops). */
IBV_FWD(int, ibv_destroy_srq, (struct ibv_srq *srq), (srq))

/* ───────────────────────── verbs: query / strings ──────────────────────── */

IBV_FWD(int, ibv_query_device,
        (struct ibv_context *context, struct ibv_device_attr *device_attr),
        (context, device_attr))
/* ibv_query_device_ex is static inline (dispatch via verbs_context ops). */

/* ibv_query_port is a macro in modern verbs.h expanding to the static inline
 * ___ibv_query_port, which calls the real exported ibv_query_port (taking a
 * struct _compat_ibv_port_attr *). Undefine the macro and forward the real
 * symbol with its compat signature so the header inline keeps working. */
#undef ibv_query_port
IBV_FWD(int, ibv_query_port,
        (struct ibv_context *context, uint8_t port_num,
         struct _compat_ibv_port_attr *port_attr),
        (context, port_num, port_attr))

IBV_FWD(int, ibv_query_gid,
        (struct ibv_context *context, uint8_t port_num, int index,
         union ibv_gid *gid),
        (context, port_num, index, gid))
IBV_FWD(const char *, ibv_wc_status_str, (enum ibv_wc_status status), (status))
IBV_FWD(const char *, ibv_event_type_str, (enum ibv_event_type event), (event))

/* ───────────────────────── mlx5 Direct Verbs (DEVX) ────────────────────── */
#if defined(HAVE_MLX5_DV)

DV_FWD(bool, mlx5dv_is_supported, (struct ibv_device *device), (device))
DV_FWD(struct ibv_context *, mlx5dv_open_device,
       (struct ibv_device *device, struct mlx5dv_context_attr *attr),
       (device, attr))
DV_FWD(int, mlx5dv_query_device,
       (struct ibv_context *ctx_in, struct mlx5dv_context *attrs_out),
       (ctx_in, attrs_out))
DV_FWD(struct ibv_qp *, mlx5dv_create_qp,
       (struct ibv_context *context, struct ibv_qp_init_attr_ex *qp_attr,
        struct mlx5dv_qp_init_attr *mlx5_qp_attr),
       (context, qp_attr, mlx5_qp_attr))
DV_FWD(struct ibv_cq_ex *, mlx5dv_create_cq,
       (struct ibv_context *context, struct ibv_cq_init_attr_ex *cq_attr,
        struct mlx5dv_cq_init_attr *mlx5_cq_attr),
       (context, cq_attr, mlx5_cq_attr))
DV_FWD(int, mlx5dv_init_obj, (struct mlx5dv_obj *obj, uint64_t obj_type),
       (obj, obj_type))
DV_FWD(struct mlx5dv_mkey *, mlx5dv_create_mkey,
       (struct mlx5dv_mkey_init_attr *mkey_init_attr), (mkey_init_attr))
DV_FWD(int, mlx5dv_destroy_mkey, (struct mlx5dv_mkey *mkey), (mkey))
DV_FWD(struct ibv_mr *, mlx5dv_reg_dmabuf_mr,
       (struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova,
        int fd, int access, int mlx5_access),
       (pd, offset, length, iova, fd, access, mlx5_access))
DV_FWD(int, mlx5dv_get_data_direct_sysfs_path,
       (struct ibv_context *context, char *buf, size_t buf_len),
       (context, buf, buf_len))

/* DEVX object / command */
DV_FWD(struct mlx5dv_devx_obj *, mlx5dv_devx_obj_create,
       (struct ibv_context *context, const void *in, size_t inlen,
        void *out, size_t outlen),
       (context, in, inlen, out, outlen))
DV_FWD(int, mlx5dv_devx_obj_destroy, (struct mlx5dv_devx_obj *obj), (obj))
DV_FWD(int, mlx5dv_devx_obj_modify,
       (struct mlx5dv_devx_obj *obj, const void *in, size_t inlen,
        void *out, size_t outlen),
       (obj, in, inlen, out, outlen))
DV_FWD(int, mlx5dv_devx_obj_query,
       (struct mlx5dv_devx_obj *obj, const void *in, size_t inlen,
        void *out, size_t outlen),
       (obj, in, inlen, out, outlen))
DV_FWD(int, mlx5dv_devx_general_cmd,
       (struct ibv_context *context, const void *in, size_t inlen,
        void *out, size_t outlen),
       (context, in, inlen, out, outlen))
DV_FWD(int, mlx5dv_devx_qp_modify,
       (struct ibv_qp *qp, const void *in, size_t inlen,
        void *out, size_t outlen),
       (qp, in, inlen, out, outlen))
DV_FWD(int, mlx5dv_devx_qp_query,
       (struct ibv_qp *qp, const void *in, size_t inlen,
        void *out, size_t outlen),
       (qp, in, inlen, out, outlen))

/* DEVX UMEM */
DV_FWD(struct mlx5dv_devx_umem *, mlx5dv_devx_umem_reg,
       (struct ibv_context *ctx, void *addr, size_t size, uint32_t access),
       (ctx, addr, size, access))
DV_FWD(struct mlx5dv_devx_umem *, mlx5dv_devx_umem_reg_ex,
       (struct ibv_context *ctx, struct mlx5dv_devx_umem_in *umem_in),
       (ctx, umem_in))
DV_FWD(int, mlx5dv_devx_umem_dereg, (struct mlx5dv_devx_umem *umem), (umem))

/* DEVX UAR */
DV_FWD(struct mlx5dv_devx_uar *, mlx5dv_devx_alloc_uar,
       (struct ibv_context *context, uint32_t flags), (context, flags))
DV_FWD_VOID(mlx5dv_devx_free_uar, (struct mlx5dv_devx_uar *devx_uar),
            (devx_uar))
DV_FWD(int, mlx5dv_devx_query_eqn,
       (struct ibv_context *context, uint32_t vector, uint32_t *eqn),
       (context, vector, eqn))

/* DEVX async events */
DV_FWD(struct mlx5dv_devx_event_channel *, mlx5dv_devx_create_event_channel,
       (struct ibv_context *context,
        enum mlx5dv_devx_create_event_channel_flags flags),
       (context, flags))
DV_FWD_VOID(mlx5dv_devx_destroy_event_channel,
            (struct mlx5dv_devx_event_channel *event_channel),
            (event_channel))
DV_FWD(int, mlx5dv_devx_subscribe_devx_event,
       (struct mlx5dv_devx_event_channel *event_channel,
        struct mlx5dv_devx_obj *obj, uint16_t events_sz,
        uint16_t events_num[], uint64_t cookie),
       (event_channel, obj, events_sz, events_num, cookie))
DV_FWD(ssize_t, mlx5dv_devx_get_event,
       (struct mlx5dv_devx_event_channel *event_channel,
        struct mlx5dv_devx_async_event_hdr *event_data, size_t event_resp_len),
       (event_channel, event_data, event_resp_len))

#endif /* HAVE_MLX5_DV */