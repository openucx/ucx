
#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "ib_dlopen.h"

#include <stddef.h>
#include <stdint.h>

#define UCT_IB_VERBS_LIB_NAME  "libibverbs.so.1"

#ifdef HAVE_NETLINK_RDMA
#define UCT_IB_VERBS_NETLINK_OPS(_op, _module, _ops) \
    _op(_module, _ops, int, -1, ibv_get_device_index, \
        (struct ibv_device *device), (device))
#else
#define UCT_IB_VERBS_NETLINK_OPS(_op, _module, _ops)
#endif

#if HAVE_DECL_IBV_REG_DMABUF_MR
#define UCT_IB_VERBS_DMABUF_OPS(_op, _module, _ops) \
    _op(_module, _ops, struct ibv_mr *, NULL, ibv_reg_dmabuf_mr, \
        (struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, \
         int fd, int access), (pd, offset, length, iova, fd, access))
#else
#define UCT_IB_VERBS_DMABUF_OPS(_op, _module, _ops)
#endif

#if HAVE_DECL_IBV_SET_ECE
#define UCT_IB_VERBS_ECE_OPS(_op, _module, _ops) \
    _op(_module, _ops, int, -1, ibv_query_ece, \
        (struct ibv_qp *qp, struct ibv_ece *ece), (qp, ece)) \
    _op(_module, _ops, int, -1, ibv_set_ece, \
        (struct ibv_qp *qp, struct ibv_ece *ece), (qp, ece))
#else
#define UCT_IB_VERBS_ECE_OPS(_op, _module, _ops)
#endif

#define UCT_IB_VERBS_DEVICE_LIST_OP(_op, _module, _ops) \
    _op(_module, _ops, struct ibv_device **, NULL, ibv_get_device_list, \
        (int *num_devices), (num_devices))

#define UCT_IB_VERBS_VOID_OPS(_op, _module, _ops) \
    _op(_module, _ops, ibv_free_device_list, (struct ibv_device **list), \
        (list)) \
    _op(_module, _ops, ibv_ack_async_event, \
        (struct ibv_async_event *event), (event)) \
    _op(_module, _ops, ibv_ack_cq_events, \
        (struct ibv_cq *cq, unsigned int nevents), (cq, nevents))

#define UCT_IB_VERBS_FWD_OPS(_op, _module, _ops) \
    _op(_module, _ops, const char *, NULL, ibv_get_device_name, \
        (struct ibv_device *device), (device)) \
    _op(_module, _ops, __be64, 0, ibv_get_device_guid, \
        (struct ibv_device *device), \
        (device)) \
    UCT_IB_VERBS_NETLINK_OPS(_op, _module, _ops) \
    _op(_module, _ops, struct ibv_context *, NULL, ibv_open_device, \
        (struct ibv_device *device), (device)) \
    _op(_module, _ops, int, -1, ibv_close_device, \
        (struct ibv_context *context), \
        (context)) \
    _op(_module, _ops, int, -1, ibv_fork_init, (void), ()) \
    _op(_module, _ops, int, -1, ibv_get_async_event, \
        (struct ibv_context *context, struct ibv_async_event *event), \
        (context, event)) \
    _op(_module, _ops, struct ibv_comp_channel *, NULL, \
        ibv_create_comp_channel, \
        (struct ibv_context *context), (context)) \
    _op(_module, _ops, int, -1, ibv_destroy_comp_channel, \
        (struct ibv_comp_channel *channel), (channel)) \
    _op(_module, _ops, struct ibv_pd *, NULL, ibv_alloc_pd, \
        (struct ibv_context *context), (context)) \
    _op(_module, _ops, int, -1, ibv_dealloc_pd, (struct ibv_pd *pd), (pd)) \
    _op(_module, _ops, struct ibv_mr *, NULL, ibv_reg_mr, \
        (struct ibv_pd *pd, void *addr, size_t length, int access), \
        (pd, addr, length, access)) \
    _op(_module, _ops, struct ibv_mr *, NULL, ibv_reg_mr_iova2, \
        (struct ibv_pd *pd, void *addr, size_t length, uint64_t iova, \
         unsigned int access), (pd, addr, length, iova, access)) \
    _op(_module, _ops, int, -1, ibv_dereg_mr, (struct ibv_mr *mr), (mr)) \
    UCT_IB_VERBS_DMABUF_OPS(_op, _module, _ops) \
    _op(_module, _ops, struct ibv_cq *, NULL, ibv_create_cq, \
        (struct ibv_context *context, int cqe, void *cq_context, \
         struct ibv_comp_channel *channel, int comp_vector), \
        (context, cqe, cq_context, channel, comp_vector)) \
    _op(_module, _ops, int, -1, ibv_destroy_cq, (struct ibv_cq *cq), (cq)) \
    _op(_module, _ops, int, -1, ibv_get_cq_event, \
        (struct ibv_comp_channel *channel, struct ibv_cq **cq, \
         void **cq_context), (channel, cq, cq_context)) \
    _op(_module, _ops, struct ibv_ah *, NULL, ibv_create_ah, \
        (struct ibv_pd *pd, struct ibv_ah_attr *attr), (pd, attr)) \
    _op(_module, _ops, int, -1, ibv_destroy_ah, (struct ibv_ah *ah), (ah)) \
    _op(_module, _ops, struct ibv_qp *, NULL, ibv_create_qp, \
        (struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr), \
        (pd, qp_init_attr)) \
    _op(_module, _ops, int, -1, ibv_destroy_qp, (struct ibv_qp *qp), (qp)) \
    _op(_module, _ops, int, -1, ibv_modify_qp, \
        (struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask), \
        (qp, attr, attr_mask)) \
    _op(_module, _ops, int, -1, ibv_query_qp, \
        (struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask, \
         struct ibv_qp_init_attr *init_attr), \
        (qp, attr, attr_mask, init_attr)) \
    UCT_IB_VERBS_ECE_OPS(_op, _module, _ops) \
    _op(_module, _ops, struct ibv_srq *, NULL, ibv_create_srq, \
        (struct ibv_pd *pd, struct ibv_srq_init_attr *srq_init_attr), \
        (pd, srq_init_attr)) \
    _op(_module, _ops, int, -1, ibv_destroy_srq, \
        (struct ibv_srq *srq), (srq)) \
    _op(_module, _ops, int, -1, ibv_query_device, \
        (struct ibv_context *context, struct ibv_device_attr *device_attr), \
        (context, device_attr)) \
    _op(_module, _ops, int, -1, ibv_query_port, \
        (struct ibv_context *context, uint8_t port_num, \
         struct _compat_ibv_port_attr *port_attr), \
        (context, port_num, port_attr)) \
    _op(_module, _ops, int, -1, ibv_query_gid, \
        (struct ibv_context *context, uint8_t port_num, int index, \
         union ibv_gid *gid), (context, port_num, index, gid)) \
    _op(_module, _ops, int, -1, ibv_query_pkey, \
        (struct ibv_context *context, uint8_t port_num, int index, \
         __be16 *pkey), (context, port_num, index, pkey)) \
    _op(_module, _ops, const char *, NULL, ibv_wc_status_str, \
        (enum ibv_wc_status status), (status)) \
    _op(_module, _ops, const char *, NULL, ibv_event_type_str, \
        (enum ibv_event_type event), (event)) \
    _op(_module, _ops, const char *, NULL, ibv_node_type_str, \
        (enum ibv_node_type node_type), (node_type))

#define UCT_IB_VERBS_OPS(_op, _void_op, _module, _ops) \
    UCT_IB_VERBS_DEVICE_LIST_OP(_op, _module, _ops) \
    UCT_IB_VERBS_VOID_OPS(_void_op, _module, _ops) \
    UCT_IB_VERBS_FWD_OPS(_op, _module, _ops)

typedef struct uct_ib_verbs_ops {
    UCT_IB_VERBS_OPS(UCT_IB_DLOPEN_OP_FIELD,
                     UCT_IB_DLOPEN_VOID_OP_FIELD, uct_ib_verbs,
                     uct_ib_verbs_ops)
} uct_ib_verbs_ops_t;

const char *uct_ib_dlopen_status_string(uct_ib_dlopen_status_t status)
{
    switch (status) {
    case UCT_IB_DLOPEN_STATUS_OK:
        return "ok";
    case UCT_IB_DLOPEN_STATUS_NO_LIB:
        return "library not found";
    case UCT_IB_DLOPEN_STATUS_MISSING_SYM:
        return "symbol not found";
    }

    return "unknown";
}

UCT_IB_DLOPEN_DEFINE_MODULE(uct_ib_verbs, UCT_IB_VERBS_LIB_NAME,
                            uct_ib_verbs_ops_t, UCT_IB_VERBS_OPS,
                            uct_ib_verbs_dlopen_check)

struct ibv_device **ibv_get_device_list(int *num_devices)
{
    if (uct_ib_verbs_init() != 0) {
        if (num_devices != NULL) {
            *num_devices = 0;
        }
        return NULL;
    }

    return uct_ib_verbs_ops.ibv_get_device_list(num_devices);
}

/* ibv_reg_mr and ibv_query_port are macros in modern verbs.h. */
#undef ibv_reg_mr
#undef ibv_query_port

UCT_IB_VERBS_VOID_OPS(UCT_IB_DLOPEN_FWD_VOID_OP, uct_ib_verbs,
                      uct_ib_verbs_ops)
UCT_IB_VERBS_FWD_OPS(UCT_IB_DLOPEN_FWD_OP, uct_ib_verbs, uct_ib_verbs_ops)
