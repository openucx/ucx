/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_IB_MLX5_COCO_H_
#define UCT_IB_MLX5_COCO_H_

#include <ucs/type/status.h>

#include <stddef.h>
#include <stdint.h>

struct mlx5dv_devx_umem;

struct uct_ib_mlx5_md;
struct uct_ib_mlx5_devx_umem;
typedef struct uct_ib_mlx5_coco_state uct_ib_mlx5_coco_state_t;

typedef struct uct_ib_mlx5_coco_shared_alloc {
    void   *addr;
    size_t requested_size;
    size_t exposed_size;
    int    dmabuf_fd;
    int    umem_registered;
    int    quarantined;
} uct_ib_mlx5_coco_shared_alloc_t;

typedef struct uct_ib_mlx5_coco_umem_record {
    uint32_t umem_id;
    void     *addr;
    size_t   requested_size;
    size_t   exposed_size;
    uint32_t access_flags;
    uint8_t  live;
} uct_ib_mlx5_coco_umem_record_t;

typedef struct uct_ib_mlx5_coco_mkey_record {
    uint32_t lkey;
    uint32_t rkey;
    void     *base;
    size_t   length;
    uint64_t access_mask;
    uint8_t  live;
} uct_ib_mlx5_coco_mkey_record_t;

typedef struct uct_ib_mlx5_coco_cq_req {
    uint32_t cq_len;
    uint32_t cqe_size;
    uint32_t cq_umem_id;
    uint64_t cq_umem_offset;
    uint32_t dbr_umem_id;
    uint64_t dbr_offset;
    uint32_t eqn;
    uint32_t uar_page;
} uct_ib_mlx5_coco_cq_req_t;

typedef struct uct_ib_mlx5_coco_qp_req {
    uint8_t  qp_type;
    uint32_t send_cqn;
    uint32_t recv_cqn;
    uint32_t rmpn;
    uint32_t wq_umem_id;
    uint32_t dbr_umem_id;
    uint32_t sq_wqe_count;
    uint32_t rq_wqe_count;
} uct_ib_mlx5_coco_qp_req_t;

typedef struct uct_ib_mlx5_coco_rmp_req {
    uint32_t wq_umem_id;
    uint32_t dbr_umem_id;
    uint32_t wq_size;
    uint32_t stride;
    uint8_t  cyclic;
    uint8_t  mp_enabled;
} uct_ib_mlx5_coco_rmp_req_t;

typedef struct uct_ib_mlx5_coco_shared_alloc_ops {
    ucs_status_t (*alloc)(size_t size, void **addr_p, int *fd_p, void *arg);
    ucs_status_t (*umem_reg)(struct uct_ib_mlx5_md *md,
                             const uct_ib_mlx5_coco_shared_alloc_t *alloc,
                             int access_mode,
                             struct mlx5dv_devx_umem **umem_p, void *arg);
    ucs_status_t (*umem_dereg)(struct mlx5dv_devx_umem *umem, void *arg);
    ucs_status_t (*unmap)(void *addr, size_t size, void *arg);
    ucs_status_t (*close_fd)(int fd, void *arg);
} uct_ib_mlx5_coco_shared_alloc_ops_t;

ucs_status_t uct_ib_mlx5_coco_exposed_size(size_t requested_size,
                                           size_t *exposed_size_p);

ucs_status_t uct_ib_mlx5_coco_state_init(struct uct_ib_mlx5_md *md);

void uct_ib_mlx5_coco_state_cleanup(struct uct_ib_mlx5_md *md);

int uct_ib_mlx5_coco_mkey_policy_ready(const struct uct_ib_mlx5_md *md);

void uct_ib_mlx5_coco_set_shared_alloc_ops(
        const uct_ib_mlx5_coco_shared_alloc_ops_t *ops, void *arg);

ucs_status_t
uct_ib_mlx5_coco_md_buf_alloc_shared(struct uct_ib_mlx5_md *md, size_t size,
                                     int silent, void **buf_p,
                                     struct uct_ib_mlx5_devx_umem *mem,
                                     int access_mode, char *name);

void
uct_ib_mlx5_coco_md_buf_free_shared(struct uct_ib_mlx5_md *md, void *buf,
                                    struct uct_ib_mlx5_devx_umem *mem);

ucs_status_t
uct_ib_mlx5_coco_umem_record_add(uct_ib_mlx5_coco_state_t *state,
                                 uint32_t umem_id, void *addr,
                                 size_t requested_size, size_t exposed_size,
                                 uint32_t access_flags);

ucs_status_t
uct_ib_mlx5_coco_umem_record_validate(const uct_ib_mlx5_coco_state_t *state,
                                      uint32_t umem_id, void *addr,
                                      size_t requested_size,
                                      size_t exposed_size,
                                      uint32_t access_flags);

ucs_status_t
uct_ib_mlx5_coco_umem_record_remove(uct_ib_mlx5_coco_state_t *state,
                                    uint32_t umem_id);

const uct_ib_mlx5_coco_umem_record_t*
uct_ib_mlx5_coco_umem_record_find(const uct_ib_mlx5_coco_state_t *state,
                                  uint32_t umem_id);

uint64_t uct_ib_mlx5_coco_mkey_sanitize_access(uint64_t access_mask);

ucs_status_t
uct_ib_mlx5_coco_mkey_record_add(uct_ib_mlx5_coco_state_t *state,
                                 uint32_t lkey, uint32_t rkey, void *base,
                                 size_t length, uint64_t access_mask);

ucs_status_t
uct_ib_mlx5_coco_mkey_record_validate(const uct_ib_mlx5_coco_state_t *state,
                                      uint32_t lkey, uint32_t rkey,
                                      void *base, size_t length,
                                      uint64_t access_mask);

ucs_status_t
uct_ib_mlx5_coco_mkey_record_remove(uct_ib_mlx5_coco_state_t *state,
                                    uint32_t lkey, uint32_t rkey);

ucs_status_t
uct_ib_mlx5_coco_mkey_record_remove_rkey(uct_ib_mlx5_coco_state_t *state,
                                         uint32_t rkey);

const uct_ib_mlx5_coco_mkey_record_t*
uct_ib_mlx5_coco_mkey_record_find_lkey(const uct_ib_mlx5_coco_state_t *state,
                                       uint32_t lkey);

ucs_status_t
uct_ib_mlx5_coco_validate_cq_output(struct uct_ib_mlx5_md *md,
                                    const uct_ib_mlx5_coco_cq_req_t *req,
                                    const void *out, size_t out_len,
                                    uint32_t *cqn_p);

ucs_status_t
uct_ib_mlx5_coco_validate_qp_output(struct uct_ib_mlx5_md *md,
                                    const uct_ib_mlx5_coco_qp_req_t *req,
                                    const void *out, size_t out_len,
                                    uint32_t *qpn_p);

ucs_status_t
uct_ib_mlx5_coco_validate_rmp_output(struct uct_ib_mlx5_md *md,
                                     const uct_ib_mlx5_coco_rmp_req_t *req,
                                     const void *out, size_t out_len,
                                     uint32_t *rmpn_p);

ucs_status_t
uct_ib_mlx5_coco_cqn_record_remove(uct_ib_mlx5_coco_state_t *state,
                                   uint32_t cqn);

ucs_status_t
uct_ib_mlx5_coco_qpn_record_remove(uct_ib_mlx5_coco_state_t *state,
                                   uint32_t qpn);

ucs_status_t
uct_ib_mlx5_coco_rmpn_record_remove(uct_ib_mlx5_coco_state_t *state,
                                    uint32_t rmpn);

#endif
