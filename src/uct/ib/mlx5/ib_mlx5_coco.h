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

typedef struct uct_ib_mlx5_md uct_ib_mlx5_md_t;
typedef struct uct_ib_mlx5_devx_umem uct_ib_mlx5_devx_umem_t;
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

typedef struct uct_ib_mlx5_coco_shared_alloc_ops {
    ucs_status_t (*alloc)(size_t size, void **addr_p, int *fd_p, void *arg);
    ucs_status_t (*umem_reg)(uct_ib_mlx5_md_t *md,
                             const uct_ib_mlx5_coco_shared_alloc_t *alloc,
                             int access_mode,
                             struct mlx5dv_devx_umem **umem_p, void *arg);
    ucs_status_t (*umem_dereg)(struct mlx5dv_devx_umem *umem, void *arg);
    ucs_status_t (*unmap)(void *addr, size_t size, void *arg);
    ucs_status_t (*close_fd)(int fd, void *arg);
} uct_ib_mlx5_coco_shared_alloc_ops_t;

ucs_status_t uct_ib_mlx5_coco_exposed_size(size_t requested_size,
                                           size_t *exposed_size_p);

ucs_status_t uct_ib_mlx5_coco_state_init(uct_ib_mlx5_md_t *md);

void uct_ib_mlx5_coco_state_cleanup(uct_ib_mlx5_md_t *md);

int uct_ib_mlx5_coco_mkey_policy_ready(const uct_ib_mlx5_md_t *md);

void uct_ib_mlx5_coco_set_shared_alloc_ops(
        const uct_ib_mlx5_coco_shared_alloc_ops_t *ops, void *arg);

ucs_status_t
uct_ib_mlx5_coco_md_buf_alloc_shared(uct_ib_mlx5_md_t *md, size_t size,
                                     int silent, void **buf_p,
                                     uct_ib_mlx5_devx_umem_t *mem,
                                     int access_mode, char *name);

void uct_ib_mlx5_coco_md_buf_free_shared(uct_ib_mlx5_md_t *md, void *buf,
                                         uct_ib_mlx5_devx_umem_t *mem);

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

#endif
