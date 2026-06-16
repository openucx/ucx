/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ib_mlx5.h"
#include "ib_mlx5_coco.h"

#include <ucs/debug/memtrack_int.h>
#include <ucs/debug/log.h>

#include <errno.h>
#include <string.h>


struct uct_ib_mlx5_coco_state {
    uct_ib_mlx5_coco_umem_record_t *umem_records;
    size_t                         umem_count;
    size_t                         umem_capacity;
    uct_ib_mlx5_coco_mkey_record_t *mkey_records;
    size_t                         mkey_count;
    size_t                         mkey_capacity;
};

typedef struct {
    uct_ib_mlx5_coco_shared_alloc_ops_t ops;
    void                                *arg;
} uct_ib_mlx5_coco_shared_alloc_backend_t;

static ucs_status_t
uct_ib_mlx5_coco_default_alloc(size_t size, void **addr_p, int *fd_p,
                               void *arg)
{
#if HAVE_LINUX_DMA_HEAP_H
    struct dma_heap_allocation_data heap_data;
    ucs_status_t status;
    int heap_fd;
    void *addr;
    int ret;

    heap_fd = open(UCT_IB_MLX5_CC_DMA_HEAP, O_RDWR | O_CLOEXEC);
    if (heap_fd < 0) {
        return (errno == ENOENT) ? UCS_ERR_UNSUPPORTED : UCS_ERR_IO_ERROR;
    }

    memset(&heap_data, 0, sizeof(heap_data));
    heap_data.len      = size;
    heap_data.fd_flags = O_RDWR | O_CLOEXEC;

    ret = ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &heap_data);
    if (ret != 0) {
        status = (errno == ENOMEM) ? UCS_ERR_NO_MEMORY : UCS_ERR_IO_ERROR;
        ret = close(heap_fd);
        if (ret != 0) {
            ucs_warn("close(dma_heap_fd=%d) failed: %m", heap_fd);
        }
        return status;
    }

    ret = close(heap_fd);
    if (ret != 0) {
        ucs_warn("close(dma_heap_fd=%d) failed: %m", heap_fd);
    }

    addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, heap_data.fd, 0);
    if (addr == MAP_FAILED) {
        status = (errno == ENOMEM) ? UCS_ERR_NO_MEMORY : UCS_ERR_IO_ERROR;
        ret = close(heap_data.fd);
        if (ret != 0) {
            ucs_warn("close(dmabuf_fd=%d) failed: %m", heap_data.fd);
        }
        return status;
    }

    *addr_p = addr;
    *fd_p   = heap_data.fd;
    return UCS_OK;
#else
    (void)size;
    (void)addr_p;
    (void)fd_p;
    (void)arg;
    return UCS_ERR_UNSUPPORTED;
#endif
}

static ucs_status_t
uct_ib_mlx5_coco_default_umem_reg(uct_ib_mlx5_md_t *md,
                                  const uct_ib_mlx5_coco_shared_alloc_t *alloc,
                                  int access_mode,
                                  struct mlx5dv_devx_umem **umem_p,
                                  void *arg)
{
#if HAVE_DECL_MLX5DV_DEVX_UMEM_REG_EX && HAVE_DECL_MLX5DV_UMEM_MASK_DMABUF && \
    HAVE_STRUCT_MLX5DV_DEVX_UMEM_IN
    struct mlx5dv_devx_umem_in umem_in;

    memset(&umem_in, 0, sizeof(umem_in));
    umem_in.addr        = NULL;
    umem_in.size        = alloc->exposed_size;
    umem_in.access      = access_mode;
    umem_in.pgsz_bitmap = UINT64_MAX & ~(ucs_get_page_size() - 1);
    umem_in.comp_mask   = MLX5DV_UMEM_MASK_DMABUF;
    umem_in.dmabuf_fd   = alloc->dmabuf_fd;

    *umem_p = mlx5dv_devx_umem_reg_ex(md->super.dev.ibv_context, &umem_in);
    return (*umem_p == NULL) ? uct_ib_mlx5_devx_umem_reg_status(errno) : UCS_OK;
#else
    (void)md;
    (void)alloc;
    (void)access_mode;
    (void)umem_p;
    (void)arg;
    return UCS_ERR_UNSUPPORTED;
#endif
}

static ucs_status_t
uct_ib_mlx5_coco_default_umem_dereg(struct mlx5dv_devx_umem *umem, void *arg)
{
    return (mlx5dv_devx_umem_dereg(umem) == 0) ? UCS_OK : UCS_ERR_IO_ERROR;
}

static ucs_status_t uct_ib_mlx5_coco_default_unmap(void *addr, size_t size,
                                                   void *arg)
{
    return (munmap(addr, size) == 0) ? UCS_OK : UCS_ERR_IO_ERROR;
}

static ucs_status_t uct_ib_mlx5_coco_default_close(int fd, void *arg)
{
    return (close(fd) == 0) ? UCS_OK : UCS_ERR_IO_ERROR;
}

static uct_ib_mlx5_coco_shared_alloc_backend_t
uct_ib_mlx5_coco_shared_alloc_backend = {
    {
        uct_ib_mlx5_coco_default_alloc,
        uct_ib_mlx5_coco_default_umem_reg,
        uct_ib_mlx5_coco_default_umem_dereg,
        uct_ib_mlx5_coco_default_unmap,
        uct_ib_mlx5_coco_default_close
    },
    NULL
};

void uct_ib_mlx5_coco_set_shared_alloc_ops(
        const uct_ib_mlx5_coco_shared_alloc_ops_t *ops, void *arg)
{
    if (ops == NULL) {
        uct_ib_mlx5_coco_shared_alloc_backend.ops.alloc =
                uct_ib_mlx5_coco_default_alloc;
        uct_ib_mlx5_coco_shared_alloc_backend.ops.umem_reg =
                uct_ib_mlx5_coco_default_umem_reg;
        uct_ib_mlx5_coco_shared_alloc_backend.ops.umem_dereg =
                uct_ib_mlx5_coco_default_umem_dereg;
        uct_ib_mlx5_coco_shared_alloc_backend.ops.unmap =
                uct_ib_mlx5_coco_default_unmap;
        uct_ib_mlx5_coco_shared_alloc_backend.ops.close_fd =
                uct_ib_mlx5_coco_default_close;
        uct_ib_mlx5_coco_shared_alloc_backend.arg = NULL;
        return;
    }

    uct_ib_mlx5_coco_shared_alloc_backend.ops = *ops;
    uct_ib_mlx5_coco_shared_alloc_backend.arg = arg;
}

ucs_status_t uct_ib_mlx5_coco_exposed_size(size_t requested_size,
                                           size_t *exposed_size_p)
{
    if (requested_size == 0) {
        return UCS_ERR_INVALID_PARAM;
    }

    *exposed_size_p = ucs_align_up(requested_size, ucs_get_page_size());
    return UCS_OK;
}

static ucs_status_t
uct_ib_mlx5_coco_grow(void **records_p, size_t *capacity_p, size_t count,
                      size_t elem_size, const char *name)
{
    size_t capacity = *capacity_p;
    void *records;

    if (count < capacity) {
        return UCS_OK;
    }

    capacity = (capacity == 0) ? 4 : capacity * 2;
    records  = ucs_realloc(*records_p, capacity * elem_size, name);
    if (records == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    *records_p  = records;
    *capacity_p = capacity;
    return UCS_OK;
}

ucs_status_t uct_ib_mlx5_coco_state_init(uct_ib_mlx5_md_t *md)
{
    if (!uct_ib_md_is_coco_hardened(&md->super)) {
        md->coco = NULL;
        return UCS_OK;
    }

    if (md->coco != NULL) {
        return UCS_OK;
    }

    md->coco = ucs_calloc(1, sizeof(*md->coco), "mlx5 CoCo state");
    return (md->coco == NULL) ? UCS_ERR_NO_MEMORY : UCS_OK;
}

void uct_ib_mlx5_coco_state_cleanup(uct_ib_mlx5_md_t *md)
{
    if (md->coco == NULL) {
        return;
    }

    ucs_free(md->coco->umem_records);
    ucs_free(md->coco->mkey_records);
    ucs_free(md->coco);
    md->coco = NULL;
}

int uct_ib_mlx5_coco_mkey_policy_ready(const uct_ib_mlx5_md_t *md)
{
    return !uct_ib_md_is_coco_hardened(&md->super) || (md->coco != NULL);
}

const uct_ib_mlx5_coco_umem_record_t*
uct_ib_mlx5_coco_umem_record_find(const uct_ib_mlx5_coco_state_t *state,
                                  uint32_t umem_id)
{
    size_t i;

    if (state == NULL) {
        return NULL;
    }

    for (i = 0; i < state->umem_count; ++i) {
        if (state->umem_records[i].live &&
            (state->umem_records[i].umem_id == umem_id)) {
            return &state->umem_records[i];
        }
    }

    return NULL;
}

ucs_status_t
uct_ib_mlx5_coco_umem_record_add(uct_ib_mlx5_coco_state_t *state,
                                 uint32_t umem_id, void *addr,
                                 size_t requested_size, size_t exposed_size,
                                 uint32_t access_flags)
{
    uct_ib_mlx5_coco_umem_record_t *record;
    ucs_status_t status;

    if (state == NULL) {
        return UCS_OK;
    }

    if (uct_ib_mlx5_coco_umem_record_find(state, umem_id) != NULL) {
        return UCS_ERR_ALREADY_EXISTS;
    }

    status = uct_ib_mlx5_coco_grow((void**)&state->umem_records,
                                   &state->umem_capacity, state->umem_count,
                                   sizeof(*state->umem_records),
                                   "CoCo UMEM records");
    if (status != UCS_OK) {
        return status;
    }

    record                 = &state->umem_records[state->umem_count++];
    record->umem_id        = umem_id;
    record->addr           = addr;
    record->requested_size = requested_size;
    record->exposed_size   = exposed_size;
    record->access_flags   = access_flags;
    record->live           = 1;
    return UCS_OK;
}

ucs_status_t
uct_ib_mlx5_coco_umem_record_validate(const uct_ib_mlx5_coco_state_t *state,
                                      uint32_t umem_id, void *addr,
                                      size_t requested_size,
                                      size_t exposed_size,
                                      uint32_t access_flags)
{
    const uct_ib_mlx5_coco_umem_record_t *record =
            uct_ib_mlx5_coco_umem_record_find(state, umem_id);

    if (record == NULL) {
        return UCS_ERR_NO_ELEM;
    }

    if ((record->addr != addr) || (requested_size > record->requested_size) ||
        (exposed_size > record->exposed_size) ||
        ((access_flags & ~record->access_flags) != 0)) {
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

ucs_status_t
uct_ib_mlx5_coco_umem_record_remove(uct_ib_mlx5_coco_state_t *state,
                                    uint32_t umem_id)
{
    size_t i;

    if (state == NULL) {
        return UCS_OK;
    }

    for (i = 0; i < state->umem_count; ++i) {
        if (state->umem_records[i].live &&
            (state->umem_records[i].umem_id == umem_id)) {
            state->umem_records[i].live = 0;
            return UCS_OK;
        }
    }

    return UCS_ERR_NO_ELEM;
}

uint64_t uct_ib_mlx5_coco_mkey_sanitize_access(uint64_t access_mask)
{
    return access_mask & ~((uint64_t)IBV_ACCESS_REMOTE_ATOMIC);
}

const uct_ib_mlx5_coco_mkey_record_t*
uct_ib_mlx5_coco_mkey_record_find_lkey(const uct_ib_mlx5_coco_state_t *state,
                                       uint32_t lkey)
{
    size_t i;

    if (state == NULL) {
        return NULL;
    }

    for (i = 0; i < state->mkey_count; ++i) {
        if (state->mkey_records[i].live &&
            (state->mkey_records[i].lkey == lkey)) {
            return &state->mkey_records[i];
        }
    }

    return NULL;
}

static const uct_ib_mlx5_coco_mkey_record_t*
uct_ib_mlx5_coco_mkey_record_find_rkey(const uct_ib_mlx5_coco_state_t *state,
                                       uint32_t rkey)
{
    size_t i;

    if (state == NULL) {
        return NULL;
    }

    for (i = 0; i < state->mkey_count; ++i) {
        if (state->mkey_records[i].live &&
            (state->mkey_records[i].rkey == rkey)) {
            return &state->mkey_records[i];
        }
    }

    return NULL;
}

static int uct_ib_mlx5_coco_range_contains(const void *record_base,
                                           size_t record_length,
                                           const void *base, size_t length)
{
    uintptr_t rec_start = (uintptr_t)record_base;
    uintptr_t req_start = (uintptr_t)base;
    uintptr_t rec_end   = rec_start + record_length;
    uintptr_t req_end   = req_start + length;

    if ((rec_end < rec_start) || (req_end < req_start)) {
        return 0;
    }

    return (req_start >= rec_start) && (req_end <= rec_end);
}

ucs_status_t
uct_ib_mlx5_coco_mkey_record_add(uct_ib_mlx5_coco_state_t *state,
                                 uint32_t lkey, uint32_t rkey, void *base,
                                 size_t length, uint64_t access_mask)
{
    uct_ib_mlx5_coco_mkey_record_t *record;
    ucs_status_t status;

    if (state == NULL) {
        return UCS_OK;
    }

    if ((length == 0) ||
        (uct_ib_mlx5_coco_mkey_record_find_lkey(state, lkey) != NULL) ||
        (uct_ib_mlx5_coco_mkey_record_find_rkey(state, rkey) != NULL)) {
        return (length == 0) ? UCS_ERR_INVALID_PARAM :
                               UCS_ERR_ALREADY_EXISTS;
    }

    status = uct_ib_mlx5_coco_grow((void**)&state->mkey_records,
                                   &state->mkey_capacity, state->mkey_count,
                                   sizeof(*state->mkey_records),
                                   "CoCo mkey records");
    if (status != UCS_OK) {
        return status;
    }

    record              = &state->mkey_records[state->mkey_count++];
    record->lkey        = lkey;
    record->rkey        = rkey;
    record->base        = base;
    record->length      = length;
    record->access_mask = uct_ib_mlx5_coco_mkey_sanitize_access(access_mask);
    record->live        = 1;
    return UCS_OK;
}

ucs_status_t
uct_ib_mlx5_coco_mkey_record_validate(const uct_ib_mlx5_coco_state_t *state,
                                      uint32_t lkey, uint32_t rkey,
                                      void *base, size_t length,
                                      uint64_t access_mask)
{
    const uct_ib_mlx5_coco_mkey_record_t *record =
            uct_ib_mlx5_coco_mkey_record_find_lkey(state, lkey);

    if ((record == NULL) || (record->rkey != rkey)) {
        return UCS_ERR_NO_ELEM;
    }

    if (!uct_ib_mlx5_coco_range_contains(record->base, record->length, base,
                                         length) ||
        ((access_mask & ~record->access_mask) != 0)) {
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

ucs_status_t
uct_ib_mlx5_coco_mkey_record_remove(uct_ib_mlx5_coco_state_t *state,
                                    uint32_t lkey, uint32_t rkey)
{
    size_t i;

    if (state == NULL) {
        return UCS_OK;
    }

    for (i = 0; i < state->mkey_count; ++i) {
        if (state->mkey_records[i].live &&
            (state->mkey_records[i].lkey == lkey) &&
            (state->mkey_records[i].rkey == rkey)) {
            state->mkey_records[i].live = 0;
            return UCS_OK;
        }
    }

    return UCS_ERR_NO_ELEM;
}

ucs_status_t
uct_ib_mlx5_coco_mkey_record_remove_rkey(uct_ib_mlx5_coco_state_t *state,
                                         uint32_t rkey)
{
    size_t i;

    if (state == NULL) {
        return UCS_OK;
    }

    for (i = 0; i < state->mkey_count; ++i) {
        if (state->mkey_records[i].live &&
            (state->mkey_records[i].rkey == rkey)) {
            state->mkey_records[i].live = 0;
            return UCS_OK;
        }
    }

    return UCS_ERR_NO_ELEM;
}

static void uct_ib_mlx5_coco_scrub(void *addr, size_t size)
{
    memset(addr, 0, size);
}

ucs_status_t
uct_ib_mlx5_coco_md_buf_alloc_shared(uct_ib_mlx5_md_t *md, size_t size,
                                     int silent, void **buf_p,
                                     uct_ib_mlx5_devx_umem_t *mem,
                                     int access_mode, char *name)
{
    const ucs_log_level_t level = silent ? UCS_LOG_LEVEL_DEBUG :
                                           UCS_LOG_LEVEL_ERROR;
    const char *alloc_name      = (name != NULL) ? name : "unknown";
    uct_ib_mlx5_coco_shared_alloc_backend_t *backend =
            &uct_ib_mlx5_coco_shared_alloc_backend;
    uct_ib_mlx5_coco_shared_alloc_t *alloc;
    ucs_status_t status;
    uint32_t umem_id;

    uct_ib_mlx5_devx_umem_reset(mem);
    *buf_p = NULL;

    alloc = ucs_calloc(1, sizeof(*alloc), "CoCo shared allocation");
    if (alloc == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    alloc->dmabuf_fd = UCT_IB_MLX5_INVALID_DMABUF_FD;
    status = uct_ib_mlx5_coco_exposed_size(size, &alloc->exposed_size);
    if (status != UCS_OK) {
        goto err_free_alloc;
    }

    alloc->requested_size = size;
    status = backend->ops.alloc(alloc->exposed_size, &alloc->addr,
                                &alloc->dmabuf_fd, backend->arg);
    if (status != UCS_OK) {
        ucs_log(level, "shared allocation for %s size %zu failed: %s",
                alloc_name, alloc->exposed_size, ucs_status_string(status));
        goto err_free_alloc;
    }

    uct_ib_mlx5_coco_scrub(alloc->addr, alloc->exposed_size);

    status = backend->ops.umem_reg(md, alloc, access_mode, &mem->mem,
                                   backend->arg);
    if (status != UCS_OK) {
        ucs_log(level, "shared DEVX UMEM registration for %s failed: %s",
                alloc_name, ucs_status_string(status));
        goto err_unmap_close;
    }

    alloc->umem_registered = 1;
    umem_id                = mem->mem->umem_id;
    status = uct_ib_mlx5_coco_umem_record_add(
            md->coco, umem_id, alloc->addr, alloc->requested_size,
            alloc->exposed_size, access_mode);
    if (status != UCS_OK) {
        goto err_umem_dereg;
    }

    mem->size        = alloc->requested_size;
    mem->mmap_size   = alloc->exposed_size;
    mem->dmabuf_fd   = alloc->dmabuf_fd;
    mem->is_dmabuf   = 1;
    mem->coco_shared = alloc;
    *buf_p           = alloc->addr;
    return UCS_OK;

err_umem_dereg:
    (void)backend->ops.umem_dereg(mem->mem, backend->arg);
err_unmap_close:
    uct_ib_mlx5_coco_scrub(alloc->addr, alloc->exposed_size);
    (void)backend->ops.unmap(alloc->addr, alloc->exposed_size, backend->arg);
    if (alloc->dmabuf_fd != UCT_IB_MLX5_INVALID_DMABUF_FD) {
        (void)backend->ops.close_fd(alloc->dmabuf_fd, backend->arg);
    }
err_free_alloc:
    uct_ib_mlx5_devx_umem_reset(mem);
    ucs_free(alloc);
    return status;
}

void uct_ib_mlx5_coco_md_buf_free_shared(uct_ib_mlx5_md_t *md, void *buf,
                                         uct_ib_mlx5_devx_umem_t *mem)
{
    uct_ib_mlx5_coco_shared_alloc_backend_t *backend =
            &uct_ib_mlx5_coco_shared_alloc_backend;
    uct_ib_mlx5_coco_shared_alloc_t *alloc = mem->coco_shared;
    struct mlx5dv_devx_umem *umem;
    size_t mmap_size;
    int dmabuf_fd;
    uint32_t umem_id;
    ucs_status_t status;

    if ((buf == NULL) || (alloc == NULL)) {
        return;
    }

    umem      = mem->mem;
    mmap_size = mem->mmap_size;
    dmabuf_fd = mem->dmabuf_fd;
    umem_id   = (umem == NULL) ? 0 : umem->umem_id;

    if (umem != NULL) {
        status = backend->ops.umem_dereg(umem, backend->arg);
        if (status != UCS_OK) {
            ucs_warn("mlx5dv_devx_umem_dereg(mem=%p) failed: %s", umem,
                     ucs_status_string(status));
        } else {
            (void)uct_ib_mlx5_coco_umem_record_remove(md->coco, umem_id);
        }
        alloc->umem_registered = 0;
    }

    alloc->quarantined = 1;
    uct_ib_mlx5_devx_umem_reset(mem);
    uct_ib_mlx5_coco_scrub(buf, mmap_size);

    status = backend->ops.unmap(buf, mmap_size, backend->arg);
    if (status != UCS_OK) {
        ucs_warn("munmap(buf=%p, len=%zu) failed: %s", buf, mmap_size,
                 ucs_status_string(status));
    }

    if (dmabuf_fd != UCT_IB_MLX5_INVALID_DMABUF_FD) {
        status = backend->ops.close_fd(dmabuf_fd, backend->arg);
        if (status != UCS_OK) {
            ucs_warn("close(dmabuf_fd=%d) failed: %s", dmabuf_fd,
                     ucs_status_string(status));
        }
    }

    ucs_free(alloc);
}
