/**
 * Copyright (C) SambaNova Systems, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_RDU_DEVMEM_H
#define UCT_RDU_DEVMEM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*
 * Declarations of the SambaNova rdu_devmem runtime functions which the RDU
 * memory domain uses. They are resolved at run time with dlsym(), as uct/cuda
 * does for NVML, so the module builds without the runtime installed.
 */

/* Return codes; only the ones the memory domain checks against are declared,
 * with the values of the runtime library. */
typedef enum {
    UCT_RDU_DEVMEM_FAILURE       = -1, /* call did not succeed */
    UCT_RDU_DEVMEM_SUCCESS       = 0, /* call succeeded */
    UCT_RDU_DEVMEM_ENOTSUPPORTED = 4 /* the driver cannot export dma-buf */
} uct_rdu_devmem_status_t;

/* Memory kinds of rdu_malloc() */
typedef enum {
    UCT_RDU_DEVMEM_MEM_HBM = 0,
    UCT_RDU_DEVMEM_MEM_DDR = 1
} uct_rdu_devmem_mem_kind_t;

/* Callback for rdu_foreach_alloc(): invoked once per live RDU allocation. */
typedef void (*uct_rdu_devmem_foreach_cb_t)(const void *address, size_t size,
                                            uint32_t rdu_id, void *arg);

/* Function-pointer table resolved from librdu_devmem.so at run time. */
typedef struct uct_rdu_devmem_ops {
    uct_rdu_devmem_status_t (*detect)(const void *address, uint32_t *rdu_id_p,
                                      uint64_t *dev_addr_p);
    uct_rdu_devmem_status_t (*get_pcie_path)(uint32_t rdu_id, char *buf,
                                             size_t buflen);
    void (*foreach_alloc)(uct_rdu_devmem_foreach_cb_t cb, void *arg);
    uct_rdu_devmem_status_t (*dmabuf_export)(const void *address, size_t size,
                                             int *fd_p, uint64_t *offset_p);
    bool (*dmabuf_export_supported)(void);
    /* Used only to allocate memory, as perftest does; the memory domain only
     * looks up memory which the application allocated */
    int (*init)(void);
    void *(*malloc)(uint32_t rdu_id, size_t size, int mem_kind);
    void (*free)(uint32_t rdu_id, void *address);
} uct_rdu_devmem_ops_t;

/*
 * Return the resolved rdu_devmem ops, loading the runtime library on first
 * call, or NULL if the runtime is not available.  Idempotent and thread-safe.
 */
const uct_rdu_devmem_ops_t *uct_rdu_devmem_get(void);

/*
 * Return nonzero if RDU memory can be exported as a dma-buf on this host. This
 * is decided once, when the runtime library is loaded, and does not change.
 */
int uct_rdu_devmem_dmabuf_supported(void);

/*
 * Find the RDU allocation which contains an address. Return nonzero and set
 * the start and length of the allocation if the runtime knows it.
 */
int uct_rdu_devmem_find_alloc(const void *address, const void **base_p,
                              size_t *length_p);

/*
 * Thin wrappers so call sites stay SDK-agnostic: they dispatch through the
 * dlopen'd ops and degrade gracefully (FAILURE / no-op) when the runtime is
 * absent, which keeps the memory domain inert rather than crashing.
 */
static inline uct_rdu_devmem_status_t
uct_rdu_devmem_detect(const void *address, uint32_t *rdu_id_p,
                      uint64_t *dev_addr_p)
{
    const uct_rdu_devmem_ops_t *ops = uct_rdu_devmem_get();
    return (ops != NULL) ? ops->detect(address, rdu_id_p, dev_addr_p) :
                           UCT_RDU_DEVMEM_FAILURE;
}

static inline uct_rdu_devmem_status_t
uct_rdu_devmem_get_pcie_path(uint32_t rdu_id, char *buf, size_t buflen)
{
    const uct_rdu_devmem_ops_t *ops = uct_rdu_devmem_get();
    return (ops != NULL) ? ops->get_pcie_path(rdu_id, buf, buflen) :
                           UCT_RDU_DEVMEM_FAILURE;
}

static inline void
uct_rdu_devmem_foreach_alloc(uct_rdu_devmem_foreach_cb_t cb, void *arg)
{
    const uct_rdu_devmem_ops_t *ops = uct_rdu_devmem_get();
    if (ops != NULL) {
        ops->foreach_alloc(cb, arg);
    }
}

static inline uct_rdu_devmem_status_t
uct_rdu_devmem_dmabuf_export(const void *address, size_t size, int *fd_p,
                             uint64_t *offset_p)
{
    const uct_rdu_devmem_ops_t *ops = uct_rdu_devmem_get();
    /* dmabuf_export is optional: it is absent on runtimes without a dma-buf
     * exporter */
    return ((ops != NULL) && (ops->dmabuf_export != NULL)) ?
                   ops->dmabuf_export(address, size, fd_p, offset_p) :
                   UCT_RDU_DEVMEM_FAILURE;
}

#endif /* UCT_RDU_DEVMEM_H */
