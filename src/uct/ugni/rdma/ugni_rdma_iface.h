/**
 * Copyright (c) UT-Battelle, LLC. 2014-2017. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_UGNI_RDMA_IFACE_H
#define UCT_UGNI_RDMA_IFACE_H

#include <uct/ugni/base/ugni_types.h>
#include <uct/ugni/base/ugni_iface.h>
#include <uct/base/uct_iface.h>

#define UCT_UGNI_RDMA_TL_NAME  "ugni_rdma"
#define UCT_UGNI_MAX_FMA       2048
#define UCT_UGNI_MAX_RDMA      (512*1024*1024);

typedef struct uct_ugni_rdma_iface {
    uct_ugni_iface_t        super;                       /**< Super type */
    ucs_mpool_t             free_desc;                   /**< Pool of FMA descriptors for
                                                              requests without bouncing buffers */
    ucs_mpool_t             free_desc_get;               /**< Pool of FMA descriptors for
                                                              unaligned get requests without
                                                              bouncing buffers */
    ucs_mpool_t             free_desc_buffer;            /**< Pool of FMA descriptors for
                                                              requests with bouncing buffer*/
    ucs_mpool_t             free_desc_famo;              /**< Pool of FMA descriptors for
                                                              64/32 bit fetched-atomic operations
                                                              (registered memory) */
    ucs_mpool_t             free_desc_get_buffer;        /**< Pool of FMA descriptors for
                                                              FMA_SIZE fetch operations
                                                              (registered memory) */
    struct {
        unsigned            fma_seg_size;                /**< FMA Segment size */
        unsigned            rdma_max_size;               /**< Max RDMA size */
    } config;
} uct_ugni_rdma_iface_t;

typedef struct uct_ugni_rdma_iface_config {
    uct_iface_config_t       super;
    uct_iface_mpool_config_t mpool;
} uct_ugni_rdma_iface_config_t;

typedef void (*ugni_desc_free_cb_t)(void *desc);

typedef struct uct_ugni_base_desc {
    gni_post_descriptor_t   desc;
    uct_completion_t       *comp_cb;
    uct_unpack_callback_t   unpack_cb;
    uct_ugni_flush_group_t *flush_group;
    ugni_desc_free_cb_t     free_cb;
} uct_ugni_base_desc_t;

typedef struct uct_ugni_rdma_fetch_desc {
    uct_ugni_base_desc_t super;
    uct_completion_t tmp;
    uct_completion_t *orig_comp_cb;
    size_t padding;

    /* Handling unalined composed get messages */
    struct uct_ugni_rdma_fetch_desc* head; /**< Pointer to the head descriptor
                                         that manages the completion of the operation */
    void *user_buffer;              /**< Pointer to user's buffer, here to ensure it's always available for composed messages */
    size_t tail;                    /**< Tail parameter to specify how many bytes at the end of a fma/rdma are garbage*/
} uct_ugni_rdma_fetch_desc_t;

#endif
