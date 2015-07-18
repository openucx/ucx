/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#ifndef UCT_UGNI_IFACE_H
#define UCT_UGNI_IFACE_H

#include "ugni_pd.h"
#include "ugni_device.h"
#include "ugni_ep.h"

#include <uct/tl/tl_base.h>


#define UCT_UGNI_HASH_SIZE   (256)
#define UCT_UGNI_MAX_FMA     (2048)
#define UCT_UGNI_MAX_RDMA    (512*1024*1024);

struct uct_ugni_iface;


typedef struct uct_ugni_iface {
    uct_base_iface_t        super;
    uct_ugni_device_t       *dev;
    gni_cdm_handle_t        cdm_handle;                  /**< Ugni communication domain */
    gni_nic_handle_t        nic_handle;                  /**< Ugni NIC handle */
    uint32_t                pe_address;                  /**< PE address for the NIC that this
                                                              function has attached to the
                                                              communication domain. */
    uint32_t                nic_addr;                    /**< PE address that is returned for the
                                                              communication domain that this NIC
                                                              is attached to. */
    gni_cq_handle_t         local_cq;                    /**< Completion queue */
    int                     domain_id;                   /**< Id for UGNI domain creation */
    uct_ugni_ep_t           *eps[UCT_UGNI_HASH_SIZE];    /**< Array of QPs */
    unsigned                outstanding;                 /**< Counter for outstanding packets
                                                              on the interface */
    ucs_mpool_h             free_desc;                   /**< Pool of FMA descriptors for
                                                              requests without bouncing buffers */
    ucs_mpool_h             free_desc_get;               /**< Pool of FMA descriptors for
                                                              unaligned get requests without
                                                              bouncing buffers */
    ucs_mpool_h             free_desc_buffer;            /**< Pool of FMA descriptors for
                                                              requests with bouncing buffer*/
    ucs_mpool_h             free_desc_famo;              /**< Pool of FMA descriptors for
                                                              64/32 bit fetched-atomic operations
                                                              (registered memory) */
    ucs_mpool_h             free_desc_get_buffer;        /**< Pool of FMA descriptors for
                                                              FMA_SIZE fetch operations
                                                              (registered memory) */
    struct {
        unsigned            fma_seg_size;                /**< FMA Segment size */
        unsigned            rdma_max_size;               /**< Max RDMA size */
    } config;
    bool                    activated;                   /**< nic status */
    /* list of ep */
} uct_ugni_iface_t;

typedef struct uct_ugni_iface_config {
    uct_iface_config_t       super;
    uct_iface_mpool_config_t mpool;
} uct_ugni_iface_config_t;

typedef struct uct_ugni_base_desc {
    gni_post_descriptor_t desc;
    uct_completion_t *comp_cb;
    uct_unpack_callback_t unpack_cb;
    uct_ugni_ep_t  *ep;
    int not_ready_to_free;
} uct_ugni_base_desc_t;

typedef struct uct_ugni_fetch_desc {
    uct_ugni_base_desc_t super;
    uct_completion_t tmp;
    uct_completion_t *orig_comp_cb;
    size_t padding;

    /* Handling unalined composed get messages */
    size_t expected_bytes;          /**< Number of bytes expected to be delivered
                                         including padding */
    size_t network_completed_bytes; /**< Total number of delivered bytes */
    struct uct_ugni_fetch_desc* head; /**< Pointer to the head descriptor
                                         that manages the completion of the operation */
    void *user_buffer;              /**< Pointer to user's buffer, here to ensure it's always available for composed messages */
    size_t tail;                    /**< Tail parameter to specify how many bytes at the end of a fma/rdma are garbage*/
} uct_ugni_fetch_desc_t;

ucs_status_t ugni_activate_iface(uct_ugni_iface_t *iface);

ucs_status_t uct_ugni_init_nic(int device_index,
                               int *domain_id,
                               gni_cdm_handle_t *cdm_handle,
                               gni_nic_handle_t *nic_handle,
                               uint32_t *address);

extern ucs_config_field_t uct_ugni_iface_config_table[];

static inline uct_ugni_device_t * uct_ugni_iface_device(uct_ugni_iface_t *iface)
{
    return iface->dev;
}
#endif
