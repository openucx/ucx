/**
 * Copyright (c) UT-Battelle, LLC. 2014-2017. ALL RIGHTS RESERVED.
 * Copyright (c) Los Alamos National Security, LLC. 2018. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_UGNI_TYPES_H
#define UCT_UGNI_TYPES_H

#include "ugni_def.h"
#include <uct/base/uct_md.h>
#include <ucs/datastruct/arbiter.h>
#include <gni_pub.h>
#include <stdbool.h>

typedef struct uct_ugni_device {
    gni_nic_device_t type;                      /**< Device type */
    char             type_name[UCT_UGNI_MAX_TYPE_NAME];  /**< Device type name */
    char             fname[UCT_DEVICE_NAME_MAX];/**< Device full name */
    uint32_t         device_id;                 /**< Device id */
    uint32_t         address;                   /**< Device address */
    uint32_t         cpu_id;                    /**< CPU attached directly
                                                  to the device */
    cpu_set_t        cpu_mask;                  /**< CPU mask */
    /* TBD - reference counter */
} uct_ugni_device_t;

typedef struct uct_ugni_cdm {
    gni_cdm_handle_t   cdm_handle; /**< Ugni communication domain */
    gni_nic_handle_t   nic_handle; /**< Ugni NIC handle */
    uct_ugni_device_t *dev;        /**< Ugni device the cdm is connected to */
    ucs_thread_mode_t  thread_mode;
    uint32_t           address; 
    uint16_t           domain_id;

#if ENABLE_MT
    ucs_spinlock_t   lock;                      /**< Device lock */
#endif
} uct_ugni_cdm_t;

/**
 * @brief UGNI Memory domain
 *
 * Ugni does not define MD, instead I use
 * device handle that "simulates" the MD.
 * Memory that is registered with one device handle
 * can be accessed with any other.
 */
typedef struct uct_ugni_md {
    uct_md_t super;         /**< Domain info */
    uct_ugni_cdm_t cdm;     /**< Communication domain for memory registration*/
    int ref_count;          /**< UGNI Domain ref count */
} uct_ugni_md_t;

typedef struct uct_devaddr_ugni_t {
    uint32_t nic_addr;
} UCS_S_PACKED uct_devaddr_ugni_t;

typedef struct uct_sockaddr_ugni {
     uint16_t   domain_id;
} UCS_S_PACKED uct_sockaddr_ugni_t;

typedef struct uct_ugni_flush_group {
    uct_completion_t flush_comp;         /**< Completion for outstanding requests 
                                                flush_comp.count is used to track outstanding sends*/
    uct_completion_t *user_comp;         /**< User completion struct */
    struct uct_ugni_flush_group *parent; /**< Used to signal the next flush_group that this group is done*/
} uct_ugni_flush_group_t;

typedef struct uct_ugni_ep {
    uct_base_ep_t           super;
    gni_ep_handle_t         ep;           /**< Endpoint for ugni api */
    uct_ugni_flush_group_t  *flush_group; /**< Flush group new sends are added to */
    uint32_t                hash_key;     /**< Hash to look up EPs with */
    uint32_t                arb_sched;    /**< Flag to make sure we don't recursively block sends*/
    ucs_arbiter_group_t     arb_group;    /**< Our group in the pending send arbiter */
    struct uct_ugni_ep      *next;
} uct_ugni_ep_t;

typedef struct uct_ugni_iface {
    uct_base_iface_t        super;
    uct_ugni_cdm_t          cdm;                         /**< Ugni communication domain and handles */
    gni_cq_handle_t         local_cq;                    /**< Completion queue */
    uct_ugni_ep_t           *eps[UCT_UGNI_HASH_SIZE];    /**< Array of QPs */
    unsigned                outstanding;                 /**< Counter for outstanding packets
                                                              on the interface */
    ucs_arbiter_t           arbiter;                     /**< arbiter structure for pending operations */
    ucs_mpool_t             flush_pool;                  /**< Memory pool for flush objects */
} uct_ugni_iface_t;

typedef struct uct_ugni_iface_config {
    uct_iface_config_t       super;
    uct_iface_mpool_config_t mpool;
} uct_ugni_iface_config_t;

#endif
