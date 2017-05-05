/**
 * Copyright (c) UT-Battelle, LLC. 2014-2017. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_UGNI_TYPES_H
#define UCT_UGNI_TYPES_H

#include "ugni_def.h"
#include <uct/base/uct_md.h>
#include <ucs/datastruct/arbiter.h>
#include <gni_pub.h>
#include <stdbool.h>

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
    gni_cdm_handle_t cdm_handle; /**< Ugni communication domain */
    gni_nic_handle_t nic_handle; /**< Ugni NIC handle */
    uint32_t address;            /**< UGNI address */
    int ref_count;               /**< UGNI Domain ref count */
} uct_ugni_md_t;

typedef struct uct_ugni_device {
    gni_nic_device_t type;                      /**< Device type */
    char             type_name[UCT_UGNI_MAX_TYPE_NAME];  /**< Device type name */
    char             fname[UCT_DEVICE_NAME_MAX];/**< Device full name */
    uint32_t         device_id;                 /**< Device id */
    int              device_index;              /**< Index of the device in the
                                                  array of devices */
    uint32_t         address;                   /**< Device address */
    uint32_t         cpu_id;                    /**< CPU attached directly
                                                  to the device */
    cpu_set_t        cpu_mask;                  /**< CPU mask */
    bool             attached;                  /**< device was attached */
    /* TBD - reference counter */
} uct_ugni_device_t;

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
    uct_ugni_device_t       *dev;                        /**< Ugni device iface is connected to */
    gni_cdm_handle_t        cdm_handle;                  /**< Ugni communication domain */
    gni_nic_handle_t        nic_handle;                  /**< Ugni NIC handle */
    gni_cq_handle_t         local_cq;                    /**< Completion queue */
    uint16_t                domain_id;                   /**< Id for UGNI domain creation */
    uct_ugni_ep_t           *eps[UCT_UGNI_HASH_SIZE];    /**< Array of QPs */
    unsigned                outstanding;                 /**< Counter for outstanding packets
                                                              on the interface */
    bool                    activated;                   /**< nic status */
    ucs_arbiter_t           arbiter;                     /**< arbiter structure for pending operations */
    ucs_mpool_t             flush_pool;                  /**< Memory pool for flush objects */
} uct_ugni_iface_t;

typedef struct uct_ugni_base_desc {
    gni_post_descriptor_t desc;
    uct_completion_t *comp_cb;
    uct_unpack_callback_t unpack_cb;
    uct_ugni_flush_group_t *flush_group;
    int not_ready_to_free;
} uct_ugni_base_desc_t;

typedef struct uct_ugni_iface_config {
    uct_iface_config_t       super;
    uct_iface_mpool_config_t mpool;
} uct_ugni_iface_config_t;

#endif
