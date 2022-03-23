/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_WORKER_H_
#define UCP_WORKER_H_

#include "ucp_ep.h"
#include "ucp_context.h"
#include "ucp_thread.h"
#include "ucp_rkey.h"

#include <ucp/core/ucp_am.h>
#include <ucp/tag/tag_match.h>
#include <ucs/datastruct/mpool.h>
#include <ucs/datastruct/mpool_set.h>
#include <ucs/datastruct/queue_types.h>
#include <ucs/datastruct/strided_alloc.h>
#include <ucs/datastruct/conn_match.h>
#include <ucs/datastruct/ptr_map.h>
#include <ucs/arch/bitops.h>


/* The size of the private buffer in UCT descriptor headroom, which UCP may
 * use for its own needs. This size does not include ucp_recv_desc_t length,
 * because it is common for all cases and protocols (TAG, STREAM). */
#define UCP_WORKER_HEADROOM_PRIV_SIZE 32


#define UCP_WORKER_HEADROOM_SIZE \
    (sizeof(ucp_recv_desc_t) + UCP_WORKER_HEADROOM_PRIV_SIZE)


#define UCP_WORKER_THREAD_CS_CHECK_IS_BLOCKED(_worker) \
    ucs_assert(ucs_async_is_blocked(&(_worker)->async))


#if ENABLE_MT

#define UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(_worker) \
    do { \
        if ((_worker)->flags & UCP_WORKER_FLAG_THREAD_MULTI) { \
            UCS_ASYNC_BLOCK(&(_worker)->async); \
        } \
    } while (0)


#define UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(_worker) \
    do { \
        if ((_worker)->flags & UCP_WORKER_FLAG_THREAD_MULTI) { \
            UCS_ASYNC_UNBLOCK(&(_worker)->async); \
        } \
    } while (0)


#define UCP_WORKER_THREAD_CS_CHECK_IS_BLOCKED_CONDITIONAL(_worker) \
    do { \
        if ((_worker)->flags & UCP_WORKER_FLAG_THREAD_MULTI) { \
            UCP_WORKER_THREAD_CS_CHECK_IS_BLOCKED(_worker); \
        } \
    } while (0)

#else

#define UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(_worker)
#define UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(_worker)
#define UCP_WORKER_THREAD_CS_CHECK_IS_BLOCKED_CONDITIONAL(_worker)

#endif


/**
 * UCP worker flags
 */
enum {
    /** Internal worker flags start from this bit index, to co-exist with user
     * flags specified when worker is created */
    UCP_WORKER_INTERNAL_FLAGS_SHIFT = 32,

    /** The worker can be accessed from multiple threads at the same time, so
        locking is required */
    UCP_WORKER_FLAG_THREAD_MULTI =
            UCS_BIT(UCP_WORKER_INTERNAL_FLAGS_SHIFT + 0),

    /** The worker can be accessed from multiple threads, but only by one thread
        at a time, so locking is not required, but IO flush may be required.
        This flag is mutually exclusive with UCP_WORKER_FLAG_THREAD_MULTI. */
    UCP_WORKER_FLAG_THREAD_SERIALIZED =
            UCS_BIT(UCP_WORKER_INTERNAL_FLAGS_SHIFT + 1),

    /** Events are edge-triggered */
    UCP_WORKER_FLAG_EDGE_TRIGGERED =
            UCS_BIT(UCP_WORKER_INTERNAL_FLAGS_SHIFT + 2),

    /** Worker event fd is external */
    UCP_WORKER_FLAG_EXTERNAL_EVENT_FD =
            UCS_BIT(UCP_WORKER_INTERNAL_FLAGS_SHIFT + 3),

    /** Indicates that AM mpool was initialized on this worker */
    UCP_WORKER_FLAG_AM_MPOOL_INITIALIZED =
            UCS_BIT(UCP_WORKER_INTERNAL_FLAGS_SHIFT + 4)
};


/**
 * UCP iface flags
 */
enum {
    UCP_WORKER_IFACE_FLAG_OFFLOAD_ACTIVATED = UCS_BIT(0), /**< UCP iface receives tag
                                                               offload messages */
    UCP_WORKER_IFACE_FLAG_ON_ARM_LIST       = UCS_BIT(1), /**< UCP iface is an element
                                                               of arm_ifaces list, so
                                                               it needs to be armed
                                                               in ucp_worker_arm(). */
    UCP_WORKER_IFACE_FLAG_UNUSED            = UCS_BIT(2)  /**< There is another UCP iface
                                                               with the same caps, but
                                                               with better performance */
};


/**
 * UCP worker statistics counters
 */
enum {
    /* Total number of received eager messages */
    UCP_WORKER_STAT_TAG_RX_EAGER_MSG,
    UCP_WORKER_STAT_TAG_RX_EAGER_SYNC_MSG,

    /* Total number of  received eager chunks (every message
     * can be split into a bunch of chunks). It is possible that
     * some chunks  of the message arrived unexpectedly and then
     * receive had been posted and the rest arrived expectedly */
    UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_EXP,
    UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_UNEXP,

    UCP_WORKER_STAT_TAG_RX_RNDV_EXP,
    UCP_WORKER_STAT_TAG_RX_RNDV_UNEXP,

    UCP_WORKER_STAT_TAG_RX_RNDV_GET_ZCOPY,
    UCP_WORKER_STAT_TAG_RX_RNDV_SEND_RTR,
    UCP_WORKER_STAT_TAG_RX_RNDV_RKEY_PTR,

    UCP_WORKER_STAT_LAST
};


/**
 * UCP worker tag offload statistics counters
 */
enum {
    UCP_WORKER_STAT_TAG_OFFLOAD_POSTED,
    UCP_WORKER_STAT_TAG_OFFLOAD_MATCHED,
    UCP_WORKER_STAT_TAG_OFFLOAD_MATCHED_SW_RNDV,
    UCP_WORKER_STAT_TAG_OFFLOAD_CANCELED,
    UCP_WORKER_STAT_TAG_OFFLOAD_BLOCK_TAG_EXCEED,
    UCP_WORKER_STAT_TAG_OFFLOAD_BLOCK_NON_CONTIG,
    UCP_WORKER_STAT_TAG_OFFLOAD_BLOCK_WILDCARD,
    UCP_WORKER_STAT_TAG_OFFLOAD_BLOCK_SW_PEND,
    UCP_WORKER_STAT_TAG_OFFLOAD_BLOCK_NO_IFACE,
    UCP_WORKER_STAT_TAG_OFFLOAD_BLOCK_MEM_REG,
    UCP_WORKER_STAT_TAG_OFFLOAD_RX_UNEXP_EGR,
    UCP_WORKER_STAT_TAG_OFFLOAD_RX_UNEXP_RNDV,
    UCP_WORKER_STAT_TAG_OFFLOAD_RX_UNEXP_SW_RNDV,
    UCP_WORKER_STAT_TAG_OFFLOAD_LAST
};


#define UCP_WORKER_STAT_EAGER_MSG(_worker, _flags) \
    UCS_STATS_UPDATE_COUNTER((_worker)->stats, \
                             ((_flags) & UCP_RECV_DESC_FLAG_EAGER_SYNC) ? \
                             UCP_WORKER_STAT_TAG_RX_EAGER_SYNC_MSG : \
                             UCP_WORKER_STAT_TAG_RX_EAGER_MSG, 1);

#define UCP_WORKER_STAT_EAGER_CHUNK(_worker, _is_exp) \
    UCS_STATS_UPDATE_COUNTER((_worker)->stats, \
                             UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_##_is_exp, 1);

#define UCP_WORKER_STAT_RNDV(_worker, _is_exp, _value) \
    UCS_STATS_UPDATE_COUNTER((_worker)->stats, \
                             UCP_WORKER_STAT_TAG_RX_RNDV_##_is_exp, _value);

#define UCP_WORKER_STAT_TAG_OFFLOAD(_worker, _name) \
    UCS_STATS_UPDATE_COUNTER((_worker)->tm_offload_stats, \
                             UCP_WORKER_STAT_TAG_OFFLOAD_##_name, 1);

#define ucp_worker_mpool_get(_mp) \
    ({ \
        ucp_mem_desc_t *_rdesc = ucs_mpool_get_inline(_mp); \
        if (_rdesc != NULL) { \
            VALGRIND_MAKE_MEM_DEFINED(_rdesc, sizeof(*_rdesc)); \
        } \
        _rdesc; \
    })


/* Hash map to find rkey config index by rkey config key, for fast rkey unpack */
KHASH_TYPE(ucp_worker_rkey_config, ucp_rkey_config_key_t, ucp_worker_cfg_index_t);
typedef khash_t(ucp_worker_rkey_config) ucp_worker_rkey_config_hash_t;


/* Hash map of UCT EPs that are being discarded on UCP Worker */
KHASH_TYPE(ucp_worker_discard_uct_ep_hash, uct_ep_h, ucp_request_t*);
typedef khash_t(ucp_worker_discard_uct_ep_hash) ucp_worker_discard_uct_ep_hash_t;


typedef struct ucp_worker_mpool_key {
    ucs_memory_type_t mem_type;  /* memory type of the buffer pool */
    ucs_sys_device_t  sys_dev;   /* identifier for the device,
                                    UINT_MAX for default device */
} ucp_worker_mpool_key_t;


/* Hash map to find mpool by mpool key */
KHASH_TYPE(ucp_worker_mpool_hash, ucp_worker_mpool_key_t, ucs_mpool_t);
typedef khash_t(ucp_worker_mpool_hash) ucp_worker_mpool_hash_t;

/**
 * UCP worker iface, which encapsulates UCT iface, its attributes and
 * some auxiliary info needed for tag matching offloads.
 */
struct ucp_worker_iface {
    uct_iface_h                   iface;         /* UCT interface */
    uct_iface_attr_t              attr;          /* UCT interface attributes */
    ucp_worker_h                  worker;        /* The parent worker */
    ucs_list_link_t               arm_list;      /* Element in arm_ifaces list */
    ucp_rsc_index_t               rsc_index;     /* Resource index */
    int                           event_fd;      /* Event FD, or -1 if undefined */
    unsigned                      activate_count;/* How many times this iface has
                                                    been activated */
    int                           check_events_id;/* Callback id for check_events */
    unsigned                      proxy_recv_count;/* Counts active messages on proxy handler */
    unsigned                      post_count;    /* Counts uncompleted requests which are
                                                    offloaded to the transport */
    uint8_t                       flags;         /* Interface flags */
};


/**
 * UCP worker CM, which encapsulates UCT CM and its auxiliary info.
 */
struct ucp_worker_cm {
    uct_cm_h                      cm;            /* UCT CM handle */
    uct_cm_attr_t                 attr;          /* UCT CM attributes */
    ucp_rsc_index_t               cmpt_idx;      /* Index of corresponding
                                                    component */
};


UCS_PTR_MAP_TYPE(ep, 1);
UCS_PTR_MAP_TYPE(request, 0);


/**
 * UCP worker (thread context).
 */
typedef struct ucp_worker {
    uint64_t                         flags;               /* Worker flags */
    ucs_async_context_t              async;               /* Async context for this worker */
    ucp_context_h                    context;             /* Back-reference to UCP context */
    uint64_t                         uuid;                /* Unique ID for wireup */
    uint64_t                         client_id;           /* Worker client id for wireup */
    uct_worker_h                     uct;                 /* UCT worker handle */
    ucs_mpool_t                      req_mp;              /* Memory pool for requests */
    ucs_mpool_t                      rkey_mp;             /* Pool for small memory keys */
    ucp_tl_bitmap_t                  atomic_tls;          /* Which resources can be used for atomics */

    int                              inprogress;
    /* Worker name for tracing and analysis */
    char                             name[UCP_ENTITY_NAME_MAX];
    /* Worker address name composed of host name and process id */
    char                             address_name[UCP_WORKER_ADDRESS_NAME_MAX];

    unsigned                         flush_ops_count;     /* Number of pending operations */

    int                              event_fd;            /* Allocated (on-demand) event fd for wakeup */
    ucs_sys_event_set_t              *event_set;          /* Allocated UCS event set for wakeup */
    int                              eventfd;             /* Event fd to support signal() calls */
    unsigned                         uct_events;          /* UCT arm events */
    ucs_list_link_t                  arm_ifaces;          /* List of interfaces to arm */

    void                             *user_data;          /* User-defined data */
    ucs_strided_alloc_t              ep_alloc;            /* Endpoint allocator */
    ucs_list_link_t                  stream_ready_eps;    /* List of EPs with received stream data */
    unsigned                         num_all_eps;         /* Number of all endpoints (except internal
                                                           * endpoints) */
    ucs_list_link_t                  all_eps;             /* List of all endpoints (except internal
                                                           * endpoints) */
    ucs_list_link_t                  internal_eps;        /* List of internal endpoints */
    ucs_conn_match_ctx_t             conn_match_ctx;      /* Endpoint-to-endpoint matching context */
    ucp_worker_iface_t               **ifaces;            /* Array of pointers to interfaces,
                                                             one for each resource */
    unsigned                         num_ifaces;          /* Number of elements in ifaces array  */
    unsigned                         num_active_ifaces;   /* Number of activated ifaces  */
    ucp_tl_bitmap_t                  scalable_tl_bitmap;  /* Map of scalable tl resources */
    ucp_worker_cm_t                  *cms;                /* Array of CMs, one for each component */
    ucs_mpool_set_t                  am_mps;              /* Memory pool set for AM receives */
    ucs_mpool_t                      reg_mp;              /* Registered memory pool */
    ucp_worker_mpool_hash_t          mpool_hash;          /* Hash table of memory pools */
    ucs_queue_head_t                 rkey_ptr_reqs;       /* Queue of submitted RKEY PTR requests that
                                                           * are in-progress */
    uct_worker_cb_id_t               rkey_ptr_cb_id;      /* RKEY PTR worker callback queue ID */
    ucp_tag_match_t                  tm;                  /* Tag-matching queues and offload info */
    ucp_am_info_t                    am;                  /* Array of AM callbacks and their data */
    uint64_t                         am_message_id;       /* For matching long AMs */
    ucp_ep_h                         mem_type_ep[UCS_MEMORY_TYPE_LAST]; /* Memory type EPs */

    UCS_STATS_NODE_DECLARE(stats)
    UCS_STATS_NODE_DECLARE(tm_offload_stats)

    ucs_cpu_set_t                    cpu_mask;            /* Save CPU mask for subsequent calls to
                                                             ucp_worker_listen */

    ucp_worker_rkey_config_hash_t    rkey_config_hash;    /* RKEY config key -> index */
    ucp_worker_discard_uct_ep_hash_t discard_uct_ep_hash; /* Hash of discarded UCT EPs */
    UCS_PTR_MAP_T(ep)                ep_map;              /* UCP ep key to ptr
                                                             mapping */
    UCS_PTR_MAP_T(request)           request_map;         /* UCP requests key to
                                                             ptr mapping */

    unsigned                         ep_config_count;     /* Current number of ep configurations */
    ucp_ep_config_t                  ep_config[UCP_WORKER_MAX_EP_CONFIG];

    unsigned                         rkey_config_count;   /* Current number of rkey configurations */
    ucp_rkey_config_t                rkey_config[UCP_WORKER_MAX_RKEY_CONFIG];

    struct {
        int                          timerfd;             /* Timer needed to signal to user's fd when
                                                           * the next keepalive round must be done */
        uct_worker_cb_id_t           cb_id;               /* Keepalive callback id */
        ucs_time_t                   last_round;          /* Last round timestamp */
        ucs_list_link_t              *iter;               /* Last EP processed keepalive */
        unsigned                     ep_count;            /* Number of EPs processed in current time slot */
        unsigned                     iter_count;          /* Number of progress iterations to skip,
                                                           * used to minimize call of ucs_get_time */
        size_t                       round_count;         /* Number of rounds done */
    } keepalive;

    struct {
        /* Number of requests to create endpoint */
        uint64_t                     ep_creations;
        /* Number of failed requests to create endpoint */
        uint64_t                     ep_creation_failures;
        /* Number of endpoint closures */
        uint64_t                     ep_closures;
        /* Number of failed endpoints */
        uint64_t                     ep_failures;
    } counters;
} ucp_worker_t;


ucs_status_t ucp_worker_get_ep_config(ucp_worker_h worker,
                                      const ucp_ep_config_key_t *key,
                                      unsigned ep_init_flags,
                                      ucp_worker_cfg_index_t *cfg_index_p);

ucs_status_t
ucp_worker_add_rkey_config(ucp_worker_h worker,
                           const ucp_rkey_config_key_t *key,
                           const ucs_sys_dev_distance_t *lanes_distance,
                           ucp_worker_cfg_index_t *cfg_index_p);

ucs_status_t ucp_worker_iface_open(ucp_worker_h worker, ucp_rsc_index_t tl_id,
                                   uct_iface_params_t *iface_params,
                                   ucp_worker_iface_t **wiface);

ucs_status_t ucp_worker_iface_init(ucp_worker_h worker, ucp_rsc_index_t tl_id,
                                   ucp_worker_iface_t *wiface);

void ucp_worker_iface_cleanup(ucp_worker_iface_t *wiface);

void ucp_worker_iface_progress_ep(ucp_worker_iface_t *wiface);

void ucp_worker_iface_unprogress_ep(ucp_worker_iface_t *wiface);

void ucp_worker_signal_internal(ucp_worker_h worker);

void ucp_worker_iface_activate(ucp_worker_iface_t *wiface, unsigned uct_flags);

void ucp_worker_keepalive_add_ep(ucp_ep_h );

/* EP should be removed from worker all_eps prior to call this function */
void ucp_worker_keepalive_remove_ep(ucp_ep_h ep);

/* must be called with async lock held */
int ucp_worker_is_uct_ep_discarding(ucp_worker_h worker, uct_ep_h uct_ep);

/* must be called with async lock held */
ucs_status_t ucp_worker_discard_uct_ep(ucp_ep_h ucp_ep, uct_ep_h uct_ep,
                                       ucp_rsc_index_t rsc_index,
                                       unsigned ep_flush_flags,
                                       uct_pending_purge_callback_t purge_cb,
                                       void *purge_arg,
                                       ucp_send_nbx_callback_t discarded_cb,
                                       void *discarded_cb_arg);

void ucp_worker_vfs_refresh(void *obj);

ucs_status_t ucp_worker_discard_uct_ep_pending_cb(uct_pending_req_t *self);

unsigned ucp_worker_discard_uct_ep_progress(void *arg);

static UCS_F_ALWAYS_INLINE void
ucp_worker_flush_ops_count_inc(ucp_worker_h worker)
{
    UCP_WORKER_THREAD_CS_CHECK_IS_BLOCKED_CONDITIONAL(worker);
    ucs_assert(worker->flush_ops_count < UINT_MAX);
    ++worker->flush_ops_count;
}

/* must be called with async lock held */
static UCS_F_ALWAYS_INLINE void
ucp_worker_flush_ops_count_dec(ucp_worker_h worker)
{
    UCP_WORKER_THREAD_CS_CHECK_IS_BLOCKED_CONDITIONAL(worker);
    ucs_assert(worker->flush_ops_count > 0);
    --worker->flush_ops_count;
}

#endif
