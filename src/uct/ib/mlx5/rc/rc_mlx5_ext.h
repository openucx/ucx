/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_RC_MLX5_EXT_H_
#define UCT_RC_MLX5_EXT_H_

#include <uct/api/uct.h>
#include <ucs/datastruct/callbackq.h>
#include <ucs/time/time.h>

#include <stdint.h>

BEGIN_C_DECLS


struct ibv_context;
struct ibv_pd;


#define UCT_RC_MLX5_EXT_API_VERSION 2


/**
 * Requirements which have to be known before the RC mlx5 interface is
 * created.
 */
enum uct_rc_mlx5_ext_requirement {
    /** Configure the existing RC TX CQ to report completion timestamps. */
    UCT_RC_MLX5_EXT_REQ_TX_CQE_TIMESTAMP = UCS_BIT(0)
};


typedef struct uct_rc_mlx5_ext_provider *uct_rc_mlx5_ext_provider_h;
typedef struct uct_rc_mlx5_ext_periodic *uct_rc_mlx5_ext_periodic_h;


/**
 * Completion callback for a timestamped SGL operation.
 *
 * The callback is invoked once for every requested checkpoint after a
 * successful post. On endpoint purge, outstanding checkpoints are completed
 * with UCS_ERR_CANCELED and a zero timestamp.
 *
 * @param [in] arg        User argument.
 * @param [in] index      Index of the completed SGL element.
 * @param [in] status     Completion status.
 * @param [in] timestamp  Raw local NIC timestamp, or zero if unavailable.
 */
typedef void (*uct_rc_mlx5_sgl_checkpoint_cb_t)(
        void *arg, unsigned index, ucs_status_t status, uint64_t timestamp);


/**
 * Parameters for posting an SGL RDMA write with timestamp checkpoints.
 *
 * All arrays are consumed before the post operation returns. The callback
 * argument and remote targets have to remain valid until all requested
 * checkpoints have completed.
 */
typedef struct uct_rc_mlx5_ext_sgl_params {
    /** Size of this structure, for ABI compatibility. */
    size_t                              struct_size;

    /** Number of entries in every SGL array. */
    size_t                              count;

    /** Local buffers. */
    void *const                        *buffers;

    /** Local buffer lengths. */
    const size_t                       *lengths;

    /** Local memory handles. */
    uct_mem_h const                    *memhs;

    /** Remote addresses. */
    const uint64_t                     *remote_addrs;

    /** Remote keys. */
    uct_rkey_t const                   *rkeys;

    /** Nonzero entries request a timestamp for the matching SGL element. */
    const uint8_t                      *checkpoints;

    /** Callback for requested timestamp checkpoints. */
    uct_rc_mlx5_sgl_checkpoint_cb_t    callback;

    /** User argument passed to @ref callback. */
    void                               *arg;
} uct_rc_mlx5_ext_sgl_params_t;


/**
 * UCT services available to an RC mlx5 extension provider.
 *
 * The service table belongs to UCT and remains valid until the matching
 * provider interface cleanup callback returns.
 */
typedef struct uct_rc_mlx5_ext_services {
    /** Size of this structure, for ABI compatibility. */
    size_t struct_size;

    /**
     * Post an SGL RDMA write on an existing RC endpoint and report selected
     * completion timestamps. The operation uses the endpoint's existing QP,
     * TX CQ, lane, and normal RC outstanding-operation tracking. It does not
     * create or poll separate transport resources. Endpoint purge completes
     * outstanding checkpoints with UCS_ERR_CANCELED. No callback is generated
     * when posting fails.
     */
    ucs_status_t (*ep_put_sgl_timestamped)(
            void *service_ctx, uct_ep_h ep,
            const uct_rc_mlx5_ext_sgl_params_t *params);

    /**
     * Schedule a coalesced periodic callback on the existing interface worker.
     * The timer only queues work and wakes the worker; it does not create a
     * progress thread or poll a CQ. At most one invocation may be queued for a
     * periodic callback at any time.
     */
    ucs_status_t (*schedule_periodic_cb)(
            void *service_ctx, ucs_time_t interval, ucs_callback_t callback,
            void *arg, uct_rc_mlx5_ext_periodic_h *periodic_p);

    /**
     * Stop a periodic callback and remove queued work before returning. The
     * callback argument can be released after this function returns.
     */
    void (*cancel_periodic_cb)(void *service_ctx,
                               uct_rc_mlx5_ext_periodic_h periodic);

    /**
     * Report an endpoint event to UCT. The call has to be made from the
     * serialized interface worker context. UCT owns dispatch to its consumer.
     * Currently only @ref UCT_EVENT_PATH_SPEED_CHANGE is supported.
     */
    ucs_status_t (*notify_ep_event)(void *service_ctx, uct_ep_h ep,
                                    uint64_t events);
} uct_rc_mlx5_ext_services_t;


/**
 * Parameters passed by UCT when attaching a provider to an RC mlx5
 * interface.
 */
typedef struct uct_rc_mlx5_ext_iface_params {
    /** Size of this structure, for ABI compatibility. */
    size_t                                struct_size;

    /** UCT interface to which the provider is attached. */
    uct_iface_h                           iface;

    /**
     * Verbs context used by the interface. The provider can use it to query
     * mlx5 clock information, but does not own the context.
     */
    struct ibv_context                    *ibv_context;

    /**
     * Verbs protection domain used by the interface. The provider can use it
     * to register probe buffers, but does not own the protection domain.
     */
    struct ibv_pd                         *pd;

    /** Mask of valid bits in a raw completion timestamp. */
    uint64_t                              completion_timestamp_mask;

    /** UCT service operations. */
    const uct_rc_mlx5_ext_services_t      *services;

    /** Opaque argument for UCT service operations. */
    void                                  *service_ctx;
} uct_rc_mlx5_ext_iface_params_t;


/**
 * Optional hooks implemented by one RC mlx5 extension provider.
 *
 * All lifecycle, wireup, query, and service calls are serialized by the
 * interface worker unless explicitly documented otherwise. The provider has
 * to remain loaded until all attached interfaces have been cleaned up. Probe
 * traffic is attached to existing RC endpoints; the provider does not create
 * a separate data path, lane, QP, CQ, or progress engine.
 */
typedef struct uct_rc_mlx5_ext_ops {
    /** Size of this structure, for ABI compatibility. */
    size_t          struct_size;

    /** Must be @ref UCT_RC_MLX5_EXT_API_VERSION. */
    uint32_t        api_version;

    /** Human-readable provider name, used only for diagnostics. */
    const char      *name;

    /** Provider wire protocol version. */
    uint16_t        wire_version;

    /** Flags from @ref uct_rc_mlx5_ext_requirement. */
    uint64_t        requirements;

    /**
     * Number of provider-private bytes appended to an endpoint address. UCT
     * adds provider identity, wire version, and payload length separately.
     */
    size_t          ep_addr_len;

    /**
     * Initialize provider state for an existing RC mlx5 interface.
     * UCS_ERR_UNSUPPORTED disables the provider on this interface without
     * preventing the underlying transport from opening.
     */
    ucs_status_t (*iface_init)(
            const uct_rc_mlx5_ext_iface_params_t *params,
            void **iface_ctx_p);

    /** Clean up provider interface state. */
    void (*iface_cleanup)(void *iface_ctx);

    /**
     * Initialize provider state for an existing RC endpoint.
     * UCS_ERR_UNSUPPORTED disables the provider on this endpoint without
     * preventing the underlying endpoint from being created.
     */
    ucs_status_t (*ep_init)(uct_ep_h ep, void *iface_ctx, void **ep_ctx_p);

    /**
     * Clean up provider endpoint state. UCT calls this only after outstanding
     * timestamp callbacks have been completed or canceled.
     */
    void (*ep_cleanup)(uct_ep_h ep, void *ep_ctx);

    /** Pack the provider-private part of an endpoint address. */
    ucs_status_t (*ep_pack_addr)(uct_ep_h ep, void *ep_ctx, void *payload,
                                 size_t payload_len);

    /**
     * Consume the remote provider-private endpoint address. A missing or
     * incompatible remote provider disables measurement on this endpoint and
     * does not make the underlying RC endpoint unreachable.
     */
    ucs_status_t (*ep_connect)(uct_ep_h ep, void *ep_ctx,
                               const void *payload, size_t payload_len);

    /**
     * Query provider-owned endpoint attributes. Currently the only supported
     * provider field is @ref UCT_EP_ATTR_FIELD_BANDWIDTH. This callback never
     * starts a measurement.
     *
     * @return UCS_OK              Every requested field was written.
     * @return UCS_ERR_UNSUPPORTED This endpoint cannot measure bandwidth.
     * @return UCS_ERR_NO_ELEM     No completed sample is available yet.
     * @return UCS_ERR_TIMED_OUT   The last completed sample is stale.
     */
    ucs_status_t (*ep_query)(uct_ep_h ep, void *ep_ctx,
                             uct_ep_attr_t *ep_attr);
} uct_rc_mlx5_ext_ops_t;


/**
 * Register one RC mlx5 extension provider.
 *
 * UCT copies all fields present in @a ops. Only one provider can own the RC
 * path-performance extension at a time.
 */
ucs_status_t uct_rc_mlx5_ext_register(const uct_rc_mlx5_ext_ops_t *ops,
                                      uct_rc_mlx5_ext_provider_h *provider_p);


/**
 * Unregister an RC mlx5 extension provider.
 *
 * @return UCS_OK       The provider was unregistered.
 * @return UCS_ERR_BUSY One or more interfaces are still attached.
 */
ucs_status_t uct_rc_mlx5_ext_unregister(uct_rc_mlx5_ext_provider_h provider);


END_C_DECLS

#endif
