/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_RC_MLX5_EXT_H_
#define UCT_RC_MLX5_EXT_H_

#include <uct/api/uct.h>

#include <stdint.h>

BEGIN_C_DECLS


#define UCT_RC_MLX5_EXT_API_VERSION 1


enum {
    /** Request timestamps on the existing RC transmit completion queue. */
    UCT_RC_MLX5_EXT_REQ_TX_CQE_TIMESTAMP = UCS_BIT(0)
};


/**
 * @brief Final callback for a timestamped SGL write, in RC worker context.
 *
 * The timestamps are raw local NIC ticks at completion of the first and last
 * SGL elements. The first element is the timing anchor and is excluded from
 * the measured byte count. UCS_OK means both timestamps are valid, including
 * a raw value of zero. Both timestamps are zero on error or cancellation.
 *
 * @param [in] arg             Provider callback argument.
 * @param [in] status          Final operation status.
 * @param [in] start_timestamp First SGL element's completion timestamp.
 * @param [in] end_timestamp   Last SGL element's completion timestamp.
 */
typedef void (*uct_rc_mlx5_sgl_completion_func_t)(
        void *arg, ucs_status_t status, uint64_t start_timestamp,
        uint64_t end_timestamp);


/**
 * @brief Completion adapter for the existing uct_ep_put_sgl_zcopy operation.
 *
 * Initialize with uct_rc_mlx5_sgl_completion_init, then pass &adapter.super
 * only to put_sgl_zcopy on the bound endpoint with the same SGL entry count.
 * Only non-strided entries are supported: counts and strides must be NULL.
 * UCT manages every field; ordinary completions retain their normal semantics.
 * Intermediate completion tracking is private to the transport.
 *
 * An accepted timestamped post returns UCS_INPROGRESS and invokes callback
 * exactly once, after every SGL element has completed or been canceled. A
 * failed post preserves the adapter state and generates no callback. SGL
 * arrays are consumed before posting returns; the adapter, callback argument,
 * local buffers, registrations and remote targets remain valid until
 * completion.
 *
 * UCT makes the adapter idle before invoking callback, so the callback may
 * release or reinitialize it. Do not modify an in-flight adapter.
 * Initialization, posting and completion share the endpoint's worker
 * serialization.
 */
typedef struct uct_rc_mlx5_sgl_completion {
    uct_completion_t                  super;
    uct_ep_h                          ep;
    size_t                            count;
    uct_rc_mlx5_sgl_completion_func_t callback;
    void                             *arg;
} uct_rc_mlx5_sgl_completion_t;


/**
 * @brief Prepare an idle adapter for an RC endpoint with completion timestamps.
 *
 * Count must be at least two: an anchor followed by measured data. UCT captures
 * the first and last completion timestamps internally, without exposing
 * intermediate callbacks or calculating bandwidth.
 *
 * @param [in]  ep         Existing RC endpoint to bind to the adapter.
 * @param [in]  count      Number of SGL entries, including the timing anchor.
 * @param [in]  callback   Final completion callback.
 * @param [in]  arg        Argument passed to callback.
 * @param [out] completion Idle adapter to initialize, unchanged on failure.
 *
 * @return UCS_OK                The adapter was initialized.
 * @return UCS_ERR_INVALID_PARAM An argument or the SGL entry count is invalid.
 * @return UCS_ERR_UNSUPPORTED   The endpoint cannot provide timestamps.
 */
ucs_status_t uct_rc_mlx5_sgl_completion_init(
        uct_ep_h ep, size_t count, uct_rc_mlx5_sgl_completion_func_t callback,
        void *arg, uct_rc_mlx5_sgl_completion_t *completion);


/**
 * @brief Optional hooks implemented by an external RC mlx5 extension provider.
 *
 * Attach this table through uct_ib_mlx5_ext_ops_t::rc_ops and register with
 * uct_ib_mlx5_ext_register. The common registry owns the provider's identity;
 * there is no separate RC registration. Table size and API version describe
 * this callback contract, not the provider-private address wire version.
 *
 * Hooks run under the existing RC worker serialization. The provider owns its
 * timers, scheduling, sampling, estimation and change detection, using existing
 * UCS/UCT primitives. UCT does not create a separate lane, QP, CQ or progress
 * engine for the provider.
 *
 * Init hooks publish their context only on success and unwind their resources
 * on failure. Cleanup hooks run only after successful initialization. Endpoint
 * cleanup runs after timestamped operations have completed or been canceled.
 * Interface cleanup stops provider-owned timers and drains its queued work.
 *
 * A nonempty address payload requires both pack and connect hooks. Legacy
 * connect requires matching provider configurations on both peers; use
 * uct_ep_connect_to_ep_v2 with the remote address length when provider
 * configurations may differ.
 */
typedef struct uct_rc_mlx5_ext_ops {
    /** Size of the provider's RC operations table, in bytes. */
    size_t          struct_size;

    /** Must be UCT_RC_MLX5_EXT_API_VERSION. */
    uint32_t        api_version;

    /** Requirements known before interface creation. */
    uint64_t        requirements;

    /** Provider-private bytes appended to the existing endpoint address. */
    size_t          ep_addr_len;

    /**
     * @brief Initialize provider state on an existing RC interface.
     *
     * @param [in]  iface       Existing RC interface.
     * @param [in]  params      Interface creation parameters.
     * @param [out] iface_ctx_p Provider context, published only on success.
     * @return UCS_OK on success, or an error if initialization failed.
     */
    ucs_status_t (*iface_init)(uct_iface_h iface,
                               const uct_iface_params_t *params,
                               void **iface_ctx_p);

    /**
     * @brief Release provider interface state after endpoint cleanup.
     *
     * @param [in] iface     Existing RC interface.
     * @param [in] iface_ctx Provider context from iface_init.
     */
    void (*iface_cleanup)(uct_iface_h iface, void *iface_ctx);

    /**
     * @brief Initialize provider state on an existing RC endpoint.
     *
     * @param [in]  ep        Existing RC endpoint.
     * @param [in]  iface_ctx Provider context from iface_init.
     * @param [out] ep_ctx_p  Provider context, published only on success.
     * @return UCS_OK on success, or an error if initialization failed.
     */
    ucs_status_t (*ep_init)(uct_ep_h ep, void *iface_ctx, void **ep_ctx_p);

    /**
     * @brief Stop endpoint sampling and release provider endpoint state.
     *
     * @param [in] ep     Existing RC endpoint.
     * @param [in] ep_ctx Provider context from ep_init.
     */
    void (*ep_cleanup)(uct_ep_h ep, void *ep_ctx);

    /**
     * @brief Pack an opaque payload; required when ep_addr_len is nonzero.
     *
     * @param [in]  ep          Existing RC endpoint.
     * @param [in]  ep_ctx      Provider context from ep_init.
     * @param [out] payload     Address payload to fill.
     * @param [in]  payload_len Available payload bytes, equal to ep_addr_len.
     * @return UCS_OK on success, or an error if packing failed.
     */
    ucs_status_t (*ep_pack_addr)(uct_ep_h ep, void *ep_ctx, void *payload,
                                 size_t payload_len);

    /**
     * @brief Validate remote payload, including the provider wire version.
     *
     * @param [in] ep          Existing RC endpoint.
     * @param [in] ep_ctx      Provider context from ep_init.
     * @param [in] payload     Remote provider address payload.
     * @param [in] payload_len Actual remote payload length, not a local guess.
     * @return UCS_OK on success, or an error for an incompatible payload.
     */
    ucs_status_t (*ep_connect)(uct_ep_h ep, void *ep_ctx,
                               const void *payload, size_t payload_len);

    /**
     * @brief Fill requested attributes from completed samples only.
     *
     * This hook never starts a measurement. Return UCS_ERR_UNSUPPORTED for
     * unsupported attributes, UCS_ERR_NO_ELEM before the first sample, or
     * UCS_ERR_TIMED_OUT when the cached sample is stale.
     *
     * @param [in]     ep      Existing RC endpoint.
     * @param [in]     ep_ctx  Provider context from ep_init.
     * @param [in,out] ep_attr Requested attributes and provider-owned outputs.
     * @return UCS_OK if the requested provider attributes were filled.
     */
    ucs_status_t (*ep_query)(uct_ep_h ep, void *ep_ctx,
                             uct_ep_attr_t *ep_attr);
} uct_rc_mlx5_ext_ops_t;


END_C_DECLS

#endif /* UCT_RC_MLX5_EXT_H_ */
