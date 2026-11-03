/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_IB_MLX5_EXT_H_
#define UCT_IB_MLX5_EXT_H_

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <stdint.h>

#include <uct/api/uct.h>
#include <uct/base/uct_iface.h>
#include <ucs/type/status.h>
#include <ucs/sys/stubs.h>

BEGIN_C_DECLS

/**
 * @brief External plugin maximum PUT SGL zero-copy entry count callback.
 *
 * @return Maximum number of SGL entries supported by the plugin's
 *         @ref uct_ib_mlx5_ext_ep_put_sgl_zcopy implementation, or 0 if
 *         unsupported.
 */
typedef size_t (*uct_ib_mlx5_ext_max_put_sgl_zcopy_count_func_t)(void);

/**
 * @brief External plugin maximum GET SGL zero-copy entry count callback.
 *
 * @return Maximum number of SGL entries supported by the plugin's
 *         @ref uct_ib_mlx5_ext_ep_get_sgl_zcopy implementation, or 0 if
 *         unsupported.
 */
typedef size_t (*uct_ib_mlx5_ext_max_get_sgl_zcopy_count_func_t)(void);


/* RC-specific provider hooks and timestamped SGL completions. */

enum {
    /**
     * Require successful provider attachment at interface creation. Unmet
     * requirements or iface_init failure abort interface creation. If unset,
     * normal RC continues without the provider when attachment fails. This
     * policy does not suppress ordinary RC errors or endpoint hook errors.
     */
    UCT_RC_MLX5_EXT_FLAG_REQUIRE_IFACE_ATTACH = UCS_BIT(0)
};


enum {
    /** Request timestamps on the existing RC transmit completion queue. */
    UCT_RC_MLX5_EXT_REQ_TX_CQE_TIMESTAMP = UCS_BIT(0)
};


/**
 * @brief Final callback for a timestamped SGL write, in RC worker context.
 *
 * The timestamps are raw local NIC ticks at completion of the first and last
 * SGL elements. UCS_OK means both timestamps are valid, including a raw value
 * of zero. Both timestamps are zero on error or cancellation.
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
    size_t                            sgl_count;
    uct_rc_mlx5_sgl_completion_func_t callback;
    void                             *arg;
} uct_rc_mlx5_sgl_completion_t;


/**
 * @brief Prepare an idle adapter for an RC endpoint with completion timestamps.
 *
 * The SGL entry count must be at least two.
 *
 * @param [in]  ep         Existing RC endpoint to bind to the adapter.
 * @param [in]  sgl_count  Number of SGL entries.
 * @param [in]  callback   Final completion callback.
 * @param [in]  arg        Argument passed to callback.
 * @param [out] completion Idle adapter to initialize, unchanged on failure.
 *
 * @return UCS_OK                The adapter was initialized.
 * @return UCS_ERR_INVALID_PARAM An argument or the SGL entry count is invalid.
 * @return UCS_ERR_UNSUPPORTED   The endpoint cannot provide timestamps.
 */
ucs_status_t uct_rc_mlx5_sgl_completion_init(
        uct_ep_h ep, size_t sgl_count,
        uct_rc_mlx5_sgl_completion_func_t callback, void *arg,
        uct_rc_mlx5_sgl_completion_t *completion);


/**
 * @brief Optional hooks implemented by an external RC mlx5 extension provider.
 *
 * Attach this table through uct_ib_mlx5_ext_ops_t::rc_ops and register with
 * uct_ib_mlx5_ext_register. The common registry owns the provider's identity;
 * there is no separate RC registration. The provider owns its address payload
 * wire format and version.
 *
 * Hooks run under the existing RC worker serialization.
 *
 * Init hooks are optional. If absent, their context is NULL in subsequent
 * hooks. Init hooks publish their context only on success and unwind resources
 * on failure. A cleanup hook runs only if its corresponding init hook returned
 * UCS_OK, even if that hook published a NULL context. Endpoint cleanup runs
 * after timestamped operations have completed or been canceled.
 * Canceled completion callbacks must not start new endpoint work. Cleanup must
 * quiesce work referencing a provider context before releasing that context.
 *
 * A nonempty address payload requires both pack and connect hooks. With
 * uct_ep_connect_to_ep_v2 and a supplied remote endpoint-address length, UCT
 * validates address bounds and provider presence/identity before calling
 * ep_connect. Missing or incompatible providers and truncated payloads are
 * rejected, not treated as interoperable. The provider validates its payload's
 * wire version inside ep_connect.
 *
 * Calls without a remote address length, including legacy connect, require
 * matching provider presence, registry identity, payload length and wire
 * version on both peers, guaranteed by the caller. Supply the length when
 * this cannot be guaranteed; a supplied length permits bounds checks, not
 * compatibility by itself.
 */
typedef struct uct_rc_mlx5_ext_ops {
    /**
     * Interface attachment policy flags; see
     * @ref UCT_RC_MLX5_EXT_FLAG_REQUIRE_IFACE_ATTACH.
     */
    uint64_t        flags;

    /**
     * Requirements known before interface creation; see
     * @ref UCT_RC_MLX5_EXT_REQ_TX_CQE_TIMESTAMP.
     * Unmet requirements follow the interface attachment policy in flags.
     */
    uint64_t        requirements;

    /**
     * Provider-private bytes appended to the existing endpoint address.
     * The complete endpoint address, including the RC address, UCX extension
     * framing and provider payload, must not exceed 255 bytes. Oversized
     * providers must be rejected before their iface_init hook is called.
     */
    size_t          ep_addr_len;

    /**
     * @brief Initialize provider state on an existing RC interface.
     *
     * @param [in]  iface       Existing RC interface.
     * @param [out] iface_ctx_p Provider context, published only on success.
     * @return UCS_OK on success, or an error if initialization failed.
     */
    ucs_status_t (*iface_init)(uct_iface_h iface, void **iface_ctx_p);

    /**
     * @brief Release provider interface state after endpoint cleanup.
     *
     * @param [in] iface_ctx Provider context from iface_init.
     */
    void (*iface_cleanup)(void *iface_ctx);

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
     * @brief Release provider endpoint state.
     *
     * @param [in] ep_ctx Provider context from ep_init.
     */
    void (*ep_cleanup)(void *ep_ctx);

    /**
     * @brief Pack an opaque payload; required when ep_addr_len is nonzero.
     *
     * @param [in]  ep_ctx      Provider context from ep_init.
     * @param [out] payload     Address payload to fill.
     * @param [in]  payload_len Available payload bytes, equal to ep_addr_len.
     * @return UCS_OK on success, or an error if packing failed.
     */
    ucs_status_t (*ep_pack_addr)(void *ep_ctx, void *payload,
                                size_t payload_len);

    /**
     * @brief Validate the remote provider payload.
     *
     * UCT validates the provider identity and payload bounds before this hook
     * when the remote address length is supplied. Otherwise, callers must
     * satisfy the matching-peer precondition above. Reject incompatible
     * payloads, including unsupported wire versions.
     *
     * Called before RC endpoint connection. Errors are returned to the caller.
     * Failed calls must unwind partial state. If RC connection subsequently
     * fails, ep_cleanup releases the retained provider state.
     *
     * @param [in] ep          Existing RC endpoint.
     * @param [in] ep_ctx      Provider context from ep_init.
     * @param [in] payload     Remote provider address payload.
     * @param [in] payload_len Remote length; see connection preconditions.
     * @return UCS_OK on success, or an error for an incompatible payload.
     */
    ucs_status_t (*ep_connect)(uct_ep_h ep, void *ep_ctx,
                               const void *payload, size_t payload_len);

    /**
     * @brief Return the latest available cached endpoint bandwidth.
     *
     * This optional hook requires the endpoint bandwidth-query attribute
     * extension. It may be omitted when that extension is unavailable.
     *
     * UCT handles standard attributes and calls this hook only when
     * UCT_EP_ATTR_FIELD_BANDWIDTH is requested. The provider fills only
     * bandwidth, leaving field_mask and all other attributes unchanged.
     * Queries without bandwidth do not depend on the provider.
     *
     * This hook never starts a measurement. Return UCS_ERR_UNSUPPORTED when
     * bandwidth measurement is unsupported, UCS_ERR_NO_ELEM before the first
     * sample, or UCS_ERR_TIMED_OUT when the cached sample is stale. Leave
     * attributes unchanged on error; these errors apply only when bandwidth
     * was requested, including in a mixed-attribute query.
     *
     * @param [in]     ep_ctx  Provider context from ep_init.
     * @param [in,out] ep_attr Requested attributes and provider-owned outputs.
     * @return UCS_OK if the requested bandwidth was filled.
     */
    ucs_status_t (*ep_query)(void *ep_ctx, uct_ep_attr_t *ep_attr);
} uct_rc_mlx5_ext_ops_t;


/**
 * @brief External plugin operations.
 */
typedef struct uct_ib_mlx5_ext_ops {
    char                                           name[UCT_COMPONENT_NAME_MAX]; /**< Plugin name */
    uct_ib_mlx5_ext_max_put_sgl_zcopy_count_func_t max_put_sgl_zcopy_count;      /**< Maximum PUT SGL zero-copy entry count callback */
    uct_ep_put_sgl_zcopy_func_t                    ep_put_sgl_zcopy;             /**< PUT SGL zero-copy callback */
    uct_ib_mlx5_ext_max_get_sgl_zcopy_count_func_t max_get_sgl_zcopy_count;      /**< Maximum GET SGL zero-copy entry count callback */
    uct_ep_get_sgl_zcopy_func_t                    ep_get_sgl_zcopy;             /**< GET SGL zero-copy callback */

    /**
     * Optional RC lifecycle, wireup and query hooks. Set to NULL when absent.
     * RC binds the first registered non-NULL table at interface creation and
     * uses that provider for the interface's lifetime. This does not change
     * the independent PUT/GET SGL dispatch above.
     *
     * The registry borrows this table. Keep it immutable and retain its code
     * and referenced data until unregistration or registry cleanup returns.
     * Use a unique provider name and register before opening RC interfaces.
     * Before unregistering or clearing the registry, destroy attached
     * interfaces and quiesce callbacks and queued work.
     * Serialize registry changes with interface creation and plugin dispatch.
     *
     * UCX and plugin consumers must be rebuilt together. This enlarged
     * transport-private registration structure has no binary compatibility
     * promise.
     */
    const uct_rc_mlx5_ext_ops_t *rc_ops;
} uct_ib_mlx5_ext_ops_t;

/**
 * @brief Release mlx5 external extension.
 */
void uct_ib_mlx5_ext_cleanup(void);

/**
 * @brief Unregister the first external plugin matching a name.
 *
 * @param [in] name Plugin name.
 */
void uct_ib_mlx5_ext_unregister(const char *name);

/**
 * @brief Register an external plugin.
 *
 * @param [in] ops Plugin operations.
 *
 * @return UCS_OK on success, or an error if registration failed.
 */
ucs_status_t uct_ib_mlx5_ext_register(const uct_ib_mlx5_ext_ops_t *ops);

size_t uct_ib_mlx5_ext_max_put_sgl_zcopy_count(void);

ucs_status_t uct_ib_mlx5_ext_ep_put_sgl_zcopy(uct_ep_h ep,
                                              void * const *buffers,
                                              const size_t *lengths,
                                              uct_mem_h const *memhs,
                                              const uint64_t *remote_addrs,
                                              uct_rkey_t const *rkeys,
                                              const size_t *counts,
                                              const size_t *strides,
                                              size_t count,
                                              uct_completion_t *comp);

size_t uct_ib_mlx5_ext_max_get_sgl_zcopy_count(void);

ucs_status_t uct_ib_mlx5_ext_ep_get_sgl_zcopy(uct_ep_h ep,
                                              void * const *buffers,
                                              const size_t *lengths,
                                              uct_mem_h const *memhs,
                                              const uint64_t *remote_addrs,
                                              uct_rkey_t const *rkeys,
                                              const size_t *counts,
                                              const size_t *strides,
                                              size_t count,
                                              uct_completion_t *comp);

END_C_DECLS

#endif /* UCT_IB_MLX5_EXT_H_ */
