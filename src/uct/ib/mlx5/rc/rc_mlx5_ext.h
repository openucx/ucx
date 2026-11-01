/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_RC_MLX5_EXT_H_
#define UCT_RC_MLX5_EXT_H_

#include <uct/api/uct_def.h>
#include <uct/base/uct_iface.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/type/status.h>

BEGIN_C_DECLS

#define UCT_RC_MLX5_EXT_API_VERSION 2

enum {
    /** Request completion timestamps on the stock RC TX CQ. */
    UCT_RC_MLX5_EXT_REQ_TX_CQE_TIMESTAMP = UCS_BIT(0)
};

/** Internal callback for one SGL checkpoint, in normal RC completion context.
 * The timestamp is raw local NIC ticks, or zero when unavailable (e.g. purge).
 * Each requested checkpoint is reported exactly once after a successful post.
 */
typedef void (*uct_rc_mlx5_sgl_checkpoint_cb_t)(
        void *arg, unsigned index, ucs_status_t status, uint64_t timestamp);


/**
 * Internal RC completion adapter for the ordinary uct_ep_put_sgl_zcopy API.
 *
 * Initialize with uct_rc_mlx5_sgl_completion_init and pass &adapter.super only
 * to put_sgl_zcopy on the bound endpoint, not to other UCT operations. The
 * unique super.func marker identifies this object before extended fields are
 * accessed. NULL and ordinary completions retain their existing semantics.
 *
 * The SGL arrays and checkpoint flags are consumed before post returns. The
 * adapter, callback state, registered buffers and remote targets must remain
 * valid until all checkpoint callbacks finish. super.count tracks remaining
 * callbacks; it is zero when idle, and super.status aggregates their status.
 * The count/status are committed before invoking a callback, so the final
 * callback may release or reinitialize the adapter. Do not reinitialize or
 * modify an in-flight adapter. A failed post neither changes its state nor
 * generates callbacks. Access is serialized by the endpoint's worker.
 */
typedef struct uct_rc_mlx5_sgl_completion {
    uct_completion_t                  super;
    uct_ep_h                          ep;
    size_t                            count;
    const uint8_t                    *checkpoints;
    uct_rc_mlx5_sgl_checkpoint_cb_t    callback;
    void                             *arg;
} uct_rc_mlx5_sgl_completion_t;


/**
 * Prepare an idle adapter for a timestamp-capable stock RC endpoint.
 * A nonzero checkpoint flag selects an element; the final element must be
 * selected. Returns unsupported without changing the output for other
 * backends or an interface without timestamp support.
 */
ucs_status_t uct_rc_mlx5_sgl_completion_init(
        uct_ep_h ep, size_t count, const uint8_t *checkpoints,
        uct_rc_mlx5_sgl_checkpoint_cb_t callback, void *arg,
        uct_rc_mlx5_sgl_completion_t *completion);

/* Export one marker across DSOs; it is not an ordinary completion callback. */
void uct_rc_mlx5_sgl_completion_marker(uct_completion_t *completion);

static inline int
uct_rc_mlx5_sgl_completion_is_adapter(const uct_completion_t *completion)
{
    return (completion != NULL) &&
           (completion->func == uct_rc_mlx5_sgl_completion_marker);
}

/**
 * Optional hooks implemented by one external RC mlx5 extension provider.
 *
 * The provider has to remain loaded until all interfaces which acquired it
 * are destroyed. Hooks run under the same worker serialization as the stock
 * RC transport. The address payload is appended to the stock RC EP address.
 * Init hooks publish their context only on success and unwind their own
 * resources on failure; cleanup hooks are called only after successful init.
 * A nonempty address payload requires both pack and connect hooks. The legacy
 * connect API requires matching provider configuration on both peers; use
 * connect-to-ep v2 for length-checked exchange with a potentially unlike peer.
 */
typedef struct uct_rc_mlx5_ext_ops {
    size_t          struct_size;
    uint32_t        api_version;
    const char      *name;
    uint64_t        requirements;
    size_t          ep_addr_len;

    ucs_status_t (*iface_init)(uct_iface_h iface,
                               const uct_iface_params_t *params,
                               void **iface_ctx_p);
    void (*iface_cleanup)(uct_iface_h iface, void *iface_ctx);

    ucs_status_t (*ep_init)(uct_ep_h ep, void *iface_ctx, void **ep_ctx_p);
    void (*ep_cleanup)(uct_ep_h ep, void *ep_ctx);

    ucs_status_t (*ep_pack_addr)(uct_ep_h ep, void *ep_ctx, void *payload,
                                 size_t payload_len);
    ucs_status_t (*ep_connect)(uct_ep_h ep, void *ep_ctx,
                               const void *payload, size_t payload_len);
    ucs_status_t (*ep_query)(uct_ep_h ep, void *ep_ctx,
                             uct_ep_attr_t *ep_attr);

    /** Optional event capabilities; absent in shorter structures means none.
     * Only UCT_IFACE_FLAG_EVENT_PATH_SPEED_CHANGE is supported, and requires
     * an ep_query hook. */
    uint64_t event_flags;
} uct_rc_mlx5_ext_ops_t;

ucs_status_t uct_rc_mlx5_ext_register(const uct_rc_mlx5_ext_ops_t *ops);
void uct_rc_mlx5_ext_unregister(const uct_rc_mlx5_ext_ops_t *ops);

/* Internal transport helpers. */
const uct_rc_mlx5_ext_ops_t *uct_rc_mlx5_ext_acquire(void);
void uct_rc_mlx5_ext_release(const uct_rc_mlx5_ext_ops_t *ops);
uint32_t uct_rc_mlx5_ext_provider_id(const uct_rc_mlx5_ext_ops_t *ops);

END_C_DECLS

#endif
