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

#define UCT_RC_MLX5_EXT_API_VERSION 1

enum {
    /** Request completion timestamps on the stock RC TX CQ. */
    UCT_RC_MLX5_EXT_REQ_TX_CQE_TIMESTAMP = UCS_BIT(0)
};

typedef struct uct_rc_mlx5_path_identity {
    uint32_t    qpn;
    const void  *tx_cq;
} uct_rc_mlx5_path_identity_t;

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

    /** Optional hook after stock RC progress; NULL adds no provider call. */
    unsigned (*progress)(uct_iface_h iface, void *iface_ctx);

    /** Optional event capabilities; absent in shorter structures means none.
     * Only UCT_IFACE_FLAG_EVENT_PATH_SPEED_CHANGE is supported, and requires
     * an ep_query hook. */
    uint64_t event_flags;
} uct_rc_mlx5_ext_ops_t;

ucs_status_t uct_rc_mlx5_ext_register(const uct_rc_mlx5_ext_ops_t *ops);
void uct_rc_mlx5_ext_unregister(const uct_rc_mlx5_ext_ops_t *ops);

ucs_status_t uct_rc_mlx5_ep_get_path_identity(
        uct_ep_h ep, uct_rc_mlx5_path_identity_t *identity);

/* Internal transport helpers. */
const uct_rc_mlx5_ext_ops_t *uct_rc_mlx5_ext_acquire(void);
void uct_rc_mlx5_ext_release(const uct_rc_mlx5_ext_ops_t *ops);
uint32_t uct_rc_mlx5_ext_provider_id(const uct_rc_mlx5_ext_ops_t *ops);

END_C_DECLS

#endif
