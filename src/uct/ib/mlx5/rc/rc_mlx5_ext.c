/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rc_mlx5_ext.h"

#include <ucs/algorithm/crc.h>
#include <ucs/debug/log.h>
#include <ucs/sys/math.h>

#include <pthread.h>
#include <string.h>


static pthread_mutex_t uct_rc_mlx5_ext_lock = PTHREAD_MUTEX_INITIALIZER;
static uct_rc_mlx5_ext_ops_t uct_rc_mlx5_ext_ops;
static const uct_rc_mlx5_ext_ops_t *uct_rc_mlx5_ext_owner;
static unsigned uct_rc_mlx5_ext_refcount;

ucs_status_t uct_rc_mlx5_ext_register(const uct_rc_mlx5_ext_ops_t *ops)
{
    size_t min_size = ucs_offsetof(uct_rc_mlx5_ext_ops_t, progress) +
                      sizeof(ops->progress);
    ucs_status_t status = UCS_OK;

    if ((ops == NULL) || (ops->name == NULL) || (ops->name[0] == '\0') ||
        (ops->api_version != UCT_RC_MLX5_EXT_API_VERSION) ||
        (ops->struct_size < min_size) || (ops->ep_addr_len > UINT16_MAX) ||
        (ops->requirements & ~UCT_RC_MLX5_EXT_REQ_TX_CQE_TIMESTAMP)) {
        return UCS_ERR_INVALID_PARAM;
    }

    pthread_mutex_lock(&uct_rc_mlx5_ext_lock);
    if ((uct_rc_mlx5_ext_owner != NULL) || (uct_rc_mlx5_ext_refcount != 0)) {
        status = UCS_ERR_BUSY;
        goto out;
    }

    memset(&uct_rc_mlx5_ext_ops, 0, sizeof(uct_rc_mlx5_ext_ops));
    memcpy(&uct_rc_mlx5_ext_ops, ops,
           ucs_min(ops->struct_size, sizeof(uct_rc_mlx5_ext_ops)));
    uct_rc_mlx5_ext_ops.struct_size = sizeof(uct_rc_mlx5_ext_ops);
    uct_rc_mlx5_ext_owner           = ops;
    ucs_debug("rc_mlx5 extension '%s' registered", ops->name);

out:
    pthread_mutex_unlock(&uct_rc_mlx5_ext_lock);
    return status;
}

void uct_rc_mlx5_ext_unregister(const uct_rc_mlx5_ext_ops_t *ops)
{
    pthread_mutex_lock(&uct_rc_mlx5_ext_lock);
    if (ops == uct_rc_mlx5_ext_owner) {
        uct_rc_mlx5_ext_owner = NULL;
        if (uct_rc_mlx5_ext_refcount != 0) {
            ucs_warn("rc_mlx5 extension '%s' unregistered with %u active "
                     "interfaces", uct_rc_mlx5_ext_ops.name,
                     uct_rc_mlx5_ext_refcount);
        }
    }
    pthread_mutex_unlock(&uct_rc_mlx5_ext_lock);
}

const uct_rc_mlx5_ext_ops_t *uct_rc_mlx5_ext_acquire(void)
{
    const uct_rc_mlx5_ext_ops_t *ops = NULL;

    pthread_mutex_lock(&uct_rc_mlx5_ext_lock);
    if (uct_rc_mlx5_ext_owner != NULL) {
        ++uct_rc_mlx5_ext_refcount;
        ops = &uct_rc_mlx5_ext_ops;
    }
    pthread_mutex_unlock(&uct_rc_mlx5_ext_lock);
    return ops;
}

void uct_rc_mlx5_ext_release(const uct_rc_mlx5_ext_ops_t *ops)
{
    if (ops == NULL) {
        return;
    }

    pthread_mutex_lock(&uct_rc_mlx5_ext_lock);
    ucs_assert(ops == &uct_rc_mlx5_ext_ops);
    ucs_assert(uct_rc_mlx5_ext_refcount > 0);
    --uct_rc_mlx5_ext_refcount;
    pthread_mutex_unlock(&uct_rc_mlx5_ext_lock);
}

uint64_t uct_rc_mlx5_ext_get_requirements(void)
{
    uint64_t requirements;

    pthread_mutex_lock(&uct_rc_mlx5_ext_lock);
    requirements = (uct_rc_mlx5_ext_owner == NULL) ?
                   0 : uct_rc_mlx5_ext_ops.requirements;
    pthread_mutex_unlock(&uct_rc_mlx5_ext_lock);
    return requirements;
}

uint32_t uct_rc_mlx5_ext_provider_id(const uct_rc_mlx5_ext_ops_t *ops)
{
    return ucs_crc32(0, ops->name, strlen(ops->name));
}
