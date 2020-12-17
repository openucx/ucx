/**
* Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "dc_mlx5.h"

#include <uct/ib/rc/accel/rc_mlx5.inl>
#include "uct/ib/rc/base/rc_iface.h"
#include "uct/ib/rc/base/rc_ep.h"


static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_update_tx_res(uct_dc_mlx5_iface_t *iface, uct_ib_mlx5_txwq_t *txwq,
                          uct_rc_txqp_t *txqp, uint16_t hw_ci)
{
    uct_rc_txqp_available_set(txqp, uct_ib_mlx5_txwq_update_bb(txwq, hw_ci));
    ucs_assert(uct_rc_txqp_available(txqp) <= txwq->bb_max);

    uct_rc_iface_update_reads(&iface->super.super);
}
