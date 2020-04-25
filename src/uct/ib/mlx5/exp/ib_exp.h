/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_IB_MLX5_EXP_H_
#define UCT_IB_MLX5_EXP_H_

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#if defined (HAVE_MLX5_HW) && defined (HAVE_VERBS_EXP_H)
void uct_ib_exp_qp_fill_attr(uct_ib_iface_t *iface, uct_ib_qp_attr_t *attr);
#else
static inline void uct_ib_exp_qp_fill_attr(uct_ib_iface_t *iface, uct_ib_qp_attr_t *attr) { }
#endif

#endif
