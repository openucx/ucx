/**
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef __CONFIG_H
#define __CONFIG_H

#include <infiniband/efadv.h>
#include <infiniband/verbs.h>


extern struct efadv_device_attr efa_dev_attr;
extern struct ibv_device_attr efa_ibv_dev_attr;
extern struct ibv_port_attr efa_ib_port_attr;
extern struct ibv_qp_attr efa_ib_qp_attr;
extern struct ibv_qp_init_attr efa_ib_qp_init_attr;

#endif
