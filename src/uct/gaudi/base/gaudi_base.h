/*
 * Copyright (C) Intel Corporation, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_GAUDI_BASE_H_
#define UCT_GAUDI_BASE_H_

#include <stdbool.h>
#include <uct/base/uct_iface.h>
#include <uct/base/uct_md.h>
#include "scal.h"

int uct_gaudi_base_get_fd(int device_id, bool *fd_created);

void uct_gaudi_base_close_fd(int fd, bool fd_created);

void uct_gaudi_base_close_dmabuf_fd(int fd);

ucs_status_t uct_gaudi_base_get_sysdev(int fd, ucs_sys_device_t *sys_dev);

ucs_status_t uct_gaudi_base_get_info(int fd,
                                     uint64_t *device_base_allocated_address,
                                     uint64_t *device_base_address,
                                     uint64_t *totalSize, int *dmabuf_fd);

ucs_status_t
uct_gaudi_base_query_devices(uct_md_h md,
                             uct_tl_device_resource_t **tl_devices_p,
                             unsigned *num_tl_devices_p);
#endif
