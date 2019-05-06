/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_IB_ALLOC_H_
#define UCT_IB_ALLOC_H_

#include <uct/api/uct.h>

BEGIN_C_DECLS

typedef struct uct_ib_device_mem *uct_ib_device_mem_h;

ucs_status_t uct_ib_md_alloc_device_mem(uct_md_h uct_md, size_t *length_p,
                                        void **address_p, unsigned flags,
                                        const char *alloc_name,
                                        uct_ib_device_mem_h *dev_mem_p);

void uct_ib_md_release_device_mem(uct_ib_device_mem_h dev_mem);

END_C_DECLS

#endif
