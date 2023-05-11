/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2023. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_IB_ALLOC_H_
#define UCT_IB_ALLOC_H_


#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <uct/api/uct.h>
#include <uct/ib/base/ib_md.h>

BEGIN_C_DECLS


#if HAVE_IBV_DM

/**
 * IB device memory allocation parameters
 */
typedef struct {
    size_t        length;      /* [in, out] allocation size*/
    unsigned      flags;       /* [in]      access flags*/
    const char    *alloc_name; /* [in]      allocation name*/
    void          *address;    /* [out]     device allocation mapped address */
    uct_ib_mem_t  *memh;       /* [out]     struct containing allocated device memory 
                                            and registered mr keys*/
} ucs_alloc_device_mem_params_t;


ucs_status_t uct_ib_md_alloc_device_mem(uct_md_h uct_md, ucs_alloc_device_mem_params_t *params);

ucs_status_t uct_ib_md_release_device_mem(uct_md_h uct_md, uct_ib_mem_t *memh);
#endif

END_C_DECLS

#endif
