/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#include "mm_pd.h"
#include "mm_iface.h"

#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include "xpmem.h"


static ucs_status_t uct_xpmem_query()
{
    int fd, ver;

    fd = open(XPMEM_DEV_PATH, O_RDWR);
    if (fd < 0) {
        ucs_debug("Could not open the XPMEM device file at /dev/xpmem: %m. Disabling xpmem resource");
        return UCS_ERR_UNSUPPORTED;
    }
    close(fd);

    ver = xpmem_version();
    if (ver < 0) {
        ucs_debug("Failed to query XPMEM version %d, %m", ver);
        return UCS_ERR_UNSUPPORTED;
    }

    return UCS_OK;
}

static ucs_status_t uct_xmpem_reg(void *address, size_t size, uct_mm_id_t *mmid_p)
{
    // TODO set *mmid_p to xpmem_seg_id
    return UCS_ERR_UNSUPPORTED;
}

static ucs_status_t uct_xpmem_dereg(uct_mm_id_t mmid)
{
    return UCS_ERR_UNSUPPORTED;
}

static ucs_status_t uct_xpmem_attach(uct_mm_id_t mmid, void **address_p)
{
    return UCS_ERR_UNSUPPORTED;
}

static ucs_status_t uct_xpmem_detach(void *address)
{
    return UCS_ERR_UNSUPPORTED;
}

static uct_mm_mapper_ops_t uct_xpmem_mapper_ops = {
    .query   = uct_xpmem_query,
    .reg     = uct_xmpem_reg,
    .dereg   = uct_xpmem_dereg,
    .alloc   = NULL,
    .attach  = uct_xpmem_attach,
    .release = uct_xpmem_detach
};

UCT_MM_COMPONENT_DEFINE(uct_xpmem_pd, "xpmem", &uct_xpmem_mapper_ops)
UCT_PD_REGISTER_TL(&uct_xpmem_pd, &uct_mm_tl);
