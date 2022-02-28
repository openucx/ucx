#ifndef UCT_UGNI_TYPES_H
#define UCT_UGNI_TYPES_H

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <uct/api/uct.h>
#include <uct/base/uct_md.h>
#include <uct/base/uct_iface.h>
#include <ucs/datastruct/arbiter.h>
#include <ucs/datastruct/bitmap.h>
#include "ofi_def.h"

typedef struct uct_ofi_md {
    uct_md_t           super; /**< Domain info */
    int                ref_count;
    struct fi_info    *fab_info;
    struct fid_fabric *fab_ctx;
    struct fid_domain *dom_ctx;
} uct_ofi_md_t;

typedef struct uct_ofi_av {
    ucs_bitmap_t(256) free;
    fi_addr_t     *table;
    struct fid_av *av;
} uct_ofi_av_t;

typedef struct uct_ofi_iface {
    uct_base_iface_t        super;
    unsigned                outstanding; /**< Counter for outstanding packets */
    ucs_arbiter_t           arbiter;     /**< arbiter structure for pending ops */
    uct_ofi_av_t            *av;         /**< libfabric address vector */
    struct fi_info          *info;
    struct fid_ep *local;
    struct fid_cq *tx_cq;
    struct fid_cq *rx_cq;
} uct_ofi_iface_t;

typedef struct uct_ofi_iface_config {
    uct_iface_config_t       super;
    uct_iface_mpool_config_t mpool;
} uct_ofi_iface_config_t;

typedef struct uct_ofi_ep {
    uct_base_ep_t           super;
    uint64_t                id;
    int                     av_index;
} uct_ofi_ep_t;

typedef struct uct_ofi_name {
    size_t  size;
    uint8_t name[FI_NAME_MAX];
} uct_ofi_name_t;

#endif
