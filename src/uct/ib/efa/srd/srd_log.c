/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <uct/ib/efa/srd/srd_log.h>
#include <uct/ib/efa/srd/srd_def.h>

#include <uct/ib/base/ib_log.h>


void uct_srd_dump_packet(uct_base_iface_t *iface, uct_am_trace_type_t type,
                         void *data, size_t length, size_t valid_length,
                         char *buffer, size_t max)
{
    uct_srd_hdr_t *hdr = data;
    char *p, *endp;
    int am_id;

    p    = buffer;
    endp = buffer + max;

    snprintf(p, endp - p, " ep_uuid 0x%" PRIx64 " psn %u", hdr->ep_uuid,
             hdr->psn);
    p += strlen(p);

    if (hdr->id < UCT_AM_ID_MAX) {
        am_id = hdr->id;
        snprintf(p, endp - p, " am %d ", am_id);
        p += strlen(p);
        uct_iface_dump_am(iface, type, am_id, hdr + 1, length - sizeof(*hdr), p,
                          endp - p);
    } else {
        snprintf(p, endp - p, " id %x", hdr->id);
    }
}
