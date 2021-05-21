/**
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "srd_iface.h"
#include "srd_ep.h"


void uct_srd_dump_packet(uct_base_iface_t *iface, uct_am_trace_type_t type,
                        void *data, size_t length, size_t valid_length,
                        char *buffer, size_t max)
{
    uct_srd_neth_t *neth = data;
    uct_srd_ctl_hdr_t *ctlh;
    char *p, *endp;
    char buf[128];
    int am_id;

    p    = buffer;
    endp = buffer + max;

    snprintf(p, endp - p, " dst %d psn %u",
            uct_srd_neth_get_dest_id(neth), neth->psn);
    p += strlen(p);

    if (neth->packet_type & UCT_SRD_PACKET_FLAG_AM) {
        am_id = uct_srd_neth_get_am_id(neth);
        snprintf(p, endp - p, " am %d ", am_id);
        p += strlen(p);
        uct_iface_dump_am(iface, type, am_id, neth + 1,
                          length - sizeof(*neth), p, endp - p);
    } else if (neth->packet_type & UCT_SRD_PACKET_FLAG_CTLX) {
        ctlh = (uct_srd_ctl_hdr_t *)(neth + 1);
        switch (ctlh->type) {
            case UCT_SRD_PACKET_CREQ:
                snprintf(p, endp - p,
                         " CREQ from %s:%d qpn 0x%x %s epid %d cid %d path %d",
                         ctlh->peer.name, ctlh->peer.pid,
                         uct_ib_unpack_uint24(ctlh->conn_req.ep_addr.iface_addr.qp_num),
                         uct_ib_address_str(uct_srd_creq_ib_addr(ctlh), buf, sizeof(buf)),
                         uct_ib_unpack_uint24(ctlh->conn_req.ep_addr.ep_id),
                         ctlh->conn_req.conn_sn, ctlh->conn_req.path_index);
                break;
            case UCT_SRD_PACKET_CREP:
                snprintf(p, endp - p, " CREP from %s:%d src_ep_id %d",
                         ctlh->peer.name, ctlh->peer.pid,
                         ctlh->conn_rep.src_ep_id);
                break;
            default:
                snprintf(p, endp - p, " <unknown control packet %d> from %s:%d",
                         ctlh->type, ctlh->peer.name, ctlh->peer.pid);
                break;
        }
    }
}
