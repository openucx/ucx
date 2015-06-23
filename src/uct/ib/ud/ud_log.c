/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ud_log.h"
#include "ud_iface.h"
#include "ud_ep.h"

static int uct_ud_dump_neth(char *p, int max, uct_ud_neth_t *neth, int pkt_len)
{
    int n, ret = 0;
    uint32_t dest_id;
    uint32_t am_id;
    uint32_t is_am;
    uint32_t ack_req;
    uint32_t is_ctl;
    uint32_t is_put;

    dest_id = uct_ud_neth_get_dest_id(neth);
    am_id   = uct_ud_neth_get_am_id(neth);
    is_am   = neth->packet_type & UCT_UD_PACKET_FLAG_AM;
    ack_req = neth->packet_type & UCT_UD_PACKET_FLAG_ACK_REQ;
    is_ctl  = neth->packet_type & UCT_UD_PACKET_FLAG_CTL;
    is_put  = neth->packet_type & UCT_UD_PACKET_FLAG_PUT;

    n = snprintf(p, max, "PKT: dst=0x%x psn=%u ack=%u dlen=%u ", 
                 (unsigned)dest_id, 
                 (unsigned)neth->psn, (unsigned)neth->ack_psn,
                 (unsigned)(pkt_len - sizeof(*neth))
            );
    p += n; max -= n; ret += n;

    if (is_am) {
        n = snprintf(p, max, "AM: %d", am_id);
        p += n; max -= n; ret += n;
    } else if (is_put) { 
        uct_ud_put_hdr_t *put_hdr;
        
        put_hdr = (uct_ud_put_hdr_t *)(neth+1);
        n = snprintf(p, max, "PUT: 0x%0lx", (unsigned long)put_hdr->rva);
        p += n; max -= n; ret += n;
    } else if (is_ctl) {
        uct_ud_ctl_hdr_t *ctl_hdr = (uct_ud_ctl_hdr_t *)(neth+1);

        ctl_hdr = (uct_ud_ctl_hdr_t *)(neth+1);
        switch(ctl_hdr->type) {
            case UCT_UD_PACKET_CREQ:
                n = snprintf(p, max, "CREQ: qp=%x lid=%d epid=%d cid=%d ",
                             ctl_hdr->conn_req.ib_addr.qp_num,
                             ctl_hdr->conn_req.ib_addr.lid,
                             ctl_hdr->conn_req.ib_addr.id,
                             ctl_hdr->conn_req.conn_id);
                p += n; max -= n; ret += n;
                break;
            case UCT_UD_PACKET_CREP:
                n = snprintf(p, max, "CREP: src_ep_id=%d ", ctl_hdr->conn_rep.src_ep_id);
                p += n; max -= n; ret += n;
                break;
            default:
                n = snprintf(p, max, "WTF_CTL");
                p += n; max -= n; ret += n;
                break;
        }
    } else if (pkt_len != sizeof(neth)) {
        n = snprintf(p, max, "WTF UKNOWN DATA");
        p += n; max -= n; ret += n;
    }

    if (ack_req) {
        snprintf(p, max, " ACK_REQ");
        p += n; max -= n; ret += n;
    }
    return ret;
}

int uct_ud_dump_ep(char *p, int max, uct_ud_ep_t *ep)
{
    int n;

    if (ep == NULL) {
        n = snprintf(p, max, "ep=%p ", ep);
    } else {
        n = snprintf(p, max, "ep=%p cid:%d 0x%x->0x%x ", 
                     ep, ep->conn_id, ep->ep_id, ep->dest_ep_id);
    }
    return n;
} 

void __uct_ud_log_packet(const char *file, int line, const char *function,
                         char *tag,
                         uct_ud_iface_t *iface, 
                         uct_ud_ep_t *ep, 
                         uct_ud_neth_t *neth, uint32_t len)
{
    char buf[256] = {0};
    char *p;
    int n, max;

    p = buf;
    max = sizeof(buf);

    n = snprintf(p, max, "%s: if=%p ", tag, iface);
    p += n; max -= n;

    n = uct_ud_dump_ep(p, max, ep);
    p += n; max -= n;

    uct_ud_dump_neth(p, max, neth, len);

    uct_log_data(file, line, function, buf);
}

