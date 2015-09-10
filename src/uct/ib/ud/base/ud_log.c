/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ud_log.h"
#include "ud_iface.h"
#include "ud_ep.h"


static void uct_ud_dump_neth(uct_ud_iface_t *iface, uct_am_trace_type_t type,
                             char *p, int max, uct_ud_neth_t *neth, int pkt_len)
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
    p += n; max -= n;

    if (is_am) {
        n = snprintf(p, max, "AM: %d", am_id);
        p += n; max -= n;
        uct_iface_dump_am(&iface->super.super, type, am_id, neth + 1,
                          pkt_len - sizeof(*neth), p, max);
        n = strlen(p);
        p += n; max -= n;
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
            p += n; max -= n;
            break;
        case UCT_UD_PACKET_CREP:
            n = snprintf(p, max, "CREP: src_ep_id=%d ", ctl_hdr->conn_rep.src_ep_id);
            p += n; max -= n;
            break;
        default:
            n = snprintf(p, max, "WTF_CTL");
            p += n; max -= n;
            break;
        }
    } else if (pkt_len != sizeof(neth)) {
        n = snprintf(p, max, "WTF UKNOWN DATA");
        p += n; max -= n;
    }

    if (ack_req) {
        n = snprintf(p, max, " ACK_REQ");
        p += n; max -= n;
    }
    /* dump raw neth since it helps out to debug sniffer traces */
    {
        int i;
        char *base = (char *)neth;
        
        n = snprintf(p, max, " NETH ");
        p += n; max -= n;
        for (i = 0; i < sizeof(*neth); i++) {
           n = snprintf(p, max, "%02X ", (unsigned)(char)base[i]);
           p += n; max -= n;
        }
    }
}

static int uct_ud_dump_ep(char *p, int max, uct_ud_ep_t *ep)
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

static char *pkt_type2str(int type)
{
    switch(type) {
    case UCT_AM_TRACE_TYPE_SEND:
        return "TX";
    case UCT_AM_TRACE_TYPE_RECV:
        return "RX";
    case UCT_AM_TRACE_TYPE_RECV_DROP:
        return "RX **DROPPED BY FILTER**";
    default:
        return "UNKNOWN";
    }
}


void uct_ud_log_packet(const char *file, int line, const char *function,
                       uct_ud_iface_t *iface, uct_ud_ep_t *ep,
                       uct_am_trace_type_t type, uct_ud_neth_t *neth, uint32_t len)
{
    char buf[256] = {0};
    char *p;
    int n, max;

    p = buf;
    max = sizeof(buf);

    n = snprintf(p, max, "%s: if=%p ",
                 pkt_type2str(type), iface);
    p += n; max -= n;

    n = uct_ud_dump_ep(p, max, ep);
    p += n; max -= n;

    uct_ud_dump_neth(iface, type, p, max, neth, len);

    uct_log_data(file, line, function, buf);
}

