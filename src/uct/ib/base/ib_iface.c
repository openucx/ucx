/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ib_iface.h"
#include "ib_log.h"

#include <uct/base/uct_md.h>
#include <ucs/arch/bitops.h>
#include <ucs/arch/cpu.h>
#include <ucs/type/class.h>
#include <ucs/type/cpu_set.h>
#include <ucs/debug/log.h>
#include <ucs/time/time.h>
#include <ucs/memory/numa.h>
#include <string.h>
#include <stdlib.h>
#include <poll.h>


static UCS_CONFIG_DEFINE_ARRAY(path_bits_spec,
                               sizeof(ucs_range_spec_t),
                               UCS_CONFIG_TYPE_RANGE_SPEC);

const char *uct_ib_mtu_values[] = {
    [UCT_IB_MTU_DEFAULT]    = "default",
    [UCT_IB_MTU_512]        = "512",
    [UCT_IB_MTU_1024]       = "1024",
    [UCT_IB_MTU_2048]       = "2048",
    [UCT_IB_MTU_4096]       = "4096",
    [UCT_IB_MTU_LAST]       = NULL
};

enum {
    UCT_IB_ADDRESS_TYPE_LINK_LOCAL,
    UCT_IB_ADDRESS_TYPE_SITE_LOCAL,
    UCT_IB_ADDRESS_TYPE_GLOBAL,
    UCT_IB_ADDRESS_TYPE_ETH,
    UCT_IB_ADDRESS_TYPE_LAST,
    UCT_IB_IFACE_ADDRESS_TYPE_AUTO  = UCT_IB_ADDRESS_TYPE_LAST,
    UCT_IB_IFACE_ADDRESS_TYPE_LAST
};

static const char *uct_ib_iface_addr_types[] = {
   [UCT_IB_ADDRESS_TYPE_LINK_LOCAL] = "ib_local",
   [UCT_IB_ADDRESS_TYPE_SITE_LOCAL] = "ib_site_local",
   [UCT_IB_ADDRESS_TYPE_GLOBAL]     = "ib_global",
   [UCT_IB_ADDRESS_TYPE_ETH]        = "eth",
   [UCT_IB_IFACE_ADDRESS_TYPE_AUTO] = "auto",
   [UCT_IB_IFACE_ADDRESS_TYPE_LAST] = NULL
};

ucs_config_field_t uct_ib_iface_config_table[] = {
  {"", "", NULL,
   ucs_offsetof(uct_ib_iface_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

  {"SEG_SIZE", "8192",
   "Size of bounce buffers used for post_send and post_recv.",
   ucs_offsetof(uct_ib_iface_config_t, seg_size), UCS_CONFIG_TYPE_MEMUNITS},

  {"TX_QUEUE_LEN", "256",
   "Length of send queue in the QP.",
   ucs_offsetof(uct_ib_iface_config_t, tx.queue_len), UCS_CONFIG_TYPE_UINT},

  {"TX_MAX_BATCH", "16",
   "Number of send WQEs to batch in one post-send list. Larger values reduce\n"
   "the CPU usage, but increase the latency and pipelining between sender and\n"
   "receiver.",
   ucs_offsetof(uct_ib_iface_config_t, tx.max_batch), UCS_CONFIG_TYPE_UINT},

  {"TX_MAX_POLL", "16",
   "Max number of receive completions to pick during TX poll",
   ucs_offsetof(uct_ib_iface_config_t, tx.max_poll), UCS_CONFIG_TYPE_UINT},

  {"TX_MIN_INLINE", "64",
   "Bytes to reserve in send WQE for inline data. Messages which are small\n"
   "enough will be sent inline.",
   ucs_offsetof(uct_ib_iface_config_t, tx.min_inline), UCS_CONFIG_TYPE_MEMUNITS},

  {"TX_INLINE_RESP", "32",
   "Bytes to reserve in send WQE for inline response. Responses which are small\n"
   "enough, such as of atomic operations and small reads, will be received inline.",
   ucs_offsetof(uct_ib_iface_config_t, tx.inl_resp), UCS_CONFIG_TYPE_MEMUNITS},

  {"TX_MIN_SGE", "3",
   "Number of SG entries to reserve in the send WQE.",
   ucs_offsetof(uct_ib_iface_config_t, tx.min_sge), UCS_CONFIG_TYPE_UINT},

#if HAVE_DECL_IBV_EXP_CQ_MODERATION
  {"TX_EVENT_MOD_COUNT", "0",
   "Number of send completions for which an event would be generated (0 - disabled).",
   ucs_offsetof(uct_ib_iface_config_t, tx.cq_moderation_count), UCS_CONFIG_TYPE_UINT},

  {"TX_EVENT_MOD_PERIOD", "0us",
   "Time period to generate send event (0 - disabled).",
   ucs_offsetof(uct_ib_iface_config_t, tx.cq_moderation_period), UCS_CONFIG_TYPE_TIME},

  {"RX_EVENT_MOD_COUNT", "0",
   "Number of received messages for which an event would be generated (0 - disabled).",
   ucs_offsetof(uct_ib_iface_config_t, rx.cq_moderation_count), UCS_CONFIG_TYPE_UINT},

  {"RX_EVENT_MOD_PERIOD", "0us",
   "Time period to generate receive event (0 - disabled).",
   ucs_offsetof(uct_ib_iface_config_t, rx.cq_moderation_period), UCS_CONFIG_TYPE_TIME},
#endif /* HAVE_DECL_IBV_EXP_CQ_MODERATION */

  UCT_IFACE_MPOOL_CONFIG_FIELDS("TX_", -1, 1024, "send",
                                ucs_offsetof(uct_ib_iface_config_t, tx.mp),
      "\nAttention: Setting this param with value != -1 is a dangerous thing\n"
      "in RC/DC and could cause deadlock or performance degradation."),

  {"RX_QUEUE_LEN", "4096",
   "Length of receive queue in the QPs.",
   ucs_offsetof(uct_ib_iface_config_t, rx.queue_len), UCS_CONFIG_TYPE_UINT},

  {"RX_MAX_BATCH", "16",
   "How many post-receives to perform in one batch.",
   ucs_offsetof(uct_ib_iface_config_t, rx.max_batch), UCS_CONFIG_TYPE_UINT},

  {"RX_MAX_POLL", "16",
   "Max number of receive completions to pick during RX poll",
   ucs_offsetof(uct_ib_iface_config_t, rx.max_poll), UCS_CONFIG_TYPE_UINT},

  {"RX_INLINE", "0",
   "Number of bytes to request for inline receive. If the maximal supported size\n"
   "is smaller, it will be used instead. If it is possible to support a larger\n"
   "size than requested with the same hardware resources, it will be used instead.",
   ucs_offsetof(uct_ib_iface_config_t, rx.inl), UCS_CONFIG_TYPE_MEMUNITS},

  UCT_IFACE_MPOOL_CONFIG_FIELDS("RX_", -1, 0, "receive",
                                ucs_offsetof(uct_ib_iface_config_t, rx.mp), ""),

  {"ADDR_TYPE", "auto",
   "Set the interface address type. \"auto\" mode detects the type according to\n"
   "link layer type and IB subnet prefix.\n"
   "Deprecated. To force use of global routing use IS_GLOBAL.",
   ucs_offsetof(uct_ib_iface_config_t, addr_type),
   UCS_CONFIG_TYPE_ENUM(uct_ib_iface_addr_types)},

  {"IS_GLOBAL", "n",
   "Force interface to use global routing.",
   ucs_offsetof(uct_ib_iface_config_t, is_global), UCS_CONFIG_TYPE_BOOL},

  {"SL", "0",
   "IB Service Level / RoCEv2 Ethernet Priority.\n",
   ucs_offsetof(uct_ib_iface_config_t, sl), UCS_CONFIG_TYPE_UINT},

  {"TRAFFIC_CLASS", "auto",
   "IB Traffic Class / RoCEv2 Differentiated Services Code Point (DSCP).\n"
   "\"auto\" option selects 106 on RoCEv2 and 0 otherwise.",
   ucs_offsetof(uct_ib_iface_config_t, traffic_class), UCS_CONFIG_TYPE_ULUNITS},

  {"HOP_LIMIT", "255",
   "IB Hop limit / RoCEv2 Time to Live. Should be between 0 and 255.\n",
   ucs_offsetof(uct_ib_iface_config_t, hop_limit), UCS_CONFIG_TYPE_UINT},

  {"LID_PATH_BITS", "0-17",
   "List of IB Path bits separated by comma (a,b,c) "
   "which will be the low portion of the LID, according to the LMC in the fabric.",
   ucs_offsetof(uct_ib_iface_config_t, lid_path_bits), UCS_CONFIG_TYPE_ARRAY(path_bits_spec)},

  {"PKEY", "0x7fff",
   "Which pkey value to use. Should be between 0 and 0x7fff.",
   ucs_offsetof(uct_ib_iface_config_t, pkey_value), UCS_CONFIG_TYPE_HEX},

#if HAVE_IBV_EXP_RES_DOMAIN
  {"RESOURCE_DOMAIN", "y",
   "Enable multiple resource domains (experimental).",
   ucs_offsetof(uct_ib_iface_config_t, enable_res_domain), UCS_CONFIG_TYPE_BOOL},
#endif


  {NULL}
};

int uct_ib_iface_is_roce(uct_ib_iface_t *iface)
{
    return uct_ib_device_is_port_roce(uct_ib_iface_device(iface),
                                      iface->config.port_num);
}

static void uct_ib_iface_recv_desc_init(uct_iface_h tl_iface, void *obj, uct_mem_h memh)
{
    uct_ib_iface_recv_desc_t *desc = obj;
    uct_ib_mem_t *ib_memh = memh;

    desc->lkey = ib_memh->lkey;
}

ucs_status_t uct_ib_iface_recv_mpool_init(uct_ib_iface_t *iface,
                                          const uct_ib_iface_config_t *config,
                                          const char *name, ucs_mpool_t *mp)
{
    unsigned grow;

    if (config->rx.queue_len < 1024) {
        grow = 1024;
    } else {
        /* We want to have some free (+10%) elements to avoid mem pool expansion */
        grow = ucs_min( (int)(1.1 * config->rx.queue_len + 0.5),
                        config->rx.mp.max_bufs);
    }

    return uct_iface_mpool_init(&iface->super, mp,
                                iface->config.rx_payload_offset + iface->config.seg_size,
                                iface->config.rx_hdr_offset,
                                UCS_SYS_CACHE_LINE_SIZE,
                                &config->rx.mp, grow,
                                uct_ib_iface_recv_desc_init,
                                name);
}

void uct_ib_iface_release_desc(uct_recv_desc_t *self, void *desc)
{
    uct_ib_iface_t *iface = ucs_container_of(self, uct_ib_iface_t, release_desc);
    void *ib_desc;

    ib_desc = desc - iface->config.rx_headroom_offset;
    ucs_mpool_put_inline(ib_desc);
}

size_t uct_ib_address_size(const union ibv_gid *gid, uint8_t is_global_addr,
                           int is_link_layer_eth)
{
    if (is_link_layer_eth) {
        return sizeof(uct_ib_address_t) +
               sizeof(union ibv_gid);  /* raw gid */
    } else if ((gid->global.subnet_prefix == UCT_IB_LINK_LOCAL_PREFIX) &&
               !is_global_addr) {
        return sizeof(uct_ib_address_t) +
               sizeof(uint16_t); /* lid */
    } else if (((gid->global.subnet_prefix & UCT_IB_SITE_LOCAL_MASK) ==
                UCT_IB_SITE_LOCAL_PREFIX) && !is_global_addr) {
        return sizeof(uct_ib_address_t) +
               sizeof(uint16_t) + /* lid */
               sizeof(uint64_t) + /* if_id */
               sizeof(uint16_t);  /* subnet16 */
    } else {
        return sizeof(uct_ib_address_t) +
               sizeof(uint16_t) + /* lid */
               sizeof(uint64_t) + /* if_id */
               sizeof(uint64_t);  /* subnet64 */
    }
}

size_t uct_ib_iface_address_size(uct_ib_iface_t *iface)
{
    return uct_ib_address_size(&iface->gid, iface->is_global_addr, uct_ib_iface_is_roce(iface));
}

void uct_ib_address_pack(const union ibv_gid *gid, uint16_t lid,
                         int is_link_layer_eth, uint8_t is_global_addr,
                         uct_ib_address_t *ib_addr)
{
    void *ptr = ib_addr + 1;

    if (is_link_layer_eth) {
        /* RoCE, in this case we don't use the lid and set the GID flag */
        ib_addr->flags = UCT_IB_ADDRESS_FLAG_LINK_LAYER_ETH |
                         UCT_IB_ADDRESS_FLAG_GID;
        /* uint8_t raw[16]; */
        memcpy(ptr, gid->raw, sizeof(gid->raw) * sizeof(uint8_t));
    } else {
        /* IB, LID */
        ib_addr->flags   = UCT_IB_ADDRESS_FLAG_LINK_LAYER_IB |
                           UCT_IB_ADDRESS_FLAG_LID;
        *(uint16_t*) ptr = lid;
        ptr             += sizeof(uint16_t);

        if ((gid->global.subnet_prefix != UCT_IB_LINK_LOCAL_PREFIX) ||
            is_global_addr) {
            ib_addr->flags  |= UCT_IB_ADDRESS_FLAG_IF_ID;
            *(uint64_t*) ptr = gid->global.interface_id;
            ptr += sizeof(uint64_t);

            if (((gid->global.subnet_prefix & UCT_IB_SITE_LOCAL_MASK) ==
                                              UCT_IB_SITE_LOCAL_PREFIX) &&
                !is_global_addr) {
                /* Site-local */
                ib_addr->flags  |= UCT_IB_ADDRESS_FLAG_SUBNET16;
                *(uint16_t*) ptr = gid->global.subnet_prefix >> 48;
            } else {
                /* Global */
                ib_addr->flags  |= UCT_IB_ADDRESS_FLAG_SUBNET64;
                *(uint64_t*) ptr = gid->global.subnet_prefix;
            }
        }
    }
}

void uct_ib_iface_address_pack(uct_ib_iface_t *iface, const union ibv_gid *gid,
                               uint16_t lid, uct_ib_address_t *ib_addr)
{
    uct_ib_address_pack(gid, lid, uct_ib_iface_is_roce(iface),
                        iface->is_global_addr, ib_addr);
}

void uct_ib_address_unpack(const uct_ib_address_t *ib_addr, uint16_t *lid,
                           union ibv_gid *gid)
{
    const void *ptr = ib_addr + 1;

    *lid                      = 0;

    if (ib_addr->flags & UCT_IB_ADDRESS_FLAG_GID) {
        memcpy(gid->raw, ptr, sizeof(gid->raw) * sizeof(uint8_t)); /* uint8_t raw[16]; */
        ucs_assert(ib_addr->flags & UCT_IB_ADDRESS_FLAG_LINK_LAYER_ETH);
        ucs_assert(!(ib_addr->flags & UCT_IB_ADDRESS_FLAG_LID));
        return;
    }

    gid->global.subnet_prefix = UCT_IB_LINK_LOCAL_PREFIX; /* Default prefix */
    gid->global.interface_id  = 0;

    if (ib_addr->flags & UCT_IB_ADDRESS_FLAG_LID) {
        *lid = *(uint16_t*)ptr;
        ptr += sizeof(uint16_t);
    }

    if (ib_addr->flags & UCT_IB_ADDRESS_FLAG_IF_ID) {
        gid->global.interface_id = *(uint64_t*)ptr;
        ptr += sizeof(uint64_t);
    }

    if (ib_addr->flags & UCT_IB_ADDRESS_FLAG_SUBNET16) {
        gid->global.subnet_prefix = UCT_IB_SITE_LOCAL_PREFIX |
                                    ((uint64_t) *(uint16_t*) ptr << 48);
        ptr += sizeof(uint16_t);
    }

    if (ib_addr->flags & UCT_IB_ADDRESS_FLAG_SUBNET64) {
        gid->global.subnet_prefix = *(uint64_t*) ptr;
    }
}

const char *uct_ib_address_str(const uct_ib_address_t *ib_addr, char *buf,
                               size_t max)
{
    union ibv_gid gid;
    uint16_t lid;
    char *p, *endp;

    uct_ib_address_unpack(ib_addr, &lid, &gid);

    p    = buf;
    endp = buf + max;
    if (lid != 0) {
        snprintf(p, endp - p, "lid %d ", lid);
        p += strlen(p);
    }
    inet_ntop(AF_INET6, &gid, p, endp - p);

    return buf;
}

ucs_status_t uct_ib_iface_get_device_address(uct_iface_h tl_iface,
                                             uct_device_addr_t *dev_addr)
{
    uct_ib_iface_t   *iface   = ucs_derived_of(tl_iface, uct_ib_iface_t);

    uct_ib_iface_address_pack(iface, &iface->gid, uct_ib_iface_port_attr(iface)->lid,
                              (void*)dev_addr);
    return UCS_OK;
}

int uct_ib_iface_is_reachable(const uct_iface_h tl_iface, const uct_device_addr_t *dev_addr,
                              const uct_iface_addr_t *iface_addr)
{
    uct_ib_iface_t *iface = ucs_derived_of(tl_iface, uct_ib_iface_t);
    int is_local_eth = uct_ib_iface_is_roce(iface);
    const uct_ib_address_t *ib_addr = (const void*)dev_addr;
    union ibv_gid gid;
    uint16_t lid;

    uct_ib_address_unpack(ib_addr, &lid, &gid);

    if (!is_local_eth && (ib_addr->flags & UCT_IB_ADDRESS_FLAG_LINK_LAYER_IB)) {
        /* same subnet prefix */
        return gid.global.subnet_prefix == iface->gid.global.subnet_prefix;
    } else if (is_local_eth && (ib_addr->flags & UCT_IB_ADDRESS_FLAG_LINK_LAYER_ETH)) {
        /* there shouldn't be a lid and the gid flag should be on */
        ucs_assert(ib_addr->flags & UCT_IB_ADDRESS_FLAG_GID);
        ucs_assert(!(ib_addr->flags & UCT_IB_ADDRESS_FLAG_LID));
        return 1;
    } else {
        /* local and remote have different link layers and therefore are unreachable */
        return 0;
    }
}

ucs_status_t uct_ib_iface_create_ah(uct_ib_iface_t *iface,
                                    struct ibv_ah_attr *ah_attr,
                                    struct ibv_ah **ah_p)
{
    return uct_ib_device_create_ah_cached(uct_ib_iface_device(iface), ah_attr,
                                          uct_ib_iface_md(iface)->pd, ah_p);
}

static ucs_status_t uct_ib_iface_init_pkey(uct_ib_iface_t *iface,
                                           const uct_ib_iface_config_t *config)
{
    uct_ib_device_t *dev  = uct_ib_iface_device(iface);
    uint16_t pkey_tbl_len = uct_ib_iface_port_attr(iface)->pkey_tbl_len;
    int pkey_already_set  = 0;
    uint16_t pkey_index, port_pkey, pkey;

    if (config->pkey_value > UCT_IB_PKEY_PARTITION_MASK) {
        ucs_error("Requested pkey 0x%x is invalid, should be in the range 0..0x%x",
                  config->pkey_value, UCT_IB_PKEY_PARTITION_MASK);
        return UCS_ERR_INVALID_PARAM;
    }

    /* get the user's pkey value and find its index in the port's pkey table */
    for (pkey_index = 0; pkey_index < pkey_tbl_len; ++pkey_index) {
        /* get the pkey values from the port's pkeys table */
        if (ibv_query_pkey(dev->ibv_context, iface->config.port_num, pkey_index,
                           &port_pkey))
        {
            ucs_error("ibv_query_pkey("UCT_IB_IFACE_FMT", index=%d) failed: %m",
                      UCT_IB_IFACE_ARG(iface), pkey_index);
        }

        pkey = ntohs(port_pkey);
        if (!(pkey & UCT_IB_PKEY_MEMBERSHIP_MASK)) {
            /* if pkey = 0x0, just skip it w/o debug trace, because 0x0
             * means that there is no real pkey configured at this index */
            if (pkey) {
                ucs_debug("skipping send-only pkey[%d]=0x%x", pkey_index, pkey);
            }
            continue;
        }

        /* take only the lower 15 bits for the comparison */
        if (!pkey_already_set ||
            ((pkey & UCT_IB_PKEY_PARTITION_MASK) == config->pkey_value)) {
            iface->pkey_index = pkey_index;
            iface->pkey_value = pkey;
            pkey_already_set  = 1;
        }
    }

    if (!pkey_already_set) {
        ucs_error("There is no valid pkey with full membership on %s:%d",
                  uct_ib_device_name(dev), iface->config.port_num);
        return UCS_ERR_INVALID_PARAM;
    }

    ucs_debug("using pkey[%d] 0x%x on "UCT_IB_IFACE_FMT, iface->pkey_index,
              iface->pkey_value, UCT_IB_IFACE_ARG(iface));
    return UCS_OK;
}

static ucs_status_t uct_ib_iface_init_lmc(uct_ib_iface_t *iface,
                                          const uct_ib_iface_config_t *config)
{
    unsigned i, j, num_path_bits;
    unsigned first, last;
    uint8_t lmc;
    int step;

    if (config->lid_path_bits.count == 0) {
        ucs_error("List of path bits must not be empty");
        return UCS_ERR_INVALID_PARAM;
    }

    /* count the number of lid_path_bits */
    num_path_bits = 0;
    for (i = 0; i < config->lid_path_bits.count; i++) {
        num_path_bits += 1 + abs((int)(config->lid_path_bits.ranges[i].first -
                                       config->lid_path_bits.ranges[i].last));
    }

    iface->path_bits = ucs_calloc(1, num_path_bits * sizeof(*iface->path_bits),
                                  "ib_path_bits");
    if (iface->path_bits == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    lmc = uct_ib_iface_port_attr(iface)->lmc;

    /* go over the list of values (ranges) for the lid_path_bits and set them */
    iface->path_bits_count = 0;
    for (i = 0; i < config->lid_path_bits.count; ++i) {

        first = config->lid_path_bits.ranges[i].first;
        last  = config->lid_path_bits.ranges[i].last;

        /* range of values or one value */
        if (first < last) {
            step = 1;
        } else {
            step = -1;
        }

        /* fill the value/s */
        for (j = first; j != (last + step); j += step) {
            if (j >= UCS_BIT(lmc)) {
                ucs_debug("Not using value %d for path_bits - must be < 2^lmc (lmc=%d)",
                          j, lmc);
                if (step == 1) {
                    break;
                } else {
                    continue;
                }
            }

            ucs_assert(iface->path_bits_count < num_path_bits);
            iface->path_bits[iface->path_bits_count] = j;
            iface->path_bits_count++;
        }
    }

    return UCS_OK;
}

void uct_ib_iface_fill_attr(uct_ib_iface_t *iface, uct_ib_qp_attr_t *attr)
{
    attr->ibv.send_cq             = iface->cq[UCT_IB_DIR_TX];
    attr->ibv.recv_cq             = iface->cq[UCT_IB_DIR_RX];

    attr->ibv.srq                 = attr->srq;
    attr->ibv.cap                 = attr->cap;
    attr->ibv.qp_type             = attr->qp_type;
    attr->ibv.sq_sig_all          = attr->sq_sig_all;

#if HAVE_DECL_IBV_EXP_CREATE_QP
    if (!(attr->ibv.comp_mask & IBV_EXP_QP_INIT_ATTR_PD)) {
        attr->ibv.comp_mask       = IBV_EXP_QP_INIT_ATTR_PD;
        attr->ibv.pd              = uct_ib_iface_md(iface)->pd;
    }
#elif HAVE_DECL_IBV_CREATE_QP_EX
    if (!(attr->ibv.comp_mask & IBV_QP_INIT_ATTR_PD)) {
        attr->ibv.comp_mask       = IBV_QP_INIT_ATTR_PD;
        attr->ibv.pd              = uct_ib_iface_md(iface)->pd;
    }
#endif

    attr->port                    = iface->config.port_num;

    if (attr->qp_type == IBV_QPT_UD) {
        return;
    }

    /* MOFED requires this to enable IB spec atomic */
#if HAVE_DECL_IBV_EXP_ATOMIC_HCA_REPLY_BE
    if (uct_ib_iface_device(iface)->dev_attr.exp_atomic_cap ==
                                     IBV_EXP_ATOMIC_HCA_REPLY_BE) {
        attr->ibv.comp_mask       |= IBV_EXP_QP_INIT_ATTR_CREATE_FLAGS;
        attr->ibv.exp_create_flags = IBV_EXP_QP_CREATE_ATOMIC_BE_REPLY;
    }
#endif
}

ucs_status_t uct_ib_iface_create_qp(uct_ib_iface_t *iface,
                                    uct_ib_qp_attr_t *attr,
                                    struct ibv_qp **qp_p)
{
    uct_ib_device_t *dev = uct_ib_iface_device(iface);
    struct ibv_qp *qp;

    uct_ib_iface_fill_attr(iface, attr);

#if HAVE_DECL_IBV_EXP_CREATE_QP
    qp = ibv_exp_create_qp(dev->ibv_context, &attr->ibv);
#elif HAVE_DECL_IBV_CREATE_QP_EX
    qp = ibv_create_qp_ex(dev->ibv_context, &attr->ibv);
#else
    qp = ibv_create_qp(uct_ib_iface_md(iface)->pd, &attr->ibv);
#endif
    if (qp == NULL) {
        ucs_error("iface=%p: failed to create %s QP TX wr:%d sge:%d inl:%d RX wr:%d sge:%d inl %d: %m",
                  iface, uct_ib_qp_type_str(attr->qp_type),
                  attr->cap.max_send_wr, attr->cap.max_send_sge, attr->cap.max_inline_data,
                  attr->cap.max_recv_wr, attr->cap.max_recv_sge, attr->max_inl_recv);
        return UCS_ERR_IO_ERROR;
    }

    attr->cap  = attr->ibv.cap;
    *qp_p      = qp;

    ucs_debug("iface=%p: created %s QP 0x%x on %s:%d TX wr:%d sge:%d inl:%d RX wr:%d sge:%d inl %d",
              iface, uct_ib_qp_type_str(attr->qp_type), qp->qp_num,
              uct_ib_device_name(dev), iface->config.port_num,
              attr->cap.max_send_wr, attr->cap.max_send_sge, attr->cap.max_inline_data,
              attr->cap.max_recv_wr, attr->cap.max_recv_sge, attr->max_inl_recv);

    return UCS_OK;
}

ucs_status_t uct_ib_verbs_create_cq(struct ibv_context *context, int cqe,
                                    struct ibv_comp_channel *channel,
                                    int comp_vector, int ignore_overrun,
                                    size_t *inl, struct ibv_cq **cq_p)
{
    struct ibv_cq *cq;
#if HAVE_DECL_IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN
    struct ibv_cq_init_attr_ex cq_attr = {};

    cq_attr.cqe = cqe;
    cq_attr.channel = channel;
    cq_attr.comp_vector = comp_vector;
    if (ignore_overrun) {
        cq_attr.comp_mask = IBV_CQ_INIT_ATTR_MASK_FLAGS;
        cq_attr.flags = IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN;
    }

    cq = ibv_cq_ex_to_cq(ibv_create_cq_ex(context, &cq_attr));
    if (!cq && (errno == ENOSYS))
#endif
    {
        *inl = 0;
        cq = ibv_create_cq(context, cqe, NULL, channel, comp_vector);
    }

    if (!cq) {
        ucs_error("ibv_create_cq(cqe=%d) failed: %m", cqe);
        return UCS_ERR_IO_ERROR;
    }

    *cq_p = cq;
    return UCS_OK;
}

static ucs_status_t uct_ib_iface_create_cq(uct_ib_iface_t *iface, int cq_length,
                                           size_t *inl, int preferred_cpu,
                                           int flags, struct ibv_cq **cq_p)
{
    uct_ib_device_t *dev = uct_ib_iface_device(iface);
    ucs_status_t status;
#if HAVE_DECL_IBV_EXP_SETENV && !HAVE_DECL_MLX5DV_CQ_INIT_ATTR_MASK_CQE_SIZE
    static const char *cqe_size_env_var = "MLX5_CQE_SIZE";
    const char *cqe_size_env_value;
    size_t cqe_size = 64;
    size_t cqe_size_min;
    char cqe_size_buf[32];
    int env_var_added = 0;
    int ret;

    cqe_size_min       = (*inl > 32) ? 128 : 64;
    cqe_size_env_value = getenv(cqe_size_env_var);

    if (cqe_size_env_value != NULL) {
        cqe_size = atol(cqe_size_env_value);
        if (cqe_size < cqe_size_min) {
            ucs_error("%s is set to %zu, but at least %zu is required (inl: %zu)",
                      cqe_size_env_var, cqe_size, cqe_size_min, *inl);
            return UCS_ERR_INVALID_PARAM;
        }
    } else {
        cqe_size = uct_ib_get_cqe_size(cqe_size_min);
        snprintf(cqe_size_buf, sizeof(cqe_size_buf),"%zu", cqe_size);
        ucs_debug("%s: setting %s=%s", uct_ib_device_name(dev), cqe_size_env_var,
                  cqe_size_buf);
        ret = ibv_exp_setenv(dev->ibv_context, cqe_size_env_var, cqe_size_buf, 1);
        if (ret) {
            ucs_error("ibv_exp_setenv(%s=%s) failed: %m", cqe_size_env_var,
                      cqe_size_buf);
            return UCS_ERR_INVALID_PARAM;
        }

        env_var_added = 1;
    }
#endif
    status = iface->ops->create_cq(dev->ibv_context, cq_length,
                                   iface->comp_channel, preferred_cpu,
                                   flags & UCT_IB_CQ_IGNORE_OVERRUN, inl, cq_p);
    if (status != UCS_OK) {
        goto out_unsetenv;
    }

    status = UCS_OK;

out_unsetenv:
#if HAVE_DECL_IBV_EXP_SETENV && !HAVE_DECL_MLX5DV_CQ_INIT_ATTR_MASK_CQE_SIZE
    *inl = cqe_size / 2;
    if (env_var_added) {
        /* if we created a new environment variable, remove it */
        ret = ibv_exp_unsetenv(dev->ibv_context, cqe_size_env_var);
        if (ret) {
            ucs_warn("unsetenv(%s) failed: %m", cqe_size_env_var);
        }
    }
#endif
    return status;
}


static ucs_status_t uct_ib_iface_set_moderation(struct ibv_cq *cq,
                                                unsigned count, double period_usec)
{
#if HAVE_DECL_IBV_EXP_CQ_MODERATION
    unsigned period = (unsigned)(period_usec * UCS_USEC_PER_SEC);

    if (count > UINT16_MAX) {
        ucs_error("CQ moderation count is too high: %u, max value: %u", count, UINT16_MAX);
        return UCS_ERR_INVALID_PARAM;
    } else if (count == 0) {
        /* in case if count value is 0 (unchanged default value) - set it to maximum
         * possible value */
        count = UINT16_MAX;
    }

    if (period > UINT16_MAX) {
        ucs_error("CQ moderation period is too high: %u, max value: %uus", period, UINT16_MAX);
        return UCS_ERR_INVALID_PARAM;
    } else if (period == 0) {
        /* in case if count value is 0 (unchanged default value) - set it to maximum
         * possible value, the same behavior as counter */
        period = UINT16_MAX;
    }

    if ((count < UINT16_MAX) || (period < UINT16_MAX)) {
        struct ibv_exp_cq_attr cq_attr = {
            .comp_mask            = IBV_EXP_CQ_ATTR_MODERATION,
            .moderation.cq_count  = (uint16_t)(count),
            .moderation.cq_period = (uint16_t)(period),
            .cq_cap_flags         = 0
        };
        if (ibv_exp_modify_cq(cq, &cq_attr, IBV_EXP_CQ_MODERATION)) {
            ucs_error("ibv_exp_modify_cq(count=%d, period=%d) failed: %m", count, period);
            return UCS_ERR_IO_ERROR;
        }
    }
#endif /* HAVE_DECL_IBV_EXP_CQ_MODERATION */

    return UCS_OK;
}

UCS_CLASS_INIT_FUNC(uct_ib_iface_t, uct_ib_iface_ops_t *ops, uct_md_h md,
                    uct_worker_h worker, const uct_iface_params_t *params,
                    const uct_ib_iface_config_t *config,
                    const uct_ib_iface_init_attr_t *init_attr)
{
    uct_ib_md_t *ib_md    = ucs_derived_of(md, uct_ib_md_t);
    uct_ib_device_t *dev = &ib_md->dev;
    size_t rx_headroom   = (params->field_mask &
                            UCT_IFACE_PARAM_FIELD_RX_HEADROOM) ?
                           params->rx_headroom : 0;
    ucs_cpu_set_t cpu_mask;
    int preferred_cpu;
    ucs_status_t status;
    uint8_t port_num;
    int is_roce_v2;
    size_t inl;

    if (!(params->open_mode & UCT_IFACE_OPEN_MODE_DEVICE)) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (params->field_mask & UCT_IFACE_PARAM_FIELD_CPU_MASK) {
        cpu_mask = params->cpu_mask;
    } else {
        memset(&cpu_mask, 0, sizeof(cpu_mask));
    }

    preferred_cpu = ucs_cpu_set_find_lcs(&cpu_mask);

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &ops->super, md, worker,
                              params, &config->super
                              UCS_STATS_ARG(((params->field_mask &
                                              UCT_IFACE_PARAM_FIELD_STATS_ROOT) &&
                                             (params->stats_root != NULL)) ?
                                            params->stats_root :
                                            dev->stats)
                              UCS_STATS_ARG(params->mode.device.dev_name));

    status = uct_ib_device_find_port(dev, params->mode.device.dev_name, &port_num);
    if (status != UCS_OK) {
        goto err;
    }

    self->ops                       = ops;

    self->config.rx_payload_offset  = sizeof(uct_ib_iface_recv_desc_t) +
                                      ucs_max(sizeof(uct_recv_desc_t) +
                                              rx_headroom,
                                              init_attr->rx_priv_len +
                                              init_attr->rx_hdr_len);
    self->config.rx_hdr_offset      = self->config.rx_payload_offset -
                                      init_attr->rx_hdr_len;
    self->config.rx_headroom_offset = self->config.rx_payload_offset -
                                      rx_headroom;
    self->config.seg_size           = init_attr->seg_size;
    self->config.tx_max_poll        = config->tx.max_poll;
    self->config.rx_max_poll        = config->rx.max_poll;
    self->config.rx_max_batch       = ucs_min(config->rx.max_batch,
                                              config->rx.queue_len / 4);
    self->config.port_num           = port_num;
    self->config.sl                 = config->sl;
    self->config.hop_limit          = config->hop_limit;
    self->release_desc.cb           = uct_ib_iface_release_desc;
    self->config.enable_res_domain  = config->enable_res_domain;
    self->config.qp_type            = init_attr->qp_type;

    if (ucs_derived_of(worker, uct_priv_worker_t)->thread_mode == UCS_THREAD_MODE_MULTI) {
        ucs_error("IB transports do not support multi-threaded worker");
        return UCS_ERR_INVALID_PARAM;
    }

    status = uct_ib_iface_init_pkey(self, config);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_ib_device_select_gid_index(dev, self->config.port_num,
                                            ib_md->config.gid_index,
                                            &self->config.gid_index);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_ib_device_query_gid(dev, self->config.port_num,
                                     self->config.gid_index, &self->gid,
                                     &is_roce_v2);
    if (status != UCS_OK) {
        goto err;
    }

    if (config->traffic_class == UCS_ULUNITS_AUTO) {
        self->config.traffic_class = is_roce_v2 ? UCT_IB_DEFAULT_ROCEV2_DSCP : 0;
    } else {
        self->config.traffic_class = config->traffic_class;
    }

    status = uct_ib_iface_init_lmc(self, config);
    if (status != UCS_OK) {
        goto err;
    }

    self->comp_channel = ibv_create_comp_channel(dev->ibv_context);
    if (self->comp_channel == NULL) {
        ucs_error("ibv_create_comp_channel() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_cleanup;
    }

    status = ucs_sys_fcntl_modfl(self->comp_channel->fd, O_NONBLOCK, 0);
    if (status != UCS_OK) {
        goto err_destroy_comp_channel;
    }

    inl = config->rx.inl;
    status = uct_ib_iface_create_cq(self, init_attr->tx_cq_len, &inl,
                                    preferred_cpu, init_attr->flags,
                                    &self->cq[UCT_IB_DIR_TX]);
    if (status != UCS_OK) {
        goto err_destroy_comp_channel;
    }
    ucs_assert_always(inl <= UINT8_MAX);
    self->config.max_inl_resp = inl;

    status = uct_ib_iface_set_moderation(self->cq[UCT_IB_DIR_TX],
                                         config->tx.cq_moderation_count,
                                         config->tx.cq_moderation_period);
    if (status != UCS_OK) {
        goto err_destroy_send_cq;
    }

    inl = config->rx.inl;
    status = uct_ib_iface_create_cq(self, init_attr->rx_cq_len, &inl,
                                    preferred_cpu, init_attr->flags,
                                    &self->cq[UCT_IB_DIR_RX]);
    if (status != UCS_OK) {
        goto err_destroy_send_cq;
    }

    status = uct_ib_iface_set_moderation(self->cq[UCT_IB_DIR_RX],
                                         config->rx.cq_moderation_count,
                                         config->rx.cq_moderation_period);
    if (status != UCS_OK) {
        goto err_destroy_recv_cq;
    }

    /* Address scope and size */
    if (uct_ib_iface_is_roce(self) || config->is_global ||
        /* check ADDR_TYPE for backward compatibility */
        (config->addr_type == UCT_IB_ADDRESS_TYPE_SITE_LOCAL) ||
        (config->addr_type == UCT_IB_ADDRESS_TYPE_GLOBAL)) {
        self->is_global_addr = 1;
    } else {
        self->is_global_addr = 0;
    }

    self->addr_size  = uct_ib_iface_address_size(self);

    ucs_debug("created uct_ib_iface_t headroom_ofs %d payload_ofs %d hdr_ofs %d data_sz %d",
              self->config.rx_headroom_offset, self->config.rx_payload_offset,
              self->config.rx_hdr_offset, self->config.seg_size);

    return UCS_OK;

err_destroy_recv_cq:
    ibv_destroy_cq(self->cq[UCT_IB_DIR_RX]);
err_destroy_send_cq:
    ibv_destroy_cq(self->cq[UCT_IB_DIR_TX]);
err_destroy_comp_channel:
    ibv_destroy_comp_channel(self->comp_channel);
err_cleanup:
    ucs_free(self->path_bits);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ib_iface_t)
{
    int ret;

    ret = ibv_destroy_cq(self->cq[UCT_IB_DIR_RX]);
    if (ret != 0) {
        ucs_warn("ibv_destroy_cq(recv_cq) returned %d: %m", ret);
    }

    ret = ibv_destroy_cq(self->cq[UCT_IB_DIR_TX]);
    if (ret != 0) {
        ucs_warn("ibv_destroy_cq(send_cq) returned %d: %m", ret);
    }

    ret = ibv_destroy_comp_channel(self->comp_channel);
    if (ret != 0) {
        ucs_warn("ibv_destroy_comp_channel(comp_channel) returned %d: %m", ret);
    }

    ucs_free(self->path_bits);
}

UCS_CLASS_DEFINE(uct_ib_iface_t, uct_base_iface_t);

int uct_ib_iface_prepare_rx_wrs(uct_ib_iface_t *iface, ucs_mpool_t *mp,
                                uct_ib_recv_wr_t *wrs, unsigned n)
{
    uct_ib_iface_recv_desc_t *desc;
    unsigned count;

    count = 0;
    while (count < n) {
        UCT_TL_IFACE_GET_RX_DESC(&iface->super, mp, desc, break);
        wrs[count].sg.addr   = (uintptr_t)uct_ib_iface_recv_desc_hdr(iface, desc);
        wrs[count].sg.length = iface->config.rx_payload_offset + iface->config.seg_size;
        wrs[count].sg.lkey   = desc->lkey;
        wrs[count].ibwr.num_sge = 1;
        wrs[count].ibwr.wr_id   = (uintptr_t)desc;
        wrs[count].ibwr.sg_list = &wrs[count].sg;
        wrs[count].ibwr.next    = &wrs[count + 1].ibwr;
        ++count;
    }

    if (count > 0) {
        wrs[count - 1].ibwr.next = NULL;
    }

    return count;
}

static ucs_status_t uct_ib_iface_get_numa_latency(uct_ib_iface_t *iface,
                                                  double *latency)
{
    uct_ib_device_t *dev = uct_ib_iface_device(iface);
    uct_ib_md_t *md      = uct_ib_iface_md(iface);
    cpu_set_t temp_cpu_mask, process_affinity;
#if HAVE_NUMA
    int distance, min_cpu_distance;
    int cpu, num_cpus;
#endif
    int ret;

    if (!md->config.prefer_nearest_device) {
        *latency = 0;
        return UCS_OK;
    }

    ret = sched_getaffinity(0, sizeof(process_affinity), &process_affinity);
    if (ret) {
        ucs_error("sched_getaffinity() failed: %m");
        return UCS_ERR_INVALID_PARAM;
    }

#if HAVE_NUMA
    /* Try to estimate the extra device latency according to NUMA distance */
    if (dev->numa_node != -1) {
        min_cpu_distance = INT_MAX;
        num_cpus         = ucs_min(CPU_SETSIZE, numa_num_configured_cpus());
        for (cpu = 0; cpu < num_cpus; ++cpu) {
            if (!CPU_ISSET(cpu, &process_affinity)) {
                continue;
            }
            distance = numa_distance(ucs_numa_node_of_cpu(cpu), dev->numa_node);
            if (distance >= UCS_NUMA_MIN_DISTANCE) {
                min_cpu_distance = ucs_min(min_cpu_distance, distance);
            }
        }

        if (min_cpu_distance != INT_MAX) {
            /* set the extra latency to (numa_distance - 10) * 20nsec */
            *latency = (min_cpu_distance - UCS_NUMA_MIN_DISTANCE) * 20e-9;
            return UCS_OK;
        }
    }
#endif

    /* Estimate the extra device latency according to its local CPUs mask */
    CPU_AND(&temp_cpu_mask, &dev->local_cpus, &process_affinity);
    if (CPU_EQUAL(&process_affinity, &temp_cpu_mask)) {
        *latency = 0;
    } else {
        *latency = 200e-9;
    }
    return UCS_OK;
}

ucs_status_t uct_ib_iface_query(uct_ib_iface_t *iface, size_t xport_hdr_len,
                                uct_iface_attr_t *iface_attr)
{
    uct_ib_device_t *dev = uct_ib_iface_device(iface);
    uct_ib_md_t     *md  = uct_ib_iface_md(iface);
    static const unsigned ib_port_widths[] = {
        [0] = 1,
        [1] = 4,
        [2] = 8,
        [3] = 12,
        [4] = 16
    };
    uint8_t active_width, active_speed, active_mtu, width_idx;
    double encoding, signal_rate, wire_speed;
    size_t mtu, width, extra_pkt_len;
    ucs_status_t status;
    double numa_latency;

    uct_base_iface_query(&iface->super, iface_attr);
    
    active_width = uct_ib_iface_port_attr(iface)->active_width;
    active_speed = uct_ib_iface_port_attr(iface)->active_speed;
    active_mtu   = uct_ib_iface_port_attr(iface)->active_mtu;

    /* Get active width */
    width_idx = ucs_ilog2(active_width);
    if (!ucs_is_pow2(active_width) ||
        (active_width < 1) || (width_idx > 4))
    {
        ucs_error("Invalid active_width on %s:%d: %d",
                  UCT_IB_IFACE_ARG(iface), active_width);
        return UCS_ERR_IO_ERROR;
    }

    iface_attr->device_addr_len = iface->addr_size;

    switch (active_speed) {
    case 1: /* SDR */
        iface_attr->latency.overhead = 5000e-9;
        signal_rate                  = 2.5e9;
        encoding                     = 8.0/10.0;
        break;
    case 2: /* DDR */
        iface_attr->latency.overhead = 2500e-9;
        signal_rate                  = 5.0e9;
        encoding                     = 8.0/10.0;
        break;
    case 4:
        iface_attr->latency.overhead = 1300e-9;
        if (uct_ib_iface_is_roce(iface)) {
            /* 10/40g Eth  */
            signal_rate              = 10.3125e9;
            encoding                 = 64.0/66.0;
        } else {
            /* QDR */
            signal_rate              = 10.0e9;
            encoding                 = 8.0/10.0;
        }
        break;
    case 8: /* FDR10 */
        iface_attr->latency.overhead = 700e-9;
        signal_rate                  = 10.3125e9;
        encoding                     = 64.0/66.0;
        break;
    case 16: /* FDR */
        iface_attr->latency.overhead = 700e-9;
        signal_rate                  = 14.0625e9;
        encoding                     = 64.0/66.0;
        break;
    case 32: /* EDR / 100g Eth */
        iface_attr->latency.overhead = 600e-9;
        signal_rate                  = 25.78125e9;
        encoding                     = 64.0/66.0;
        break;
    case 64: /* 50g Eth */
        iface_attr->latency.overhead = 600e-9;
        signal_rate                  = 25.78125e9 * 2;
        encoding                     = 64.0/66.0;
        break;
    default:
        ucs_error("Invalid active_speed on %s:%d: %d",
                  UCT_IB_IFACE_ARG(iface), active_speed);
        return UCS_ERR_IO_ERROR;
    }

    status = uct_ib_iface_get_numa_latency(iface, &numa_latency);
    if (status != UCS_OK) {
        return status;
    }

    iface_attr->latency.overhead += numa_latency;
    iface_attr->latency.growth    = 0;

    /* Wire speed calculation: Width * SignalRate * Encoding */
    width                 = ib_port_widths[width_idx];
    wire_speed            = (width * signal_rate * encoding) / 8.0;

    /* Calculate packet overhead  */
    mtu                   = ucs_min(uct_ib_mtu_value(active_mtu),
                                    iface->config.seg_size);

    extra_pkt_len = UCT_IB_BTH_LEN + xport_hdr_len +  UCT_IB_ICRC_LEN + UCT_IB_VCRC_LEN + UCT_IB_DELIM_LEN;

    if (uct_ib_iface_is_roce(iface)) {
        extra_pkt_len += UCT_IB_GRH_LEN + UCT_IB_ROCE_LEN;
        iface_attr->latency.overhead += 200e-9;
    } else {
        /* TODO check if UCT_IB_DELIM_LEN is present in RoCE as well */
        extra_pkt_len += UCT_IB_LRH_LEN;
    }

    iface_attr->bandwidth.shared    = ucs_min((wire_speed * mtu) / (mtu + extra_pkt_len), md->pci_bw);
    iface_attr->bandwidth.dedicated = 0;
    iface_attr->priority            = uct_ib_device_spec(dev)->priority;

    return UCS_OK;
}

ucs_status_t uct_ib_iface_event_fd_get(uct_iface_h tl_iface, int *fd_p)
{
    uct_ib_iface_t *iface = ucs_derived_of(tl_iface, uct_ib_iface_t);
    *fd_p  = iface->comp_channel->fd;
    return UCS_OK;
}

ucs_status_t uct_ib_iface_pre_arm(uct_ib_iface_t *iface)
{
    int res, send_cq_count, recv_cq_count;
    struct ibv_cq *cq;
    void *cq_context;

    send_cq_count = 0;
    recv_cq_count = 0;
    do {
        res = ibv_get_cq_event(iface->comp_channel, &cq, &cq_context);
        if (0 == res) {
            if (iface->cq[UCT_IB_DIR_TX] == cq) {
                iface->ops->event_cq(iface, UCT_IB_DIR_TX);
                ++send_cq_count;
            }
            if (iface->cq[UCT_IB_DIR_RX] == cq) {
                iface->ops->event_cq(iface, UCT_IB_DIR_RX);
                ++recv_cq_count;
            }
        }
    } while (res == 0);

    if (errno != EAGAIN) {
        return UCS_ERR_IO_ERROR;
    }

    if (send_cq_count > 0) {
        ibv_ack_cq_events(iface->cq[UCT_IB_DIR_TX], send_cq_count);
    }

    if (recv_cq_count > 0) {
        ibv_ack_cq_events(iface->cq[UCT_IB_DIR_RX], recv_cq_count);
    }

    /* avoid re-arming the interface if any events exists */
    if ((send_cq_count > 0) || (recv_cq_count > 0)) {
        ucs_trace("arm_cq: got %d send and %d recv events, returning BUSY",
                  send_cq_count, recv_cq_count);
        return UCS_ERR_BUSY;
    }

    return UCS_OK;
}

ucs_status_t uct_ib_iface_arm_cq(uct_ib_iface_t *iface,
                                 uct_ib_dir_t dir,
                                 int solicited_only)
{
    int ret;

    ret = ibv_req_notify_cq(iface->cq[dir], solicited_only);
    if (ret != 0) {
        ucs_error("ibv_req_notify_cq("UCT_IB_IFACE_FMT", %d, sol=%d) failed: %m",
                  UCT_IB_IFACE_ARG(iface), dir, solicited_only);
        return UCS_ERR_IO_ERROR;
    }
    return UCS_OK;
}
