/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

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
#include <ucs/sys/sock.h>
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

  {"TX_INLINE_RESP", "0",
   "Bytes to reserve in send WQE for inline response. Responses which are small\n"
   "enough, such as of atomic operations and small reads, will be received inline.",
   ucs_offsetof(uct_ib_iface_config_t, inl[UCT_IB_DIR_TX]), UCS_CONFIG_TYPE_MEMUNITS},

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
   ucs_offsetof(uct_ib_iface_config_t, inl[UCT_IB_DIR_RX]), UCS_CONFIG_TYPE_MEMUNITS},

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

  {"SL", "auto",
   "InfiniBand: Service level. 'auto' will select a value matching UCX_IB_AR configuration.\n"
   "RoCEv2: Ethernet Priority. 'auto' will select 0 by default.",
   ucs_offsetof(uct_ib_iface_config_t, sl), UCS_CONFIG_TYPE_ULUNITS},

  {"TRAFFIC_CLASS", "auto",
   "IB Traffic Class / RoCEv2 Differentiated Services Code Point (DSCP).\n"
   "\"auto\" option selects 106 on RoCEv2 and 0 otherwise.",
   ucs_offsetof(uct_ib_iface_config_t, traffic_class), UCS_CONFIG_TYPE_ULUNITS},

  {"HOP_LIMIT", "255",
   "IB Hop limit / RoCEv2 Time to Live. Should be between 0 and 255.\n",
   ucs_offsetof(uct_ib_iface_config_t, hop_limit), UCS_CONFIG_TYPE_UINT},

  {"NUM_PATHS", "auto",
   "Number of connections that should be created between a pair of communicating\n"
   "endpoints for optimal performance. The default value 'auto' behaves according\n"
   "to the port link layer:\n"
   " RoCE       - "UCS_PP_MAKE_STRING(UCT_IB_DEV_MAX_PORTS) " for LAG port, otherwise - 1.\n"
   " InfiniBand - As the number of path bits enabled by fabric's LMC value and selected\n"
   "              by "UCS_DEFAULT_ENV_PREFIX UCT_IB_CONFIG_PREFIX"LID_PATH_BITS configuration.",
   ucs_offsetof(uct_ib_iface_config_t, num_paths), UCS_CONFIG_TYPE_ULUNITS},

  {"ROCE_PATH_FACTOR", "1",
   "Multiplier for RoCE LAG UDP source port calculation. The UDP source port\n"
   "is typically used by switches and network adapters to select a different\n"
   "path for the same pair of endpoints.",
   ucs_offsetof(uct_ib_iface_config_t, roce_path_factor), UCS_CONFIG_TYPE_UINT},

  {"LID_PATH_BITS", "0",
   "List of IB Path bits separated by comma (a,b,c) "
   "which will be the low portion of the LID, according to the LMC in the fabric.",
   ucs_offsetof(uct_ib_iface_config_t, lid_path_bits), UCS_CONFIG_TYPE_ARRAY(path_bits_spec)},

  {"PKEY", "auto",
   "Which pkey value to use. Should be between 0 and 0x7fff.\n"
   "\"auto\" option selects a first valid pkey value with full membership.",
   ucs_offsetof(uct_ib_iface_config_t, pkey), UCS_CONFIG_TYPE_HEX},

#ifdef HAVE_IBV_EXP_RES_DOMAIN
  {"RESOURCE_DOMAIN", "y",
   "Enable multiple resource domains (experimental).",
   ucs_offsetof(uct_ib_iface_config_t, enable_res_domain), UCS_CONFIG_TYPE_BOOL},
#endif

  {"PATH_MTU", "default",
   "Path MTU. \"default\" will select the best MTU for the device.",
   ucs_offsetof(uct_ib_iface_config_t, path_mtu),
                UCS_CONFIG_TYPE_ENUM(uct_ib_mtu_values)},

  {NULL}
};

int uct_ib_iface_is_roce(uct_ib_iface_t *iface)
{
    return uct_ib_device_is_port_roce(uct_ib_iface_device(iface),
                                      iface->config.port_num);
}

int uct_ib_iface_is_ib(uct_ib_iface_t *iface)
{
    return uct_ib_device_is_port_ib(uct_ib_iface_device(iface),
                                    iface->config.port_num);
}

static void uct_ib_iface_recv_desc_init(uct_iface_h tl_iface, void *obj, uct_mem_h memh)
{
    uct_ib_iface_recv_desc_t *desc = obj;

    desc->lkey = uct_ib_memh_get_lkey(memh);
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

    ib_desc = UCS_PTR_BYTE_OFFSET(desc, -(ptrdiff_t)iface->config.rx_headroom_offset);
    ucs_mpool_put_inline(ib_desc);
}

static inline uct_ib_roce_version_t
uct_ib_address_flags_get_roce_version(uint8_t flags)
{
    ucs_assert(flags & UCT_IB_ADDRESS_FLAG_LINK_LAYER_ETH);
    return (uct_ib_roce_version_t)(flags >> ucs_ilog2(UCT_IB_ADDRESS_FLAG_ETH_LAST));
}

static inline sa_family_t
uct_ib_address_flags_get_roce_af(uint8_t flags)
{
    ucs_assert(flags & UCT_IB_ADDRESS_FLAG_LINK_LAYER_ETH);
    return (flags & UCT_IB_ADDRESS_FLAG_ROCE_IPV6) ?
           AF_INET6 : AF_INET;
}

size_t uct_ib_address_size(const uct_ib_address_pack_params_t *params)
{
    size_t size = sizeof(uct_ib_address_t);

    if (params->flags & UCT_IB_ADDRESS_PACK_FLAG_ETH) {
        /* Ethernet: address contains only raw GID */
        size += sizeof(union ibv_gid);
    } else {
        /* InfiniBand: address always contains LID */
        size += sizeof(uint16_t); /* lid */

        if (params->flags & UCT_IB_ADDRESS_PACK_FLAG_INTERFACE_ID) {
            /* Add GUID */
            UCS_STATIC_ASSERT(sizeof(params->gid.global.interface_id) == sizeof(uint64_t));
            size += sizeof(uint64_t);
        }

        if (params->flags & UCT_IB_ADDRESS_PACK_FLAG_SUBNET_PREFIX) {
            if ((params->gid.global.subnet_prefix & UCT_IB_SITE_LOCAL_MASK) ==
                                                    UCT_IB_SITE_LOCAL_PREFIX) {
                /* 16-bit subnet prefix */
                size += sizeof(uint16_t);
            } else if (params->gid.global.subnet_prefix != UCT_IB_LINK_LOCAL_PREFIX) {
                /* 64-bit subnet prefix */
                size += sizeof(uint64_t);
            }
            /* Note: if subnet prefix is LINK_LOCAL, no need to pack it because
             * it's a well-known value defined by IB specification.
             */
        }
    }

    if (params->flags & UCT_IB_ADDRESS_PACK_FLAG_PATH_MTU) {
        size += sizeof(uint8_t);
    }

    if (params->flags & UCT_IB_ADDRESS_PACK_FLAG_GID_INDEX) {
        size += sizeof(uint8_t);
    }

    if (params->flags & UCT_IB_ADDRESS_PACK_FLAG_PKEY) {
        size += sizeof(uint16_t);
    }

    return size;
}

void uct_ib_address_pack(const uct_ib_address_pack_params_t *params,
                         uct_ib_address_t *ib_addr)
{
    void *ptr = ib_addr + 1;

    if (params->flags & UCT_IB_ADDRESS_PACK_FLAG_ETH) {
        /* RoCE, in this case we don't use the lid, we pack the gid, the RoCE
         * version, address family and set the ETH flag */
        ib_addr->flags = UCT_IB_ADDRESS_FLAG_LINK_LAYER_ETH |
                         (params->roce_info.ver <<
                          ucs_ilog2(UCT_IB_ADDRESS_FLAG_ETH_LAST));

        if (params->roce_info.addr_family == AF_INET6) {
            ib_addr->flags |= UCT_IB_ADDRESS_FLAG_ROCE_IPV6;
        }

        /* uint8_t raw[16]; */
        memcpy(ptr, params->gid.raw, sizeof(params->gid.raw));
        ptr = UCS_PTR_TYPE_OFFSET(ptr, params->gid.raw);
    } else {
        /* IB, LID */
        ib_addr->flags   = 0;
        *(uint16_t*)ptr  = params->lid;
        ptr              = UCS_PTR_TYPE_OFFSET(ptr, uint16_t);

        if (params->flags & UCT_IB_ADDRESS_PACK_FLAG_INTERFACE_ID) {
            /* Pack GUID */
            ib_addr->flags  |= UCT_IB_ADDRESS_FLAG_IF_ID;
            *(uint64_t*) ptr = params->gid.global.interface_id;
            ptr              = UCS_PTR_TYPE_OFFSET(ptr, uint64_t);
        }

        if (params->flags & UCT_IB_ADDRESS_PACK_FLAG_SUBNET_PREFIX) {
            if ((params->gid.global.subnet_prefix & UCT_IB_SITE_LOCAL_MASK) ==
                                                    UCT_IB_SITE_LOCAL_PREFIX) {
                /* Site-local */
                ib_addr->flags |= UCT_IB_ADDRESS_FLAG_SUBNET16;
                *(uint16_t*)ptr = params->gid.global.subnet_prefix >> 48;
                ptr             = UCS_PTR_TYPE_OFFSET(ptr, uint16_t);
            } else if (params->gid.global.subnet_prefix != UCT_IB_LINK_LOCAL_PREFIX) {
                /* Global */
                ib_addr->flags |= UCT_IB_ADDRESS_FLAG_SUBNET64;
                *(uint64_t*)ptr = params->gid.global.subnet_prefix;
                ptr             = UCS_PTR_TYPE_OFFSET(ptr, uint64_t);
            }
        }
    }

    if (params->flags & UCT_IB_ADDRESS_PACK_FLAG_PATH_MTU) {
        ucs_assert((int)params->path_mtu < UINT8_MAX);
        ib_addr->flags |= UCT_IB_ADDRESS_FLAG_PATH_MTU;
        *(uint8_t*)ptr  = (uint8_t)params->path_mtu;
        ptr             = UCS_PTR_TYPE_OFFSET(ptr, uint8_t);
    }

    if (params->flags & UCT_IB_ADDRESS_PACK_FLAG_GID_INDEX) {
        ib_addr->flags |= UCT_IB_ADDRESS_FLAG_GID_INDEX;
        *(uint8_t*)ptr  = params->gid_index;
    }

    if (params->flags & UCT_IB_ADDRESS_PACK_FLAG_PKEY) {
        ucs_assert(params->pkey != UCT_IB_ADDRESS_DEFAULT_PKEY);
        ib_addr->flags |= UCT_IB_ADDRESS_FLAG_PKEY;
        *(uint16_t*)ptr = params->pkey;
    }
}

unsigned uct_ib_iface_address_pack_flags(uct_ib_iface_t *iface)
{
    unsigned pack_flags = 0;

    if (iface->pkey != UCT_IB_ADDRESS_DEFAULT_PKEY) {
        pack_flags |= UCT_IB_ADDRESS_PACK_FLAG_PKEY;
    }

    if (uct_ib_iface_is_roce(iface)) {
        /* pack Ethernet address */
        pack_flags |= UCT_IB_ADDRESS_PACK_FLAG_ETH;
    } else if (iface->config.force_global_addr) {
        /* pack full IB address */
        pack_flags |= UCT_IB_ADDRESS_PACK_FLAG_SUBNET_PREFIX |
                      UCT_IB_ADDRESS_PACK_FLAG_INTERFACE_ID;
    } else {
        /* pack only subnet prefix for reachability test */
        pack_flags |= UCT_IB_ADDRESS_PACK_FLAG_SUBNET_PREFIX;
    }

    return pack_flags;
}

size_t uct_ib_iface_address_size(uct_ib_iface_t *iface)
{
    uct_ib_address_pack_params_t params;

    params.flags     = uct_ib_iface_address_pack_flags(iface);
    params.gid       = iface->gid_info.gid;
    params.roce_info = iface->gid_info.roce_info;
    return uct_ib_address_size(&params);
}

void uct_ib_iface_address_pack(uct_ib_iface_t *iface, uct_ib_address_t *ib_addr)
{
    uct_ib_address_pack_params_t params;

    params.flags     = uct_ib_iface_address_pack_flags(iface);
    params.gid       = iface->gid_info.gid;
    params.lid       = uct_ib_iface_port_attr(iface)->lid;
    params.roce_info = iface->gid_info.roce_info;
    /* to suppress gcc 4.3.4 warning */
    params.path_mtu  = UCT_IB_ADDRESS_INVALID_PATH_MTU;
    params.gid_index = UCT_IB_ADDRESS_INVALID_GID_INDEX;
    params.pkey      = iface->pkey;
    uct_ib_address_pack(&params, ib_addr);
}

void uct_ib_address_unpack(const uct_ib_address_t *ib_addr,
                           uct_ib_address_pack_params_t *params_p)
{
    const void *ptr                     = ib_addr + 1;
    /* silence cppcheck warning */
    uct_ib_address_pack_params_t params = {0};

    params.gid_index = UCT_IB_ADDRESS_INVALID_GID_INDEX;
    params.path_mtu  = UCT_IB_ADDRESS_INVALID_PATH_MTU;
    params.pkey      = UCT_IB_ADDRESS_DEFAULT_PKEY;

    if (ib_addr->flags & UCT_IB_ADDRESS_FLAG_LINK_LAYER_ETH) {
        /* uint8_t raw[16]; */
        memcpy(params.gid.raw, ptr, sizeof(params.gid.raw));
        ptr           = UCS_PTR_BYTE_OFFSET(ptr, sizeof(params.gid.raw));
        params.flags |= UCT_IB_ADDRESS_PACK_FLAG_ETH;

        params.roce_info.addr_family =
            uct_ib_address_flags_get_roce_af(ib_addr->flags);
        params.roce_info.ver         =
            uct_ib_address_flags_get_roce_version(ib_addr->flags);
    } else {
        /* Default prefix */
        params.gid.global.subnet_prefix = UCT_IB_LINK_LOCAL_PREFIX;
        params.gid.global.interface_id  = 0;
        params.flags                   |= UCT_IB_ADDRESS_PACK_FLAG_SUBNET_PREFIX |
                                          UCT_IB_ADDRESS_PACK_FLAG_INTERFACE_ID;

        /* If the link layer is not ETHERNET, then it is IB and a lid
         * must be present */
        params.lid                      = *(const uint16_t*)ptr;
        ptr                             = UCS_PTR_TYPE_OFFSET(ptr, uint16_t);

        if (ib_addr->flags & UCT_IB_ADDRESS_FLAG_IF_ID) {
            params.gid.global.interface_id = *(uint64_t*)ptr;
            ptr                            = UCS_PTR_TYPE_OFFSET(ptr, uint64_t);
        }

        if (ib_addr->flags & UCT_IB_ADDRESS_FLAG_SUBNET16) {
            params.gid.global.subnet_prefix = UCT_IB_SITE_LOCAL_PREFIX |
                                              ((uint64_t)*(uint16_t*)ptr << 48);
            ptr                             = UCS_PTR_TYPE_OFFSET(ptr, uint16_t);
            ucs_assert(!(ib_addr->flags & UCT_IB_ADDRESS_FLAG_SUBNET64));
        }

        if (ib_addr->flags & UCT_IB_ADDRESS_FLAG_SUBNET64) {
            params.gid.global.subnet_prefix = *(uint64_t*)ptr;
            ptr                             = UCS_PTR_TYPE_OFFSET(ptr, uint64_t);
            params.flags                   |= UCT_IB_ADDRESS_PACK_FLAG_SUBNET_PREFIX;
        }
    }

    if (ib_addr->flags & UCT_IB_ADDRESS_FLAG_PATH_MTU) {
        params.path_mtu = (enum ibv_mtu)*(const uint8_t*)ptr;
        ptr             = UCS_PTR_TYPE_OFFSET(ptr, const uint8_t);
        params.flags   |= UCT_IB_ADDRESS_PACK_FLAG_PATH_MTU;
    }

    if (ib_addr->flags & UCT_IB_ADDRESS_FLAG_GID_INDEX) {
        params.gid_index = *(const uint8_t*)ptr;
        ptr              = UCS_PTR_TYPE_OFFSET(ptr, const uint16_t);
        params.flags    |= UCT_IB_ADDRESS_PACK_FLAG_GID_INDEX;
    }

    if (ib_addr->flags & UCT_IB_ADDRESS_FLAG_PKEY) {
        params.pkey = *(const uint16_t*)ptr;
    }
    /* PKEY is always in params */
    params.flags |= UCT_IB_ADDRESS_PACK_FLAG_PKEY;

    *params_p = params;
}

const char *uct_ib_address_str(const uct_ib_address_t *ib_addr, char *buf,
                               size_t max)
{
    uct_ib_address_pack_params_t params;
    char *p, *endp;

    uct_ib_address_unpack(ib_addr, &params);

    p    = buf;
    endp = buf + max;
    if (params.lid != 0) {
        snprintf(p, endp - p, "lid %d ", params.lid);
        p += strlen(p);
    }

    uct_ib_gid_str(&params.gid, p, endp - p);
    p += strlen(p);

    if (params.flags & UCT_IB_ADDRESS_PACK_FLAG_GID_INDEX) {
        ucs_assert(params.gid_index != UCT_IB_ADDRESS_INVALID_GID_INDEX);
        snprintf(p, endp - p, "gid index %u ", params.gid_index);
        p += strlen(p);
    }

    if (params.flags & UCT_IB_ADDRESS_PACK_FLAG_PATH_MTU) {
        ucs_assert(params.path_mtu != UCT_IB_ADDRESS_INVALID_PATH_MTU);
        snprintf(p, endp - p, "mtu %zu ", uct_ib_mtu_value(params.path_mtu));
        p += strlen(p);
    }

    ucs_assert((params.flags & UCT_IB_ADDRESS_PACK_FLAG_PKEY) &&
               (params.flags != UCT_IB_ADDRESS_INVALID_PKEY));
    snprintf(p, endp - p, "pkey 0x%x ", params.pkey);

    return buf;
}

ucs_status_t uct_ib_iface_get_device_address(uct_iface_h tl_iface,
                                             uct_device_addr_t *dev_addr)
{
    uct_ib_iface_t *iface = ucs_derived_of(tl_iface, uct_ib_iface_t);

    uct_ib_iface_address_pack(iface, (void*)dev_addr);

    return UCS_OK;
}

static int uct_ib_iface_roce_is_reachable(const uct_ib_device_gid_info_t *local_gid_info,
                                          const uct_ib_address_t *remote_ib_addr)
{
    sa_family_t local_ib_addr_af         = local_gid_info->roce_info.addr_family;
    uct_ib_roce_version_t local_roce_ver = local_gid_info->roce_info.ver;
    uint8_t remote_ib_addr_flags         = remote_ib_addr->flags;
    uct_ib_roce_version_t remote_roce_ver;
    sa_family_t remote_ib_addr_af;
    char local_gid_str[128], remote_gid_str[128];

    if ((uct_ib_address_flags_get_roce_version(remote_ib_addr_flags)) ==
         UCT_IB_DEVICE_ROCE_ANY) {
        return 1;
    }

    /* check the address family */
    remote_ib_addr_af = uct_ib_address_flags_get_roce_af(remote_ib_addr_flags);

    if (local_ib_addr_af != remote_ib_addr_af) {
        ucs_assert(local_ib_addr_af != 0);
        ucs_debug("different addr_family detected. local %s remote %s",
                  ucs_sockaddr_address_family_str(local_ib_addr_af),
                  ucs_sockaddr_address_family_str(remote_ib_addr_af));
        return 0;
    }

    /* check the RoCE version */
    ucs_assert(local_roce_ver != UCT_IB_DEVICE_ROCE_ANY);

    remote_roce_ver = uct_ib_address_flags_get_roce_version(remote_ib_addr_flags);

    if (local_roce_ver != remote_roce_ver) {
        ucs_trace("different RoCE versions detected. local %s (gid=%s)"
                  "remote %s (gid=%s)",
                  uct_ib_roce_version_str(local_roce_ver),
                  uct_ib_gid_str(&local_gid_info->gid, local_gid_str,
                                 sizeof(local_gid_str)),
                  uct_ib_roce_version_str(remote_roce_ver),
                  uct_ib_gid_str((union ibv_gid *)(remote_ib_addr + 1), remote_gid_str,
                                 sizeof(remote_gid_str)));
        return 0;
    }

    return 1;
}

int uct_ib_iface_is_reachable(const uct_iface_h tl_iface,
                              const uct_device_addr_t *dev_addr,
                              const uct_iface_addr_t *iface_addr)
{
    uct_ib_iface_t *iface           = ucs_derived_of(tl_iface, uct_ib_iface_t);
    int is_local_eth                = uct_ib_iface_is_roce(iface);
    const uct_ib_address_t *ib_addr = (const void*)dev_addr;
    uct_ib_address_pack_params_t params;

    uct_ib_address_unpack(ib_addr, &params);

    if (/* at least one PKEY has to be with full membership */
        !((params.pkey | iface->pkey) & UCT_IB_PKEY_MEMBERSHIP_MASK) ||
        /* PKEY values have to be equal */
        ((params.pkey ^ iface->pkey) & UCT_IB_PKEY_PARTITION_MASK)) {
        return 0;
    }

    if (!is_local_eth && !(ib_addr->flags & UCT_IB_ADDRESS_FLAG_LINK_LAYER_ETH)) {
        /* same subnet prefix */
        return params.gid.global.subnet_prefix ==
               iface->gid_info.gid.global.subnet_prefix;
    } else if (is_local_eth && (ib_addr->flags & UCT_IB_ADDRESS_FLAG_LINK_LAYER_ETH)) {
        /* there shouldn't be a lid and the UCT_IB_ADDRESS_FLAG_LINK_LAYER_ETH
         * flag should be on. If reachable, the remote and local RoCE versions
         * and address families have to be the same */
        return uct_ib_iface_roce_is_reachable(&iface->gid_info, ib_addr);
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

void uct_ib_iface_fill_ah_attr_from_gid_lid(uct_ib_iface_t *iface, uint16_t lid,
                                            const union ibv_gid *gid,
                                            uint8_t gid_index,
                                            unsigned path_index,
                                            struct ibv_ah_attr *ah_attr)
{
    uint8_t path_bits;
    char buf[128];

    memset(ah_attr, 0, sizeof(*ah_attr));

    ucs_assert(iface->config.sl != UCT_IB_SL_INVALID);

    ah_attr->sl                = iface->config.sl;
    ah_attr->port_num          = iface->config.port_num;
    ah_attr->grh.traffic_class = iface->config.traffic_class;

    if (uct_ib_iface_is_roce(iface)) {
        ah_attr->dlid          = UCT_IB_ROCE_UDP_SRC_PORT_BASE |
                                 (iface->config.roce_path_factor * path_index);
        /* Workaround rdma-core issue of calling rand() which affects global
         * random state in glibc */
        ah_attr->grh.flow_label = 1;
    } else {
        /* TODO iface->path_bits should be removed and replaced by path_index */
        path_bits              = iface->path_bits[path_index %
                                                  iface->path_bits_count];
        ah_attr->dlid          = lid | path_bits;
        ah_attr->src_path_bits = path_bits;
    }

    if (iface->config.force_global_addr ||
        (iface->gid_info.gid.global.subnet_prefix != gid->global.subnet_prefix)) {
        ucs_assert_always(gid->global.interface_id != 0);
        ah_attr->is_global      = 1;
        ah_attr->grh.dgid       = *gid;
        ah_attr->grh.sgid_index = gid_index;
        ah_attr->grh.hop_limit  = iface->config.hop_limit;
    } else {
        ah_attr->is_global      = 0;
    }

    ucs_debug("iface %p: ah_attr %s", iface,
              uct_ib_ah_attr_str(buf, sizeof(buf), ah_attr));
}

void uct_ib_iface_fill_ah_attr_from_addr(uct_ib_iface_t *iface,
                                         const uct_ib_address_t *ib_addr,
                                         unsigned path_index,
                                         struct ibv_ah_attr *ah_attr,
                                         enum ibv_mtu *path_mtu)
{
    uct_ib_address_pack_params_t params;

    ucs_assert(!uct_ib_iface_is_roce(iface) ==
               !(ib_addr->flags & UCT_IB_ADDRESS_FLAG_LINK_LAYER_ETH));

    uct_ib_address_unpack(ib_addr, &params);

    if (params.flags & UCT_IB_ADDRESS_PACK_FLAG_PATH_MTU) {
        ucs_assert(params.path_mtu != UCT_IB_ADDRESS_INVALID_PATH_MTU);
        *path_mtu = params.path_mtu;
    } else {
        *path_mtu = iface->config.path_mtu;
    }

    if (params.flags & UCT_IB_ADDRESS_PACK_FLAG_GID_INDEX) {
        ucs_assert(params.gid_index != UCT_IB_ADDRESS_INVALID_GID_INDEX);
    } else {
        params.gid_index = iface->gid_info.gid_index;
    }

    uct_ib_iface_fill_ah_attr_from_gid_lid(iface, params.lid, &params.gid,
                                           params.gid_index, path_index,
                                           ah_attr);
}

static ucs_status_t uct_ib_iface_init_pkey(uct_ib_iface_t *iface,
                                           const uct_ib_iface_config_t *config)
{
    uct_ib_device_t *dev    = uct_ib_iface_device(iface);
    uint16_t pkey_tbl_len   = uct_ib_iface_port_attr(iface)->pkey_tbl_len;
    int pkey_found          = 0;
    uint16_t lim_pkey       = UCT_IB_ADDRESS_INVALID_PKEY;
    uint16_t lim_pkey_index = UINT16_MAX;
    uint16_t pkey_index, port_pkey, pkey;

    if ((config->pkey != UCS_HEXUNITS_AUTO) &&
        (config->pkey > UCT_IB_PKEY_PARTITION_MASK)) {
        ucs_error("requested pkey 0x%x is invalid, should be in the range 0..0x%x",
                  config->pkey, UCT_IB_PKEY_PARTITION_MASK);
        return UCS_ERR_INVALID_PARAM;
    }

    /* get the user's pkey value and find its index in the port's pkey table */
    for (pkey_index = 0; pkey_index < pkey_tbl_len; ++pkey_index) {
        /* get the pkey values from the port's pkeys table */
        if (ibv_query_pkey(dev->ibv_context, iface->config.port_num, pkey_index,
                           &port_pkey))
        {
            ucs_debug("ibv_query_pkey("UCT_IB_IFACE_FMT", index=%d) failed: %m",
                      UCT_IB_IFACE_ARG(iface), pkey_index);
            continue;
        }

        pkey = ntohs(port_pkey);
        /* if pkey = 0x0, just skip it w/o debug trace, because 0x0
         * means that there is no real pkey configured at this index */
        if (pkey == UCT_IB_ADDRESS_INVALID_PKEY) {
            continue;
        }

        if ((config->pkey == UCS_HEXUNITS_AUTO) ||
            /* take only the lower 15 bits for the comparison */
            ((pkey & UCT_IB_PKEY_PARTITION_MASK) == config->pkey)) {
            if (!(pkey & UCT_IB_PKEY_MEMBERSHIP_MASK) &&
                /* limited PKEY has not yet been found */ 
                (lim_pkey == UCT_IB_ADDRESS_INVALID_PKEY)) {
                lim_pkey_index = pkey_index;
                lim_pkey       = pkey;
                continue;
            }

            iface->pkey_index = pkey_index;
            iface->pkey       = pkey;
            pkey_found        = 1;
            break;
        }
    }

    if (!pkey_found) {
        if (lim_pkey == UCT_IB_ADDRESS_INVALID_PKEY) {
            /* PKEY neither with full nor with limited membership was found */
            if (config->pkey == UCS_HEXUNITS_AUTO) {
                ucs_error("there is no valid pkey to use on "
                          UCT_IB_IFACE_FMT, UCT_IB_IFACE_ARG(iface));
            } else {
                ucs_error("unable to find specified pkey 0x%x on "UCT_IB_IFACE_FMT,
                          config->pkey, UCT_IB_IFACE_ARG(iface));
            }

            return UCS_ERR_NO_ELEM;
        } else {
            ucs_assert(lim_pkey_index != UINT16_MAX);
            iface->pkey_index = lim_pkey_index;
            iface->pkey       = lim_pkey;
        }
    }

    ucs_debug("using pkey[%d] 0x%x on "UCT_IB_IFACE_FMT, iface->pkey_index,
              iface->pkey, UCT_IB_IFACE_ARG(iface));

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
    attr->ibv.qp_type             = (enum ibv_qp_type)attr->qp_type;
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
        ucs_error("iface=%p: failed to create %s QP "
                  "TX wr:%d sge:%d inl:%d resp:%d RX wr:%d sge:%d resp:%d: %m",
                  iface, uct_ib_qp_type_str(attr->qp_type),
                  attr->cap.max_send_wr, attr->cap.max_send_sge,
                  attr->cap.max_inline_data, attr->max_inl_cqe[UCT_IB_DIR_TX],
                  attr->cap.max_recv_wr, attr->cap.max_recv_sge,
                  attr->max_inl_cqe[UCT_IB_DIR_RX]);
        return UCS_ERR_IO_ERROR;
    }

    attr->cap  = attr->ibv.cap;
    *qp_p      = qp;

    ucs_debug("iface=%p: created %s QP 0x%x on %s:%d "
              "TX wr:%d sge:%d inl:%d resp:%d RX wr:%d sge:%d resp:%d",
              iface, uct_ib_qp_type_str(attr->qp_type), qp->qp_num,
              uct_ib_device_name(dev), iface->config.port_num,
              attr->cap.max_send_wr, attr->cap.max_send_sge,
              attr->cap.max_inline_data, attr->max_inl_cqe[UCT_IB_DIR_TX],
              attr->cap.max_recv_wr, attr->cap.max_recv_sge,
              attr->max_inl_cqe[UCT_IB_DIR_RX]);

    return UCS_OK;
}

ucs_status_t uct_ib_verbs_create_cq(uct_ib_iface_t *iface, uct_ib_dir_t dir,
                                    const uct_ib_iface_init_attr_t *init_attr,
                                    int preferred_cpu, size_t inl)
{
    uct_ib_device_t *dev = uct_ib_iface_device(iface);
    struct ibv_cq *cq;
#if HAVE_DECL_IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN
    struct ibv_cq_init_attr_ex cq_attr = {};

    cq_attr.cqe         = init_attr->cq_len[dir];
    cq_attr.channel     = iface->comp_channel;
    cq_attr.comp_vector = preferred_cpu;
    if (init_attr->flags & UCT_IB_CQ_IGNORE_OVERRUN) {
        cq_attr.comp_mask = IBV_CQ_INIT_ATTR_MASK_FLAGS;
        cq_attr.flags     = IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN;
    }

    cq = ibv_cq_ex_to_cq(ibv_create_cq_ex(dev->ibv_context, &cq_attr));
    if (!cq && (errno == ENOSYS))
#endif
    {
        iface->config.max_inl_cqe[dir] = 0;
        cq = ibv_create_cq(dev->ibv_context, init_attr->cq_len[dir], NULL,
                           iface->comp_channel, preferred_cpu);
    }

    if (!cq) {
        ucs_error("ibv_create_cq(cqe=%d) failed: %m", init_attr->cq_len[dir]);
        return UCS_ERR_IO_ERROR;
    }

    iface->cq[dir]                 = cq;
    iface->config.max_inl_cqe[dir] = inl;
    return UCS_OK;
}

static ucs_status_t
uct_ib_iface_create_cq(uct_ib_iface_t *iface, uct_ib_dir_t dir,
                       const uct_ib_iface_init_attr_t *init_attr,
                       const uct_ib_iface_config_t *config,
                       int preferred_cpu)
{
    ucs_status_t status;
    size_t inl                          = config->inl[dir];
#if HAVE_DECL_IBV_EXP_SETENV && !HAVE_DECL_MLX5DV_CQ_INIT_ATTR_MASK_CQE_SIZE
    uct_ib_device_t *dev                = uct_ib_iface_device(iface);
    static const char *cqe_size_env_var = "MLX5_CQE_SIZE";
    size_t cqe_size                     = 64;
    int env_var_added                   = 0;
    const char *cqe_size_env_value;
    size_t cqe_size_min;
    char cqe_size_buf[32];
    int ret;

    cqe_size_min       = (inl > 32) ? 128 : 64;
    cqe_size_env_value = getenv(cqe_size_env_var);

    if (cqe_size_env_value != NULL) {
        cqe_size = atol(cqe_size_env_value);
        if (cqe_size < cqe_size_min) {
            ucs_error("%s is set to %zu, but at least %zu is required (inl: %zu)",
                      cqe_size_env_var, cqe_size, cqe_size_min, inl);
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
    status = iface->ops->create_cq(iface, dir, init_attr, preferred_cpu, inl);
    if (status != UCS_OK) {
        goto out_unsetenv;
    }

    status = UCS_OK;

out_unsetenv:
#if HAVE_DECL_IBV_EXP_SETENV && !HAVE_DECL_MLX5DV_CQ_INIT_ATTR_MASK_CQE_SIZE
    iface->config.max_inl_cqe[dir] = cqe_size / 2;
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

static void uct_ib_iface_set_num_paths(uct_ib_iface_t *iface,
                                       const uct_ib_iface_config_t *config)
{
    uct_ib_device_t *dev = uct_ib_iface_device(iface);

    if (config->num_paths == UCS_ULUNITS_AUTO) {
        if (uct_ib_iface_is_roce(iface)) {
            /* RoCE - number of paths is RoCE LAG level */
            iface->num_paths =
                    uct_ib_device_get_roce_lag_level(dev, iface->config.port_num);
        } else {
            /* IB - number of paths is LMC level */
            ucs_assert(iface->path_bits_count > 0);
            iface->num_paths = iface->path_bits_count;
        }
    } else {
        iface->num_paths = config->num_paths;
    }
}

int uct_ib_iface_is_roce_v2(uct_ib_iface_t *iface, uct_ib_device_t *dev)
{
    return uct_ib_iface_is_roce(iface) &&
           (iface->gid_info.roce_info.ver == UCT_IB_DEVICE_ROCE_V2);
}

ucs_status_t uct_ib_iface_init_roce_gid_info(uct_ib_iface_t *iface,
                                             size_t md_config_index)
{
    uct_ib_device_t *dev = uct_ib_iface_device(iface);
    uint8_t port_num     = iface->config.port_num;

    ucs_assert(uct_ib_iface_is_roce(iface));

    if (md_config_index == UCS_ULUNITS_AUTO) {
        return uct_ib_device_select_gid(dev, port_num, &iface->gid_info);
    }

    return uct_ib_device_query_gid_info(dev->ibv_context, uct_ib_device_name(dev),
                                        port_num, md_config_index,
                                        &iface->gid_info);
}

static ucs_status_t uct_ib_iface_init_gid_info(uct_ib_iface_t *iface,
                                               size_t md_config_index)
{
    uct_ib_device_gid_info_t *gid_info = &iface->gid_info;
    ucs_status_t status;

    /* Fill the gid index and the RoCE version */
    if (uct_ib_iface_is_roce(iface)) {
        status = uct_ib_iface_init_roce_gid_info(iface, md_config_index);
        if (status != UCS_OK) {
            goto out;
        }
    } else {
        gid_info->gid_index             = (md_config_index ==
                                           UCS_ULUNITS_AUTO) ?
                                          UCT_IB_MD_DEFAULT_GID_INDEX :
                                          md_config_index;
        gid_info->roce_info.ver         = UCT_IB_DEVICE_ROCE_ANY;
        gid_info->roce_info.addr_family = 0;
    }

    /* Fill the gid */
    status = uct_ib_device_query_gid(uct_ib_iface_device(iface),
                                     iface->config.port_num,
                                     gid_info->gid_index, &gid_info->gid);
    if (status != UCS_OK) {
        goto out;
    }

out:
    return status;
}

static void uct_ib_iface_set_path_mtu(uct_ib_iface_t *iface,
                                      const uct_ib_iface_config_t *config)
{
    enum ibv_mtu port_mtu = uct_ib_iface_port_attr(iface)->active_mtu;
    uct_ib_device_t *dev  = uct_ib_iface_device(iface);

    /* MTU is set by user configuration */
    if (config->path_mtu != UCT_IB_MTU_DEFAULT) {
        /* cast from uct_ib_mtu_t to ibv_mtu */
        iface->config.path_mtu = (enum ibv_mtu)(config->path_mtu +
                                                (IBV_MTU_512 - UCT_IB_MTU_512));
    } else if ((port_mtu > IBV_MTU_2048) &&
               (IBV_DEV_ATTR(dev, vendor_id) == 0x02c9) &&
               ((IBV_DEV_ATTR(dev, vendor_part_id) == 4099) ||
                (IBV_DEV_ATTR(dev, vendor_part_id) == 4100) ||
                (IBV_DEV_ATTR(dev, vendor_part_id) == 4103) ||
                (IBV_DEV_ATTR(dev, vendor_part_id) == 4104))) {
        /* On some devices optimal path_mtu is 2048 */
        iface->config.path_mtu = IBV_MTU_2048;
    } else {
        iface->config.path_mtu = port_mtu;
    }
}

uint8_t uct_ib_iface_config_select_sl(const uct_ib_iface_config_t *ib_config)
{
    ucs_assert((ib_config->sl <= UCT_IB_SL_MAX) ||
               (ib_config->sl == UCS_ULUNITS_AUTO));
    return (ib_config->sl == UCS_ULUNITS_AUTO) ? 0 : (uint8_t)ib_config->sl;
}

UCS_CLASS_INIT_FUNC(uct_ib_iface_t, uct_ib_iface_ops_t *ops, uct_md_h md,
                    uct_worker_h worker, const uct_iface_params_t *params,
                    const uct_ib_iface_config_t *config,
                    const uct_ib_iface_init_attr_t *init_attr)
{
    uct_ib_md_t *ib_md   = ucs_derived_of(md, uct_ib_md_t);
    uct_ib_device_t *dev = &ib_md->dev;
    size_t rx_headroom   = (params->field_mask &
                            UCT_IFACE_PARAM_FIELD_RX_HEADROOM) ?
                           params->rx_headroom : 0;
    ucs_cpu_set_t cpu_mask;
    int preferred_cpu;
    ucs_status_t status;
    uint8_t port_num;

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

    status = uct_ib_device_find_port(dev, params->mode.device.dev_name,
                                     &port_num);
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
    self->config.roce_path_factor   = config->roce_path_factor;
    self->config.tx_max_poll        = config->tx.max_poll;
    self->config.rx_max_poll        = config->rx.max_poll;
    self->config.rx_max_batch       = ucs_min(config->rx.max_batch,
                                              config->rx.queue_len / 4);
    self->config.port_num           = port_num;
    self->config.sl                 = UCT_IB_SL_INVALID;
    self->config.hop_limit          = config->hop_limit;
    self->release_desc.cb           = uct_ib_iface_release_desc;
    self->config.enable_res_domain  = config->enable_res_domain;
    self->config.qp_type            = init_attr->qp_type;
    uct_ib_iface_set_path_mtu(self, config);

    if (ucs_derived_of(worker, uct_priv_worker_t)->thread_mode == UCS_THREAD_MODE_MULTI) {
        ucs_error("IB transports do not support multi-threaded worker");
        return UCS_ERR_INVALID_PARAM;
    }

    status = uct_ib_iface_init_pkey(self, config);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_ib_iface_init_gid_info(self, ib_md->config.gid_index);
    if (status != UCS_OK) {
        goto err;
    }

    if (config->traffic_class == UCS_ULUNITS_AUTO) {
        self->config.traffic_class = uct_ib_iface_is_roce_v2(self, dev) ?
                                     UCT_IB_DEFAULT_ROCEV2_DSCP : 0;
    } else {
        self->config.traffic_class = config->traffic_class;
    }

    status = uct_ib_iface_init_lmc(self, config);
    if (status != UCS_OK) {
        goto err;
    }

    uct_ib_iface_set_num_paths(self, config);

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

    status = uct_ib_iface_create_cq(self, UCT_IB_DIR_TX, init_attr,
                                    config, preferred_cpu);
    if (status != UCS_OK) {
        goto err_destroy_comp_channel;
    }

    status = uct_ib_iface_set_moderation(self->cq[UCT_IB_DIR_TX],
                                         config->tx.cq_moderation_count,
                                         config->tx.cq_moderation_period);
    if (status != UCS_OK) {
        goto err_destroy_send_cq;
    }

    status = uct_ib_iface_create_cq(self, UCT_IB_DIR_RX, init_attr,
                                    config, preferred_cpu);
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
        uct_ib_grh_required(uct_ib_iface_port_attr(self)) ||
        /* check ADDR_TYPE for backward compatibility */
        (config->addr_type == UCT_IB_ADDRESS_TYPE_SITE_LOCAL) ||
        (config->addr_type == UCT_IB_ADDRESS_TYPE_GLOBAL)) {
        self->config.force_global_addr = 1;
    } else {
        self->config.force_global_addr = 0;
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
    ucs_sys_cpuset_t temp_cpu_mask, process_affinity;
#if HAVE_NUMA
    int distance, min_cpu_distance;
    int cpu, num_cpus;
#endif
    int ret;

    if (!md->config.prefer_nearest_device) {
        *latency = 0;
        return UCS_OK;
    }

    ret = ucs_sys_getaffinity(&process_affinity);
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
    iface_attr->dev_num_paths   = iface->num_paths;

    switch (active_speed) {
    case 1: /* SDR */
        iface_attr->latency.c = 5000e-9;
        signal_rate           = 2.5e9;
        encoding              = 8.0/10.0;
        break;
    case 2: /* DDR */
        iface_attr->latency.c = 2500e-9;
        signal_rate           = 5.0e9;
        encoding              = 8.0/10.0;
        break;
    case 4:
        iface_attr->latency.c = 1300e-9;
        if (uct_ib_iface_is_roce(iface)) {
            /* 10/40g Eth  */
            signal_rate       = 10.3125e9;
            encoding          = 64.0/66.0;
        } else {
            /* QDR */
            signal_rate       = 10.0e9;
            encoding          = 8.0/10.0;
        }
        break;
    case 8: /* FDR10 */
        iface_attr->latency.c = 700e-9;
        signal_rate           = 10.3125e9;
        encoding              = 64.0/66.0;
        break;
    case 16: /* FDR */
        iface_attr->latency.c = 700e-9;
        signal_rate           = 14.0625e9;
        encoding              = 64.0/66.0;
        break;
    case 32: /* EDR / 100g Eth */
        iface_attr->latency.c = 600e-9;
        signal_rate           = 25.78125e9;
        encoding              = 64.0/66.0;
        break;
    case 64: /* 50g Eth */
        iface_attr->latency.c = 600e-9;
        signal_rate           = 25.78125e9 * 2;
        encoding              = 64.0/66.0;
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

    iface_attr->latency.c += numa_latency;
    iface_attr->latency.m  = 0;

    /* Wire speed calculation: Width * SignalRate * Encoding */
    width                 = ib_port_widths[width_idx];
    wire_speed            = (width * signal_rate * encoding) / 8.0;

    /* Calculate packet overhead  */
    mtu                   = ucs_min(uct_ib_mtu_value((enum ibv_mtu)active_mtu),
                                    iface->config.seg_size);

    extra_pkt_len = UCT_IB_BTH_LEN + xport_hdr_len +  UCT_IB_ICRC_LEN + UCT_IB_VCRC_LEN + UCT_IB_DELIM_LEN;

    if (uct_ib_iface_is_roce(iface)) {
        extra_pkt_len += UCT_IB_GRH_LEN + UCT_IB_ROCE_LEN;
        iface_attr->latency.c += 200e-9;
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
    *fd_p                 = iface->comp_channel->fd;
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
