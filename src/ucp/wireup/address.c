/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2016. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "address.h"
#include "wireup_ep.h"

#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_ep.inl>
#include <ucs/arch/bitops.h>
#include <ucs/datastruct/array.h>
#include <ucs/debug/log.h>
#include <ucs/type/serialize.h>
#include <ucs/type/float8.h>
#include <inttypes.h>

#define ucp_address_error(_pack_flags, _fmt, ...) \
    if (!((_pack_flags) & UCP_ADDRESS_PACK_FLAG_NO_TRACE)) { \
        ucs_error(_fmt, ##__VA_ARGS__); \
    }

#define ucp_address_trace(_pack_flags, _fmt, ...) \
    if (!((_pack_flags) & UCP_ADDRESS_PACK_FLAG_NO_TRACE)) { \
        ucs_trace(_fmt, ##__VA_ARGS__); \
    }


/*
 * Packed address layout:
 *
 * [ header(8bit) | uuid(64bit) | client_id | worker_name(string) ]
 * [ device1_md_index | device1_address(var) ]
 *    [ tl1_name_csum(string) | tl1_info | tl1_address(var) ]
 *    [ tl2_name_csum(string) | tl2_info | tl2_address(var) ]
 *    ...
 * [ device2_md_index | device2_address(var) ]
 *    ...
 *
 *   * Worker name is packed if UCX_ADDRESS_DEBUG_INFO is enabled.
 *   * In unified mode tl_info contains just rsc_index and iface latency overhead.
 *     For last address in the tl address list, it will have LAST flag set.
 *   * For ep address, lane index contains the LAST flag.
 *   * In non unified mode tl_info contains iface attributes. LAST flag is set in
 *     iface address length.
 *   * If a device does not have tl addresses, it's md_index will have the flag
 *     EMPTY.
 *   * If the address list is empty, then it will contain only a single md_index
 *     which equals to UCP_NULL_RESOURCE.
 *   * For non-unified mode, ep address contains length with flags. Multiple ep
 *     addresses could be present and the last one is marked with the flag
 *     UCP_ADDRESS_FLAG_LAST. For unified mode, there could not be more than one
 *     ep address.
 *   * For any mode, ep address is followed by a lane index.
 */

/* Address version 2 format:
*
 *            addr_version
 *                ^
 * proto_version  |    flags    worker_uuid     worker_name
 *         ^      |       ^          ^               ^
 *      +------+------+---------+---------------+---------+
 *      |  4   |  4   |   8     |     64        | string  +---------+
 *      +------+------+---------+---------------+---------+         |
 *                                                                  |
 *                                     for each device              |
 *   +--------------------------------------------------------------+
 *   |
 *   |           md_idx(*1)         dev_addr_len(*2)
 *   |           extension             extension
 *   |    md_idx      ^    dev_addr_len   ^       npath   sys_dev  dev_addr
 *   |        ^       |          ^        |        ^        ^        ^
 *   |  +---+-----+---------+---+-----+--------+--------+--------+-------------+
 *   +->| 1 |  7  |   8     | 3 |  5  |   8    |   8    |   8    |dev_addr_len +-+
 *      +---+-----+---------+---+-----+--------+--------+--------+-------------+ |
 *        v                   v                                                  |
 *     md_flags             dev_flags                                            |
 *                                                                for each iface
|
 *   +---------------------------------------------------------------------------+
 *   |         iface_attr(*3)        if_addr_len(*4)
 *   |iface_id       ^    if_addr_len   extension    if_addr
 *   |    ^          |          ^           ^           ^
 *   |  +---------+--------+--+-------+----------+-----------+
 *   +->|   8     |attr_len|2 |  6    |    8     |if_addr_len+-+
 *      +---------+--------+--+-------+----------+-----------+ |
 *                          v                                  |
 *                         if_flags                            |
 *                                                 for each ep |
 *   +---------------------------------------------------------+
 *   |  ep_addr_len    ep_addr   lane_idx
 *   |     ^             ^          ^
 *   |  +---------+-----------+-+-------+
 *   +->|    8    |ep_addr_len|1|   7   |
 *      +---------+-----------+-+-------+
 *                             v
 *                           ep_flags
 *
 *    (*1) - present and contains actual md id, if md_idx == 127
 *    (*2) - present and contains actual device address length,
 *           if dev_addr_len == 31
 *    (*3) - iface attrs format defined by ucp_address_v2_packed_iface_attr_t
 *    (*4) - present and contains actual iface address length,
 *           if if_addr_len == 63
 */


typedef struct {
    size_t           dev_addr_len;
    ucp_tl_bitmap_t  tl_bitmap;
    ucp_rsc_index_t  rsc_index;
    ucp_rsc_index_t  tl_count;
    unsigned         num_paths;
    ucs_sys_device_t sys_dev;
    size_t           tl_addrs_size;
} ucp_address_packed_device_t;


typedef struct {
    float            overhead;
    float            bandwidth;
    float            lat_ovh;
    uint32_t         prio_cap_flags; /* 8 lsb : prio
                                      * 22 msb:
                                      *        - iface flags
                                      *        - iface event flags
                                      * 2 hsb :
                                      *        - amo32
                                      *        - amo64 */
} UCS_S_PACKED ucp_address_packed_iface_attr_t;


typedef struct {
    ucs_fp8_t        overhead;
    ucs_fp8_t        bandwidth;
    ucs_fp8_t        latency;
    uint8_t          prio;
    /* Maximal segment size than can be received by this iface */
    uint16_t         seg_size;
    /* Includes caps, event and atomic flags */
    uint16_t         flags;
} UCS_S_PACKED ucp_address_v2_packed_iface_attr_t;


/* In unified mode we pack resource index instead of iface attrs to the address,
 * so the peer can get all attrs from the local device with the same resource
 * index.
 * Also we send information which depends on device NUMA locality,
 * which may be different on peers (processes which do address pack
 * and address unpack):
 * - latency overhead
 * - Indication whether resource can be used for atomics or not (packed to the
 *   signed bit of lat_ovh).
 *
 * TODO: Revise/fix this when NUMA locality is exposed in UCP.
 * */
typedef struct {
    ucp_rsc_index_t  rsc_index;
    float            lat_ovh;
} UCS_S_PACKED ucp_address_unified_iface_attr_t;

/* The following structs are used for manipulating device index, in order to
 * include both dev_index and sys_dev into it. */
typedef struct {
    ucp_rsc_index_t  dev_index;
    ucs_sys_device_t sys_dev;
} ucp_address_remote_device_t;

UCS_ARRAY_DECLARE_TYPE(ucp_address_remote_device_array_t, unsigned,
                       ucp_address_remote_device_t);

#define UCP_ADDRESS_V1_FLAG_ATOMIC32  UCS_BIT(30) /* 32bit atomic operations */
#define UCP_ADDRESS_V1_FLAG_ATOMIC64  UCS_BIT(31) /* 64bit atomic operations */

#define UCP_ADDRESS_FLAG_LAST         0x80u  /* Last address in the list */
#define UCP_ADDRESS_FLAG_HAS_EP_ADDR  0x40u  /* For iface address:
                                                Indicates that ep addr is packed
                                                right after iface addr */
#define UCP_ADDRESS_FLAG_NUM_PATHS    0x40u  /* For device address:
                                                Indicates that number of paths on the
                                                device is packed right after device
                                                address, otherwise number of paths
                                                defaults to 1. */
#define UCP_ADDRESS_FLAG_SYS_DEVICE   0x20u  /* For device address:
                                                Indicates that system device is
                                                packed after device address or
                                                number of paths (if present) */

/* Mask for iface and endpoint address length */
#define UCP_ADDRESS_IFACE_LEN_MASK   (UCS_MASK(8) ^ \
                                      (UCP_ADDRESS_FLAG_HAS_EP_ADDR | \
                                       UCP_ADDRESS_FLAG_LAST))

/* Mask for device address length */
#define UCP_ADDRESS_DEVICE_LEN_MASK  (UCS_MASK(8) ^ \
                                      (UCP_ADDRESS_FLAG_SYS_DEVICE | \
                                       UCP_ADDRESS_FLAG_NUM_PATHS | \
                                       UCP_ADDRESS_FLAG_LAST))

#define UCP_ADDRESS_FLAG_MD_EMPTY_DEV 0x80u  /* Device without TL addresses */
#define UCP_ADDRESS_FLAG_MD_ALLOC     0x40u  /* MD can register  */
#define UCP_ADDRESS_FLAG_MD_REG       0x20u  /* MD can allocate */
#define UCP_ADDRESS_FLAG_MD_MASK_V1   (UCS_MASK(8) ^ \
                                        (UCP_ADDRESS_FLAG_MD_EMPTY_DEV | \
                                         UCP_ADDRESS_FLAG_MD_ALLOC | \
                                         UCP_ADDRESS_FLAG_MD_REG))
#define UCP_ADDRESS_FLAG_MD_MASK      (UCS_MASK(8) ^ \
                                       UCP_ADDRESS_FLAG_MD_EMPTY_DEV)

#define UCP_ADDRESS_RELEASE_VERSION_BITS  4
#define UCP_ADDRESS_RELEASE_VERSION_MAX   UCS_BIT(UCP_ADDRESS_RELEASE_VERSION_BITS)

#define UCP_ADDRESS_HEADER_VERSION_MASK     UCS_MASK(4) /* Version - 4 bits */
#define UCP_ADDRESS_HEADER_SHIFT            4

#define UCP_ADDRESS_DEFAULT_WORKER_UUID     0
#define UCP_ADDRESS_DEFAULT_CLIENT_ID       0

enum {
    UCP_ADDRESS_HEADER_FLAG_DEBUG_INFO  = UCS_BIT(0),  /* Address has debug info */
    UCP_ADDRESS_HEADER_FLAG_WORKER_UUID = UCS_BIT(1),  /* Worker unique id */
    UCP_ADDRESS_HEADER_FLAG_CLIENT_ID   = UCS_BIT(2),  /* Worker client id */
    UCP_ADDRESS_HEADER_FLAG_AM_ONLY     = UCS_BIT(3)   /* Only AM lane info */
};

static size_t ucp_address_iface_attr_size(ucp_worker_t *worker, uint64_t flags,
                                          ucp_object_version_t addr_version)
{
    size_t rsc_id_size = (flags & UCP_ADDRESS_PACK_FLAG_TL_RSC_IDX) ?
                         sizeof(uint8_t) : 0ul;

    if (ucp_worker_is_unified_mode(worker)) {
        return sizeof(ucp_address_unified_iface_attr_t);
    } else if (addr_version == UCP_OBJECT_VERSION_V1) {
        return sizeof(ucp_address_packed_iface_attr_t) + rsc_id_size;
    } else {
        return sizeof(ucp_address_v2_packed_iface_attr_t) + rsc_id_size;
    }
}

static uint64_t ucp_worker_iface_can_connect(uct_iface_attr_t *attrs)
{
    return attrs->cap.flags &
           (UCT_IFACE_FLAG_CONNECT_TO_IFACE | UCT_IFACE_FLAG_CONNECT_TO_EP);
}

/* Pack a string and return a pointer to storage right after the string */
static void *
ucp_address_pack_worker_address_name(ucp_worker_h worker, void *dest)
{
    const char *s;
    size_t length;

    s      = ucp_worker_get_address_name(worker);
    length = strlen(s);
    ucs_assert(length <= UINT8_MAX);
    *(uint8_t*)dest = length;
    memcpy(UCS_PTR_TYPE_OFFSET(dest, uint8_t), s, length);
    return UCS_PTR_BYTE_OFFSET(UCS_PTR_TYPE_OFFSET(dest, uint8_t), length);
}

/* Unpack a string and return pointer to next storage byte */
static const void *
ucp_address_unpack_worker_address_name(const void *src, char *s)
{
    size_t length, avail;

    length   = *(const uint8_t*)src;
    avail    = ucs_min(length, UCP_WORKER_ADDRESS_NAME_MAX - 1);
    memcpy(s, UCS_PTR_TYPE_OFFSET(src, uint8_t), avail);
    s[avail] = '\0';
    return UCS_PTR_TYPE_OFFSET(UCS_PTR_BYTE_OFFSET(src, length), uint8_t);
}

static uint8_t ucp_address_pack_release_version()
{
    return UCP_API_MINOR % UCP_ADDRESS_RELEASE_VERSION_MAX;
}

/* Release version is packed into 4 bits. Unpack it assuming that we do not
 * support wire compatibility for more than 8 versions (therefore we take the
 * value which is closest to our release version)
 */
static unsigned ucp_address_unpack_release_version(uint8_t packed_version)
{
    int range = UCP_ADDRESS_RELEASE_VERSION_MAX;
    int delta = ucp_address_pack_release_version() - packed_version;
    int base;

    base = (UCP_API_MINOR / range) + (delta >= range / 2) -
                                               (delta <= -range / 2);

    return packed_version + ucs_max(0, base) * range;
}

static ucp_address_packed_device_t*
ucp_address_get_device(ucp_context_h context, ucp_rsc_index_t rsc_index,
                       ucp_address_packed_device_t *devices,
                       ucp_rsc_index_t *num_devices_p)
{
    const ucp_tl_resource_desc_t *tl_rsc = context->tl_rscs;
    ucp_address_packed_device_t *dev;

    for (dev = devices; dev < devices + *num_devices_p; ++dev) {
        if ((tl_rsc[rsc_index].md_index == tl_rsc[dev->rsc_index].md_index) &&
            (tl_rsc[rsc_index].dev_index == tl_rsc[dev->rsc_index].dev_index)) {
            goto out;
        }
    }

    dev = &devices[(*num_devices_p)++];
    memset(dev, 0, sizeof(*dev));
out:
    return dev;
}

static ssize_t ucp_address_packed_value_size(size_t value, size_t max_value,
                                             ucp_object_version_t addr_version,
                                             const char *dev_name)
{
    if (addr_version == UCP_OBJECT_VERSION_V1) {
        /* Address version 1 does not support value extension */
        if (value > max_value) {
            ucs_debug("device %s: value %zu > max_value %zu, use "
                      "UCX_ADDRESS_VERSION=v2 to allow packing such addresses",
                      dev_name, value, max_value);
            return UCS_ERR_UNSUPPORTED;
        }

        return sizeof(uint8_t);
    } else if (value < max_value) {
        /* The value fits into a partial byte, up to max_value */
        return sizeof(uint8_t);
    } else {
        /* The value needs to be extended to a full byte */
        ucs_assertv_always(value <= UINT8_MAX, "value %zu", value);
        return sizeof(uint8_t) * 2;
    }
}

static ssize_t ucp_address_packed_length_size(ucp_worker_h worker,
                                              size_t length, size_t max_length,
                                              ucp_object_version_t addr_version,
                                              const char *dev_name)
{
    if (ucp_worker_is_unified_mode(worker)) {
        return 0;
    }

    return ucp_address_packed_value_size(length, max_length, addr_version,
                                         dev_name);
}

static int ucp_address_pack_v1_extra_info(ucp_object_version_t addr_version,
                                          uint64_t pack_flags)
{
    return (addr_version == UCP_OBJECT_VERSION_V1) &&
           (pack_flags & UCP_ADDRESS_PACK_FLAG_RELEASE_VER_V1);
}

static ucs_status_t
ucp_address_gather_devices(ucp_worker_h worker, const ucp_ep_config_key_t *key,
                           const ucp_tl_bitmap_t *tl_bitmap, uint64_t flags,
                           ucp_object_version_t addr_version,
                           unsigned max_num_paths,
                           ucp_address_packed_device_t **devices_p,
                           ucp_rsc_index_t *num_devices_p)
{
    ucp_context_h context = worker->context;
    ucp_tl_bitmap_t current_tl_bitmap = *tl_bitmap;
    ucp_address_packed_device_t *dev, *devices;
    uct_iface_attr_t *iface_attr;
    ucp_rsc_index_t num_devices;
    ucp_rsc_index_t rsc_index;
    ucp_lane_index_t lane;
    ssize_t length_size;

    devices = ucs_calloc(context->num_tls, sizeof(*devices), "packed_devices");
    if (devices == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    num_devices = 0;
    UCS_STATIC_BITMAP_AND_INPLACE(&current_tl_bitmap, context->tl_bitmap);
    UCS_STATIC_BITMAP_FOR_EACH_BIT(rsc_index, &current_tl_bitmap) {
        iface_attr = ucp_worker_iface_get_attr(worker, rsc_index);
        if (!ucp_worker_iface_can_connect(iface_attr)) {
            continue;
        }

        dev = ucp_address_get_device(context, rsc_index, devices, &num_devices);

        if (flags & UCP_ADDRESS_PACK_FLAG_EP_ADDR) {
            ucs_assert(key != NULL);
            /* Each lane which matches the resource index adds an ep address
             * entry. The length and flags is packed in non-unified mode only.
             */
            for (lane = 0; lane < key->num_lanes; ++lane) {
                if ((key->lanes[lane].rsc_index == rsc_index) &&
                    ucp_ep_config_connect_p2p(worker, key, rsc_index)) {
                    dev->tl_addrs_size += !ucp_worker_is_unified_mode(worker);
                    dev->tl_addrs_size += iface_attr->ep_addr_len;
                    dev->tl_addrs_size += sizeof(uint8_t); /* lane index */
                }
            }
        }

        dev->tl_addrs_size += sizeof(uint16_t); /* tl name checksum */

        if (flags & UCP_ADDRESS_PACK_FLAG_IFACE_ADDR) {
            /* iface address (its length will be packed in non-unified mode only) */
            dev->tl_addrs_size += iface_attr->iface_addr_len;
            /* iface address length (+flags) can take 2 bytes with address
             * version 2 in non-unified mode
             */
            length_size         = ucp_address_packed_length_size(
                    worker, iface_attr->iface_addr_len,
                    UCP_ADDRESS_IFACE_LEN_MASK, addr_version,
                    worker->context->tl_rscs[rsc_index].tl_rsc.dev_name);
            if (length_size < 0) {
                ucs_free(devices);
                return length_size;
            }

            dev->tl_addrs_size += length_size;
            dev->tl_addrs_size += ucp_address_iface_attr_size(worker, flags,
                                                              addr_version);
        } else {
            dev->tl_addrs_size += 1; /* 0-value for valid unpacking */
        }

        if (flags & UCP_ADDRESS_PACK_FLAG_DEVICE_ADDR) {
            dev->dev_addr_len = iface_attr->device_addr_len;
        } else {
            dev->dev_addr_len = 0;
        }

        if (flags & UCP_ADDRESS_PACK_FLAG_SYS_DEVICE) {
            dev->sys_dev = context->tl_rscs[rsc_index].tl_rsc.sys_device;
        } else {
            dev->sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;
        }

        if (iface_attr->dev_num_paths > UINT8_MAX) {
            ucs_error("only up to %d paths are supported by address pack (got: %u)",
                      UINT8_MAX, iface_attr->dev_num_paths);
            ucs_free(devices);
            return UCS_ERR_UNSUPPORTED;
        }

        dev->rsc_index  = rsc_index;
        UCS_STATIC_BITMAP_SET(&dev->tl_bitmap, rsc_index);
        dev->num_paths  = ucs_min(max_num_paths, iface_attr->dev_num_paths);
    }

    *devices_p     = devices;
    *num_devices_p = num_devices;
    return UCS_OK;
}

static ssize_t
ucp_address_packed_size(ucp_worker_h worker,
                        const ucp_address_packed_device_t *devices,
                        ucp_rsc_index_t num_devices, uint64_t pack_flags,
                        ucp_object_version_t addr_version)
{
    size_t size = 0;
    size_t md_mask;
    ssize_t value_size;
    const ucp_address_packed_device_t *dev;
    ucp_md_index_t md_index;
    const ucp_tl_resource_desc_t *rsc;

    /* header: version and flags */
    if (addr_version == UCP_OBJECT_VERSION_V1) {
        size   += sizeof(uint8_t);
        md_mask = UCP_ADDRESS_FLAG_MD_MASK_V1;
    } else {
        size   += sizeof(uint16_t);
        md_mask = UCP_ADDRESS_FLAG_MD_MASK;
    }

    if (pack_flags & UCP_ADDRESS_PACK_FLAG_WORKER_UUID) {
        size += sizeof(uint64_t);
    }

    if (pack_flags & UCP_ADDRESS_PACK_FLAG_CLIENT_ID) {
        size += sizeof(uint64_t);
    }

    if ((worker->context->config.ext.address_debug_info) &&
        (pack_flags & UCP_ADDRESS_PACK_FLAG_WORKER_NAME)) {
        size += strlen(ucp_worker_get_address_name(worker)) + 1;
    }

    if (num_devices == 0) {
        size += 1; /* NULL md_index */
    } else {
        for (dev = devices; dev < (devices + num_devices); ++dev) {
            rsc        = &worker->context->tl_rscs[dev->rsc_index];
            /* device md_index */
            md_index   = rsc->md_index;
            /* md index (+flags) can take 2 bytes with address version 2 */
            value_size = ucp_address_packed_value_size(md_index, md_mask,
                                                       addr_version,
                                                       rsc->tl_rsc.dev_name);
            if (value_size < 0) {
                return value_size;
            }

            size += value_size;
            if (pack_flags & UCP_ADDRESS_PACK_FLAG_DEVICE_ADDR) {
                /* device address length */
                value_size = ucp_address_packed_value_size(
                        dev->dev_addr_len, UCP_ADDRESS_DEVICE_LEN_MASK,
                        addr_version, rsc->tl_rsc.dev_name);
                if (value_size < 0) {
                    return value_size;
                }

                size += value_size;
                size += dev->dev_addr_len;  /* device address */
            } else {
                size += 1; /* 0 device address length */
            }
            if (dev->num_paths > 1) {
                size += 1; /* number of paths */
            }
            if (dev->sys_dev != UCS_SYS_DEVICE_ID_UNKNOWN) {
                size += 1; /* system device */
            }
            size += dev->tl_addrs_size; /* transport addresses */
        }

        if (ucp_address_pack_v1_extra_info(addr_version, pack_flags)) {
           /* Pack empty device which contains wire version. This way it can be
            * parsed by UCX versions starting from 1.16 and ignored by older
            * ones. */
            size += 2;
        }
    }
    return size;
}

static void ucp_address_memcheck(ucp_context_h context, void *ptr, size_t size,
                                 ucp_rsc_index_t rsc_index)
{

    void *undef_ptr;

    undef_ptr = (void*)VALGRIND_CHECK_MEM_IS_DEFINED(ptr, size);
    if (undef_ptr != NULL) {
        ucs_error(UCT_TL_RESOURCE_DESC_FMT
                  " address contains undefined bytes at offset %zd",
                  UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[rsc_index].tl_rsc),
                  UCS_PTR_BYTE_DIFF(ptr, undef_ptr));
    }
}

static void *ucp_address_pack_byte_extended(void *ptr, size_t value,
                                            size_t max_value,
                                            ucp_object_version_t addr_version)
{
    if ((addr_version != UCP_OBJECT_VERSION_V1) && (value >= max_value)) {
        /* Set maximal possible value, indicating that the actual value is in
         * the next byte. */
        *ucs_serialize_next(&ptr, uint8_t) = max_value;
        max_value                          = UINT8_MAX;
    }

    ucs_assertv_always(value <= max_value, "value=%zu, max_value %zu", value,
                       max_value);

    *ucs_serialize_next(&ptr, uint8_t) = value;

    return ptr;
}

static void *
ucp_address_unpack_byte_extended(const void *ptr, size_t value_mask,
                                 ucp_object_version_t addr_version,
                                 uint8_t *value_p)
{
    uint8_t value = *ucs_serialize_next(&ptr, const uint8_t) & value_mask;

    if ((addr_version != UCP_OBJECT_VERSION_V1) && (value == value_mask)) {
        value = *ucs_serialize_next(&ptr, const uint8_t);
    }

    *value_p = value;
    return (void*)ptr;
}

static size_t ucp_address_md_mask(ucp_object_version_t addr_version)
{
    return (addr_version == UCP_OBJECT_VERSION_V1) ?
           UCP_ADDRESS_FLAG_MD_MASK_V1 : UCP_ADDRESS_FLAG_MD_MASK;
}

static void *
ucp_address_pack_md_info(void *ptr, int is_empty_dev, uint64_t md_flags,
                         ucp_md_index_t md_index,
                         ucp_object_version_t addr_version)
{
    uint8_t *flags_ptr = ptr;
    size_t mask        = ucp_address_md_mask(addr_version);

    ptr = ucp_address_pack_byte_extended(ptr, md_index, mask, addr_version);

    if (is_empty_dev) {
        *flags_ptr |= UCP_ADDRESS_FLAG_MD_EMPTY_DEV;
    }

    if (addr_version == UCP_OBJECT_VERSION_V1) {
        /* Preserve wire protocol even though these flags are not used */
        if (md_flags & UCT_MD_FLAG_ALLOC) {
            *flags_ptr |= UCP_ADDRESS_FLAG_MD_ALLOC;
        }

        if (md_flags & UCT_MD_FLAG_REG) {
            *flags_ptr |= UCP_ADDRESS_FLAG_MD_REG;
        }
    }

    return ptr;
}

static void *ucp_address_unpack_md_info(const void *ptr,
                                        ucp_object_version_t addr_version,
                                        ucp_md_index_t *md_index,
                                        int *empty_dev)
{
    uint8_t md_byte = *(uint8_t*)ptr;
    size_t mask;

    *empty_dev = md_byte & UCP_ADDRESS_FLAG_MD_EMPTY_DEV;
    mask       = ucp_address_md_mask(addr_version);

    return ucp_address_unpack_byte_extended(ptr, mask, addr_version, md_index);
}

static uint32_t ucp_address_pack_flags(uint64_t input_flags,
                                       uint64_t cap_mask,
                                       uint8_t output_start_bit)
{
    uint32_t result_flags = 0;
    uint32_t packed_flag;
    uint8_t cap_index;

    ucs_assert((ucs_popcount(cap_mask) + output_start_bit) < 32);
    packed_flag = UCS_BIT(output_start_bit);

    ucs_for_each_bit(cap_index, cap_mask) {
        if (input_flags & UCS_BIT(cap_index)) {
            result_flags |= packed_flag;
        }

        packed_flag <<= 1;
    }

    return result_flags;
}

static uint64_t ucp_address_unpack_flags(uint32_t input_flags,
                                         uint64_t cap_mask,
                                         uint8_t input_start_bit)
{
    uint64_t result_flags = 0;
    uint32_t packed_flag;
    uint8_t cap_index;

    ucs_assert((ucs_popcount(cap_mask) + input_start_bit) < 32);
    packed_flag = UCS_BIT(input_start_bit);

    ucs_for_each_bit(cap_index, cap_mask) {
        if (input_flags & packed_flag) {
            result_flags |= UCS_BIT(cap_index);
        }

        packed_flag <<= 1;
    }

    return result_flags;
}

static uint64_t ucp_address_flags_from_iface_flags(uint64_t iface_cap_flags,
                                                   uint64_t iface_event_flags)
{
    uint64_t iface_flags = 0;

    if (iface_cap_flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
        iface_flags |= UCP_ADDR_IFACE_FLAG_CONNECT_TO_IFACE;
    }

    if (iface_cap_flags & UCT_IFACE_FLAG_CB_ASYNC) {
        iface_flags |= UCP_ADDR_IFACE_FLAG_CB_ASYNC;
    }

    if (ucs_test_all_flags(iface_cap_flags,
                           UCT_IFACE_FLAG_CB_SYNC | UCT_IFACE_FLAG_AM_BCOPY)) {
        iface_flags |= UCP_ADDR_IFACE_FLAG_AM_SYNC;
    }

    if (iface_cap_flags & (UCT_IFACE_FLAG_PUT_SHORT | UCT_IFACE_FLAG_PUT_BCOPY |
                           UCT_IFACE_FLAG_PUT_ZCOPY)) {
        iface_flags |= UCP_ADDR_IFACE_FLAG_PUT;
    }

    if (iface_cap_flags & (UCT_IFACE_FLAG_GET_SHORT | UCT_IFACE_FLAG_GET_BCOPY |
                           UCT_IFACE_FLAG_GET_ZCOPY)) {
        iface_flags |= UCP_ADDR_IFACE_FLAG_GET;
    }

    if (iface_cap_flags & (UCT_IFACE_FLAG_TAG_EAGER_SHORT |
                           UCT_IFACE_FLAG_TAG_EAGER_BCOPY |
                           UCT_IFACE_FLAG_TAG_EAGER_ZCOPY)) {
        iface_flags |= UCP_ADDR_IFACE_FLAG_TAG_EAGER;
    }

    if (iface_cap_flags & UCT_IFACE_FLAG_TAG_RNDV_ZCOPY) {
        iface_flags |= UCP_ADDR_IFACE_FLAG_TAG_RNDV;
    }

    if (iface_event_flags & UCT_IFACE_FLAG_EVENT_RECV) {
        iface_flags |= UCP_ADDR_IFACE_FLAG_EVENT_RECV;
    }

    return iface_flags;
}

static unsigned
ucp_address_pack_iface_attr_v1(const ucp_worker_iface_t *wiface, void *ptr,
                               unsigned atomic_flags)
{
    const uct_iface_attr_t *iface_attr      = &wiface->attr;
    ucp_address_packed_iface_attr_t *packed = ptr;

    packed->lat_ovh   = ucp_wireup_iface_lat_distance_v1(wiface);
    packed->bandwidth = ucp_wireup_iface_bw_distance(wiface);
    packed->overhead  = iface_attr->overhead;
    /* Pack prio, capability and atomic flags */
    packed->prio_cap_flags = (uint8_t)iface_attr->priority |
                             ucp_address_pack_flags(iface_attr->cap.flags,
                                                    UCP_ADDRESS_IFACE_FLAGS, 8);
    /* Keep only the bits defined by UCP_ADDRESS_IFACE_EVENT_FLAGS to shrink
     * address. */
    packed->prio_cap_flags |= ucp_address_pack_flags(
            iface_attr->cap.event_flags, UCP_ADDRESS_IFACE_EVENT_FLAGS,
            8 + ucs_popcount(UCP_ADDRESS_IFACE_FLAGS));

    if (atomic_flags & UCP_ADDR_IFACE_FLAG_ATOMIC32) {
        packed->prio_cap_flags |= UCP_ADDRESS_V1_FLAG_ATOMIC32;
    }

    if (atomic_flags & UCP_ADDR_IFACE_FLAG_ATOMIC64) {
        packed->prio_cap_flags |= UCP_ADDRESS_V1_FLAG_ATOMIC64;
    }

    ucs_assert_always((ucs_popcount(UCP_ADDRESS_IFACE_FLAGS) +
                ucs_popcount(UCP_ADDRESS_IFACE_EVENT_FLAGS)) <= 22);

    return sizeof(*packed);
}

size_t ucp_address_iface_seg_size(const uct_iface_attr_t *iface_attr)
{
    /* To be replaced by iface_attr.cap.am.max_recv when it is added to the
     * UCT API */
    if (iface_attr->cap.flags & UCT_IFACE_FLAG_AM_BCOPY) {
        return iface_attr->cap.am.max_bcopy;
    } else if (iface_attr->cap.flags & UCT_IFACE_FLAG_AM_ZCOPY) {
       return iface_attr->cap.am.max_zcopy;
    } else if (iface_attr->cap.flags & UCT_IFACE_FLAG_AM_SHORT) {
        return iface_attr->cap.am.max_short;
    } else {
        return 0ul;
    }
}

static unsigned
ucp_address_pack_iface_attr_v2(const ucp_worker_iface_t *wiface, void *ptr,
                               unsigned atomic_flags)
{
    const uct_iface_attr_t *iface_attr         = &wiface->attr;
    ucp_address_v2_packed_iface_attr_t *packed = ptr;

    uint64_t addr_iface_flags;
    double latency_nsec, overhead_nsec, latency, bandwidth;
    size_t seg_size;

    latency   = ucp_wireup_iface_lat_distance_v2(wiface);
    bandwidth = ucp_wireup_iface_bw_distance(wiface);

    latency_nsec  = latency * UCS_NSEC_PER_SEC;
    overhead_nsec = iface_attr->overhead * UCS_NSEC_PER_SEC;

    packed->overhead  = UCS_FP8_PACK(OVERHEAD, overhead_nsec);
    packed->bandwidth = UCS_FP8_PACK(BANDWIDTH, bandwidth);
    packed->latency   = UCS_FP8_PACK(LATENCY, latency_nsec);
    packed->prio      = ucs_min(UINT8_MAX, iface_attr->priority);
    addr_iface_flags  = ucp_address_flags_from_iface_flags(
                            iface_attr->cap.flags, iface_attr->cap.event_flags);
    packed->flags     = (uint16_t)(addr_iface_flags | atomic_flags);
    seg_size          = ucp_address_iface_seg_size(iface_attr) /
                        UCP_ADDRESS_IFACE_SEG_SIZE_FACTOR;
    packed->seg_size  = (uint16_t)seg_size;

    ucs_assertv(seg_size <= UINT16_MAX, "seg_size %zu", seg_size);

    return sizeof(*packed);
}

static int ucp_address_pack_iface_attr(const ucp_worker_iface_t *wiface,
                                       void *ptr, ucp_rsc_index_t rsc_index,
                                       unsigned pack_flags,
                                       ucp_object_version_t addr_version,
                                       int enable_atomics)
{
    const uct_iface_attr_t *iface_attr = &wiface->attr;
    ucp_worker_h worker                = wiface->worker;
    unsigned atomic_flags              = 0;
    unsigned packed_len;
    double lat_ovh;
    ucp_address_unified_iface_attr_t *unified;

    if (ucp_worker_is_unified_mode(worker)) {
        /* In unified mode all workers have the same transports and tl bitmap.
         * Just send rsc index, so the remote peer could fetch iface attributes
         * from its local iface. Also send latency overhead, because it
         * depends on device NUMA locality. */
        unified            = ptr;
        unified->rsc_index = rsc_index;
        lat_ovh            = ucp_wireup_iface_lat_distance_v1(wiface);
        if (enable_atomics) {
            lat_ovh = -lat_ovh;
        }
        unified->lat_ovh = lat_ovh;

        return sizeof(*unified);
    }

    if (enable_atomics) {
        if (ucs_test_all_flags(iface_attr->cap.atomic32.op_flags,
                               UCP_ATOMIC_OP_MASK) &&
            ucs_test_all_flags(iface_attr->cap.atomic32.fop_flags,
                               UCP_ATOMIC_FOP_MASK)) {
            atomic_flags |= UCP_ADDR_IFACE_FLAG_ATOMIC32;
        }
        if (ucs_test_all_flags(iface_attr->cap.atomic64.op_flags,
                               UCP_ATOMIC_OP_MASK) &&
            ucs_test_all_flags(iface_attr->cap.atomic64.fop_flags,
                               UCP_ATOMIC_FOP_MASK)) {
            atomic_flags |= UCP_ADDR_IFACE_FLAG_ATOMIC64;
        }
    }

    if (addr_version == UCP_OBJECT_VERSION_V1) {
        packed_len = ucp_address_pack_iface_attr_v1(wiface, ptr, atomic_flags);
    } else {
        packed_len = ucp_address_pack_iface_attr_v2(wiface, ptr, atomic_flags);
    }

    if (pack_flags & UCP_ADDRESS_PACK_FLAG_TL_RSC_IDX) {
        ptr             = UCS_PTR_BYTE_OFFSET(ptr, packed_len);
        *(uint8_t*)ptr  = rsc_index;
        packed_len     += sizeof(uint8_t);
    }

    return packed_len;
}

static unsigned
ucp_address_unpack_iface_attr_v1(ucp_worker_t *worker,
                                 ucp_address_iface_attr_t *iface_attr,
                                 const void *ptr)
{
    const ucp_address_packed_iface_attr_t *packed = ptr;
    uct_ppn_bandwidth_t bandwidth;
    uint64_t iface_flags, event_flags;

    iface_attr->overhead    = packed->overhead;
    iface_attr->lat_ovh     = packed->lat_ovh;
    iface_attr->priority    = packed->prio_cap_flags & UCS_MASK(8);
    /* UCP address v1 does not carry segment size, MAX will not affect ep
     * threshold calculations (which are trimmed by this value). */
    iface_attr->seg_size    = UINT_MAX;
    iface_flags             = ucp_address_unpack_flags(packed->prio_cap_flags,
                                                       UCP_ADDRESS_IFACE_FLAGS,
                                                       8);
    event_flags             = ucp_address_unpack_flags(
                                  packed->prio_cap_flags,
                                  UCP_ADDRESS_IFACE_EVENT_FLAGS,
                                  8 + ucs_popcount(UCP_ADDRESS_IFACE_FLAGS));
    iface_attr->flags       = ucp_address_flags_from_iface_flags(iface_flags,
                                                                 event_flags);

    if (packed->prio_cap_flags & UCP_ADDRESS_V1_FLAG_ATOMIC32) {
        iface_attr->flags  |= UCP_ADDR_IFACE_FLAG_ATOMIC32;
    }

    if (packed->prio_cap_flags & UCP_ADDRESS_V1_FLAG_ATOMIC64) {
        iface_attr->flags  |= UCP_ADDR_IFACE_FLAG_ATOMIC64;
    }

    if (packed->bandwidth < 0.0) {
        /* The received value of the bandwidth is "dedicated - shared" which
         * doesn't consider ppn and could be sent only by the peer which uses
         * UCX version <= 1.11. Calculate the bandwidth value considering our
         * ppn value to be the same value as on the peer */
        bandwidth.shared      = fabs(packed->bandwidth);
        bandwidth.dedicated   = 0.0;
        iface_attr->bandwidth = ucp_tl_iface_bandwidth(worker->context,
                                                       &bandwidth);
    } else {
        /* The received value is either a dedicated bandwidth value (when the
         * peer is using an older version) or the total bandwidth considering
         * remote ppn value (when the peer is using a newer version) */
        iface_attr->bandwidth = packed->bandwidth;
    }

    return sizeof(*packed);
}

static unsigned
ucp_address_unpack_iface_attr_v2(ucp_worker_t *worker,
                                 ucp_address_iface_attr_t *iface_attr,
                                 const void *ptr)
{
    const ucp_address_v2_packed_iface_attr_t *packed = ptr;

    iface_attr->priority    = packed->prio;
    iface_attr->seg_size    = packed->seg_size *
                              UCP_ADDRESS_IFACE_SEG_SIZE_FACTOR;
    iface_attr->overhead    = UCS_FP8_UNPACK(OVERHEAD, packed->overhead) /
                                             UCS_NSEC_PER_SEC;
    iface_attr->lat_ovh     = UCS_FP8_UNPACK(LATENCY, packed->latency) /
                                             UCS_NSEC_PER_SEC;
    iface_attr->bandwidth   = UCS_FP8_UNPACK(BANDWIDTH, packed->bandwidth);
    iface_attr->flags       = packed->flags;

    return sizeof(*packed);
}

static ucs_status_t
ucp_address_unpack_iface_attr(ucp_worker_t *worker,
                              ucp_address_iface_attr_t *iface_attr,
                              const void *ptr, unsigned unpack_flags,
                              ucp_object_version_t addr_version, size_t *size_p)
{
    const ucp_address_unified_iface_attr_t *unified;
    ucp_worker_iface_t *wiface;
    ucp_rsc_index_t rsc_idx;
    int iface_attr_len;

    if (ucp_worker_is_unified_mode(worker)) {
        /* Address contains resources index and iface latency overhead
         * (not all iface attrs). */
        unified             = ptr;
        rsc_idx             = unified->rsc_index & UCP_ADDRESS_IFACE_LEN_MASK;
        iface_attr->lat_ovh = fabs(unified->lat_ovh);
        if (!UCS_STATIC_BITMAP_GET(worker->context->tl_bitmap, rsc_idx)) {
            ucp_address_error(
                    unpack_flags,
                    "failed to unpack address, resource[%d] is not valid",
                    rsc_idx);
            return UCS_ERR_INVALID_ADDR;
        }

        /* Just take the rest of iface attrs from the local resource. */
        wiface                    = ucp_worker_iface(worker, rsc_idx);
        iface_attr->flags         = ucp_address_flags_from_iface_flags(
                                        wiface->attr.cap.flags,
                                        wiface->attr.cap.event_flags);
        iface_attr->priority      = wiface->attr.priority;
        iface_attr->overhead      = wiface->attr.overhead;
        iface_attr->bandwidth     =
                ucp_tl_iface_bandwidth(worker->context,
                                       &wiface->attr.bandwidth);
        iface_attr->seg_size      = wiface->attr.cap.am.max_bcopy;

        if (signbit(unified->lat_ovh)) {
            iface_attr->atomic.atomic32.op_flags  = wiface->attr.cap.atomic32.op_flags;
            iface_attr->atomic.atomic32.fop_flags = wiface->attr.cap.atomic32.fop_flags;
            iface_attr->atomic.atomic64.op_flags  = wiface->attr.cap.atomic64.op_flags;
            iface_attr->atomic.atomic64.fop_flags = wiface->attr.cap.atomic64.fop_flags;
        }

        *size_p = sizeof(*unified);
        return UCS_OK;
    }

    if (addr_version == UCP_OBJECT_VERSION_V1) {
        iface_attr_len = ucp_address_unpack_iface_attr_v1(worker, iface_attr,
                                                          ptr);
    } else {
        iface_attr_len = ucp_address_unpack_iface_attr_v2(worker, iface_attr,
                                                          ptr);
    }

    if (iface_attr->bandwidth <= 0) {
        ucp_address_error(unpack_flags,
                          "failed to unpack address, invalid bandwidth %.2f",
                          iface_attr->bandwidth);
        return UCS_ERR_INVALID_ADDR;
    }

    /* Unpack iface 32-bit atomic operations */
    if (iface_attr->flags & UCP_ADDR_IFACE_FLAG_ATOMIC32) {
        iface_attr->atomic.atomic32.op_flags  |= UCP_ATOMIC_OP_MASK;
        iface_attr->atomic.atomic32.fop_flags |= UCP_ATOMIC_FOP_MASK;
    }

    /* Unpack iface 64-bit atomic operations */
    if (iface_attr->flags & UCP_ADDR_IFACE_FLAG_ATOMIC64) {
        iface_attr->atomic.atomic64.op_flags  |= UCP_ATOMIC_OP_MASK;
        iface_attr->atomic.atomic64.fop_flags |= UCP_ATOMIC_FOP_MASK;
    }

    *size_p = iface_attr_len;

    if (unpack_flags & UCP_ADDRESS_PACK_FLAG_TL_RSC_IDX) {
        /* dst_rsc_index was removed from unpacked address, but still exists
         * in packed address (for wire compatibility), so we need to skip it */
        *size_p += sizeof(uint8_t);
    }

    return UCS_OK;
}

static void*
ucp_address_iface_flags_ptr(ucp_worker_h worker, void *attr_ptr, int attr_len)
{
    if (ucp_worker_is_unified_mode(worker)) {
        /* In unified mode, rsc_index is packed instead of attrs. Address flags
         * will be packed in the end of rsc_index byte. */
        UCS_STATIC_ASSERT(ucs_offsetof(ucp_address_unified_iface_attr_t,
                                       rsc_index) == 0);
        return attr_ptr;
    }

    /* In non-unified mode, address flags will be packed in the end of
     * iface addr length byte, which is packed right after iface attrs. */
    return UCS_PTR_BYTE_OFFSET(attr_ptr, attr_len);
}

static void *ucp_address_pack_tl_length(ucp_worker_h worker, void *ptr,
                                        unsigned max_length, size_t addr_length,
                                        ucp_object_version_t addr_version,
                                        int is_extendable)
{
    if (ucp_worker_is_unified_mode(worker)) {
        return ptr;
    }

    if (is_extendable) {
        return ucp_address_pack_byte_extended(ptr, addr_length, max_length,
                                              addr_version);
    }

    *ucs_serialize_next(&ptr, uint8_t) = addr_length;

    return ptr;
}

static const void *
ucp_address_unpack_tl_length(ucp_worker_h worker, const void *flags_ptr,
                             const void *ptr, ucp_object_version_t addr_version,
                             uint8_t *addr_length, int is_ep_addr,
                             int *is_last_iface)
{
    ucp_rsc_index_t rsc_index;
    uct_iface_attr_t *attr;
    const ucp_address_unified_iface_attr_t *unified;

    /* Caller should not use *is_last_iface for ep address, because for ep
     * address last flag is part of lane index */
    ucs_assert(!is_ep_addr || is_last_iface == NULL);

    if (ucp_worker_is_unified_mode(worker)) {
        /* In unified mode:
         * - flags are packed with rsc index in ucp_address_unified_iface_attr_t
         * - iface and ep addr lengths are not packed, need to take them from
         *   local iface attrs */
        unified   = flags_ptr;
        rsc_index = unified->rsc_index & UCP_ADDRESS_IFACE_LEN_MASK;
        attr      = ucp_worker_iface_get_attr(worker, rsc_index);

        ucs_assert(&unified->rsc_index == flags_ptr);

        if (is_ep_addr) {
            *addr_length = attr->ep_addr_len;
        } else {
            *addr_length   = attr->iface_addr_len;
            *is_last_iface = unified->rsc_index & UCP_ADDRESS_FLAG_LAST;
        }
        return ptr;
    }

    if (is_ep_addr) {
        *addr_length = *ucs_serialize_next(&ptr, uint8_t);
        return ptr;
    }

    *is_last_iface = *(uint8_t*)ptr & UCP_ADDRESS_FLAG_LAST;

    return ucp_address_unpack_byte_extended(ptr, UCP_ADDRESS_IFACE_LEN_MASK,
                                            addr_version, addr_length);
}

static void ucp_address_pack_header_flags(uint8_t *address_header,
                                          ucp_object_version_t addr_version,
                                          uint8_t flags)
{
    if (addr_version == UCP_OBJECT_VERSION_V1) {
        *address_header |= (flags << UCP_ADDRESS_HEADER_SHIFT);
    } else {
        address_header += 1;
        *address_header = flags;
    }
}

static void *ucp_address_pack_header(uint8_t *ptr,
                                     ucp_object_version_t addr_version)
{
    uint8_t *addr_header = ptr;

    *addr_header = addr_version;

    if (addr_version == UCP_OBJECT_VERSION_V1) {
        return UCS_PTR_TYPE_OFFSET(ptr, uint8_t);
    }

    *addr_header |= ucp_address_pack_release_version() << UCP_ADDRESS_HEADER_SHIFT;

    return UCS_PTR_TYPE_OFFSET(ptr, uint16_t);
}

static void *ucp_address_unpack_header(const void *ptr,
                                       ucp_object_version_t *addr_version,
                                       uint8_t *addr_flags,
                                       unsigned *dst_version)
{
    const uint8_t *addr_header = ptr;

    *addr_version = *addr_header & UCP_ADDRESS_HEADER_VERSION_MASK;

    if (*addr_version == UCP_OBJECT_VERSION_V1) {
        *addr_flags  = *addr_header >> UCP_ADDRESS_HEADER_SHIFT;
        /* With address v1 it is packed after devices */
        *dst_version = 0;
        return UCS_PTR_TYPE_OFFSET(ptr, uint8_t);
    }

    ucs_assertv_always(*addr_version == UCP_OBJECT_VERSION_V2,
                       "addr version %u", *addr_version);

    *addr_flags  = *(addr_header + 1);
    *dst_version = ucp_address_unpack_release_version(
                       *addr_header >> UCP_ADDRESS_HEADER_SHIFT);

    return UCS_PTR_TYPE_OFFSET(ptr, uint16_t);
}

uint64_t ucp_address_get_uuid(const void *address)
{
    uint64_t *uuid;
    ucp_object_version_t address_version;
    unsigned dst_version;
    uint8_t flags;

    uuid = ucp_address_unpack_header(address, &address_version, &flags,
                                     &dst_version);

    return (flags & UCP_ADDRESS_HEADER_FLAG_WORKER_UUID) ?
           *uuid : UCP_ADDRESS_DEFAULT_WORKER_UUID;
}

uint64_t ucp_address_get_client_id(const void *address)
{
    const void *offset;
    ucp_object_version_t address_version;
    unsigned dst_version;
    uint8_t flags;

    offset = ucp_address_unpack_header(address, &address_version, &flags,
                                       &dst_version);
    if (!(flags & UCP_ADDRESS_HEADER_FLAG_CLIENT_ID)) {
        return UCP_ADDRESS_DEFAULT_CLIENT_ID;
    }

    if (flags & UCP_ADDRESS_HEADER_FLAG_WORKER_UUID) {
        offset = UCS_PTR_TYPE_OFFSET(offset, uint64_t);
    }

    return *ucs_serialize_next(&offset, uint64_t);
}

uint8_t ucp_address_is_am_only(const void *address)
{
    uint8_t addr_flags;
    ucp_object_version_t addr_version;
    unsigned dst_version;

    ucp_address_unpack_header(address, &addr_version, &addr_flags,
                              &dst_version);
    return addr_flags & UCP_ADDRESS_HEADER_FLAG_AM_ONLY;
}

static ucs_status_t
ucp_address_do_pack(ucp_worker_h worker, ucp_ep_h ep, void *buffer, size_t size,
                    unsigned pack_flags, ucp_object_version_t addr_version,
                    const ucp_lane_index_t *lanes2remote,
                    const ucp_address_packed_device_t *devices,
                    ucp_rsc_index_t num_devices)
{
    ucp_context_h context       = worker->context;
    uint64_t md_flags_pack_mask = (UCT_MD_FLAG_REG | UCT_MD_FLAG_ALLOC);
    const ucp_address_packed_device_t *dev;
    uint8_t *address_header_p;
    uct_iface_attr_t *iface_attr;
    ucp_md_index_t md_index;
    ucp_worker_iface_t *wiface;
    ucp_rsc_index_t rsc_index;
    ucp_lane_index_t lane, remote_lane;
    ucp_tl_bitmap_t dev_tl_bitmap;
    unsigned num_ep_addrs;
    ucs_status_t status;
    size_t iface_addr_len;
    size_t ep_addr_len;
    uint64_t md_flags;
    uint8_t *ep_lane_ptr;
    void *flags_ptr, *dev_flags_ptr;
    unsigned addr_index;
    int attr_len;
    void *ptr;
    int enable_amo;
    uint8_t addr_flags;

    ptr               = buffer;
    addr_index        = 0;
    addr_flags        = 0;
    address_header_p  = ptr;
    ptr               = ucp_address_pack_header(address_header_p, addr_version);
    dev_flags_ptr     = NULL;

    if (pack_flags & UCP_ADDRESS_PACK_FLAG_AM_ONLY) {
        addr_flags |= UCP_ADDRESS_HEADER_FLAG_AM_ONLY;
    }

    if (pack_flags & UCP_ADDRESS_PACK_FLAG_WORKER_UUID) {
        *(uint64_t*)ptr = worker->uuid;
        ptr             = UCS_PTR_TYPE_OFFSET(ptr, uint64_t);
        addr_flags     |= UCP_ADDRESS_HEADER_FLAG_WORKER_UUID;
    }

    if (pack_flags & UCP_ADDRESS_PACK_FLAG_CLIENT_ID) {
        *(uint64_t*)ptr = worker->client_id;
        ptr             = UCS_PTR_TYPE_OFFSET(ptr, uint64_t);
        addr_flags     |= UCP_ADDRESS_HEADER_FLAG_CLIENT_ID;
    }

    if (worker->context->config.ext.address_debug_info) {
        /* Add debug information to the packed address, and set the corresponding
         * flag in address header.
         */
        addr_flags |= UCP_ADDRESS_HEADER_FLAG_DEBUG_INFO;

        if (pack_flags & UCP_ADDRESS_PACK_FLAG_WORKER_NAME) {
            ptr            = ucp_address_pack_worker_address_name(worker, ptr);
        }
    }

    ucp_address_pack_header_flags(address_header_p, addr_version, addr_flags);

    if (num_devices == 0) {
        *((uint8_t*)ptr) = UCP_NULL_RESOURCE;
        ptr = UCS_PTR_TYPE_OFFSET(ptr, UCP_NULL_RESOURCE);
        goto out;
    }

    for (dev = devices; dev < (devices + num_devices); ++dev) {
        dev_tl_bitmap = context->tl_bitmap;
        UCS_STATIC_BITMAP_AND_INPLACE(&dev_tl_bitmap, dev->tl_bitmap);

        /* MD index */
        md_index      = context->tl_rscs[dev->rsc_index].md_index;
        md_flags      = context->tl_mds[md_index].attr.flags &
                            md_flags_pack_mask;
        ptr           = ucp_address_pack_md_info(ptr,
                                                 UCS_STATIC_BITMAP_IS_ZERO(dev_tl_bitmap),
                                                 md_flags, md_index, addr_version);
        dev_flags_ptr = ptr;
        ucs_assert_always((pack_flags & UCP_ADDRESS_PACK_FLAG_DEVICE_ADDR) ||
                          (dev->dev_addr_len == 0));
        ptr = ucp_address_pack_byte_extended(ptr, dev->dev_addr_len,
                                             UCP_ADDRESS_DEVICE_LEN_MASK,
                                             addr_version);

        /* Device number of paths flag and value */
        ucs_assert(dev->num_paths >= 1);
        if (dev->num_paths > 1) {
            ucs_assert(dev->num_paths <= UINT8_MAX);
            *(uint8_t*)dev_flags_ptr |= UCP_ADDRESS_FLAG_NUM_PATHS;
            *(uint8_t*)ptr            = dev->num_paths;
            ptr                       = UCS_PTR_TYPE_OFFSET(ptr, uint8_t);
        }

        /* System device */
        if (dev->sys_dev != UCS_SYS_DEVICE_ID_UNKNOWN) {
            *(uint8_t*)dev_flags_ptr |= UCP_ADDRESS_FLAG_SYS_DEVICE;
            *(uint8_t*)ptr            = dev->sys_dev;
            ptr                       = UCS_PTR_TYPE_OFFSET(ptr, uint8_t);
        }

        /* Device address */
        if (pack_flags & UCP_ADDRESS_PACK_FLAG_DEVICE_ADDR) {
            wiface = ucp_worker_iface(worker, dev->rsc_index);
            status = uct_iface_get_device_address(wiface->iface,
                                                  (uct_device_addr_t*)ptr);
            if (status != UCS_OK) {
                ucp_address_error(
                        pack_flags, "failed to get %s device address %s",
                        context->tl_rscs[dev->rsc_index].tl_rsc.dev_name,
                        ucs_status_string(status));
                return status;
            }

            ucp_address_memcheck(context, ptr, dev->dev_addr_len, dev->rsc_index);
            ptr = UCS_PTR_BYTE_OFFSET(ptr, dev->dev_addr_len);
        }

        flags_ptr = NULL;
        UCS_STATIC_BITMAP_FOR_EACH_BIT(rsc_index, &dev_tl_bitmap) {
            wiface     = ucp_worker_iface(worker, rsc_index);
            iface_attr = &wiface->attr;

            if (!ucp_worker_iface_can_connect(iface_attr)) {
                ucp_address_error(pack_flags,
                                  UCT_TL_RESOURCE_DESC_FMT
                                  " doesn't have connect caps: 0x%lx",
                                  UCT_TL_RESOURCE_DESC_ARG(
                                          &context->tl_rscs[rsc_index].tl_rsc),
                                  iface_attr->cap.flags);
                return UCS_ERR_INVALID_ADDR;
            }

            /* Transport name checksum */
            *(uint16_t*)ptr = context->tl_rscs[rsc_index].tl_name_csum;
            ptr = UCS_PTR_TYPE_OFFSET(ptr,
                                      context->tl_rscs[rsc_index].tl_name_csum);

            /* Transport information */
            enable_amo = UCS_STATIC_BITMAP_GET(worker->atomic_tls, rsc_index);
            attr_len   = ucp_address_pack_iface_attr(wiface, ptr, rsc_index,
                                                     pack_flags, addr_version,
                                                     enable_amo);
            if (attr_len < 0) {
                return UCS_ERR_INVALID_ADDR;
            }

            ucp_address_memcheck(context, ptr, attr_len, rsc_index);

            if (pack_flags & UCP_ADDRESS_PACK_FLAG_IFACE_ADDR) {
                iface_addr_len = iface_attr->iface_addr_len;
            } else {
                iface_addr_len = 0;
            }

            flags_ptr = ucp_address_iface_flags_ptr(worker, ptr, attr_len);
            ptr       = UCS_PTR_BYTE_OFFSET(ptr, attr_len);

            /* Pack iface address */
            ptr = ucp_address_pack_tl_length(worker, ptr,
                                             UCP_ADDRESS_IFACE_LEN_MASK,
                                             iface_addr_len, addr_version, 1);
            if (pack_flags & UCP_ADDRESS_PACK_FLAG_IFACE_ADDR) {
                status = uct_iface_get_address(wiface->iface,
                                               (uct_iface_addr_t*)ptr);
                if (status != UCS_OK) {
                    ucp_address_error(
                            pack_flags,
                            UCT_TL_RESOURCE_DESC_FMT
                            " failed to get iface address %s",
                            UCT_TL_RESOURCE_DESC_ARG(
                                    &context->tl_rscs[rsc_index].tl_rsc),
                            ucs_status_string(status));
                    return status;
                }

                ucp_address_memcheck(context, ptr, iface_addr_len, rsc_index);
                ptr = UCS_PTR_BYTE_OFFSET(ptr, iface_addr_len);
            }

            /* Pack ep address if present: iterate over all lanes which use the
             * current resource (rsc_index) and pack their addresses. The last
             * one is marked with UCP_ADDRESS_FLAG_LAST in its length field.
             */
            num_ep_addrs = 0;
            if (pack_flags & UCP_ADDRESS_PACK_FLAG_EP_ADDR) {
                ucs_assert(ep != NULL);
                ep_addr_len = iface_attr->ep_addr_len;
                ep_lane_ptr = NULL;

                ucs_for_each_bit(lane, ucp_ep_config(ep)->p2p_lanes) {
                    ucs_assert(lane < UCP_MAX_LANES);
                    if (ucp_ep_get_rsc_index(ep, lane) != rsc_index) {
                        continue;
                    }

                    /* pack ep address length */
                    ptr = ucp_address_pack_tl_length(worker, ptr, UINT8_MAX,
                                                     ep_addr_len, addr_version,
                                                     0);

                    /* pack ep address */
                    status = uct_ep_get_address(ucp_ep_get_lane(ep, lane), ptr);
                    if (status != UCS_OK) {
                        ucp_address_error(
                                pack_flags,
                                UCT_TL_RESOURCE_DESC_FMT
                                " failed to get ep address %s",
                                UCT_TL_RESOURCE_DESC_ARG(
                                        &context->tl_rscs[rsc_index].tl_rsc),
                                ucs_status_string(status));
                        return status;
                    }

                    ucp_address_memcheck(context, ptr, ep_addr_len, rsc_index);
                    ptr = UCS_PTR_BYTE_OFFSET(ptr, ep_addr_len);

                    /* pack ep lane index, and save the pointer for lane index
                     * of last ep in 'ep_last_ptr' to set UCP_ADDRESS_FLAG_LAST.
                     */
                    remote_lane  = (lanes2remote == NULL) ? lane :
                                   lanes2remote[lane];
                    ucs_assertv(remote_lane <= UCP_ADDRESS_IFACE_LEN_MASK,
                                "remote_lane=%d", remote_lane);
                    ep_lane_ptr  = ptr;
                    *ep_lane_ptr = remote_lane;
                    ptr          = UCS_PTR_TYPE_OFFSET(ptr, uint8_t);

                    ucp_address_trace(
                            pack_flags,
                            "pack addr[%d].ep_addr[%d] : len %zu lane %d->%d",
                            addr_index, num_ep_addrs, ep_addr_len, lane,
                            remote_lane);

                    ++num_ep_addrs;
                }

                if (num_ep_addrs > 0) {
                    /* set LAST flag for the last ep address */
                    ucs_assert(ep_lane_ptr != NULL);
                    *ep_lane_ptr         |= UCP_ADDRESS_FLAG_LAST;
                    /* indicate that the iface has ep address */
                    *(uint8_t*)flags_ptr |= UCP_ADDRESS_FLAG_HAS_EP_ADDR;
                }
            }

            ucs_assert((num_ep_addrs > 0) ||
                       !(*(uint8_t*)flags_ptr & UCP_ADDRESS_FLAG_HAS_EP_ADDR));

            ucp_address_trace(pack_flags,
                              "pack addr[%d] : " UCT_TL_RESOURCE_DESC_FMT
                              " sysdev %d paths %d eps %u md_flags 0x%" PRIx64
                              " tl_flags 0x%" PRIx64 " bw %.2f+%.2f/nMBs"
                              " ovh %.0fns lat_ovh %.0fns dev_priority %d"
                              " a32 0x%" PRIx64 "/0x%" PRIx64 " a64 0x%" PRIx64
                              "/0x%" PRIx64,
                              addr_index,
                              UCT_TL_RESOURCE_DESC_ARG(
                                      &context->tl_rscs[rsc_index].tl_rsc),
                              dev->sys_dev, dev->num_paths, num_ep_addrs,
                              md_flags,
                              iface_attr->cap.flags & UCP_ADDRESS_IFACE_FLAGS,
                              iface_attr->bandwidth.dedicated / UCS_MBYTE,
                              iface_attr->bandwidth.shared / UCS_MBYTE,
                              iface_attr->overhead * 1e9,
                              iface_attr->latency.c * 1e9, iface_attr->priority,
                              iface_attr->cap.atomic32.op_flags,
                              iface_attr->cap.atomic32.fop_flags,
                              iface_attr->cap.atomic64.op_flags,
                              iface_attr->cap.atomic64.fop_flags);

            ++addr_index;
            ucs_assert(addr_index <= UCP_MAX_RESOURCES);
        }

        /* flags_ptr is a valid pointer to the flags set to the last entry
         * during the above loop So, set the LAST flag for the flags_ptr
         * from the last iteration */
        if (flags_ptr != NULL) {
            ucs_assert(!UCS_STATIC_BITMAP_IS_ZERO(dev_tl_bitmap));
            *(uint8_t*)flags_ptr |= UCP_ADDRESS_FLAG_LAST;
        } else {
            /* cppcheck-suppress internalAstError */
            ucs_assert(UCS_STATIC_BITMAP_IS_ZERO(dev_tl_bitmap));
        }
    }

    if (ucp_address_pack_v1_extra_info(addr_version, pack_flags)) {
        /* Pack our release version in a wire compatible way:
         * pack a pseudo empty device after all real devices and specify
         * release version instead of md index.
         */
        ucs_assertv(addr_version == UCP_OBJECT_VERSION_V1, "addr_version=%u",
                    addr_version);
        ptr = ucp_address_pack_md_info(
                ptr, 1, 0, ucp_address_pack_release_version(), addr_version);
        /* Mark this device as the last one with 0 length address */
        *ucs_serialize_next(&ptr, uint8_t) = UCP_ADDRESS_FLAG_LAST;
    } else if (num_devices > 0) {
        ucs_assert(dev_flags_ptr != NULL);
        *(uint8_t*)dev_flags_ptr |= UCP_ADDRESS_FLAG_LAST;
    }

out:
    ucs_assertv(UCS_PTR_BYTE_OFFSET(buffer, size) == ptr,
                "buffer=%p size=%zu ptr=%p ptr-buffer=%zd",
                buffer, size, ptr, UCS_PTR_BYTE_DIFF(buffer, ptr));
    return UCS_OK;
}

ucs_status_t
ucp_address_length(ucp_worker_h worker, const ucp_ep_config_key_t *key,
                   const ucp_tl_bitmap_t *tl_bitmap, unsigned pack_flags,
                   ucp_object_version_t addr_version, size_t *size_p)
{
    ucp_address_packed_device_t *devices;
    ucp_rsc_index_t num_devices;
    ucs_status_t status;
    ssize_t size;

    /* Collect all devices required to pack their address */
    status = ucp_address_gather_devices(worker, key, tl_bitmap, pack_flags,
                                        addr_version, UINT_MAX, &devices,
                                        &num_devices);
    if (status != UCS_OK) {
        goto out;
    }

    /* Calculate the required ucp address length */
    size = ucp_address_packed_size(worker, devices, num_devices, pack_flags,
                                   addr_version);
    if (size > -1) {
        *size_p = size;
    } else {
        status = size;
    }

    ucs_free(devices);

out:
    return status;
}

ucs_status_t ucp_address_pack(ucp_worker_h worker, ucp_ep_h ep,
                              const ucp_tl_bitmap_t *tl_bitmap,
                              unsigned pack_flags,
                              ucp_object_version_t addr_version,
                              const ucp_lane_index_t *lanes2remote,
                              unsigned max_num_paths, size_t *size_p,
                              void **buffer_p)
{
    ucp_address_packed_device_t *devices;
    ucp_rsc_index_t num_devices;
    const ucp_ep_config_key_t *key;
    ucs_status_t status;
    void *buffer;
    ssize_t size;

    if (ep == NULL) {
        pack_flags &= ~UCP_ADDRESS_PACK_FLAG_EP_ADDR;
        key         = NULL;
    } else {
        key         = &ucp_ep_config(ep)->key;
    }

    /* Collect all devices we want to pack */
    status = ucp_address_gather_devices(worker, key, tl_bitmap, pack_flags,
                                        addr_version, max_num_paths, &devices,
                                        &num_devices);
    if (status != UCS_OK) {
        goto out;
    }

    /* Calculate packed size */
    size = ucp_address_packed_size(worker, devices, num_devices, pack_flags,
                                   addr_version);
    if (size < 0) {
        status = (ucs_status_t)size;
        goto out_free_devices;
    }

    /* Allocate address */
    buffer = ucs_malloc(size, "ucp_address");
    if (buffer == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out_free_devices;
    }

    memset(buffer, 0, size);

    /* Pack the address */
    status = ucp_address_do_pack(worker, ep, buffer, size, pack_flags,
                                 addr_version, lanes2remote, devices,
                                 num_devices);
    if (status != UCS_OK) {
        ucs_free(buffer);
        goto out_free_devices;
    }

    VALGRIND_CHECK_MEM_IS_DEFINED(buffer, size);

    *size_p   = size;
    *buffer_p = buffer;
    status    = UCS_OK;

out_free_devices:
    ucs_free(devices);
out:
    return status;
}

static ucp_rsc_index_t ucp_address_get_remote_device_index(
        ucp_address_remote_device_array_t *device_array,
        ucp_rsc_index_t dev_index, ucs_sys_device_t sys_dev)
{
    ucp_rsc_index_t unique_dev_index = 0;
    ucp_address_remote_device_t *device;

    ucs_array_for_each(device, device_array) {
        /* Devices with different dev_index are still considered identical,
         * if the sys_dev is equal */
        if ((device->dev_index == dev_index) ||
            ((device->sys_dev != UCS_SYS_DEVICE_ID_UNKNOWN) &&
             (device->sys_dev == sys_dev))) {
            return unique_dev_index;
        }

        unique_dev_index++;
    }

    device            = ucs_array_append_fixed(device_array);
    device->sys_dev   = sys_dev;
    device->dev_index = dev_index;
    return ucs_array_length(device_array) - 1;
}

ucs_status_t ucp_address_unpack(ucp_worker_t *worker, const void *buffer,
                                unsigned unpack_flags,
                                ucp_unpacked_address_t *unpacked_address)
{
    UCS_ARRAY_DEFINE_ONSTACK(ucp_address_remote_device_array_t,
                             remote_device_array, UCP_MAX_RESOURCES);
    ucp_address_entry_t *address_list, *address;
    uint8_t addr_flags;
    ucp_object_version_t addr_version;
    unsigned dst_version;
    ucp_address_entry_ep_addr_t *ep_addr;
    int last_dev, last_tl, last_ep_addr;
    const uct_device_addr_t *dev_addr;
    ucp_rsc_index_t dev_index;
    ucs_sys_device_t sys_dev;
    ucp_md_index_t md_index;
    unsigned dev_num_paths;
    ucs_status_t status;
    int empty_dev;
    uint8_t dev_addr_len, iface_addr_len, ep_addr_len;
    size_t attr_len;
    uint8_t flags;
    const void *ptr;
    const void *flags_ptr;

    /* Initialize the unpacked address to empty */
    unpacked_address->address_count = 0;
    unpacked_address->address_list  = NULL;

    ptr = ucp_address_unpack_header(buffer, &addr_version, &addr_flags,
                                    &dst_version);

    ucp_address_trace(unpack_flags,
                      "unpacking address version %u dst version %u flags 0x%x",
                      addr_version, dst_version, addr_flags);

    if (((unpack_flags & UCP_ADDRESS_PACK_FLAG_WORKER_UUID) &&
         (addr_version == UCP_OBJECT_VERSION_V1)) ||
        (addr_flags & UCP_ADDRESS_HEADER_FLAG_WORKER_UUID)) {
        /* NOTE:
         * 1. addr_flags may not contain UCP_ADDRESS_HEADER_FLAG_WORKER_UUID
         *    even though the worker uuid is packed, because this flags was
         *    introduced in UCX v1.12.
         * 2. Unpack worker uuid if addr_flags contains
         *    UCP_ADDRESS_HEADER_FLAG_WORKER_UUID, even if there is no
         *    UCP_ADDRESS_PACK_FLAG_WORKER_UUID bit in unpack_flags, to
         *    correctly unpack the address.
         */
        unpacked_address->uuid = ucp_address_get_uuid(buffer);
        ptr                    = UCS_PTR_TYPE_OFFSET(ptr,
                                                     unpacked_address->uuid);
    } else {
        unpacked_address->uuid = 0ul;
    }

    if (addr_flags & UCP_ADDRESS_HEADER_FLAG_CLIENT_ID) {
        ptr = UCS_PTR_TYPE_OFFSET(ptr, uint64_t);
    }

    if ((addr_flags & UCP_ADDRESS_HEADER_FLAG_DEBUG_INFO) &&
        (unpack_flags & UCP_ADDRESS_PACK_FLAG_WORKER_NAME)) {
        ptr = ucp_address_unpack_worker_address_name(ptr,
                                                     unpacked_address->name);
    } else {
        ucs_strncpy_safe(unpacked_address->name, UCP_WIREUP_EMPTY_PEER_NAME,
                         sizeof(unpacked_address->name));
    }

    /* Empty address list */
    if (*(uint8_t*)ptr == UCP_NULL_RESOURCE) {
        return UCS_OK;
    }

    /* Allocate address list */
    address_list = ucs_calloc(UCP_MAX_RESOURCES, sizeof(*address_list),
                              "ucp_address_list");
    if (address_list == NULL) {
        ucs_error("failed to allocate address list");
        return UCS_ERR_NO_MEMORY;
    }

    /* Unpack addresses */
    address   = address_list;
    dev_index = 0;

    do {
        /* md_index and empty flag */
        ptr      = ucp_address_unpack_md_info(ptr, addr_version, &md_index,
                                              &empty_dev);

        /* device address length */
        flags    = (*(uint8_t*)ptr) & ~UCP_ADDRESS_DEVICE_LEN_MASK;
        last_dev = flags & UCP_ADDRESS_FLAG_LAST;
        ptr = ucp_address_unpack_byte_extended(ptr, UCP_ADDRESS_DEVICE_LEN_MASK,
                                               addr_version, &dev_addr_len);

        if (flags & UCP_ADDRESS_FLAG_NUM_PATHS) {
            dev_num_paths = *(uint8_t*)ptr;
            ptr           = UCS_PTR_TYPE_OFFSET(ptr, uint8_t);
        } else {
            dev_num_paths = 1;
        }
        if (flags & UCP_ADDRESS_FLAG_SYS_DEVICE) {
            sys_dev = *(uint8_t*)ptr;
            ptr     = UCS_PTR_TYPE_OFFSET(ptr, uint8_t);
        } else {
            sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;
        }

        if (empty_dev && last_dev) {
            /* Wire version takes just 4 least significant bits, there are still
             * 3 bits left in md index for future extensions (the most
             * significant bit is occupied by UCP_ADDRESS_FLAG_MD_EMPTY_DEV).
             */
            dst_version = ucp_address_unpack_release_version(
                         md_index & UCS_MASK(UCP_ADDRESS_RELEASE_VERSION_BITS));
            ucp_address_trace(unpack_flags,
                              "unpacked dst release version %u", dst_version);
        }

        dev_addr = ptr;
        ptr      = UCS_PTR_BYTE_OFFSET(ptr, dev_addr_len);

        last_tl = empty_dev;
        while (!last_tl) {
            if (address >= &address_list[UCP_MAX_RESOURCES]) {
                ucp_address_error(unpack_flags,
                                  "failed to parse address: number of addresses"
                                  " exceeds %d",
                                  UCP_MAX_RESOURCES);
                goto err_free;
            }

            /* tl_name_csum */
            address->tl_name_csum = *(uint16_t*)ptr;
            ptr = UCS_PTR_TYPE_OFFSET(ptr, address->tl_name_csum);

            address->dev_addr      = (dev_addr_len > 0) ? dev_addr : NULL;
            address->dev_addr_len  = dev_addr_len;
            address->md_index      = md_index;
            address->sys_dev       = sys_dev;
            /* We create a new dev_index based on both sys_dev and dev_index fields */
            address->dev_index     = ucp_address_get_remote_device_index(
                    &remote_device_array, dev_index, sys_dev);
            address->dev_num_paths = dev_num_paths;

            status = ucp_address_unpack_iface_attr(worker, &address->iface_attr,
                                                   ptr, unpack_flags,
                                                   addr_version, &attr_len);
            if (status != UCS_OK) {
                goto err_free;
            }

            flags_ptr = ucp_address_iface_flags_ptr(worker, (void*)ptr,
                                                    attr_len);
            ptr       = UCS_PTR_BYTE_OFFSET(ptr, attr_len);
            ptr       = ucp_address_unpack_tl_length(
                                          worker, flags_ptr, ptr, addr_version,
                                          &iface_addr_len, 0, &last_tl);
            address->iface_addr   = (iface_addr_len > 0) ? ptr : NULL;
            address->num_ep_addrs = 0;
            ptr                   = UCS_PTR_BYTE_OFFSET(ptr, iface_addr_len);
            last_ep_addr          = !(*(uint8_t*)flags_ptr &
                                      UCP_ADDRESS_FLAG_HAS_EP_ADDR);
            while (!last_ep_addr) {
                if (address->num_ep_addrs >= UCP_MAX_LANES) {
                    ucp_address_error(
                            unpack_flags,
                            "failed to parse address: number of ep addresses"
                            " exceeds %d",
                            UCP_MAX_LANES);
                    goto err_free;
                }

                ptr = ucp_address_unpack_tl_length(worker, flags_ptr, ptr,
                                                   addr_version, &ep_addr_len,
                                                   1, NULL);

                ep_addr       = &address->ep_addrs[address->num_ep_addrs++];
                ep_addr->addr = ptr;
                ep_addr->len  = ep_addr_len;
                ptr           = UCS_PTR_BYTE_OFFSET(ptr, ep_addr_len);

                ep_addr->lane = *(uint8_t*)ptr & UCP_ADDRESS_IFACE_LEN_MASK;
                last_ep_addr  = *(uint8_t*)ptr & UCP_ADDRESS_FLAG_LAST;

                ucp_address_trace(
                        unpack_flags,
                        "unpack addr[%d].ep_addr[%d] : len %d lane %d",
                        (int)(address - address_list),
                        (int)(ep_addr - address->ep_addrs), ep_addr_len,
                        ep_addr->lane);

                ptr           = UCS_PTR_TYPE_OFFSET(ptr, uint8_t);
            }

            ucp_address_trace(unpack_flags,
                              "unpack addr[%d] : sysdev %d paths %d eps %u"
                              " tl_flags 0x%" PRIx64 " bw %.2f/nMBs"
                              " ovh %.0fns lat_ovh %.0fns dev_priority %d"
                              " a32 0x%" PRIx64 "/0x%" PRIx64 " a64 0x%" PRIx64
                              "/0x%" PRIx64,
                              (int)(address - address_list), address->sys_dev,
                              address->dev_num_paths, address->num_ep_addrs,
                              address->iface_attr.flags,
                              address->iface_attr.bandwidth / UCS_MBYTE,
                              address->iface_attr.overhead * 1e9,
                              address->iface_attr.lat_ovh * 1e9,
                              address->iface_attr.priority,
                              address->iface_attr.atomic.atomic32.op_flags,
                              address->iface_attr.atomic.atomic32.fop_flags,
                              address->iface_attr.atomic.atomic64.op_flags,
                              address->iface_attr.atomic.atomic64.fop_flags);

            ++address;
        }

        ++dev_index;
    } while (!last_dev);

    unpacked_address->addr_version  = addr_version;
    unpacked_address->dst_version   = dst_version;
    unpacked_address->address_count = address - address_list;
    unpacked_address->address_list  = address_list;
    return UCS_OK;

err_free:
    ucs_free(address_list);
    return UCS_ERR_INVALID_PARAM;
}
