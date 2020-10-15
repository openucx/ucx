/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
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
#include <ucs/debug/log.h>
#include <inttypes.h>


/*
 * Packed address layout:
 *
 * [ header(8bit) | uuid(64bit) | worker_name(string) ]
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


typedef struct {
    size_t           dev_addr_len;
    uint64_t         tl_bitmap;
    ucp_rsc_index_t  rsc_index;
    ucp_rsc_index_t  tl_count;
    unsigned         num_paths;
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
} ucp_address_packed_iface_attr_t;


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
} ucp_address_unified_iface_attr_t;


#define UCP_ADDRESS_FLAG_ATOMIC32     UCS_BIT(30) /* 32bit atomic operations */
#define UCP_ADDRESS_FLAG_ATOMIC64     UCS_BIT(31) /* 64bit atomic operations */

#define UCP_ADDRESS_FLAG_LAST         0x80u  /* Last address in the list */
#define UCP_ADDRESS_FLAG_HAS_EP_ADDR  0x40u  /* For iface address:
                                                Indicates that ep addr is packed
                                                right after iface addr */
#define UCP_ADDRESS_FLAG_HAVE_PATHS   0x40u  /* For device address:
                                                Indicates that number of paths on the
                                                device is packed right after device
                                                address, otherwise number of paths
                                                defaults to 1. */
#define UCP_ADDRESS_FLAG_LEN_MASK     (UCS_MASK(8) ^ \
                                        (UCP_ADDRESS_FLAG_HAS_EP_ADDR | \
                                         UCP_ADDRESS_FLAG_HAVE_PATHS  | \
                                         UCP_ADDRESS_FLAG_LAST))

#define UCP_ADDRESS_FLAG_MD_EMPTY_DEV 0x80u  /* Device without TL addresses */
#define UCP_ADDRESS_FLAG_MD_ALLOC     0x40u  /* MD can register  */
#define UCP_ADDRESS_FLAG_MD_REG       0x20u  /* MD can allocate */
#define UCP_ADDRESS_FLAG_MD_MASK      (UCS_MASK(8) ^ \
                                        (UCP_ADDRESS_FLAG_MD_EMPTY_DEV | \
                                         UCP_ADDRESS_FLAG_MD_ALLOC | \
                                         UCP_ADDRESS_FLAG_MD_REG))

#define UCP_ADDRESS_HEADER_VERSION_MASK     UCS_MASK(4) /* Version - 4 bits */
#define UCP_ADDRESS_HEADER_FLAG_DEBUG_INFO  UCS_BIT(4)  /* Address has debug info */

/* Enumeration of UCP address versions.
 * Every release which changes the address binary format must bump this number.
 */
enum {
    UCP_ADDRESS_VERSION_V1      = 0,
    UCP_ADDRESS_VERSION_LAST,
    UCP_ADDRESS_VERSION_CURRENT = UCP_ADDRESS_VERSION_LAST - 1
};


static size_t ucp_address_iface_attr_size(ucp_worker_t *worker,
                                          uint64_t flags)
{
    return ucp_worker_is_unified_mode(worker) ?
           sizeof(ucp_address_unified_iface_attr_t) :
           (sizeof(ucp_address_packed_iface_attr_t) +
            ((flags & UCP_ADDRESS_PACK_FLAG_TL_RSC_IDX) ?
             sizeof(uint8_t) : 0));
}

static uint64_t ucp_worker_iface_can_connect(uct_iface_attr_t *attrs)
{
    return attrs->cap.flags &
           (UCT_IFACE_FLAG_CONNECT_TO_IFACE | UCT_IFACE_FLAG_CONNECT_TO_EP);
}

/* Pack a string and return a pointer to storage right after the string */
static void* ucp_address_pack_worker_name(ucp_worker_h worker, void *dest)
{
    const char *s;
    size_t length;

    s      = ucp_worker_get_name(worker);
    length = strlen(s);
    ucs_assert(length <= UINT8_MAX);
    *(uint8_t*)dest = length;
    memcpy(UCS_PTR_TYPE_OFFSET(dest, uint8_t), s, length);
    return UCS_PTR_BYTE_OFFSET(UCS_PTR_TYPE_OFFSET(dest, uint8_t), length);
}

/* Unpack a string and return pointer to next storage byte */
static const void*
ucp_address_unpack_worker_name(const void *src, char *s)
{
    size_t length, avail;

    length   = *(const uint8_t*)src;
    avail    = ucs_min(length, UCP_WORKER_NAME_MAX - 1);
    memcpy(s, UCS_PTR_TYPE_OFFSET(src, uint8_t), avail);
    s[avail] = '\0';
    return UCS_PTR_TYPE_OFFSET(UCS_PTR_BYTE_OFFSET(src, length), uint8_t);
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
            !strcmp(tl_rsc[rsc_index].tl_rsc.dev_name,
                    tl_rsc[dev->rsc_index].tl_rsc.dev_name)) {
            goto out;
        }
    }

    dev = &devices[(*num_devices_p)++];
    memset(dev, 0, sizeof(*dev));
out:
    return dev;
}

static ucs_status_t
ucp_address_gather_devices(ucp_worker_h worker, ucp_ep_h ep, uint64_t tl_bitmap,
                           uint64_t flags, ucp_address_packed_device_t **devices_p,
                           ucp_rsc_index_t *num_devices_p)
{
    ucp_context_h context = worker->context;
    ucp_address_packed_device_t *dev, *devices;
    uct_iface_attr_t *iface_attr;
    ucp_rsc_index_t num_devices;
    ucp_rsc_index_t rsc_index;
    ucp_lane_index_t lane;

    devices = ucs_calloc(context->num_tls, sizeof(*devices), "packed_devices");
    if (devices == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    num_devices = 0;
    tl_bitmap  &= context->tl_bitmap;
    ucs_for_each_bit(rsc_index, tl_bitmap) {
        iface_attr = ucp_worker_iface_get_attr(worker, rsc_index);
        if (!ucp_worker_iface_can_connect(iface_attr)) {
            continue;
        }

        dev = ucp_address_get_device(context, rsc_index, devices, &num_devices);

        if (flags & UCP_ADDRESS_PACK_FLAG_EP_ADDR) {
            ucs_assert(ep != NULL);
            /* Each lane which matches the resource index adds an ep address
             * entry. The length and flags is packed in non-unified mode only.
             */
            ucs_for_each_bit(lane, ucp_ep_config(ep)->p2p_lanes) {
                if (ucp_ep_get_rsc_index(ep, lane) == rsc_index) {
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
            dev->tl_addrs_size += !ucp_worker_is_unified_mode(worker); /* if addr length */
            dev->tl_addrs_size += ucp_address_iface_attr_size(worker, flags);
        } else {
            dev->tl_addrs_size += 1; /* 0-value for valid unpacking */
        }

        if (flags & UCP_ADDRESS_PACK_FLAG_DEVICE_ADDR) {
            dev->dev_addr_len = iface_attr->device_addr_len;
        } else {
            dev->dev_addr_len = 0;
        }

        if (iface_attr->dev_num_paths > UINT8_MAX) {
            ucs_error("only up to %d paths are supported by address pack (got: %u)",
                      UINT8_MAX, iface_attr->dev_num_paths);
            ucs_free(devices);
            return UCS_ERR_UNSUPPORTED;
        }

        dev->rsc_index  = rsc_index;
        dev->tl_bitmap |= UCS_BIT(rsc_index);
        dev->num_paths  = iface_attr->dev_num_paths;
    }

    *devices_p     = devices;
    *num_devices_p = num_devices;
    return UCS_OK;
}

static size_t ucp_address_packed_size(ucp_worker_h worker,
                                      const ucp_address_packed_device_t *devices,
                                      ucp_rsc_index_t num_devices,
                                      uint64_t pack_flags)
{
    size_t size = 0;
    const ucp_address_packed_device_t *dev;

    /* header: version and flags */
    size += 1;

    if (pack_flags & UCP_ADDRESS_PACK_FLAG_WORKER_UUID) {
        size += sizeof(uint64_t);
    }

    if ((worker->context->config.ext.address_debug_info) &&
        (pack_flags & UCP_ADDRESS_PACK_FLAG_WORKER_NAME)) {
        size += strlen(ucp_worker_get_name(worker)) + 1;
    }

    if (num_devices == 0) {
        size += 1;                      /* NULL md_index */
    } else {
        for (dev = devices; dev < (devices + num_devices); ++dev) {
            size += 1;                  /* device md_index */
            size += 1;                  /* device address length */
            if (pack_flags & UCP_ADDRESS_PACK_FLAG_DEVICE_ADDR) {
                size += dev->dev_addr_len;  /* device address */
            }
            if (dev->num_paths > 1) {
                size += 1;                  /* number of paths */
            }
            size += dev->tl_addrs_size; /* transport addresses */
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

static int ucp_address_pack_iface_attr(ucp_worker_h worker, void *ptr,
                                       ucp_rsc_index_t rsc_index,
                                       const uct_iface_attr_t *iface_attr,
                                       unsigned pack_flags,
                                       int enable_atomics)
{
    int packed_len;
    ucp_address_packed_iface_attr_t  *packed;
    ucp_address_unified_iface_attr_t *unified;

    /* check if at least one of bandwidth values is 0 */
    if ((iface_attr->bandwidth.dedicated * iface_attr->bandwidth.shared) != 0) {
        ucs_error("Incorrect bandwidth value: one of bandwidth dedicated/shared must be zero");
        return -1;
    }


    if (ucp_worker_is_unified_mode(worker)) {
        /* In unified mode all workers have the same transports and tl bitmap.
         * Just send rsc index, so the remote peer could fetch iface attributes
         * from its local iface. Also send latency overhead, because it
         * depends on device NUMA locality. */
        unified            = ptr;
        unified->rsc_index = rsc_index;
        unified->lat_ovh   = enable_atomics ? -iface_attr->latency.c :
                                               iface_attr->latency.c;

        return sizeof(*unified);
    }

    packed                 = ptr;
    packed->prio_cap_flags = (uint8_t)iface_attr->priority;
    packed->overhead       = iface_attr->overhead;
    packed->bandwidth      = iface_attr->bandwidth.dedicated - iface_attr->bandwidth.shared;
    packed->lat_ovh        = iface_attr->latency.c;

    ucs_assert((ucs_popcount(UCP_ADDRESS_IFACE_FLAGS) +
                ucs_popcount(UCP_ADDRESS_IFACE_EVENT_FLAGS)) <= 22);

    /* Keep only the bits defined by UCP_ADDRESS_IFACE_FLAGS
     * to shrink address. */
    packed->prio_cap_flags |=
        ucp_address_pack_flags(iface_attr->cap.flags,
                               UCP_ADDRESS_IFACE_FLAGS, 8);

    /* Keep only the bits defined by UCP_ADDRESS_IFACE_EVENT_FLAGS
     * to shrink address. */
    packed->prio_cap_flags |=
        ucp_address_pack_flags(iface_attr->cap.event_flags,
                               UCP_ADDRESS_IFACE_EVENT_FLAGS,
                               8 + ucs_popcount(UCP_ADDRESS_IFACE_FLAGS));

    if (enable_atomics) {
        if (ucs_test_all_flags(iface_attr->cap.atomic32.op_flags, UCP_ATOMIC_OP_MASK) &&
            ucs_test_all_flags(iface_attr->cap.atomic32.fop_flags, UCP_ATOMIC_FOP_MASK)) {
            packed->prio_cap_flags |= UCP_ADDRESS_FLAG_ATOMIC32;
        }
        if (ucs_test_all_flags(iface_attr->cap.atomic64.op_flags, UCP_ATOMIC_OP_MASK) &&
            ucs_test_all_flags(iface_attr->cap.atomic64.fop_flags, UCP_ATOMIC_FOP_MASK)) {
            packed->prio_cap_flags |= UCP_ADDRESS_FLAG_ATOMIC64;
        }
    }

    packed_len = sizeof(*packed);

    if (pack_flags & UCP_ADDRESS_PACK_FLAG_TL_RSC_IDX) {
        ptr             = packed + 1;
        *(uint8_t*)ptr  = rsc_index;
        packed_len     += sizeof(uint8_t);
    }

    return packed_len;
}

static ucs_status_t
ucp_address_unpack_iface_attr(ucp_worker_t *worker,
                              ucp_address_iface_attr_t *iface_attr,
                              const void *ptr, unsigned unpack_flags,
                              size_t *size_p)
{
    const ucp_address_packed_iface_attr_t *packed;
    const ucp_address_unified_iface_attr_t *unified;
    ucp_worker_iface_t *wiface;
    ucp_rsc_index_t rsc_idx;

    if (ucp_worker_is_unified_mode(worker)) {
        /* Address contains resources index and iface latency overhead
         * (not all iface attrs). */
        unified               = ptr;
        rsc_idx               = unified->rsc_index & UCP_ADDRESS_FLAG_LEN_MASK;
        iface_attr->lat_ovh   = fabs(unified->lat_ovh);
        if (!(worker->context->tl_bitmap & UCS_BIT(rsc_idx))) {
            if (!(unpack_flags & UCP_ADDRESS_PACK_FLAG_NO_TRACE)) {
                ucs_error("failed to unpack address, resource[%d] is not valid",
                          rsc_idx);
            }
            return UCS_ERR_INVALID_ADDR;
        }

        /* Just take the rest of iface attrs from the local resource. */
        wiface                    = ucp_worker_iface(worker, rsc_idx);
        iface_attr->cap_flags     = wiface->attr.cap.flags;
        iface_attr->event_flags   = wiface->attr.cap.event_flags;
        iface_attr->priority      = wiface->attr.priority;
        iface_attr->overhead      = wiface->attr.overhead;
        iface_attr->bandwidth     = wiface->attr.bandwidth;
        iface_attr->dst_rsc_index = rsc_idx;
        if (signbit(unified->lat_ovh)) {
            iface_attr->atomic.atomic32.op_flags  = wiface->attr.cap.atomic32.op_flags;
            iface_attr->atomic.atomic32.fop_flags = wiface->attr.cap.atomic32.fop_flags;
            iface_attr->atomic.atomic64.op_flags  = wiface->attr.cap.atomic64.op_flags;
            iface_attr->atomic.atomic64.fop_flags = wiface->attr.cap.atomic64.fop_flags;
        }

        *size_p = sizeof(*unified);
        return UCS_OK;
    }

    packed                          = ptr;
    iface_attr->priority            = packed->prio_cap_flags & UCS_MASK(8);
    iface_attr->overhead            = packed->overhead;
    iface_attr->bandwidth.dedicated = ucs_max(0.0, packed->bandwidth);
    iface_attr->bandwidth.shared    = ucs_max(0.0, -packed->bandwidth);
    iface_attr->lat_ovh             = packed->lat_ovh;

    /* Unpack iface flags */
    iface_attr->cap_flags =
        ucp_address_unpack_flags(packed->prio_cap_flags,
                                 UCP_ADDRESS_IFACE_FLAGS, 8);

    /* Unpack iface event flags */
    iface_attr->event_flags =
        ucp_address_unpack_flags(packed->prio_cap_flags,
                                 UCP_ADDRESS_IFACE_EVENT_FLAGS,
                                 8 + ucs_popcount(UCP_ADDRESS_IFACE_FLAGS));

    /* Unpack iface 32-bit atomic operations */
    if (packed->prio_cap_flags & UCP_ADDRESS_FLAG_ATOMIC32) {
        iface_attr->atomic.atomic32.op_flags  |= UCP_ATOMIC_OP_MASK;
        iface_attr->atomic.atomic32.fop_flags |= UCP_ATOMIC_FOP_MASK;
    }
    
    /* Unpack iface 64-bit atomic operations */
    if (packed->prio_cap_flags & UCP_ADDRESS_FLAG_ATOMIC64) {
        iface_attr->atomic.atomic64.op_flags  |= UCP_ATOMIC_OP_MASK;
        iface_attr->atomic.atomic64.fop_flags |= UCP_ATOMIC_FOP_MASK;
    }

    *size_p = sizeof(*packed);

    if (unpack_flags & UCP_ADDRESS_PACK_FLAG_TL_RSC_IDX) {
        ptr                       = packed + 1;
        iface_attr->dst_rsc_index = *(uint8_t*)ptr;
        *size_p                  += sizeof(uint8_t);
    } else {
        iface_attr->dst_rsc_index = UCP_NULL_RESOURCE;
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

static void*
ucp_address_pack_length(ucp_worker_h worker, void *ptr, size_t addr_length)
{
    if (ucp_worker_is_unified_mode(worker)) {
        return ptr;
    }

    ucs_assertv(addr_length <= UCP_ADDRESS_FLAG_LEN_MASK, "addr_length=%zu",
                addr_length);
    *(uint8_t*)ptr = addr_length;

    return UCS_PTR_TYPE_OFFSET(ptr, uint8_t);
}

static const void*
ucp_address_unpack_length(ucp_worker_h worker, const void* flags_ptr, const void *ptr,
                          size_t *addr_length, int is_ep_addr, int *is_last_iface)
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
        rsc_index = unified->rsc_index & UCP_ADDRESS_FLAG_LEN_MASK;
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

    if (!is_ep_addr) {
        *is_last_iface = *(uint8_t*)ptr & UCP_ADDRESS_FLAG_LAST;
    }

    *addr_length = *(uint8_t*)ptr & UCP_ADDRESS_FLAG_LEN_MASK;

    return UCS_PTR_TYPE_OFFSET(ptr, uint8_t);
}

static ucs_status_t ucp_address_do_pack(ucp_worker_h worker, ucp_ep_h ep,
                                        void *buffer, size_t size,
                                        uint64_t tl_bitmap, unsigned pack_flags,
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
    uint64_t dev_tl_bitmap;
    unsigned num_ep_addrs;
    ucs_status_t status;
    size_t iface_addr_len;
    size_t ep_addr_len;
    uint64_t md_flags;
    uint8_t *ep_lane_ptr;
    void *flags_ptr;
    unsigned addr_index;
    int attr_len;
    void *ptr;
    int enable_amo;

    ptr               = buffer;
    addr_index        = 0;
    address_header_p  = ptr;
    *address_header_p = UCP_ADDRESS_VERSION_CURRENT;
    ptr               = UCS_PTR_TYPE_OFFSET(ptr, uint8_t);

    if (pack_flags & UCP_ADDRESS_PACK_FLAG_WORKER_UUID) {
        *(uint64_t*)ptr = worker->uuid;
        ptr             = UCS_PTR_TYPE_OFFSET(ptr, worker->uuid);
    }

    if (worker->context->config.ext.address_debug_info) {
        /* Add debug information to the packed address, and set the corresponding
         * flag in address header.
         */
        *address_header_p |= UCP_ADDRESS_HEADER_FLAG_DEBUG_INFO;

        if (pack_flags & UCP_ADDRESS_PACK_FLAG_WORKER_NAME) {
            ptr            = ucp_address_pack_worker_name(worker, ptr);
        }
    }

    if (num_devices == 0) {
        *((uint8_t*)ptr) = UCP_NULL_RESOURCE;
        ptr = UCS_PTR_TYPE_OFFSET(ptr, UCP_NULL_RESOURCE);
        goto out;
    }

    for (dev = devices; dev < (devices + num_devices); ++dev) {

        dev_tl_bitmap = context->tl_bitmap & dev->tl_bitmap;

        /* MD index */
        md_index       = context->tl_rscs[dev->rsc_index].md_index;
        md_flags       = context->tl_mds[md_index].attr.cap.flags & md_flags_pack_mask;
        ucs_assertv_always(md_index <= UCP_ADDRESS_FLAG_MD_MASK,
                           "md_index=%d", md_index);

        *(uint8_t*)ptr = md_index |
                         ((dev_tl_bitmap == 0)           ? UCP_ADDRESS_FLAG_MD_EMPTY_DEV : 0) |
                         ((md_flags & UCT_MD_FLAG_ALLOC) ? UCP_ADDRESS_FLAG_MD_ALLOC     : 0) |
                         ((md_flags & UCT_MD_FLAG_REG)   ? UCP_ADDRESS_FLAG_MD_REG       : 0);
        ptr = UCS_PTR_TYPE_OFFSET(ptr, md_index);

        /* Device address length */
        *(uint8_t*)ptr = (dev == (devices + num_devices - 1)) ?
                         UCP_ADDRESS_FLAG_LAST : 0;
        if (pack_flags & UCP_ADDRESS_PACK_FLAG_DEVICE_ADDR) {
            ucs_assert(dev->dev_addr_len <= UCP_ADDRESS_FLAG_LEN_MASK);
            *(uint8_t*)ptr |= dev->dev_addr_len;
        }

        /* Device number of paths flag and value */
        ucs_assert(dev->num_paths >= 1);
        ucs_assert(dev->num_paths <= UINT8_MAX);

        if (dev->num_paths > 1) {
            *(uint8_t*)ptr |= UCP_ADDRESS_FLAG_HAVE_PATHS;
            ptr = UCS_PTR_TYPE_OFFSET(ptr, uint8_t);
            *(uint8_t*)ptr = dev->num_paths;
        }
        ptr = UCS_PTR_TYPE_OFFSET(ptr, uint8_t);

        /* Device address */
        if (pack_flags & UCP_ADDRESS_PACK_FLAG_DEVICE_ADDR) {
            wiface = ucp_worker_iface(worker, dev->rsc_index);
            status = uct_iface_get_device_address(wiface->iface,
                                                  (uct_device_addr_t*)ptr);
            if (status != UCS_OK) {
                return status;
            }

            ucp_address_memcheck(context, ptr, dev->dev_addr_len, dev->rsc_index);
            ptr = UCS_PTR_BYTE_OFFSET(ptr, dev->dev_addr_len);
        }

        flags_ptr = NULL;
        ucs_for_each_bit(rsc_index, dev_tl_bitmap) {

            wiface     = ucp_worker_iface(worker, rsc_index);
            iface_attr = &wiface->attr;

            if (!ucp_worker_iface_can_connect(iface_attr)) {
                return UCS_ERR_INVALID_ADDR;
            }

            /* Transport name checksum */
            *(uint16_t*)ptr = context->tl_rscs[rsc_index].tl_name_csum;
            ptr = UCS_PTR_TYPE_OFFSET(ptr,
                                      context->tl_rscs[rsc_index].tl_name_csum);

            /* Transport information */
            enable_amo = worker->atomic_tls & UCS_BIT(rsc_index);
            attr_len   = ucp_address_pack_iface_attr(worker, ptr, rsc_index,
                                                     iface_attr, pack_flags,
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
            ptr = ucp_address_pack_length(worker, ptr, iface_addr_len);
            if (pack_flags & UCP_ADDRESS_PACK_FLAG_IFACE_ADDR) {
                status = uct_iface_get_address(wiface->iface,
                                               (uct_iface_addr_t*)ptr);
                if (status != UCS_OK) {
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
                    if (ucp_ep_get_rsc_index(ep, lane) != rsc_index) {
                        continue;
                    }

                    /* pack ep address length and save pointer to flags */
                    ptr = ucp_address_pack_length(worker, ptr, ep_addr_len);

                    /* pack ep address */
                    status = uct_ep_get_address(ep->uct_eps[lane], ptr);
                    if (status != UCS_OK) {
                        return status;
                    }

                    ucp_address_memcheck(context, ptr, ep_addr_len, rsc_index);
                    ptr = UCS_PTR_BYTE_OFFSET(ptr, ep_addr_len);

                    /* pack ep lane index, and save the pointer for lane index
                     * of last ep in 'ep_last_ptr' to set UCP_ADDRESS_FLAG_LAST.
                     */
                    remote_lane  = (lanes2remote == NULL) ? lane :
                                   lanes2remote[lane];
                    ucs_assertv(remote_lane <= UCP_ADDRESS_FLAG_LEN_MASK,
                                "remote_lane=%d", remote_lane);
                    ep_lane_ptr  = ptr;
                    *ep_lane_ptr = remote_lane;
                    ptr          = UCS_PTR_TYPE_OFFSET(ptr, uint8_t);

                    if (!(pack_flags & UCP_ADDRESS_PACK_FLAG_NO_TRACE)) {
                        ucs_trace("pack addr[%d].ep_addr[%d] : len %zu lane %d->%d",
                                  addr_index, num_ep_addrs, ep_addr_len, lane,
                                  remote_lane);
                    }

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

            if (!(pack_flags & UCP_ADDRESS_PACK_FLAG_NO_TRACE)) {
               ucs_trace("pack addr[%d] : "UCT_TL_RESOURCE_DESC_FMT" "
                          "eps %u md_flags 0x%"PRIx64" tl_flags 0x%"PRIx64
                          " bw %e + %e/n ovh %e lat_ovh %e dev_priority %d a32 "
                          "0x%"PRIx64"/0x%"PRIx64" a64 0x%"PRIx64"/0x%"PRIx64,
                          addr_index,
                          UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[rsc_index].tl_rsc),
                          num_ep_addrs, md_flags, iface_attr->cap.flags,
                          iface_attr->bandwidth.dedicated,
                          iface_attr->bandwidth.shared,
                          iface_attr->overhead,
                          iface_attr->latency.c,
                          iface_attr->priority,
                          iface_attr->cap.atomic32.op_flags,
                          iface_attr->cap.atomic32.fop_flags,
                          iface_attr->cap.atomic64.op_flags,
                          iface_attr->cap.atomic64.fop_flags);
            }

            ++addr_index;
            ucs_assert(addr_index <= UCP_MAX_RESOURCES);
        }

        /* flags_ptr is a valid pointer to the flags set to the last entry
         * during the above loop So, set the LAST flag for the flags_ptr
         * from the last iteration */
        if (flags_ptr != NULL) {
            ucs_assert(dev_tl_bitmap != 0);
            *(uint8_t*)flags_ptr |= UCP_ADDRESS_FLAG_LAST;
        } else {
            /* cppcheck-suppress internalAstError */
            ucs_assert(dev_tl_bitmap == 0);
        }
    }

out:
    ucs_assertv(UCS_PTR_BYTE_OFFSET(buffer, size) == ptr,
                "buffer=%p size=%zu ptr=%p ptr-buffer=%zd",
                buffer, size, ptr, UCS_PTR_BYTE_DIFF(buffer, ptr));
    return UCS_OK;
}

ucs_status_t ucp_address_pack(ucp_worker_h worker, ucp_ep_h ep,
                              uint64_t tl_bitmap, unsigned pack_flags,
                              const ucp_lane_index_t *lanes2remote,
                              size_t *size_p, void **buffer_p)
{
    ucp_address_packed_device_t *devices;
    ucp_rsc_index_t num_devices;
    ucs_status_t status;
    void *buffer;
    size_t size;

    if (ep == NULL) {
        pack_flags &= ~UCP_ADDRESS_PACK_FLAG_EP_ADDR;
    }

    /* Collect all devices we want to pack */
    status = ucp_address_gather_devices(worker, ep, tl_bitmap, pack_flags,
                                        &devices, &num_devices);
    if (status != UCS_OK) {
        goto out;
    }

    /* Calculate packed size */
    size = ucp_address_packed_size(worker, devices, num_devices, pack_flags);

    /* Allocate address */
    buffer = ucs_malloc(size, "ucp_address");
    if (buffer == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out_free_devices;
    }

    memset(buffer, 0, size);

    /* Pack the address */
    status = ucp_address_do_pack(worker, ep, buffer, size, tl_bitmap, pack_flags,
                                 lanes2remote, devices, num_devices);
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

ucs_status_t ucp_address_unpack(ucp_worker_t *worker, const void *buffer,
                                unsigned unpack_flags,
                                ucp_unpacked_address_t *unpacked_address)
{
    ucp_address_entry_t *address_list, *address;
    uint8_t address_header, address_version;
    ucp_address_entry_ep_addr_t *ep_addr;
    int last_dev, last_tl, last_ep_addr;
    const uct_device_addr_t *dev_addr;
    ucp_rsc_index_t dev_index;
    ucp_md_index_t md_index;
    unsigned dev_num_paths;
    ucs_status_t status;
    int empty_dev;
    uint64_t md_flags;
    size_t dev_addr_len;
    size_t iface_addr_len;
    size_t ep_addr_len;
    size_t attr_len;
    uint8_t md_byte;
    const void *ptr;
    const void *flags_ptr;

    /* Initialize the unpacked address to empty */
    unpacked_address->address_count = 0;
    unpacked_address->address_list  = NULL;

    ptr                             = buffer;
    address_header                  = *(const uint8_t *)ptr;
    ptr                             = UCS_PTR_TYPE_OFFSET(ptr, uint8_t);

    /* Check address version */
    address_version = address_header & UCP_ADDRESS_HEADER_VERSION_MASK;
    if (address_version != UCP_ADDRESS_VERSION_CURRENT) {
        ucs_error("address version mismatch: expected %u, actual %u",
                  UCP_ADDRESS_VERSION_CURRENT, address_version);
        return UCS_ERR_UNREACHABLE;
    }

    if (unpack_flags & UCP_ADDRESS_PACK_FLAG_WORKER_UUID) {
        unpacked_address->uuid = *(uint64_t*)ptr;
        ptr = UCS_PTR_TYPE_OFFSET(ptr, unpacked_address->uuid);
    } else {
        unpacked_address->uuid = 0;
    }

    if ((address_header & UCP_ADDRESS_HEADER_FLAG_DEBUG_INFO) &&
        (unpack_flags & UCP_ADDRESS_PACK_FLAG_WORKER_NAME)) {
        ptr = ucp_address_unpack_worker_name(ptr, unpacked_address->name);
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
        /* md_index */
        md_byte      = (*(uint8_t*)ptr);
        md_index     = md_byte & UCP_ADDRESS_FLAG_MD_MASK;
        md_flags     = (md_byte & UCP_ADDRESS_FLAG_MD_ALLOC) ? UCT_MD_FLAG_ALLOC : 0;
        md_flags    |= (md_byte & UCP_ADDRESS_FLAG_MD_REG)   ? UCT_MD_FLAG_REG   : 0;
        empty_dev    = md_byte & UCP_ADDRESS_FLAG_MD_EMPTY_DEV;
        ptr          = UCS_PTR_TYPE_OFFSET(ptr, md_byte);

        /* device address length */
        dev_addr_len = (*(uint8_t*)ptr) & UCP_ADDRESS_FLAG_LEN_MASK;
        last_dev     = (*(uint8_t*)ptr) & UCP_ADDRESS_FLAG_LAST;
        if ((*(uint8_t*)ptr) & UCP_ADDRESS_FLAG_HAVE_PATHS) {
            ptr           = UCS_PTR_TYPE_OFFSET(ptr, uint8_t);
            dev_num_paths = *(uint8_t*)ptr;
        } else {
            dev_num_paths = 1;
        }
        ptr      = UCS_PTR_TYPE_OFFSET(ptr, uint8_t);

        dev_addr = ptr;
        ptr      = UCS_PTR_BYTE_OFFSET(ptr, dev_addr_len);

        last_tl = empty_dev;
        while (!last_tl) {
            if (address >= &address_list[UCP_MAX_RESOURCES]) {
                if (!(unpack_flags & UCP_ADDRESS_PACK_FLAG_NO_TRACE)) {
                    ucs_error("failed to parse address: number of addresses"
                              "exceeds %d", UCP_MAX_RESOURCES);
                }
                goto err_free;
            }

            /* tl_name_csum */
            address->tl_name_csum = *(uint16_t*)ptr;
            ptr = UCS_PTR_TYPE_OFFSET(ptr, address->tl_name_csum);

            address->dev_addr      = (dev_addr_len > 0) ? dev_addr : NULL;
            address->md_index      = md_index;
            address->dev_index     = dev_index;
            address->md_flags      = md_flags;
            address->dev_num_paths = dev_num_paths;

            status = ucp_address_unpack_iface_attr(worker, &address->iface_attr,
                                                   ptr, unpack_flags, &attr_len);
            if (status != UCS_OK) {
                goto err_free;
            }

            flags_ptr = ucp_address_iface_flags_ptr(worker, (void*)ptr, attr_len);
            ptr       = UCS_PTR_BYTE_OFFSET(ptr, attr_len);
            ptr       = ucp_address_unpack_length(worker, flags_ptr, ptr,
                                                  &iface_addr_len, 0, &last_tl);
            address->iface_addr   = (iface_addr_len > 0) ? ptr : NULL;
            address->num_ep_addrs = 0;
            ptr                   = UCS_PTR_BYTE_OFFSET(ptr, iface_addr_len);

            last_ep_addr = !(*(uint8_t*)flags_ptr & UCP_ADDRESS_FLAG_HAS_EP_ADDR);
            while (!last_ep_addr) {
                if (address->num_ep_addrs >= UCP_MAX_LANES) {
                    if (!(unpack_flags & UCP_ADDRESS_PACK_FLAG_NO_TRACE)) {
                        ucs_error("failed to parse address: number of ep addresses"
                                  " exceeds %d", UCP_MAX_LANES);
                    }
                    goto err_free;
                }

                ep_addr       = &address->ep_addrs[address->num_ep_addrs++];
                ptr           = ucp_address_unpack_length(worker, flags_ptr, ptr,
                                                          &ep_addr_len, 1, NULL);
                ep_addr->addr = ptr;
                ptr           = UCS_PTR_BYTE_OFFSET(ptr, ep_addr_len);

                ep_addr->lane = *(uint8_t*)ptr & UCP_ADDRESS_FLAG_LEN_MASK;
                last_ep_addr  = *(uint8_t*)ptr & UCP_ADDRESS_FLAG_LAST;

                if (!(unpack_flags & UCP_ADDRESS_PACK_FLAG_NO_TRACE)) {
                    ucs_trace("unpack addr[%d].ep_addr[%d] : len %zu lane %d",
                              (int)(address - address_list),
                              (int)(ep_addr - address->ep_addrs),
                              ep_addr_len, ep_addr->lane);
                }

                ptr           = UCS_PTR_TYPE_OFFSET(ptr, uint8_t);
            }

            if (!(unpack_flags & UCP_ADDRESS_PACK_FLAG_NO_TRACE)) {
                ucs_trace("unpack addr[%d] : eps %u md_flags 0x%"PRIx64
                          " tl_iface_flags 0x%"PRIx64" tl_event_flags 0x%"PRIx64
                          " bw %e + %e/n ovh %e lat_ovh %e dev_priority %d a32 "
                          "0x%"PRIx64"/0x%"PRIx64" a64 0x%"PRIx64"/0x%"PRIx64,
                          (int)(address - address_list), address->num_ep_addrs,
                          address->md_flags, address->iface_attr.cap_flags,
                          address->iface_attr.event_flags,
                          address->iface_attr.bandwidth.dedicated,
                          address->iface_attr.bandwidth.shared,
                          address->iface_attr.overhead,
                          address->iface_attr.lat_ovh,
                          address->iface_attr.priority,
                          address->iface_attr.atomic.atomic32.op_flags,
                          address->iface_attr.atomic.atomic32.fop_flags,
                          address->iface_attr.atomic.atomic64.op_flags,
                          address->iface_attr.atomic.atomic64.fop_flags);
            }

            ++address;
        }

        ++dev_index;
    } while (!last_dev);

    unpacked_address->address_count = address - address_list;
    unpacked_address->address_list  = address_list;
    return UCS_OK;

err_free:
    ucs_free(address_list);
    return UCS_ERR_INVALID_PARAM;
}
