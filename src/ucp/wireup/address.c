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
 * [ uuid(64bit) | worker_name(string) ]
 * [ device1_md_index | device1_address(var) ]
 *    [ tl1_name_csum(string) | tl1_info | tl1_address(var) ]
 *    [ tl2_name_csum(string) | tl2_info | tl2_address(var) ]
 *    ...
 * [ device2_md_index | device2_address(var) ]
 *    ...
 *
 *   * worker_name is packed if ENABLE_DEBUG is set.
 *   * In unified mode tl_info contains just rsc_index and iface latency overhead.
 *     For last address in the tl address list, it will have LAST flag set.
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
    size_t           tl_addrs_size;
} ucp_address_packed_device_t;


typedef struct {
    float            overhead;
    float            bandwidth;
    float            lat_ovh;
    uint32_t         prio_cap_flags; /* 8 lsb: prio, 22 msb: cap flags, 2 hsb: amo */
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


#define UCT_ADDRESS_FLAG_ATOMIC32     UCS_BIT(30) /* 32bit atomic operations */
#define UCT_ADDRESS_FLAG_ATOMIC64     UCS_BIT(31) /* 64bit atomic operations */

#define UCP_ADDRESS_FLAG_LAST         0x80   /* Last address in the list */
#define UCP_ADDRESS_FLAG_HAVE_EP_ADDR 0x40   /* Indicates that ep addr is packed
                                                right after iface addr */
#define UCP_ADDRESS_FLAG_LEN_MASK     ~(UCP_ADDRESS_FLAG_HAVE_EP_ADDR | \
                                        UCP_ADDRESS_FLAG_LAST)

#define UCP_ADDRESS_FLAG_EMPTY        0x80   /* Device without TL addresses */
#define UCP_ADDRESS_FLAG_MD_ALLOC     0x40   /* MD can register  */
#define UCP_ADDRESS_FLAG_MD_REG       0x20   /* MD can allocate */
#define UCP_ADDRESS_FLAG_MD_MASK      ~(UCP_ADDRESS_FLAG_EMPTY | \
                                        UCP_ADDRESS_FLAG_MD_ALLOC | \
                                        UCP_ADDRESS_FLAG_MD_REG)

static size_t ucp_address_worker_name_size(ucp_worker_h worker, uint64_t flags)
{
#if ENABLE_DEBUG_DATA
    return (flags & UCP_ADDRESS_PACK_FLAG_WORKER_NAME) ?
           strlen(ucp_worker_get_name(worker)) + 1 : 0;
#else
    return 0;
#endif
}

static size_t ucp_address_iface_attr_size(ucp_worker_t *worker)
{
    return ucp_worker_unified_mode(worker) ?
           sizeof(ucp_address_unified_iface_attr_t) :
           sizeof(ucp_address_packed_iface_attr_t);
}

static uint64_t ucp_worker_iface_can_connect(uct_iface_attr_t *attrs)
{
    return attrs->cap.flags &
           (UCT_IFACE_FLAG_CONNECT_TO_IFACE | UCT_IFACE_FLAG_CONNECT_TO_EP);
}

/* Pack a string and return a pointer to storage right after the string */
static void* ucp_address_pack_worker_name(ucp_worker_h worker, void *dest,
                                          uint64_t flags)
{
#if ENABLE_DEBUG_DATA
    const char *s;
    size_t length;

    if (!(flags & UCP_ADDRESS_PACK_FLAG_WORKER_NAME)) {
        return dest;
    }

    s      = ucp_worker_get_name(worker);
    length = strlen(s);
    ucs_assert(length <= UINT8_MAX);
    *(uint8_t*)dest = length;
    memcpy(UCS_PTR_TYPE_OFFSET(dest, uint8_t), s, length);
    return UCS_PTR_BYTE_OFFSET(UCS_PTR_TYPE_OFFSET(dest, uint8_t), length);
#else
    return dest;
#endif
}

/* Unpack a string and return pointer to next storage byte */
static const void* ucp_address_unpack_worker_name(const void *src, char *s,
                                                  size_t max, uint64_t flags)
{
#if ENABLE_DEBUG_DATA
    size_t length, avail;

    if (!(flags & UCP_ADDRESS_PACK_FLAG_WORKER_NAME)) {
        s[0] = '\0';
        return src;
    }

    ucs_assert(max >= 1);
    length   = *(const uint8_t*)src;
    avail    = ucs_min(length, max - 1);
    memcpy(s, UCS_PTR_TYPE_OFFSET(src, uint8_t), avail);
    s[avail] = '\0';
    return UCS_PTR_TYPE_OFFSET(UCS_PTR_BYTE_OFFSET(src, length), uint8_t);
#else
    s[0] = '\0';
    return src;
#endif
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
    unsigned num_ep_addrs;

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

        if ((flags & UCP_ADDRESS_PACK_FLAG_EP_ADDR) &&
            ucp_worker_iface_is_tl_p2p(iface_attr)) {
            /* Each lane which matches the resource index adds an ep address
             * entry. The length and flags is packed in non-unified mode only.
             */
            ucs_assert(ep != NULL);
            num_ep_addrs = 0;
            for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
                if (ucp_ep_get_rsc_index(ep, lane) == rsc_index) {
                    dev->tl_addrs_size += !ucp_worker_unified_mode(worker);
                    dev->tl_addrs_size += iface_attr->ep_addr_len;
                    dev->tl_addrs_size += sizeof(uint8_t); /* lane index */
                }
            }
            if (ucp_worker_unified_mode(worker)) {
                ucs_assertv_always(
                    num_ep_addrs <= 1,
                    "unexpected multiple ep addresses in unified mode");
            }
        }

        dev->tl_addrs_size += sizeof(uint16_t); /* tl name checksum */

        if (flags & UCP_ADDRESS_PACK_FLAG_IFACE_ADDR) {
            /* iface address (its length will be packed in non-unified mode only) */
            dev->tl_addrs_size += iface_attr->iface_addr_len;
            dev->tl_addrs_size += !ucp_worker_unified_mode(worker); /* if addr length */
            dev->tl_addrs_size += ucp_address_iface_attr_size(worker);
        } else {
            dev->tl_addrs_size += 1; /* 0-value for valid unpacking */
        }

        if (flags & UCP_ADDRESS_PACK_FLAG_DEVICE_ADDR) {
            dev->dev_addr_len = iface_attr->device_addr_len;
        } else {
            dev->dev_addr_len = 0;
        }

        dev->rsc_index  = rsc_index;
        dev->tl_bitmap |= UCS_BIT(rsc_index);
    }

    *devices_p     = devices;
    *num_devices_p = num_devices;
    return UCS_OK;
}

static size_t ucp_address_packed_size(ucp_worker_h worker,
                                      const ucp_address_packed_device_t *devices,
                                      ucp_rsc_index_t num_devices,
                                      uint64_t flags)
{
    size_t size = 0;
    const ucp_address_packed_device_t *dev;

    if (flags & UCP_ADDRESS_PACK_FLAG_WORKER_UUID) {
        size += sizeof(uint64_t);
    }

    size += ucp_address_worker_name_size(worker, flags);

    if (num_devices == 0) {
        size += 1;                      /* NULL md_index */
    } else {
        for (dev = devices; dev < (devices + num_devices); ++dev) {
            size += 1;                  /* device md_index */
            size += 1;                  /* device address length */
            if (flags & UCP_ADDRESS_PACK_FLAG_DEVICE_ADDR) {
                size += dev->dev_addr_len;  /* device address */
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

static int ucp_address_pack_iface_attr(ucp_worker_h worker, void *ptr,
                                       ucp_rsc_index_t index,
                                       const uct_iface_attr_t *iface_attr,
                                       int enable_atomics)
{
    ucp_address_packed_iface_attr_t  *packed;
    ucp_address_unified_iface_attr_t *unified;
    uint32_t packed_flag;
    uint64_t cap_flags;
    uint64_t bit;

    /* check if at least one of bandwidth values is 0 */
    if ((iface_attr->bandwidth.dedicated * iface_attr->bandwidth.shared) != 0) {
        ucs_error("Incorrect bandwidth value: one of bandwidth dedicated/shared must be zero");
        return -1;
    }


    if (ucp_worker_unified_mode(worker)) {
        /* In unified mode all workers have the same transports and tl bitmap.
         * Just send rsc index, so the remote peer could fetch iface attributes
         * from its local iface. Also send latency overhead, because it
         * depends on device NUMA locality. */
        unified            = ptr;
        unified->rsc_index = index;
        unified->lat_ovh   = enable_atomics ? -iface_attr->latency.overhead :
                                               iface_attr->latency.overhead;

        return sizeof(*unified);
    }

    packed    = ptr;
    cap_flags = iface_attr->cap.flags;

    packed->prio_cap_flags = ((uint8_t)iface_attr->priority);
    packed->overhead       = iface_attr->overhead;
    packed->bandwidth      = iface_attr->bandwidth.dedicated - iface_attr->bandwidth.shared;
    packed->lat_ovh        = iface_attr->latency.overhead;

    /* Keep only the bits defined by UCP_ADDRESS_IFACE_FLAGS, to shrink address. */
    packed_flag = UCS_BIT(8);
    bit         = 1;
    while (UCP_ADDRESS_IFACE_FLAGS & ~(bit - 1)) {
        if (UCP_ADDRESS_IFACE_FLAGS & bit) {
            if (cap_flags & bit) {
                packed->prio_cap_flags |= packed_flag;
            }
            packed_flag <<= 1;
        }
        bit <<= 1;
    }

    if (enable_atomics) {
        if (ucs_test_all_flags(iface_attr->cap.atomic32.op_flags, UCP_ATOMIC_OP_MASK) &&
            ucs_test_all_flags(iface_attr->cap.atomic32.fop_flags, UCP_ATOMIC_FOP_MASK)) {
            packed->prio_cap_flags |= UCT_ADDRESS_FLAG_ATOMIC32;
        }
        if (ucs_test_all_flags(iface_attr->cap.atomic64.op_flags, UCP_ATOMIC_OP_MASK) &&
            ucs_test_all_flags(iface_attr->cap.atomic64.fop_flags, UCP_ATOMIC_FOP_MASK)) {
            packed->prio_cap_flags |= UCT_ADDRESS_FLAG_ATOMIC64;
        }
    }

    return sizeof(*packed);
}

static int
ucp_address_unpack_iface_attr(ucp_worker_t *worker,
                              ucp_address_iface_attr_t *iface_attr,
                              const void *ptr)
{
    const ucp_address_packed_iface_attr_t *packed;
    const ucp_address_unified_iface_attr_t *unified;
    ucp_worker_iface_t *wiface;
    uint32_t packed_flag;
    ucp_rsc_index_t rsc_idx;
    uint64_t bit;

    if (ucp_worker_unified_mode(worker)) {
        /* Address contains resources index and iface latency overhead
         * (not all iface attrs). */
        unified               = ptr;
        rsc_idx               = unified->rsc_index & UCP_ADDRESS_FLAG_LEN_MASK;
        iface_attr->lat_ovh   = fabs(unified->lat_ovh);
        wiface                = ucp_worker_iface(worker, rsc_idx);

        /* Just take the rest of iface attrs from the local resource. */
        iface_attr->cap_flags = wiface->attr.cap.flags;
        iface_attr->priority  = wiface->attr.priority;
        iface_attr->overhead  = wiface->attr.overhead;
        iface_attr->bandwidth = wiface->attr.bandwidth;
        if (signbit(unified->lat_ovh)) {
            iface_attr->atomic.atomic32.op_flags  = wiface->attr.cap.atomic32.op_flags;
            iface_attr->atomic.atomic32.fop_flags = wiface->attr.cap.atomic32.fop_flags;
            iface_attr->atomic.atomic64.op_flags  = wiface->attr.cap.atomic64.op_flags;
            iface_attr->atomic.atomic64.fop_flags = wiface->attr.cap.atomic64.fop_flags;
        }

        return sizeof(*unified);
    }

    packed                          = ptr;
    iface_attr->cap_flags           = 0;
    iface_attr->priority            = packed->prio_cap_flags & UCS_MASK(8);
    iface_attr->overhead            = packed->overhead;
    iface_attr->bandwidth.dedicated = ucs_max(0.0, packed->bandwidth);
    iface_attr->bandwidth.shared    = ucs_max(0.0, -packed->bandwidth);
    iface_attr->lat_ovh             = packed->lat_ovh;

    packed_flag = UCS_BIT(8);
    bit         = 1;
    while (UCP_ADDRESS_IFACE_FLAGS & ~(bit - 1)) {
        if (UCP_ADDRESS_IFACE_FLAGS & bit) {
            if (packed->prio_cap_flags & packed_flag) {
                iface_attr->cap_flags |= bit;
            }
            packed_flag <<= 1;
        }
        bit <<= 1;
    }

    if (packed->prio_cap_flags & UCT_ADDRESS_FLAG_ATOMIC32) {
        iface_attr->atomic.atomic32.op_flags  |= UCP_ATOMIC_OP_MASK;
        iface_attr->atomic.atomic32.fop_flags |= UCP_ATOMIC_FOP_MASK;
    }
    if (packed->prio_cap_flags & UCT_ADDRESS_FLAG_ATOMIC64) {
        iface_attr->atomic.atomic64.op_flags  |= UCP_ATOMIC_OP_MASK;
        iface_attr->atomic.atomic64.fop_flags |= UCP_ATOMIC_FOP_MASK;
    }

    return sizeof(*packed);
}

static void*
ucp_address_iface_flags_ptr(ucp_worker_h worker, void *attr_ptr, int attr_len)
{
    if (ucp_worker_unified_mode(worker)) {
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
    if (ucp_worker_unified_mode(worker)) {
        return ptr;
    }

    ucs_assert(addr_length <= UCP_ADDRESS_FLAG_LEN_MASK);
    *(uint8_t*)ptr = addr_length;

    return UCS_PTR_TYPE_OFFSET(ptr, uint8_t);
}

static const void*
ucp_address_unpack_length(ucp_worker_h worker, const void* flags_ptr, const void *ptr,
                          size_t *addr_length, int is_ep_addr, int *is_last)
{
    ucp_rsc_index_t rsc_index;
    uct_iface_attr_t *attr;
    const ucp_address_unified_iface_attr_t *unified;

    if (ucp_worker_unified_mode(worker)) {
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
            *is_last     = 1; /* in unified mode, there's only 1 ep address */
        } else {
            *addr_length = attr->iface_addr_len;
            *is_last     = unified->rsc_index & UCP_ADDRESS_FLAG_LAST;
        }
        return ptr;
    }

    *is_last     = *(uint8_t*)ptr & UCP_ADDRESS_FLAG_LAST;
    *addr_length = *(uint8_t*)ptr & UCP_ADDRESS_FLAG_LEN_MASK;

    return UCS_PTR_TYPE_OFFSET(ptr, uint8_t);
}

static ucs_status_t ucp_address_do_pack(ucp_worker_h worker, ucp_ep_h ep,
                                        void *buffer, size_t size,
                                        uint64_t tl_bitmap, uint64_t flags,
                                        const ucp_lane_index_t *lanes2remote,
                                        const ucp_address_packed_device_t *devices,
                                        ucp_rsc_index_t num_devices)
{
    ucp_context_h context       = worker->context;
    uint64_t md_flags_pack_mask = (UCT_MD_FLAG_REG | UCT_MD_FLAG_ALLOC);
    const ucp_address_packed_device_t *dev;
    uct_iface_attr_t *iface_attr;
    ucp_rsc_index_t md_index;
    ucp_worker_iface_t *wiface;
    ucp_rsc_index_t rsc_index;
    ucp_lane_index_t lane, remote_lane;
    void *flags_ptr, *ep_flags_ptr;
    uint64_t dev_tl_bitmap;
    unsigned num_ep_addrs;
    ucs_status_t status;
    size_t iface_addr_len;
    size_t ep_addr_len;
    uint64_t md_flags;
    unsigned index;
    int attr_len;
    void *ptr;
    int enable_amo;

    ptr   = buffer;
    index = 0;

    if (flags & UCP_ADDRESS_PACK_FLAG_WORKER_UUID) {
        *(uint64_t*)ptr = worker->uuid;
        ptr = UCS_PTR_TYPE_OFFSET(ptr, worker->uuid);
    }

    ptr = ucp_address_pack_worker_name(worker, ptr, flags);

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
        ucs_assert_always(!(md_index & ~UCP_ADDRESS_FLAG_MD_MASK));

        *(uint8_t*)ptr = md_index |
                         ((dev_tl_bitmap == 0)           ? UCP_ADDRESS_FLAG_EMPTY    : 0) |
                         ((md_flags & UCT_MD_FLAG_ALLOC) ? UCP_ADDRESS_FLAG_MD_ALLOC : 0) |
                         ((md_flags & UCT_MD_FLAG_REG)   ? UCP_ADDRESS_FLAG_MD_REG   : 0);
        ptr = UCS_PTR_TYPE_OFFSET(ptr, md_index);

        /* Device address length */
        *(uint8_t*)ptr = (dev == (devices + num_devices - 1)) ?
                         UCP_ADDRESS_FLAG_LAST : 0;
        if (flags & UCP_ADDRESS_PACK_FLAG_DEVICE_ADDR) {
            ucs_assert(dev->dev_addr_len < UCP_ADDRESS_FLAG_LAST);
            *(uint8_t*)ptr |= dev->dev_addr_len;
        }
        ptr = UCS_PTR_TYPE_OFFSET(ptr, uint8_t);

        /* Device address */
        if (flags & UCP_ADDRESS_PACK_FLAG_DEVICE_ADDR) {
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
                                                     iface_attr, enable_amo);
            if (attr_len < 0) {
                return UCS_ERR_INVALID_ADDR;
            }

            ucp_address_memcheck(context, ptr, attr_len, rsc_index);

            if (flags & UCP_ADDRESS_PACK_FLAG_IFACE_ADDR) {
                iface_addr_len = iface_attr->iface_addr_len;
            } else {
                iface_addr_len = 0;
            }

            flags_ptr = ucp_address_iface_flags_ptr(worker, ptr, attr_len);
            ptr       = UCS_PTR_BYTE_OFFSET(ptr, attr_len);
            ucs_assertv(iface_addr_len < UCP_ADDRESS_FLAG_HAVE_EP_ADDR,
                        "iface_addr_len=%zu", iface_addr_len);

            /* Pack iface address */
            ptr = ucp_address_pack_length(worker, ptr, iface_addr_len);
            if (flags & UCP_ADDRESS_PACK_FLAG_IFACE_ADDR) {
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
            if ((flags & UCP_ADDRESS_PACK_FLAG_EP_ADDR) &&
                ucp_worker_iface_is_tl_p2p(iface_attr)) {

                ucs_assert(ep != NULL);
                ep_addr_len  = iface_attr->ep_addr_len;
                ep_flags_ptr = NULL;

                for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
                    if (ucp_ep_get_rsc_index(ep, lane) != rsc_index) {
                        continue;
                    }

                    /* pack ep address length and save pointer to flags */
                    ep_flags_ptr = ptr;
                    ptr          = ucp_address_pack_length(worker, ptr,
                                                           ep_addr_len);

                    /* pack ep address */
                    status = uct_ep_get_address(ep->uct_eps[lane], ptr);
                    if (status != UCS_OK) {
                        return status;
                    }

                    ucp_address_memcheck(context, ptr, ep_addr_len, rsc_index);
                    ptr = UCS_PTR_BYTE_OFFSET(ptr, ep_addr_len);

                    /* pack ep lane index */
                    remote_lane    = (lanes2remote == NULL) ? lane :
                                     lanes2remote[lane];
                    *(uint8_t*)ptr = remote_lane;
                    ptr            = UCS_PTR_TYPE_OFFSET(ptr, uint8_t);

                    ucs_trace("pack addr[%d].ep_addr[%d] : len %zu lane %d->%d",
                               index, num_ep_addrs, ep_addr_len, lane,
                               remote_lane);

                    ++num_ep_addrs;
                }

                if (num_ep_addrs > 0) {
                    ucs_assert(ep_flags_ptr != NULL);
                    *(uint8_t*)flags_ptr    |= UCP_ADDRESS_FLAG_HAVE_EP_ADDR;
                    if (!ucp_worker_unified_mode(worker)) {
                        *(uint8_t*)ep_flags_ptr |= UCP_ADDRESS_FLAG_LAST;
                    }
                }
            }

            ucs_assert((num_ep_addrs > 0) ||
                       !(*(uint8_t*)flags_ptr & UCP_ADDRESS_FLAG_HAVE_EP_ADDR));

            if (flags & UCP_ADDRESS_PACK_FLAG_TRACE) {
                ucs_trace("pack addr[%d] : "UCT_TL_RESOURCE_DESC_FMT" "
                          "eps %u md_flags 0x%"PRIx64" tl_flags 0x%"PRIx64" bw %e + %e/n ovh %e "
                          "lat_ovh %e dev_priority %d a32 0x%lx/0x%lx a64 0x%lx/0x%lx",
                          index,
                          UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[rsc_index].tl_rsc),
                          num_ep_addrs, md_flags, iface_attr->cap.flags,
                          iface_attr->bandwidth.dedicated,
                          iface_attr->bandwidth.shared,
                          iface_attr->overhead,
                          iface_attr->latency.overhead,
                          iface_attr->priority,
                          iface_attr->cap.atomic32.op_flags,
                          iface_attr->cap.atomic32.fop_flags,
                          iface_attr->cap.atomic64.op_flags,
                          iface_attr->cap.atomic64.fop_flags);
            }

            ++index;
            ucs_assert(index <= UCP_MAX_RESOURCES);
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
                              uint64_t tl_bitmap, uint64_t flags,
                              const ucp_lane_index_t *lanes2remote,
                              size_t *size_p, void **buffer_p)
{
    ucp_address_packed_device_t *devices;
    ucp_rsc_index_t num_devices;
    ucs_status_t status;
    void *buffer;
    size_t size;

    if (ep == NULL) {
        flags &= ~UCP_ADDRESS_PACK_FLAG_EP_ADDR;
    }

    /* Collect all devices we want to pack */
    status = ucp_address_gather_devices(worker, ep, tl_bitmap, flags, &devices,
                                        &num_devices);
    if (status != UCS_OK) {
        goto out;
    }

    /* Calculate packed size */
    size = ucp_address_packed_size(worker, devices, num_devices, flags);

    /* Allocate address */
    buffer = ucs_malloc(size, "ucp_address");
    if (buffer == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out_free_devices;
    }

    memset(buffer, 0, size);

    /* Pack the address */
    status = ucp_address_do_pack(worker, ep, buffer, size, tl_bitmap, flags,
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
                                uint64_t flags,
                                ucp_unpacked_address_t *unpacked_address)
{
    ucp_address_entry_t *address_list, *address;
    ucp_address_entry_ep_addr_t *ep_addr;
    int last_dev, last_tl, last_ep_addr;
    const uct_device_addr_t *dev_addr;
    ucp_rsc_index_t dev_index;
    ucp_rsc_index_t md_index;
    unsigned address_count;
    int empty_dev;
    uint64_t md_flags;
    size_t dev_addr_len;
    size_t iface_addr_len;
    size_t ep_addr_len;
    size_t attr_len;
    uint8_t md_byte;
    const void *ptr;
    const void *aptr;
    const void *flags_ptr;

    ptr = buffer;
    if (flags & UCP_ADDRESS_PACK_FLAG_WORKER_UUID) {
        unpacked_address->uuid = *(uint64_t*)ptr;
        ptr = UCS_PTR_TYPE_OFFSET(ptr, unpacked_address->uuid);
    } else {
        unpacked_address->uuid = 0;
    }

    aptr = ucp_address_unpack_worker_name(ptr, unpacked_address->name,
                                          sizeof(unpacked_address->name),
                                          flags);

    /* Count addresses */
    ptr           = aptr;
    address_count = 0;

    last_dev = (*(uint8_t*)ptr == UCP_NULL_RESOURCE);
    while (!last_dev) {
        /* md_index */
        empty_dev    = (*(uint8_t*)ptr) & UCP_ADDRESS_FLAG_EMPTY;
        ptr          = UCS_PTR_TYPE_OFFSET(ptr, uint8_t);

        /* device address length */
        dev_addr_len = (*(uint8_t*)ptr) & ~UCP_ADDRESS_FLAG_LAST;
        last_dev     = (*(uint8_t*)ptr) & UCP_ADDRESS_FLAG_LAST;
        ptr          = UCS_PTR_TYPE_OFFSET(ptr, uint8_t);
        ptr          = UCS_PTR_BYTE_OFFSET(ptr, dev_addr_len);

        last_tl = empty_dev;
        while (!last_tl) {
            ptr       = UCS_PTR_TYPE_OFFSET(ptr, uint16_t); /* tl_name_csum */
            attr_len  = ucp_address_iface_attr_size(worker);
            flags_ptr = ucp_address_iface_flags_ptr(worker, (void*)ptr, attr_len);
            ptr       = UCS_PTR_BYTE_OFFSET(ptr, attr_len);
            ptr       = ucp_address_unpack_length(worker, flags_ptr, ptr,
                                                  &iface_addr_len, 0, &last_tl);
            ptr       = UCS_PTR_BYTE_OFFSET(ptr, iface_addr_len);

            last_ep_addr = !(*(uint8_t*)flags_ptr & UCP_ADDRESS_FLAG_HAVE_EP_ADDR);
            while (!last_ep_addr) {
                ptr = ucp_address_unpack_length(worker, flags_ptr, ptr,
                                                &ep_addr_len, 1, &last_ep_addr);
                ucs_assert(flags & UCP_ADDRESS_PACK_FLAG_EP_ADDR);
                ucs_assert(ep_addr_len > 0);
                ptr = UCS_PTR_BYTE_OFFSET(ptr, ep_addr_len);
                ptr = UCS_PTR_TYPE_OFFSET(ptr, uint8_t);
            }

            ++address_count;
            ucs_assert(address_count <= UCP_MAX_RESOURCES);
        }
    }

    if (address_count == 0) {
        address_list = NULL;
        goto out;
    }

    /* Allocate address list */
    address_list = ucs_calloc(address_count, sizeof(*address_list),
                              "ucp_address_list");
    if (address_list == NULL) {
        ucs_error("failed to allocate address list");
        return UCS_ERR_NO_MEMORY;
    }

    /* Unpack addresses */
    address   = address_list;
    ptr       = aptr;
    dev_index = 0;

    do {
        /* md_index */
        md_byte      = (*(uint8_t*)ptr);
        md_index     = md_byte & UCP_ADDRESS_FLAG_MD_MASK;
        md_flags     = (md_byte & UCP_ADDRESS_FLAG_MD_ALLOC) ? UCT_MD_FLAG_ALLOC : 0;
        md_flags    |= (md_byte & UCP_ADDRESS_FLAG_MD_REG)   ? UCT_MD_FLAG_REG   : 0;
        empty_dev    = md_byte & UCP_ADDRESS_FLAG_EMPTY;
        ptr          = UCS_PTR_TYPE_OFFSET(ptr, md_byte);

        /* device address length */
        dev_addr_len = (*(uint8_t*)ptr) & ~UCP_ADDRESS_FLAG_LAST;
        last_dev     = (*(uint8_t*)ptr) & UCP_ADDRESS_FLAG_LAST;
        ptr          = UCS_PTR_TYPE_OFFSET(ptr, uint8_t);

        dev_addr = ptr;
        ptr      = UCS_PTR_BYTE_OFFSET(ptr, dev_addr_len);

        last_tl = empty_dev;
        while (!last_tl) {
            /* tl_name_csum */
            address->tl_name_csum = *(uint16_t*)ptr;
            ptr = UCS_PTR_TYPE_OFFSET(ptr, address->tl_name_csum);

            address->dev_addr   = (dev_addr_len > 0) ? dev_addr : NULL;
            address->md_index   = md_index;
            address->dev_index  = dev_index;
            address->md_flags   = md_flags;

            attr_len  = ucp_address_unpack_iface_attr(worker, &address->iface_attr, ptr);
            flags_ptr = ucp_address_iface_flags_ptr(worker, (void*)ptr, attr_len);
            ptr       = UCS_PTR_BYTE_OFFSET(ptr, attr_len);
            ptr       = ucp_address_unpack_length(worker, flags_ptr, ptr,
                                                  &iface_addr_len, 0, &last_tl);
            address->iface_addr   = (iface_addr_len > 0) ? ptr : NULL;
            address->num_ep_addrs = 0;
            ptr                   = UCS_PTR_BYTE_OFFSET(ptr, iface_addr_len);

            last_ep_addr = !(*(uint8_t*)flags_ptr & UCP_ADDRESS_FLAG_HAVE_EP_ADDR);
            while (!last_ep_addr) {
                ucs_assert(address->num_ep_addrs < UCP_MAX_LANES);
                ep_addr       = &address->ep_addrs[address->num_ep_addrs++];
                ptr           = ucp_address_unpack_length(worker, flags_ptr, ptr,
                                                          &ep_addr_len, 1,
                                                          &last_ep_addr);
                ep_addr->addr = ptr;
                ptr           = UCS_PTR_BYTE_OFFSET(ptr, ep_addr_len);

                ep_addr->lane = *(uint8_t*)ptr;
                ptr           = UCS_PTR_TYPE_OFFSET(ptr, uint8_t);
            }

            if (flags & UCP_ADDRESS_PACK_FLAG_TRACE) {
                ucs_trace("unpack addr[%d] : eps %u md_flags 0x%"PRIx64" tl_flags 0x%"PRIx64" bw %e + %e/n ovh %e "
                          "lat_ovh %e dev_priority %d a32 0x%lx/0x%lx a64 0x%lx/0x%lx",
                          (int)(address - address_list), address->num_ep_addrs,
                          address->md_flags, address->iface_attr.cap_flags,
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

    ucs_assert((unsigned)(address - address_list) == address_count);

out:
    unpacked_address->address_count = address_count;
    unpacked_address->address_list  = address_list;
    return UCS_OK;
}

