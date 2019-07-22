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
 *   * In unified mode tl_info contains just rsc_index. For last address in the
 *     tl address list, it will have LAST flag set.
 *   * In non unified mode tl_info contains iface attributes. LAST flag is set in
 *     iface address length.
 *   * If a device does not have tl addresses, it's md_index will have the flag
 *     EMPTY.
 *   * If the address list is empty, then it will contain only a single md_index
 *     which equals to UCP_NULL_RESOURCE.
 *
 */


typedef struct {
    const char       *dev_name;
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

#define UCT_ADDRESS_FLAG_ATOMIC32     UCS_BIT(30) /* 32bit atomic operations */
#define UCT_ADDRESS_FLAG_ATOMIC64     UCS_BIT(31) /* 64bit atomic operations */

#define UCP_ADDRESS_FLAG_LAST         0x80   /* Last address in the list */
#define UCP_ADDRESS_FLAG_EP_ADDR      0x40   /* Indicates that ep addr is packed
                                                right after iface addr */
#define UCP_ADDRESS_FLAG_LEN_MASK     ~(UCP_ADDRESS_FLAG_EP_ADDR | \
                                        UCP_ADDRESS_FLAG_LAST)

#define UCP_ADDRESS_FLAG_EMPTY        0x80   /* Device without TL addresses */
#define UCP_ADDRESS_FLAG_MD_ALLOC     0x40   /* MD can register  */
#define UCP_ADDRESS_FLAG_MD_REG       0x20   /* MD can allocate */
#define UCP_ADDRESS_FLAG_MD_MASK      ~(UCP_ADDRESS_FLAG_EMPTY | \
                                        UCP_ADDRESS_FLAG_MD_ALLOC | \
                                        UCP_ADDRESS_FLAG_MD_REG)

static size_t ucp_address_worker_name_size(ucp_worker_h worker)
{
#if ENABLE_DEBUG_DATA
    return strlen(ucp_worker_get_name(worker)) + 1;
#else
    return 0;
#endif
}

static size_t ucp_address_iface_attr_size(ucp_worker_t *worker)
{
    return ucp_worker_unified_mode(worker) ?
           sizeof(ucp_rsc_index_t) : sizeof(ucp_address_packed_iface_attr_t);
}

static uint64_t ucp_worker_iface_can_connect(uct_iface_attr_t *attrs)
{
    return attrs->cap.flags &
           (UCT_IFACE_FLAG_CONNECT_TO_IFACE | UCT_IFACE_FLAG_CONNECT_TO_EP);
}

/* Pack a string and return a pointer to storage right after the string */
static void* ucp_address_pack_worker_name(ucp_worker_h worker, void *dest)
{
#if ENABLE_DEBUG_DATA
    const char *s = ucp_worker_get_name(worker);
    size_t length = strlen(s);

    ucs_assert(length <= UINT8_MAX);
    *(uint8_t*)dest = length;
    memcpy(dest + 1, s, length);
    return dest + 1 + length;
#else
    return dest;
#endif
}

/* Unpack a string and return pointer to next storage byte */
static const void* ucp_address_unpack_worker_name(const void *src, char *s, size_t max)
{
#if ENABLE_DEBUG_DATA
    size_t length, avail;

    ucs_assert(max >= 1);
    length   = *(const uint8_t*)src;
    avail    = ucs_min(length, max - 1);
    memcpy(s, src + 1, avail);
    s[avail] = '\0';
    return src + length + 1;
#else
    s[0] = '\0';
    return src;
#endif
}

static ucp_address_packed_device_t*
ucp_address_get_device(const char *name, ucp_address_packed_device_t *devices,
                       ucp_rsc_index_t *num_devices_p)
{
    ucp_address_packed_device_t *dev;

    for (dev = devices; dev < devices + *num_devices_p; ++dev) {
        if (!strcmp(name, dev->dev_name)) {
            goto out;
        }
    }

    dev = &devices[(*num_devices_p)++];
    memset(dev, 0, sizeof(*dev));
    dev->dev_name   = name;
out:
    return dev;
}

static ucs_status_t
ucp_address_gather_devices(ucp_worker_h worker, uint64_t tl_bitmap, int has_ep,
                           ucp_address_packed_device_t **devices_p,
                           ucp_rsc_index_t *num_devices_p)
{
    ucp_context_h context = worker->context;
    ucp_address_packed_device_t *dev, *devices;
    uct_iface_attr_t *iface_attr;
    ucp_rsc_index_t num_devices;
    ucp_rsc_index_t i;
    uint64_t mask;

    devices = ucs_calloc(context->num_tls, sizeof(*devices), "packed_devices");
    if (devices == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    num_devices = 0;
    ucs_for_each_bit(i, context->tl_bitmap) {
        mask = UCS_BIT(i);

        if (!(mask & tl_bitmap)) {
            continue;
        }

        iface_attr = ucp_worker_iface_get_attr(worker, i);

        if (!ucp_worker_iface_can_connect(iface_attr)) {
            continue;
        }

        dev = ucp_address_get_device(context->tl_rscs[i].tl_rsc.dev_name,
                                     devices, &num_devices);

        if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) && has_ep) {
            /* ep address (its length will be packed in non-unified mode only) */
            dev->tl_addrs_size += iface_attr->ep_addr_len;
            dev->tl_addrs_size += !ucp_worker_unified_mode(worker);
        }

        dev->tl_addrs_size += sizeof(uint16_t); /* tl name checksum */

        /* iface address (its length will be packed in non-unified mode only) */
        dev->tl_addrs_size += iface_attr->iface_addr_len;
        dev->tl_addrs_size += !ucp_worker_unified_mode(worker); /* if addr length */
        dev->tl_addrs_size += ucp_address_iface_attr_size(worker);
        dev->rsc_index      = i;
        dev->dev_addr_len   = iface_attr->device_addr_len;
        dev->tl_bitmap     |= mask;
    }

    *devices_p     = devices;
    *num_devices_p = num_devices;
    return UCS_OK;
}

static size_t ucp_address_packed_size(ucp_worker_h worker,
                                      const ucp_address_packed_device_t *devices,
                                      ucp_rsc_index_t num_devices)
{
    const ucp_address_packed_device_t *dev;
    size_t size;

    size = sizeof(uint64_t) + ucp_address_worker_name_size(worker);

    if (num_devices == 0) {
        size += 1;                      /* NULL md_index */
    } else {
        for (dev = devices; dev < devices + num_devices; ++dev) {
            size += 1;                  /* device md_index */
            size += 1;                  /* device address length */
            size += dev->dev_addr_len;  /* device address */
            size += dev->tl_addrs_size; /* transport addresses */
        }
    }
    return size;
}

static void ucp_address_memchek(void *ptr, size_t size,
                                const uct_tl_resource_desc_t *rsc)
{
    void *undef_ptr;

    undef_ptr = (void*)VALGRIND_CHECK_MEM_IS_DEFINED(ptr, size);
    if (undef_ptr != NULL) {
        ucs_error(UCT_TL_RESOURCE_DESC_FMT
                  " address contains undefined bytes at offset %zd",
                  UCT_TL_RESOURCE_DESC_ARG(rsc), undef_ptr - ptr);
    }
}

static ucs_status_t
ucp_address_pack_ep_address(ucp_ep_h ep, ucp_rsc_index_t tl_index,
                            uct_ep_addr_t *addr)
{
    ucp_lane_index_t lane;

    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        if (ucp_ep_get_rsc_index(ep, lane) == tl_index) {
            /*
             * If this is a wireup endpoint, it will return the underlying next_ep
             * address, and the length will be correct because the resource index
             * is of the next_ep.
             */
            return uct_ep_get_address(ep->uct_eps[lane], addr);
        }
    }

    ucs_bug("provided ucp_ep without required transport");
    return UCS_ERR_INVALID_ADDR;
}

static int ucp_address_pack_iface_attr(ucp_worker_h worker, void *ptr,
                                       ucp_rsc_index_t index,
                                       const uct_iface_attr_t *iface_attr,
                                       int enable_atomics)
{
    ucp_address_packed_iface_attr_t *packed;
    uint32_t packed_flag;
    uint64_t cap_flags;
    uint64_t bit;

    if (ucp_worker_unified_mode(worker)) {
        /* In unified mode all workers have the same transports and tl bitmap.
         * Just send rsc index, so the remote peer could fetch iface attributes
         * from its local iface. */
        *(ucp_rsc_index_t*)ptr = index;
        return sizeof(ucp_rsc_index_t);
    }

    packed    = ptr;
    cap_flags = iface_attr->cap.flags;

    packed->prio_cap_flags = ((uint8_t)iface_attr->priority);
    packed->overhead       = iface_attr->overhead;
    packed->bandwidth      = iface_attr->bandwidth;
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
    ucp_worker_iface_t *wiface;
    uint32_t packed_flag;
    ucp_rsc_index_t rsc_idx;
    uint64_t bit;

    if (ucp_worker_unified_mode(worker)) {
        /* Address contains resources index, not iface attrs.
         * Just take iface attrs from the local resource. */
        rsc_idx               = (*(ucp_rsc_index_t*)ptr) & UCP_ADDRESS_FLAG_LEN_MASK;
        wiface                = ucp_worker_iface(worker, rsc_idx);
        iface_attr->cap_flags = wiface->attr.cap.flags;
        iface_attr->priority  = wiface->attr.priority;
        iface_attr->overhead  = wiface->attr.overhead;
        iface_attr->bandwidth = wiface->attr.bandwidth;
        iface_attr->lat_ovh   = wiface->attr.latency.overhead;
        if (worker->atomic_tls & UCS_BIT(rsc_idx)) {
            iface_attr->atomic.atomic32.op_flags  = wiface->attr.cap.atomic32.op_flags;
            iface_attr->atomic.atomic32.fop_flags = wiface->attr.cap.atomic32.fop_flags;
            iface_attr->atomic.atomic64.op_flags  = wiface->attr.cap.atomic64.op_flags;
            iface_attr->atomic.atomic64.fop_flags = wiface->attr.cap.atomic64.fop_flags;
        }
        return sizeof(rsc_idx);
    }

    packed                = ptr;
    iface_attr->cap_flags = 0;
    iface_attr->priority  = packed->prio_cap_flags & UCS_MASK(8);
    iface_attr->overhead  = packed->overhead;
    iface_attr->bandwidth = packed->bandwidth;
    iface_attr->lat_ovh   = packed->lat_ovh;

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

static const void*
ucp_address_iface_flags_ptr(ucp_worker_h worker, const void *attr_ptr, int attr_len)
{
    if (ucp_worker_unified_mode(worker)) {
        /* In unified mode, rsc_index is packed instead of attrs. Address flags
         * will be packed in the end of rsc_index byte. */
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

    ucs_assert(addr_length < UINT8_MAX);
    *(uint8_t*)ptr = addr_length;

    return UCS_PTR_BYTE_OFFSET(ptr, 1);
}

static const void*
ucp_address_unpack_length(ucp_worker_h worker, const void* flags_ptr, const void *ptr,
                          size_t *addr_length, int is_ep_addr)
{
    ucp_rsc_index_t rsc_index;
    uct_iface_attr_t *attr;

    if (ucp_worker_unified_mode(worker)) {
        /* In unified mode:
         * - flags are packed with rsc index
         * - iface and ep addr lengths are not packed, need to take them from
         *   local iface attrs */
        rsc_index = (*(ucp_rsc_index_t*)flags_ptr) & UCP_ADDRESS_FLAG_LEN_MASK;
        attr      = &ucp_worker_iface(worker, rsc_index)->attr;

        if (is_ep_addr) {
            *addr_length = ((*(uint8_t*)flags_ptr) & UCP_ADDRESS_FLAG_EP_ADDR) ?
                           attr->ep_addr_len : 0;
        } else {
            *addr_length = attr->iface_addr_len;
        }
        return ptr;
    }

    if (is_ep_addr && !((*(uint8_t*)flags_ptr) & UCP_ADDRESS_FLAG_EP_ADDR)) {
        /* No ep address packed */
        *addr_length = 0;
        return ptr;
    }

    *addr_length = (*(uint8_t*)ptr) & UCP_ADDRESS_FLAG_LEN_MASK;

    return UCS_PTR_BYTE_OFFSET(ptr, 1);
}

static ucs_status_t ucp_address_do_pack(ucp_worker_h worker, ucp_ep_h ep,
                                        void *buffer, size_t size,
                                        uint64_t tl_bitmap, unsigned *order,
                                        const ucp_address_packed_device_t *devices,
                                        ucp_rsc_index_t num_devices)
{
    ucp_context_h context = worker->context;
    const ucp_address_packed_device_t *dev;
    uct_iface_attr_t *iface_attr;
    ucp_rsc_index_t md_index;
    ucp_worker_iface_t *wiface;
    ucs_status_t status;
    ucp_rsc_index_t i;
    size_t iface_addr_len;
    size_t ep_addr_len;
    uint64_t md_flags;
    unsigned index;
    int attr_len;
    void *ptr;
    const void *flags_ptr;

    ptr = buffer;
    index = 0;

    *(uint64_t*)ptr = worker->uuid;
    ptr += sizeof(uint64_t);
    ptr = ucp_address_pack_worker_name(worker, ptr);

    if (num_devices == 0) {
        *((uint8_t*)ptr) = UCP_NULL_RESOURCE;
        ++ptr;
        goto out;
    }

    for (dev = devices; dev < devices + num_devices; ++dev) {

        /* MD index */
        md_index       = context->tl_rscs[dev->rsc_index].md_index;
        md_flags       = context->tl_mds[md_index].attr.cap.flags;
        ucs_assert_always(!(md_index & ~UCP_ADDRESS_FLAG_MD_MASK));

        *(uint8_t*)ptr = md_index |
                         ((dev->tl_bitmap == 0)          ? UCP_ADDRESS_FLAG_EMPTY    : 0) |
                         ((md_flags & UCT_MD_FLAG_ALLOC) ? UCP_ADDRESS_FLAG_MD_ALLOC : 0) |
                         ((md_flags & UCT_MD_FLAG_REG)   ? UCP_ADDRESS_FLAG_MD_REG   : 0);
        ++ptr;

        /* Device address length */
        ucs_assert(dev->dev_addr_len < UCP_ADDRESS_FLAG_LAST);
        *(uint8_t*)ptr = dev->dev_addr_len | ((dev == (devices + num_devices - 1)) ?
                                              UCP_ADDRESS_FLAG_LAST : 0);
        ++ptr;

        /* Device address */
        wiface = ucp_worker_iface(worker, dev->rsc_index);
        status = uct_iface_get_device_address(wiface->iface, (uct_device_addr_t*)ptr);
        if (status != UCS_OK) {
            return status;
        }

        ucp_address_memchek(ptr, dev->dev_addr_len,
                            &context->tl_rscs[dev->rsc_index].tl_rsc);
        ptr += dev->dev_addr_len;

        ucs_for_each_bit(i, context->tl_bitmap) {

            if (!(UCS_BIT(i) & dev->tl_bitmap)) {
                continue;
            }

            wiface     = ucp_worker_iface(worker, i);
            iface_attr = &wiface->attr;

            if (!ucp_worker_iface_can_connect(iface_attr)) {
                return UCS_ERR_INVALID_ADDR;
            }

            /* Transport name checksum */
            *(uint16_t*)ptr = context->tl_rscs[i].tl_name_csum;
            ptr += sizeof(uint16_t);

            /* Transport information */
            attr_len = ucp_address_pack_iface_attr(worker, ptr, i, iface_attr,
                                                   worker->atomic_tls & UCS_BIT(i));
            ucp_address_memchek(ptr, attr_len,
                                &context->tl_rscs[dev->rsc_index].tl_rsc);

            iface_addr_len = iface_attr->iface_addr_len;
            flags_ptr      = ucp_address_iface_flags_ptr(worker, ptr, attr_len);
            ptr           += attr_len;
            ucs_assert(iface_addr_len < UCP_ADDRESS_FLAG_EP_ADDR);

            /* Pack iface address */
            ptr    = ucp_address_pack_length(worker, ptr, iface_addr_len);
            status = uct_iface_get_address(wiface->iface, (uct_iface_addr_t*)ptr);
            if (status != UCS_OK) {
                return status;
            }
            ucp_address_memchek(ptr, iface_addr_len,
                                &context->tl_rscs[dev->rsc_index].tl_rsc);
            ptr += iface_addr_len;

            /* cppcheck-suppress internalAstError */
            if (i == ucs_ilog2(dev->tl_bitmap)) {
                 *(uint8_t*)flags_ptr |= UCP_ADDRESS_FLAG_LAST;
            }

            /* Pack ep address if present */
            if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) &&
                (ep != NULL)) {

                ep_addr_len           = iface_attr->ep_addr_len;
                *(uint8_t*)flags_ptr |= UCP_ADDRESS_FLAG_EP_ADDR;

                ptr    = ucp_address_pack_length(worker, ptr, ep_addr_len);
                status = ucp_address_pack_ep_address(ep, i, ptr);
                if (status != UCS_OK) {
                    return status;
                }
                ucp_address_memchek(ptr, ep_addr_len,
                                    &context->tl_rscs[dev->rsc_index].tl_rsc);
                ptr += ep_addr_len;
            }

            /* Save the address index of this transport */
            if (order != NULL) {
                order[ucs_bitmap2idx(tl_bitmap, i)] = index;
            }

            ucs_trace("pack addr[%d] : "UCT_TL_RESOURCE_DESC_FMT
                      " md_flags 0x%"PRIx64" tl_flags 0x%"PRIx64" bw %e ovh %e "
                      "lat_ovh %e dev_priority %d a32 0x%lx/0x%lx a64 0x%lx/0x%lx",
                      index,
                      UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[i].tl_rsc),
                      md_flags, iface_attr->cap.flags,
                      iface_attr->bandwidth,
                      iface_attr->overhead,
                      iface_attr->latency.overhead,
                      iface_attr->priority,
                      iface_attr->cap.atomic32.op_flags,
                      iface_attr->cap.atomic32.fop_flags,
                      iface_attr->cap.atomic64.op_flags,
                      iface_attr->cap.atomic64.fop_flags);
            ++index;
        }
    }

out:
    ucs_assertv(buffer + size == ptr, "buffer=%p size=%zu ptr=%p ptr-buffer=%zd",
                buffer, size, ptr, ptr - buffer);
    return UCS_OK;
}

ucs_status_t ucp_address_pack(ucp_worker_h worker, ucp_ep_h ep, uint64_t tl_bitmap,
                              unsigned *order, size_t *size_p, void **buffer_p)
{
    ucp_address_packed_device_t *devices;
    ucp_rsc_index_t num_devices;
    ucs_status_t status;
    void *buffer;
    size_t size;

    /* Collect all devices we want to pack */
    status = ucp_address_gather_devices(worker, tl_bitmap, ep != NULL,
                                        &devices, &num_devices);
    if (status != UCS_OK) {
        goto out;
    }

    /* Calculate packed size */
    size = ucp_address_packed_size(worker, devices, num_devices);

    /* Allocate address */
    buffer = ucs_malloc(size, "ucp_address");
    if (buffer == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out_free_devices;
    }

    memset(buffer, 0, size);

    /* Pack the address */
    status = ucp_address_do_pack(worker, ep, buffer, size, tl_bitmap, order,
                                 devices, num_devices);
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
                                ucp_unpacked_address_t *unpacked_address)
{
    ucp_address_entry_t *address_list, *address;
    const uct_device_addr_t *dev_addr;
    ucp_rsc_index_t dev_index;
    ucp_rsc_index_t md_index;
    unsigned address_count;
    int last_dev, last_tl;
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
    unpacked_address->uuid = *(uint64_t*)ptr;
    ptr += sizeof(uint64_t);

    aptr = ucp_address_unpack_worker_name(ptr, unpacked_address->name,
                                          sizeof(unpacked_address->name));

    address_count = 0;

    /* Count addresses */
    ptr = aptr;
    do {
        if (*(uint8_t*)ptr == UCP_NULL_RESOURCE) {
            break;
        }

        /* md_index */
        empty_dev    = (*(uint8_t*)ptr) & UCP_ADDRESS_FLAG_EMPTY;
        ++ptr;

        /* device address length */
        dev_addr_len = (*(uint8_t*)ptr) & ~UCP_ADDRESS_FLAG_LAST;
        last_dev     = (*(uint8_t*)ptr) & UCP_ADDRESS_FLAG_LAST;
        ++ptr;

        ptr += dev_addr_len;

        last_tl = empty_dev;
        while (!last_tl) {
            ptr      += sizeof(uint16_t);  /* tl_name_csum */
            attr_len  = ucp_address_iface_attr_size(worker);
            flags_ptr = ucp_address_iface_flags_ptr(worker, ptr, attr_len);
            ptr      += attr_len;
            ptr       = ucp_address_unpack_length(worker, flags_ptr, ptr,
                                                  &iface_addr_len, 0);
            ptr      += iface_addr_len;
            ptr       = ucp_address_unpack_length(worker, flags_ptr, ptr,
                                                  &ep_addr_len, 1);
            ptr      += ep_addr_len;
            last_tl   = (*(uint8_t*)flags_ptr) & UCP_ADDRESS_FLAG_LAST;

            ++address_count;
            ucs_assert(address_count <= UCP_MAX_RESOURCES);
        }
    } while (!last_dev);

    if (!address_count) {
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
    address = address_list;
    ptr     = aptr;
    dev_index = 0;
    do {
        if (*(uint8_t*)ptr == UCP_NULL_RESOURCE) {
            break;
        }

        /* md_index */
        md_byte      = (*(uint8_t*)ptr);
        md_index     = md_byte & UCP_ADDRESS_FLAG_MD_MASK;
        md_flags     = (md_byte & UCP_ADDRESS_FLAG_MD_ALLOC) ? UCT_MD_FLAG_ALLOC : 0;
        md_flags    |= (md_byte & UCP_ADDRESS_FLAG_MD_REG)   ? UCT_MD_FLAG_REG   : 0;
        empty_dev    = md_byte & UCP_ADDRESS_FLAG_EMPTY;
        ++ptr;

        /* device address length */
        dev_addr_len = (*(uint8_t*)ptr) & ~UCP_ADDRESS_FLAG_LAST;
        last_dev     = (*(uint8_t*)ptr) & UCP_ADDRESS_FLAG_LAST;
        ++ptr;

        dev_addr = ptr;

        ptr += dev_addr_len;

        last_tl = empty_dev;
        while (!last_tl) {
            /* tl_name_csum */
            address->tl_name_csum = *(uint16_t*)ptr;
            ptr += sizeof(uint16_t);

            address->dev_addr   = (dev_addr_len > 0) ? dev_addr : NULL;
            address->md_index   = md_index;
            address->dev_index  = dev_index;
            address->md_flags   = md_flags;

            attr_len  = ucp_address_unpack_iface_attr(worker, &address->iface_attr, ptr);
            flags_ptr = ucp_address_iface_flags_ptr(worker, ptr, attr_len);
            ptr      += attr_len;
            ptr       = ucp_address_unpack_length(worker, flags_ptr, ptr,
                                                  &iface_addr_len, 0);
            address->iface_addr = (iface_addr_len > 0) ? ptr : NULL;

            ptr      += iface_addr_len;
            ptr       = ucp_address_unpack_length(worker, flags_ptr, ptr,
                                                  &ep_addr_len, 1);
            address->ep_addr = (ep_addr_len > 0) ? ptr : NULL;
            ptr      += ep_addr_len;
            last_tl   = (*(uint8_t*)flags_ptr) & UCP_ADDRESS_FLAG_LAST;

            ucs_trace("unpack addr[%d] : md_flags 0x%"PRIx64" tl_flags 0x%"PRIx64" bw %e ovh %e "
                      "lat_ovh %e dev_priority %d a32 0x%lx/0x%lx a64 0x%lx/0x%lx",
                      (int)(address - address_list),
                      address->md_flags, address->iface_attr.cap_flags,
                      address->iface_attr.bandwidth, address->iface_attr.overhead,
                      address->iface_attr.lat_ovh,
                      address->iface_attr.priority,
                      address->iface_attr.atomic.atomic32.op_flags,
                      address->iface_attr.atomic.atomic32.fop_flags,
                      address->iface_attr.atomic.atomic64.op_flags,
                      address->iface_attr.atomic.atomic64.fop_flags);
            ++address;
        }

        ++dev_index;
    } while (!last_dev);

out:
    unpacked_address->address_count = address_count;
    unpacked_address->address_list  = address_list;
    return UCS_OK;
}

