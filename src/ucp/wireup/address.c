/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

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
 *   * Last address in the tl address list, it's address will have the flag LAST.
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
    uint32_t         prio_cap_flags; /* 8 lsb: prio, 24 msb - cap flags */
} ucp_address_packed_iface_attr_t;


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

static size_t ucp_address_string_packed_size(const char *s)
{
    return strlen(s) + 1;
}

/* Pack a string and return a pointer to storage right after the string */
static void* ucp_address_pack_string(const char *s, void *dest)
{
    size_t length = strlen(s);

    ucs_assert(length <= UINT8_MAX);
    *(uint8_t*)dest = length;
    memcpy(dest + 1, s, length);
    return dest + 1 + length;
}

/* Unpack a string and return pointer to next storage byte */
static const void* ucp_address_unpack_string(const void *src, char *s, size_t max)
{
    size_t length, avail;

    ucs_assert(max >= 1);
    length   = *(const uint8_t*)src;
    avail    = ucs_min(length, max - 1);
    memcpy(s, src + 1, avail);
    s[avail]  = '\0';
    return src + length + 1;
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
    for (i = 0; i < context->num_tls; ++i) {
        mask = UCS_BIT(i);

        if (!(mask & tl_bitmap)) {
            continue;
        }

        iface_attr = &worker->ifaces[i].attr;

        if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) &&
            !(iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP)) {
            continue;
        }

        dev = ucp_address_get_device(context->tl_rscs[i].tl_rsc.dev_name,
                                     devices, &num_devices);

        dev->tl_addrs_size += iface_attr->iface_addr_len;

        if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) && has_ep) {
            /* ep address and its length */
            dev->tl_addrs_size += 1 + iface_attr->ep_addr_len;
        }

        dev->tl_addrs_size += sizeof(uint16_t); /* tl name checksum */
        dev->tl_addrs_size += sizeof(ucp_address_packed_iface_attr_t); /* iface attr */
        dev->tl_addrs_size += 1;                /* iface address length */
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

    size = sizeof(uint64_t) +
           ucp_address_string_packed_size(ucp_worker_get_name(worker));

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

static void ucp_address_pack_iface_attr(ucp_address_packed_iface_attr_t *packed,
                                        const uct_iface_attr_t *iface_attr,
                                        int enable_atomics)
{
    uint32_t packed_flag;
    uint64_t cap_flags;
    uint64_t bit;

    cap_flags = iface_attr->cap.flags;
    if (!enable_atomics) {
        cap_flags &= ~(UCP_UCT_IFACE_ATOMIC32_FLAGS | UCP_UCT_IFACE_ATOMIC64_FLAGS);
    }

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
}

static void
ucp_address_unpack_iface_attr(ucp_address_iface_attr_t *iface_attr,
                              const ucp_address_packed_iface_attr_t *packed)
{
    uint32_t packed_flag;
    uint64_t bit;

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
    ucs_status_t status;
    ucp_rsc_index_t i;
    size_t iface_addr_len;
    size_t ep_addr_len;
    uint64_t md_flags;
    unsigned index;
    void *ptr;
    uint8_t *iface_addr_len_ptr;

    ptr = buffer;
    index = 0;

    *(uint64_t*)ptr = worker->uuid;
    ptr += sizeof(uint64_t);
    ptr = ucp_address_pack_string(ucp_worker_get_name(worker), ptr);

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
        status = uct_iface_get_device_address(worker->ifaces[dev->rsc_index].iface,
                                              (uct_device_addr_t*)ptr);
        if (status != UCS_OK) {
            return status;
        }

        ucp_address_memchek(ptr, dev->dev_addr_len,
                            &context->tl_rscs[dev->rsc_index].tl_rsc);
        ptr += dev->dev_addr_len;

        for (i = 0; i < context->num_tls; ++i) {

            if (!(UCS_BIT(i) & dev->tl_bitmap)) {
                continue;
            }

            /* Transport name checksum */
            *(uint16_t*)ptr = context->tl_rscs[i].tl_name_csum;
            ptr += sizeof(uint16_t);

            /* Transport information */
            ucp_address_pack_iface_attr(ptr, &worker->ifaces[i].attr,
                                        worker->atomic_tls & UCS_BIT(i));
            ucp_address_memchek(ptr, sizeof(ucp_address_packed_iface_attr_t),
                                &context->tl_rscs[dev->rsc_index].tl_rsc);
            ptr += sizeof(ucp_address_packed_iface_attr_t);

            iface_attr = &worker->ifaces[i].attr;

            if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) &&
                !(iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP)) {
                return UCS_ERR_INVALID_ADDR;
            }

            /* Pack iface address */
            iface_addr_len = iface_attr->iface_addr_len;
            ucs_assert(iface_addr_len < UCP_ADDRESS_FLAG_EP_ADDR);

            status = uct_iface_get_address(worker->ifaces[i].iface,
                                           (uct_iface_addr_t*)(ptr + 1));
            if (status != UCS_OK) {
                return status;
            }
            ucp_address_memchek(ptr + 1, iface_addr_len,
                                &context->tl_rscs[dev->rsc_index].tl_rsc);
            iface_addr_len_ptr  = ptr;
            *iface_addr_len_ptr = iface_addr_len | ((i == ucs_ilog2(dev->tl_bitmap)) ?
                                                    UCP_ADDRESS_FLAG_LAST : 0);
            ptr += 1 + iface_addr_len;

            /* Pack ep address if present */
            if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) &&
                (ep != NULL)) {
                *iface_addr_len_ptr |= UCP_ADDRESS_FLAG_EP_ADDR;

                ep_addr_len = iface_attr->ep_addr_len;
                ucs_assert(ep_addr_len < UINT8_MAX);
                *(uint8_t*)ptr = ep_addr_len;

                status      = ucp_address_pack_ep_address(ep, i, ptr + 1);
                if (status != UCS_OK) {
                    return status;
                }
                ucp_address_memchek(ptr + 1, ep_addr_len,
                                    &context->tl_rscs[dev->rsc_index].tl_rsc);
                ptr += 1 + ep_addr_len;
            }

            /* Save the address index of this transport */
            if (order != NULL) {
                order[ucs_count_one_bits(tl_bitmap & UCS_MASK(i))] = index;
            }

            ucs_trace("pack addr[%d] : "UCT_TL_RESOURCE_DESC_FMT
                      " md_flags 0x%"PRIx64" tl_flags 0x%"PRIx64" bw %e ovh %e "
                      "lat_ovh: %e dev_priority %d",
                      index,
                      UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[i].tl_rsc),
                      md_flags, worker->ifaces[i].attr.cap.flags,
                      worker->ifaces[i].attr.bandwidth,
                      worker->ifaces[i].attr.overhead,
                      worker->ifaces[i].attr.latency.overhead,
                      worker->ifaces[i].attr.priority);
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

ucs_status_t ucp_address_unpack(const void *buffer, uint64_t *remote_uuid_p,
                                char *remote_name, size_t max,
                                unsigned *address_count_p,
                                ucp_address_entry_t **address_list_p)
{
    ucp_address_entry_t *address_list, *address;
    const uct_device_addr_t *dev_addr;
    ucp_rsc_index_t dev_index;
    ucp_rsc_index_t md_index;
    unsigned address_count;
    int last_dev, last_tl, ep_addr_present;
    int empty_dev;
    uint64_t md_flags;
    size_t dev_addr_len;
    size_t iface_addr_len;
    size_t ep_addr_len;
    uint8_t md_byte;
    const void *ptr;
    const void *aptr;

    ptr = buffer;
    *remote_uuid_p = *(uint64_t*)ptr;
    ptr += sizeof(uint64_t);

    aptr = ucp_address_unpack_string(ptr, remote_name, max);

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
            ptr += sizeof(uint16_t);                        /* tl_name_csum */
            ptr += sizeof(ucp_address_packed_iface_attr_t); /* iface attr */

            /* iface and ep address lengths */
            iface_addr_len  = (*(uint8_t*)ptr) & UCP_ADDRESS_FLAG_LEN_MASK;
            last_tl         = (*(uint8_t*)ptr) & UCP_ADDRESS_FLAG_LAST;
            ep_addr_present = (*(uint8_t*)ptr) & UCP_ADDRESS_FLAG_EP_ADDR;
            ptr            += 1 + iface_addr_len;

            if (ep_addr_present) {
                ep_addr_len = *(uint8_t*)ptr;
                ptr        += 1 + ep_addr_len;
            }

            ++address_count;
            ucs_assert(address_count <= UCP_MAX_RESOURCES);
        }

    } while (!last_dev);


    /* Allocate address list */
    address_list = ucs_calloc(address_count, sizeof(*address_list),
                              "ucp_address_list");
    if (address_list == NULL) {
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
            /* tl name */
            address->tl_name_csum = *(uint16_t*)ptr;
            ptr += sizeof(uint16_t);

            /* tl_name_csum */
            ucp_address_unpack_iface_attr(&address->iface_attr, ptr);
            ptr += sizeof(ucp_address_packed_iface_attr_t);

            /* tl address length */
            iface_addr_len  = (*(uint8_t*)ptr) & UCP_ADDRESS_FLAG_LEN_MASK;
            last_tl         = (*(uint8_t*)ptr) & UCP_ADDRESS_FLAG_LAST;
            ep_addr_present = (*(uint8_t*)ptr) & UCP_ADDRESS_FLAG_EP_ADDR;
            ++ptr;

            address->dev_addr   = (dev_addr_len > 0) ? dev_addr : NULL;
            address->md_index   = md_index;
            address->dev_index  = dev_index;
            address->md_flags   = md_flags;
            address->iface_addr = (iface_addr_len > 0) ? ptr : NULL;
            ptr                += iface_addr_len;

            if (ep_addr_present) {
                ep_addr_len      = *(uint8_t*)ptr;
                address->ep_addr = (ep_addr_len > 0) ? ptr + 1 : NULL;
                ptr             += 1 + ep_addr_len;
            } else {
                address->ep_addr = NULL;
            }

            ucs_trace("unpack addr[%d] : md_flags 0x%"PRIx64" tl_flags 0x%"PRIx64" bw %e ovh %e "
                      "lat_ovh %e dev_priority %d",
                      (int)(address - address_list),
                      address->md_flags, address->iface_attr.cap_flags,
                      address->iface_attr.bandwidth, address->iface_attr.overhead,
                      address->iface_attr.lat_ovh,
                      address->iface_attr.priority);
            ++address;
        }

        ++dev_index;
    } while (!last_dev);

    *address_count_p = address_count;
    *address_list_p  = address_list;
    return UCS_OK;
}

