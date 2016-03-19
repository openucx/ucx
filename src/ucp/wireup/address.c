/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "address.h"
#include "stub_ep.h"

#include <ucp/core/ucp_worker.h>
#include <ucs/arch/bitops.h>
#include <ucs/debug/log.h>
#include <string.h>


/*
 * Packed address layout:
 *
 * [ uuid(64bit) | worker_name(string) ]
 * [ device1_pd_index | device1_address(var) ]
 *    [ tl1_name(string) | tl1_address(var) ]
 *    [ tl2_name(string) | tl2_address(var) ]
 *    ...
 * [ device2_pd_index | device2_address(var) ]
 *    ...
 *
 *   * Last address in the tl address list, it's address will have the flag LAST.
 *   * If a device does not have tl addresses, it's pd_index will have the flag
 *     EMPTY.
 *   * If the address list is empty, then it will contain only a single pd_index
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


#define UCP_ADDRESS_FLAG_LAST         0x80   /* Last address in the list */
#define UCP_ADDRESS_FLAG_EMPTY        0x80   /* Device without TL addresses */


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

static const void* ucp_address_skip_string(const void *src)
{
    return src + (*(const uint8_t*)src) + 1;
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

        dev = ucp_address_get_device(context->tl_rscs[i].tl_rsc.dev_name,
                                     devices, &num_devices);

        iface_attr = &worker->iface_attrs[i];
        if (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
                    dev->tl_addrs_size += iface_attr->iface_addr_len;
        } else if (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
            if (has_ep) {
                dev->tl_addrs_size += iface_attr->ep_addr_len;
            } else {
                /* Empty address */
            }
        } else  {
            continue;
        }

        dev->rsc_index      = i;
        dev->dev_addr_len   = iface_attr->device_addr_len;
        dev->tl_bitmap     |= mask;

        dev->tl_addrs_size += 1; /* address length */
        dev->tl_addrs_size += ucp_address_string_packed_size(context->tl_rscs[i].tl_rsc.tl_name);
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
        size += 1;                      /* NULL pd_index */
    } else {
        for (dev = devices; dev < devices + num_devices; ++dev) {
            size += 1;                  /* device pd_index */
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

static ucs_status_t ucp_address_do_pack(ucp_worker_h worker, ucp_ep_h ep,
                                        void *buffer, size_t size,
                                        uint64_t tl_bitmap, unsigned *order,
                                        const ucp_address_packed_device_t *devices,
                                        ucp_rsc_index_t num_devices)
{
    ucp_context_h context = worker->context;
    const ucp_address_packed_device_t *dev;
    uct_iface_attr_t *iface_attr;
    ucs_status_t status;
    ucp_rsc_index_t i;
    size_t tl_addr_len;
    unsigned index;
    uct_ep_h uct_ep;
    void *ptr;

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

        /* PD index */
        *(uint8_t*)ptr = context->tl_rscs[dev->rsc_index].pd_index |
                         ((dev->tl_bitmap == 0) ? UCP_ADDRESS_FLAG_EMPTY : 0);
        ++ptr;

        /* Device address length */
        ucs_assert(dev->dev_addr_len < UCP_ADDRESS_FLAG_LAST);
        *(uint8_t*)ptr = dev->dev_addr_len | ((dev == (devices + num_devices - 1)) ?
                                              UCP_ADDRESS_FLAG_LAST : 0);
        ++ptr;

        /* Device address */
        status = uct_iface_get_device_address(worker->ifaces[dev->rsc_index],
                                              (uct_device_addr_t*)ptr);
        if (status != UCS_OK) {
            return status;
        }

        ptr += dev->dev_addr_len;

        for (i = 0; i < context->num_tls; ++i) {

            if (!(UCS_BIT(i) & dev->tl_bitmap)) {
                continue;
            }

            /* Transport name */
            ptr = ucp_address_pack_string(context->tl_rscs[i].tl_rsc.tl_name, ptr);

            /* Transport address length */
            iface_attr = &worker->iface_attrs[i];
            if (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
                tl_addr_len = iface_attr->iface_addr_len;
                status = uct_iface_get_address(worker->ifaces[i],
                                               (uct_iface_addr_t*)(ptr + 1));
            } else if (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
                if (ep != NULL) {
                    tl_addr_len = iface_attr->ep_addr_len;
                    ucs_assert(ep->rsc_index == i);
                    if (ep->state & UCP_EP_STATE_STUB_EP) {
                        uct_ep = ucp_ep_get_stub_ep(ep)->next_ep;
                    } else {
                        uct_ep = ep->uct_ep;
                    }
                    status = uct_ep_get_address(uct_ep, (uct_ep_addr_t*)(ptr + 1));
                } else {
                    tl_addr_len = 0;
                    status      = UCS_OK;
                }
            } else {
                tl_addr_len = 0;
                status      = UCS_ERR_INVALID_ADDR;
            }
            if (status != UCS_OK) {
                return status;
            }

            ucp_address_memchek(ptr + 1, tl_addr_len,
                                &context->tl_rscs[dev->rsc_index].tl_rsc);

            /* Save the address index of this transport */
            if (order != NULL) {
                order[ucs_count_one_bits(tl_bitmap & UCS_MASK(i))] = index++;
            }

            ucs_assert(tl_addr_len < UCP_ADDRESS_FLAG_LAST);
            *(uint8_t*)ptr = tl_addr_len | ((i == ucs_ilog2(dev->tl_bitmap)) ?
                                            UCP_ADDRESS_FLAG_LAST : 0);
            ptr += 1 + tl_addr_len;
        }
    }

out:
    ucs_assertv(buffer + size == ptr, "buffer=%p size=%zu ptr=%p", buffer, size,
                ptr);
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
    ucp_rsc_index_t pd_index;
    unsigned address_count;
    int last_dev, last_tl;
    int empty_dev;
    size_t dev_addr_len;
    size_t tl_addr_len;
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

        /* pd_index */
        pd_index     = (*(uint8_t*)ptr) & ~UCP_ADDRESS_FLAG_EMPTY;
        empty_dev    = (*(uint8_t*)ptr) & UCP_ADDRESS_FLAG_EMPTY;
        ++ptr;

        /* device address length */
        dev_addr_len = (*(uint8_t*)ptr) & ~UCP_ADDRESS_FLAG_LAST;
        last_dev     = (*(uint8_t*)ptr) & UCP_ADDRESS_FLAG_LAST;
        ++ptr;

        ptr += dev_addr_len;

        last_tl = empty_dev;
        while (!last_tl) {
            ptr = ucp_address_skip_string(ptr); /* tl_name */

            /* tl address length */
            tl_addr_len = (*(uint8_t*)ptr) & ~UCP_ADDRESS_FLAG_LAST;
            last_tl     = (*(uint8_t*)ptr) & UCP_ADDRESS_FLAG_LAST;
            ++ptr;

            ++address_count;
            ucs_assert(address_count <= UCP_MAX_RESOURCES);

            ptr += tl_addr_len;
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
    do {
        if (*(uint8_t*)ptr == UCP_NULL_RESOURCE) {
            break;
        }

        /* pd_index */
        pd_index     = (*(uint8_t*)ptr) & ~UCP_ADDRESS_FLAG_EMPTY;
        empty_dev    = (*(uint8_t*)ptr) & UCP_ADDRESS_FLAG_EMPTY;
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
            ptr = ucp_address_unpack_string(ptr, address->tl_name,
                                            UCT_TL_NAME_MAX);

            /* tl address length */
            tl_addr_len = (*(uint8_t*)ptr) & ~UCP_ADDRESS_FLAG_LAST;
            last_tl     = (*(uint8_t*)ptr) & UCP_ADDRESS_FLAG_LAST;
            ++ptr;

            address->dev_addr     = dev_addr;
            address->dev_addr_len = dev_addr_len;
            address->pd_index     = pd_index;
            address->tl_addr      = ptr;
            address->tl_addr_len  = tl_addr_len;
            ++address;

            ptr += tl_addr_len;
        }
    } while (!last_dev);

    *address_count_p = address_count;
    *address_list_p  = address_list;
    return UCS_OK;
}

