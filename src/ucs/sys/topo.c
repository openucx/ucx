/**
* Copyright (C) NVIDIA Corporation. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/sys/topo.h>
#include <ucs/type/status.h>
#include <stdio.h>
#include <ucs/datastruct/khash.h>
#include <ucs/type/spinlock.h>
#include <ucs/debug/log.h>
#include <ucs/debug/assert.h>

#define UCS_TOPO_MAX_SYS_DEVICES 1024

typedef int64_t ucs_bus_id_bit_rep_t;

typedef struct ucs_topo_sys_dev_to_bus_arr {
    ucs_sys_bus_id_t bus_arr[UCS_TOPO_MAX_SYS_DEVICES];
    unsigned         count;
} ucs_topo_sys_dev_to_bus_arr_t;

KHASH_MAP_INIT_INT64(bus_to_sys_dev, ucs_sys_device_t);

typedef struct ucs_topo_global_ctx {
    khash_t(bus_to_sys_dev)       bus_to_sys_dev_hash;
    ucs_spinlock_t                lock;
    ucs_topo_sys_dev_to_bus_arr_t sys_dev_to_bus_lookup;
} ucs_topo_global_ctx_t;

static ucs_topo_global_ctx_t ucs_topo_ctx;

/* TODO: can this conflict with a valid BDF? */
ucs_sys_bus_id_t ucs_sys_bus_id_unknown = { .domain   = 0xffff,
                                            .bus      = 0xff,
                                            .slot     = 0xff,
                                            .function = 0xff
                                          };

static ucs_bus_id_bit_rep_t ucs_topo_get_bus_id_bit_repr(const ucs_sys_bus_id_t *bus_id)
{
    return (((uint64_t)bus_id->domain << 24) |
            ((uint64_t)bus_id->bus << 16)    |
            ((uint64_t)bus_id->slot << 8)    |
            (bus_id->function));
}

void ucs_topo_init()
{
    ucs_spinlock_init(&ucs_topo_ctx.lock, 0);
    kh_init_inplace(bus_to_sys_dev, &ucs_topo_ctx.bus_to_sys_dev_hash);
    ucs_topo_ctx.sys_dev_to_bus_lookup.count = 0;
}

void ucs_topo_cleanup()
{
    ucs_status_t status;

    kh_destroy_inplace(bus_to_sys_dev, &ucs_topo_ctx.bus_to_sys_dev_hash);

    status = ucs_spinlock_destroy(&ucs_topo_ctx.lock);
    if (status != UCS_OK) {
        ucs_warn("ucs_recursive_spinlock_destroy() failed: %s",
                 ucs_status_string(status));
    }
}

static int ucs_topo_compare_bus_id(const ucs_sys_bus_id_t *bus_id1, 
                                   const ucs_sys_bus_id_t *bus_id2)
{
    return ((bus_id1->domain == bus_id2->domain) &&
            (bus_id1->bus == bus_id2->bus) &&
            (bus_id1->slot == bus_id2->slot) &&
            (bus_id1->function == bus_id2->function));
}

ucs_status_t ucs_topo_find_device_by_bus_id(const ucs_sys_bus_id_t *bus_id,
                                            ucs_sys_device_t *sys_dev)
{
    khiter_t hash_it;
    ucs_kh_put_t kh_put_status;
    ucs_bus_id_bit_rep_t bus_id_bit_rep;

    *sys_dev        = UCS_SYS_DEVICE_ID_UNKNOWN;
    bus_id_bit_rep  = ucs_topo_get_bus_id_bit_repr(bus_id);

    if (ucs_topo_compare_bus_id(bus_id, &ucs_sys_bus_id_unknown)) {
        ucs_debug("found unknown device index %u for bus id %ld",
                  *sys_dev, bus_id_bit_rep);
        return UCS_OK;
    }
    
    ucs_debug("find device index for bus id %ld", bus_id_bit_rep);

    ucs_spin_lock(&ucs_topo_ctx.lock);
    hash_it = kh_put(bus_to_sys_dev /*name*/,
                     &ucs_topo_ctx.bus_to_sys_dev_hash /*pointer to hashmap*/,
                     bus_id_bit_rep /*key*/,
                     &kh_put_status);

    if (kh_put_status == UCS_KH_PUT_KEY_PRESENT) {
        *sys_dev = kh_value(&ucs_topo_ctx.bus_to_sys_dev_hash, hash_it);
        ucs_debug("bus id %ld exists. sys_dev = %u", bus_id_bit_rep, *sys_dev);
    } else if ((kh_put_status == UCS_KH_PUT_BUCKET_EMPTY) ||
               (kh_put_status == UCS_KH_PUT_BUCKET_CLEAR)) {
        *sys_dev = ucs_topo_ctx.sys_dev_to_bus_lookup.count;
        kh_value(&ucs_topo_ctx.bus_to_sys_dev_hash, hash_it) = *sys_dev;
        ucs_debug("bus id %ld doesn't exist. sys_dev = %u", bus_id_bit_rep,
                  *sys_dev);

        ucs_topo_ctx.sys_dev_to_bus_lookup.bus_arr[*sys_dev] = *bus_id;
        ucs_topo_ctx.sys_dev_to_bus_lookup.count++;
    }

    ucs_spin_unlock(&ucs_topo_ctx.lock);
    return UCS_OK;
}


ucs_status_t ucs_topo_get_distance(ucs_sys_device_t device1,
                                   ucs_sys_device_t device2,
                                   ucs_sys_dev_distance_t *distance)
{
    return UCS_OK;
}


void ucs_topo_print_info(FILE *stream)
{
}
