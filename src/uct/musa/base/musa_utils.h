/*
 * Copyright (C) MThreads 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_MUSA_UTILS_H_
#define UCT_MUSA_UTILS_H_

#include <ucs/sys/compiler_def.h>
#include <ucs/sys/topo/base/topo.h>

#include <stdio.h>
#include <stddef.h>
#include <stdint.h>

BEGIN_C_DECLS

static UCS_F_ALWAYS_INLINE int
uct_musa_range_is_valid(uintptr_t base, size_t alloc_length,
                        uintptr_t address, size_t length)
{
    uintptr_t alloc_end;
    uintptr_t address_end;

    if ((alloc_length == 0) || (length == 0) || (address < base)) {
        return 0;
    }

    if (base > (UINTPTR_MAX - alloc_length)) {
        return 0;
    }
    alloc_end = base + alloc_length;

    if (address > (UINTPTR_MAX - length)) {
        return 0;
    }
    address_end = address + length;

    return address_end <= alloc_end;
}

static UCS_F_ALWAYS_INLINE int
uct_musa_parse_pci_bus_id(const char *pci_bus_id, ucs_sys_bus_id_t *bus_id)
{
    unsigned domain;
    unsigned bus;
    unsigned slot;
    unsigned function;
    char trailing;

    if ((pci_bus_id == NULL) || (bus_id == NULL)) {
        return 0;
    }

    if (sscanf(pci_bus_id, "%4x:%2x:%2x.%1x%c", &domain, &bus, &slot,
               &function, &trailing) != 4) {
        return 0;
    }

    if ((domain > UINT16_MAX) || (bus > UINT8_MAX) ||
        (slot > UINT8_MAX) || (function > UINT8_MAX)) {
        return 0;
    }

    bus_id->domain   = (uint16_t)domain;
    bus_id->bus      = (uint8_t)bus;
    bus_id->slot     = (uint8_t)slot;
    bus_id->function = (uint8_t)function;
    return 1;
}

END_C_DECLS

#endif
