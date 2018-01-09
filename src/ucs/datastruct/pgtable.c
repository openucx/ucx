/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "pgtable.h"

#include <ucs/arch/bitops.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/sys/math.h>
#include <string.h>


#define ucs_pgt_entry_clear(_pte) \
    { (_pte)->value = 0; }

#define ucs_pgt_entry_value(_pte) \
    ((void*)((_pte)->value & UCS_PGT_ENTRY_PTR_MASK))

#define ucs_pgt_entry_test(_pte, _flag) \
    ((_pte)->value & (_flag))

#define ucs_pgt_entry_present(_pte) \
    ucs_pgt_entry_test(_pte, UCS_PGT_ENTRY_FLAG_REGION | UCS_PGT_ENTRY_FLAG_DIR)

#define ucs_pgt_is_addr_aligned(_addr) \
    (!((_addr) & (UCS_PGT_ADDR_ALIGN - 1)))

#define ucs_pgt_check_ptr(_ptr) \
    do { \
        ucs_assertv(!((uintptr_t)(_ptr) & (UCS_PGT_ENTRY_MIN_ALIGN - 1)), \
                    "ptr=%p", (_ptr)); \
    } while (0)

#define ucs_pgt_entry_set_region(_pte, _region) \
    do { \
        ucs_pgt_region_t *tmp = (_region); \
        ucs_pgt_check_ptr(tmp); \
        (_pte)->value = ((uintptr_t)tmp) | UCS_PGT_ENTRY_FLAG_REGION; \
    } while (0)

#define ucs_pgt_entry_set_dir(_pte, _dir) \
    do { \
        ucs_pgt_dir_t *tmp = (_dir); \
        ucs_pgt_check_ptr(tmp); \
        (_pte)->value = ((uintptr_t)tmp) | UCS_PGT_ENTRY_FLAG_DIR; \
    } while (0)

#define ucs_pgt_entry_get_region(_pte) \
    ({ \
        ucs_assert(ucs_pgt_entry_test(_pte, UCS_PGT_ENTRY_FLAG_REGION)); \
        (ucs_pgt_region_t*)ucs_pgt_entry_value(_pte); \
    })

#define ucs_pgt_entry_get_dir(_pte) \
    ({ \
        ucs_assert(ucs_pgt_entry_test(_pte, UCS_PGT_ENTRY_FLAG_DIR)); \
        (ucs_pgt_dir_t*)ucs_pgt_entry_value(_pte); \
    })


static inline ucs_pgt_dir_t* ucs_pgt_dir_alloc(ucs_pgtable_t *pgtable)
{
    ucs_pgt_dir_t *pgd;

    pgd = pgtable->pgd_alloc_cb(pgtable);
    if (pgd == NULL) {
        ucs_fatal("Failed to allocate page table directory");
    }

    ucs_pgt_check_ptr(pgd);
    memset(pgd, 0, sizeof(*pgd));
    return pgd;
}

static inline void ucs_pgt_dir_release(ucs_pgtable_t *pgtable, ucs_pgt_dir_t* pgd)
{
    pgtable->pgd_release_cb(pgtable, pgd);
}

static void ucs_pgt_entry_dump_recurs(const ucs_pgtable_t *pgtable, unsigned indent,
                                      const ucs_pgt_entry_t *pte, unsigned pte_index,
                                      ucs_pgt_addr_t base, ucs_pgt_addr_t mask,
                                      unsigned shift, ucs_log_level_t log_level)
{
    ucs_pgt_region_t *region;
    ucs_pgt_dir_t *pgd;
    size_t i;

    if (ucs_pgt_entry_test(pte, UCS_PGT_ENTRY_FLAG_REGION)) {
        region = ucs_pgt_entry_value(pte);
        ucs_log(log_level, "%*s[%3u] region " UCS_PGT_REGION_FMT, indent, "",
                pte_index, UCS_PGT_REGION_ARG(region));
    } else if (ucs_pgt_entry_test(pte, UCS_PGT_ENTRY_FLAG_DIR)) {
        pgd = ucs_pgt_entry_get_dir(pte);
        ucs_log(log_level, "%*s[%3u] dir %p for [0x%lx..0x%lx], count %u shift %u mask 0x%lx",
                indent, " ", pte_index, pgd, base, (base + (1 << shift)) & mask,
                pgd->count, shift, mask);
        shift -= UCS_PGT_ENTRY_SHIFT;
        mask  |= UCS_PGT_ENTRY_MASK << shift;
        for (i = 0; i < UCS_PGT_ENTRIES_PER_DIR; ++i) {
            ucs_pgt_entry_dump_recurs(pgtable, indent + 2, &pgd->entries[i], i,
                                      base | (i << shift), mask, shift, log_level);
            ++base;
        }
    } else {
        ucs_log(log_level, "%*s[%3u] not present", indent, " ", pte_index);
    }
}

static void ucs_pgtable_log(const ucs_pgtable_t *pgtable,
                            ucs_log_level_t log_level, const char *message)
{
    ucs_log(log_level, "pgtable %p %s: base 0x%lx/0x%lx shift %u count %u",
            pgtable, message, pgtable->base, pgtable->mask, pgtable->shift,
            pgtable->num_regions);
}

void ucs_pgtable_dump(const ucs_pgtable_t *pgtable, ucs_log_level_t log_level)
{
    ucs_pgtable_log(pgtable, log_level, "dump");
    ucs_pgt_entry_dump_recurs(pgtable, 0, &pgtable->root, 0, pgtable->base,
                              pgtable->mask, pgtable->shift, log_level);
}

static void ucs_pgtable_trace(ucs_pgtable_t *pgtable, const char *message)
{
    ucs_pgtable_log(pgtable, UCS_LOG_LEVEL_TRACE_DATA, message);
}

static void ucs_pgtable_reset(ucs_pgtable_t *pgtable)
{
    pgtable->base  = 0;
    pgtable->mask  = ((ucs_pgt_addr_t)-1) << UCS_PGT_ADDR_SHIFT;
    pgtable->shift = UCS_PGT_ADDR_SHIFT;
}

/**
 * Make the page table map a wider range of addresses - expands by UCS_PGT_ENTRY_SHIFT.
 */
static void ucs_pgtable_expand(ucs_pgtable_t *pgtable)
{
    ucs_pgt_dir_t *pgd;

    ucs_assertv(pgtable->shift <= (UCS_PGT_ADDR_ORDER - UCS_PGT_ENTRY_SHIFT),
                "shift=%u", pgtable->shift);

    if (ucs_pgt_entry_present(&pgtable->root)) {
        pgd = ucs_pgt_dir_alloc(pgtable);
        pgd->entries[(pgtable->base >> pgtable->shift) & UCS_PGT_ENTRY_MASK] =
                        pgtable->root;
        pgd->count = 1;
        ucs_pgt_entry_set_dir(&pgtable->root, pgd);
    }

    pgtable->shift += UCS_PGT_ENTRY_SHIFT;
    pgtable->mask <<= UCS_PGT_ENTRY_SHIFT;
    pgtable->base  &= pgtable->mask;
    ucs_pgtable_trace(pgtable, "expand");
}

/**
 * Shrink the page table address span if possible
 *
 * @return Whether it was shrinked.
 */
static int ucs_pgtable_shrink(ucs_pgtable_t *pgtable)
{
    ucs_pgt_entry_t *pte;
    ucs_pgt_dir_t *pgd;
    unsigned pte_idx;

    if (!ucs_pgt_entry_present(&pgtable->root)) {
        ucs_pgtable_reset(pgtable);
        ucs_pgtable_trace(pgtable, "shrink");
        return 0;
    } else if (!ucs_pgt_entry_test(&pgtable->root, UCS_PGT_ENTRY_FLAG_DIR)) {
        return 0;
    }

    pgd = ucs_pgt_entry_get_dir(&pgtable->root);
    ucs_assert(pgd->count > 0); /* should be empty */

    /* If there is just one PTE, we can reduce the page table to map
     * this PTE only.
     */
    if (pgd->count != 1) {
        return 0;
    }

    /* Search for the single PTE in dir */
    for (pte_idx = 0, pte = pgd->entries; !ucs_pgt_entry_present(pte); ++pte_idx, ++pte) {
        ucs_assert(pte_idx < UCS_PGT_ENTRIES_PER_DIR);
    }

    /* Remove one level */
    pgtable->shift -= UCS_PGT_ENTRY_SHIFT;
    pgtable->base  |= (ucs_pgt_addr_t)pte_idx << pgtable->shift;
    pgtable->mask  |= UCS_PGT_ENTRY_MASK << pgtable->shift;
    pgtable->root   = *pte;
    ucs_pgtable_trace(pgtable, "shrink");
    ucs_pgt_dir_release(pgtable, pgd);
    return 1;
}

static void ucs_pgtable_check_page(ucs_pgt_addr_t address, unsigned order)
{
    ucs_assert( (address & ((1ul << order) - 1)) == 0 );
    ucs_assertv( ((order - UCS_PGT_ADDR_SHIFT) % UCS_PGT_ENTRY_SHIFT) == 0, "order=%u", order);
}

/**
 * @return Order of the next whole page starting in "start" and ending before "end"
 *         If both are 0, return the full word size.
 */
static unsigned ucs_pgtable_get_next_page_order(ucs_pgt_addr_t start, ucs_pgt_addr_t end)
{
    unsigned log2_len;

    ucs_assertv(ucs_pgt_is_addr_aligned(start), "start=0x%lx", start);
    ucs_assertv(ucs_pgt_is_addr_aligned(end),   "end=0x%lx",   end);

    if (end - start == 0) {
        log2_len = UCS_PGT_ADDR_ORDER; /* entire range */
    } else {
        log2_len = ucs_ilog2(end - start);
    }
    if (start != 0) {
        log2_len = ucs_min(ucs_ffs64(start), log2_len);
    }
    ucs_assertv(log2_len >= UCS_PGT_ADDR_SHIFT, "log2_len=%u start=0x%lx end=0x%lx",
                log2_len, start, end);

    /* Order should be: [ADDR_SHIFT + k * ENTRY_SHIFT] */
    return (((log2_len - UCS_PGT_ADDR_SHIFT) / UCS_PGT_ENTRY_SHIFT) * UCS_PGT_ENTRY_SHIFT)
            + UCS_PGT_ADDR_SHIFT;
}

/**
 * Insert a variable-size page to the page table.
 *
 * @param address  address to insert
 * @param order    page size to insert - should be k*PTE_SHIFT for a certain k
 * @param region   region to insert
 */
static ucs_status_t
ucs_pgtable_insert_page(ucs_pgtable_t *pgtable, ucs_pgt_addr_t address,
                        unsigned order, ucs_pgt_region_t *region)
{
    ucs_pgt_dir_t dummy_pgd;
    ucs_pgt_entry_t *pte;
    ucs_pgt_dir_t *pgd;
    unsigned shift;

    ucs_pgtable_check_page(address, order);

    ucs_trace_data("insert page 0x%lx order %u, for region " UCS_PGT_REGION_FMT,
                   address, order, UCS_PGT_REGION_ARG(region));

    /* Make root map addresses which include our interval */
    while (pgtable->shift < order) {
        ucs_pgtable_expand(pgtable);
    }

    if (ucs_pgt_entry_present(&pgtable->root)) {
        while ((address & pgtable->mask) != pgtable->base) {
            ucs_pgtable_expand(pgtable);
        }
    } else {
        pgtable->base = address & pgtable->mask;
        ucs_pgtable_trace(pgtable, "initialize");
    }

    /* Insert the page in the PTE */
    pgd   = &dummy_pgd;
    shift = pgtable->shift;
    pte   = &pgtable->root;
    while (1) {
        if (order == shift) {
            if (ucs_pgt_entry_present(pte)) {
                goto err;
            }
            ucs_pgt_entry_set_region(pte, region);
            ++pgd->count;
            break;
        } else {
            ucs_assert(!ucs_pgt_entry_test(pte, UCS_PGT_ENTRY_FLAG_REGION));
            ucs_assertv(shift >= UCS_PGT_ENTRY_SHIFT + order,
                        "shift=%u order=%u", shift, order);  /* sub PTE should be able to hold it */

            if (!ucs_pgt_entry_present(pte)) {
                ++pgd->count;
                ucs_pgt_entry_set_dir(pte, ucs_pgt_dir_alloc(pgtable));
            }

            pgd    = ucs_pgt_entry_get_dir(pte);
            shift -= UCS_PGT_ENTRY_SHIFT;
            pte    = &pgd->entries[(address >> shift) & UCS_PGT_ENTRY_MASK];
        }
    }

    return UCS_OK;

err:
    while (ucs_pgtable_shrink(pgtable));
    return UCS_ERR_ALREADY_EXISTS;
}

/*
 * `region' is only used to compare pointers
 */
static ucs_status_t
ucs_pgtable_remove_page_recurs(ucs_pgtable_t *pgtable, ucs_pgt_addr_t address,
                               unsigned order, ucs_pgt_dir_t *pgd,
                               ucs_pgt_entry_t *pte, unsigned shift,
                               ucs_pgt_region_t *region)
{
    ucs_pgt_dir_t *next_dir;
    ucs_pgt_entry_t *next_pte;
    ucs_status_t status;
    unsigned next_shift;

    if (ucs_pgt_entry_test(pte, UCS_PGT_ENTRY_FLAG_REGION)) {
        ucs_assertv(shift == order, "shift=%u order=%u", shift, order);
        if (ucs_pgt_entry_get_region(pte) != region) {
            goto no_elem;
        }

        --pgd->count;
        ucs_pgt_entry_clear(pte);
        return UCS_OK;
    } else if (ucs_pgt_entry_test(pte, UCS_PGT_ENTRY_FLAG_DIR)) {
        next_dir   = ucs_pgt_entry_get_dir(pte);
        next_shift = shift - UCS_PGT_ENTRY_SHIFT;
        next_pte   = &next_dir->entries[(address >> next_shift) & UCS_PGT_ENTRY_MASK];

        status = ucs_pgtable_remove_page_recurs(pgtable, address, order, next_dir,
                                                next_pte, next_shift, region);
        if (status != UCS_OK) {
            goto no_elem;
        }

        if (next_dir->count == 0) {
            ucs_pgt_entry_clear(pte);
            --pgd->count;
            ucs_pgt_dir_release(pgtable, next_dir);
        }
        return UCS_OK;
    }

no_elem:
    return UCS_ERR_NO_ELEM;
}

static ucs_status_t
ucs_pgtable_remove_page(ucs_pgtable_t *pgtable, ucs_pgt_addr_t address,
                        unsigned order, ucs_pgt_region_t *region)
{
    ucs_pgt_dir_t dummy_dir;
    ucs_status_t status;

    ucs_pgtable_check_page(address, order);

    if ((address & pgtable->mask) != pgtable->base) {
        return UCS_ERR_NO_ELEM;
    }

    status = ucs_pgtable_remove_page_recurs(pgtable, address, order, &dummy_dir,
                                            &pgtable->root, pgtable->shift,
                                            region);
    if (status != UCS_OK) {
        return status;
    }

    while (ucs_pgtable_shrink(pgtable));
    return UCS_OK;
}

ucs_status_t ucs_pgtable_insert(ucs_pgtable_t *pgtable, ucs_pgt_region_t *region)
{
    ucs_pgt_addr_t address = region->start;
    ucs_pgt_addr_t end     = region->end;
    ucs_status_t status;
    unsigned order;

    ucs_trace_data("add region " UCS_PGT_REGION_FMT, UCS_PGT_REGION_ARG(region));

    if ((address >= end) || !ucs_pgt_is_addr_aligned(address) ||
        !ucs_pgt_is_addr_aligned(end))
    {
        return UCS_ERR_INVALID_PARAM;
    }

    ucs_assert(address != end);
    while (address < end) {
        order = ucs_pgtable_get_next_page_order(address, end);
        status = ucs_pgtable_insert_page(pgtable, address, order, region);
        if (status != UCS_OK) {
            goto err;
        }
        address += 1ul << order;
    }
    ++pgtable->num_regions;

    ucs_pgtable_trace(pgtable, "insert");
    return UCS_OK;

err:
    /* Revert all pages we've inserted by now */
    end     = address;
    address = region->start;
    while (address < end) {
        order = ucs_pgtable_get_next_page_order(address, end);
        ucs_pgtable_remove_page(pgtable, address, order, region);
        address += 1ul << order;
    }
    return status;
}

ucs_status_t ucs_pgtable_remove(ucs_pgtable_t *pgtable, ucs_pgt_region_t *region)
{
    ucs_pgt_addr_t address = region->start;
    ucs_pgt_addr_t end     = region->end;
    ucs_status_t status;
    unsigned order;

    ucs_trace_data("remove region " UCS_PGT_REGION_FMT, UCS_PGT_REGION_ARG(region));

    if ((address >= end) || !ucs_pgt_is_addr_aligned(address) ||
        !ucs_pgt_is_addr_aligned(end))
    {
        return UCS_ERR_NO_ELEM;
    }

    while (address < end) {
        order = ucs_pgtable_get_next_page_order(address, end);
        status = ucs_pgtable_remove_page(pgtable, address, order, region);
        if (status != UCS_OK) {
            ucs_assert(address == region->start); /* Cannot be partially removed */
            return status;
        }
        address += 1ul << order;
    }

    ucs_assert(pgtable->num_regions > 0);
    --pgtable->num_regions;

    ucs_pgtable_trace(pgtable, "remove");
    return UCS_OK;
}

ucs_pgt_region_t *ucs_pgtable_lookup(const ucs_pgtable_t *pgtable,
                                     ucs_pgt_addr_t address)
{
    const ucs_pgt_entry_t *pte;
    ucs_pgt_region_t *region;
    ucs_pgt_dir_t *dir;
    unsigned shift;

    ucs_trace_func("pgtable=%p address=0x%lx", pgtable, address);

    /* Check if the address is mapped by the page table */
    if ((address & pgtable->mask) != pgtable->base) {
        return NULL;
    }

    /* Descend into the page table */
    pte   = &pgtable->root;
    shift = pgtable->shift;
    for (;;) {
        if (ucs_pgt_entry_test(pte, UCS_PGT_ENTRY_FLAG_REGION)) {
            region = ucs_pgt_entry_get_region(pte);
            ucs_assert((address >= region->start) && (address < region->end));
            return region;
        } else if (ucs_pgt_entry_test(pte, UCS_PGT_ENTRY_FLAG_DIR)) {
            dir = ucs_pgt_entry_get_dir(pte);
            shift -= UCS_PGT_ENTRY_SHIFT;
            pte = &dir->entries[(address >> shift) & UCS_PGT_ENTRY_MASK];
        } else {
            return NULL;
        }
    }
}

static void ucs_pgtable_search_recurs(const ucs_pgtable_t *pgtable,
                                      ucs_pgt_addr_t address, unsigned order,
                                      const ucs_pgt_entry_t *pte, unsigned shift,
                                      ucs_pgt_search_callback_t cb, void *arg,
                                      ucs_pgt_region_t **last_p)
{
    ucs_pgt_entry_t *next_pte;
    ucs_pgt_region_t *region;
    ucs_pgt_dir_t *dir;
    unsigned next_shift;
    unsigned i;

    if (ucs_pgt_entry_test(pte, UCS_PGT_ENTRY_FLAG_REGION)) {
        region = ucs_pgt_entry_value(pte);

        /* Check that we are not continuing with the previous region */
        if (*last_p == region) {
            return;
        } else if (*last_p != NULL) {
            ucs_assert(region->start >= (*last_p)->end);
        }
        *last_p = region;

        /* Assert that the region actually overlaps the address */
        ucs_assertv(ucs_max(region->start,   address) <=
                    ucs_min(region->end - 1, address + UCS_MASK_SAFE(order)),
                    UCS_PGT_REGION_FMT " address=0x%lx order=%d mask 0x%lx",
                    UCS_PGT_REGION_ARG(region), address, order,
                    (ucs_pgt_addr_t)UCS_MASK_SAFE(order));

        /* Call the callback */
        cb(pgtable, region, arg);

    } else if (ucs_pgt_entry_test(pte, UCS_PGT_ENTRY_FLAG_DIR)) {
        dir = ucs_pgt_entry_get_dir(pte);
        ucs_assert(shift >= UCS_PGT_ENTRY_SHIFT);
        next_shift = shift - UCS_PGT_ENTRY_SHIFT;

        if (order < shift) {
            /* One of the sub-ptes maps the region */
            ucs_assert(order <= next_shift);
            next_pte = &dir->entries[(address >> next_shift) & UCS_PGT_ENTRY_MASK];
            ucs_pgtable_search_recurs(pgtable, address, order, next_pte,
                                      next_shift, cb, arg, last_p);
        } else {
            /* All sub-ptes contained in the region */
            for (i = 0; i < UCS_PGT_ENTRIES_PER_DIR; ++i) {
                next_pte = &dir->entries[i];
                ucs_pgtable_search_recurs(pgtable, address, order, next_pte,
                                          next_shift, cb, arg, last_p);
            }
        }
    }
}

void ucs_pgtable_search_range(const ucs_pgtable_t *pgtable,
                              ucs_pgt_addr_t from, ucs_pgt_addr_t to,
                              ucs_pgt_search_callback_t cb, void *arg)
{
    ucs_pgt_addr_t address = ucs_align_down_pow2(from, UCS_PGT_ADDR_ALIGN);
    ucs_pgt_addr_t end     = ucs_align_up_pow2(to, UCS_PGT_ADDR_ALIGN);
    ucs_pgt_region_t *last;
    unsigned order = 0;

    last = NULL;
    while ((address <= to) && (order != UCS_PGT_ADDR_ORDER)) {
        order = ucs_pgtable_get_next_page_order(address, end);
        if ((address & pgtable->mask) == pgtable->base) {
            ucs_pgtable_search_recurs(pgtable, address, order, &pgtable->root,
                                      pgtable->shift, cb, arg, &last);
        }
        address += 1ul << order;
    }
}

static void ucs_pgtable_purge_callback(const ucs_pgtable_t *pgtable,
                                       ucs_pgt_region_t *region,
                                       void *arg)
{
    ucs_pgt_region_t ***region_pp = arg;
    **region_pp = region;
    ++(*region_pp);
}

void ucs_pgtable_purge(ucs_pgtable_t *pgtable, ucs_pgt_search_callback_t cb,
                       void *arg)
{
    unsigned num_regions = pgtable->num_regions;
    ucs_pgt_region_t **all_regions, **next_region, *region;
    ucs_pgt_addr_t from, to;
    unsigned i;

    all_regions = ucs_calloc(num_regions, sizeof(*all_regions),
                             "pgt_purge_regions");
    if (all_regions == NULL) {
        ucs_warn("failed to allocate array to collect all regions, will leak");
        return;
    }

    next_region = all_regions;
    from = pgtable->base;
    to   = pgtable->base + ((1ul << pgtable->shift) & pgtable->mask) - 1;
    ucs_pgtable_search_range(pgtable, from, to, ucs_pgtable_purge_callback,
                             &next_region);
    ucs_assertv(next_region == all_regions + num_regions,
                "next_region=%p all_regions=%p num_regions=%u",
                next_region, all_regions, num_regions);

    for (i = 0; i < num_regions; ++i) {
        region = all_regions[i];
        ucs_pgtable_remove(pgtable, region);
        cb(pgtable, region, arg);
    }

    ucs_free(all_regions);

    /* Page table should be totally empty */
    ucs_assert(!ucs_pgt_entry_present(&pgtable->root));
    ucs_assertv(pgtable->shift       == UCS_PGT_ADDR_SHIFT, "shift=%u", pgtable->shift);
    ucs_assertv(pgtable->base        == 0, "value=0x%lx", pgtable->base);
    ucs_assertv(pgtable->num_regions == 0, "num_regions=%u", pgtable->num_regions);
}

ucs_status_t ucs_pgtable_init(ucs_pgtable_t *pgtable,
                              ucs_pgt_dir_alloc_callback_t alloc_cb,
                              ucs_pgt_dir_release_callback_t release_cb)
{
    UCS_STATIC_ASSERT(ucs_is_pow2(UCS_PGT_ENTRY_MIN_ALIGN));

    /* ADDR_MAX+1 must be power of 2, or wrap around to 0. */
    UCS_STATIC_ASSERT(ucs_is_pow2_or_zero(UCS_PGT_ADDR_MAX + 1));

    /* We must cover all bits of the address up to ADDR_MAX */
    UCS_STATIC_ASSERT(((ucs_ilog2(UCS_PGT_ADDR_MAX) + 1 - UCS_PGT_ADDR_SHIFT) %
                      UCS_PGT_ENTRY_SHIFT) == 0);

    ucs_pgt_entry_clear(&pgtable->root);
    ucs_pgtable_reset(pgtable);
    pgtable->num_regions    = 0;
    pgtable->pgd_alloc_cb   = alloc_cb;
    pgtable->pgd_release_cb = release_cb;
    return UCS_OK;
}

void ucs_pgtable_cleanup(ucs_pgtable_t *pgtable)
{
    if (pgtable->num_regions != 0) {
        ucs_warn("page table not empty during cleanup");
    }
}
