/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_PGTABLE_H_
#define UCS_PGTABLE_H_

#include <ucs/config/types.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/type/status.h>

/*
 * The Page Table data structure organizes non-overlapping regions of memory in
 * an efficient radix tree, optimized for large and/or aligned regions.
 *
 * A page table entry can point to either a region (indicated by setting the
 * UCS_PGT_PTE_FLAG_REGION bit), or another entry (indicated by UCS_PGT_PTE_FLAG_DIR),
 * or be empty - if none of these bits is set.
 *
 */



/* Address alignment requirements */
#define UCS_PGT_ADDR_SHIFT         4
#define UCS_PGT_ADDR_ALIGN         (1ul << UCS_PGT_ADDR_SHIFT)
#define UCS_PGT_ADDR_ORDER          (sizeof(ucs_pgt_addr_t) * 8)
#define UCS_PGT_ADDR_MAX           ((ucs_pgt_addr_t)-1)

/* Page table entry/directory constants */
#define UCS_PGT_ENTRY_SHIFT        4
#define UCS_PGT_ENTRIES_PER_DIR    (1ul << (UCS_PGT_ENTRY_SHIFT))
#define UCS_PGT_ENTRY_MASK         (UCS_PGT_ENTRIES_PER_DIR - 1)

/* Page table pointers constants and flags */
#define UCS_PGT_ENTRY_FLAG_REGION  UCS_BIT(0)
#define UCS_PGT_ENTRY_FLAG_DIR     UCS_BIT(1)
#define UCS_PGT_ENTRY_FLAGS_MASK   (UCS_PGT_ENTRY_FLAG_REGION|UCS_PGT_ENTRY_FLAG_DIR)
#define UCS_PGT_ENTRY_PTR_MASK     (~UCS_PGT_ENTRY_FLAGS_MASK)
#define UCS_PGT_ENTRY_MIN_ALIGN    (UCS_PGT_ENTRY_FLAGS_MASK + 1)

/* Declare a variable as aligned so it could be placed in page table entry */
#define UCS_PGT_ENTRY_V_ALIGNED    UCS_V_ALIGNED(UCS_PGT_ENTRY_MIN_ALIGN > sizeof(long) ? \
                                                 UCS_PGT_ENTRY_MIN_ALIGN : sizeof(long))


#define UCS_PGT_REGION_FMT            "%p [0x%lx..0x%lx]"
#define UCS_PGT_REGION_ARG(_region)   (_region), (_region)->start, (_region)->end


/* Define the address type */
typedef unsigned long              ucs_pgt_addr_t;

/* Forward declarations */
typedef struct ucs_pgtable         ucs_pgtable_t;
typedef struct ucs_pgt_region      ucs_pgt_region_t;
typedef struct ucs_pgt_entry       ucs_pgt_entry_t;
typedef struct ucs_pgt_dir         ucs_pgt_dir_t;


/**
 * Callback for allocating a page table directory.
 *
 * @param [in]  pgtable  Pointer to the page table to allocate the directory for.
 *
 * @return Pointer to newly allocated pgdir, or NULL if failed. The pointer must
 *         be aligned to UCS_PGT_ENTRY_ALIGN boundary.
 * */
typedef ucs_pgt_dir_t* (*ucs_pgt_dir_alloc_callback_t)(const ucs_pgtable_t *pgtable);


/**
 * Callback for releasing a page table directory.
 *
 * @param [in]  pgtable  Pointer to the page table to in which the directory was
 *                       allocated.
 * @param [in]  pgdir    Page table directory to release.
 */
typedef void (*ucs_pgt_dir_release_callback_t)(const ucs_pgtable_t *pgtable,
                                               ucs_pgt_dir_t *pgdir);


/**
 * Callback for searching for regions in the page table.
 *
 * @param [in]  pgtable  The page table.
 * @param [in]  region   Found region.
 * @param [in]  arg      User-defined argument.
 */
typedef void (*ucs_pgt_search_callback_t)(const ucs_pgtable_t *pgtable,
                                          ucs_pgt_region_t *region, void *arg);


/**
 * Memory region in the page table.
 * The structure itself, and the pointers in it, must be aligned to 2^PTR_SHIFT.
 */
struct ucs_pgt_region {
    ucs_pgt_addr_t                 start; /**< Region start address */
    ucs_pgt_addr_t                 end;   /**< Region end address */
} UCS_PGT_ENTRY_V_ALIGNED;


/**
 * Page table entry:
 *
 * +--------------------+---+---+
 * |    pointer (MSB)   | d | r |
 * +--------------------+---+---+
 * |                    |   |   |
 * 64                   2   1   0
 *
 */
struct ucs_pgt_entry {
    ucs_pgt_addr_t                 value;  /**< Pointer + type bits. Can point
                                                to either a @ref ucs_pgt_dir_t or
                                                a @ref ucs_pgt_region_t. */
};


/**
 * Page table directory.
 */
struct ucs_pgt_dir {
    ucs_pgt_entry_t                entries[UCS_PGT_ENTRIES_PER_DIR];
    unsigned                       count;       /**< Number of valid entries */
};


/* Page table structure */
struct ucs_pgtable {

    /* Maps addresses whose (63-shift) high bits equal to value
     * This means: value * (2**shift) .. value * (2**(shift+1)) - 1
     */
    ucs_pgt_entry_t                root;        /**< root entry */
    ucs_pgt_addr_t                 base;        /**< base address */
    ucs_pgt_addr_t                 mask;        /**< mask for page table address range */
    unsigned                       shift;       /**< page table address span is 2**shift */
    unsigned                       num_regions; /**< total number of regions */
    ucs_pgt_dir_alloc_callback_t   pgd_alloc_cb;
    ucs_pgt_dir_release_callback_t pgd_release_cb;
};


/**
 * Initialize a page table.
 *
 * @param [in]  pgtable     Page table to initialize.
 * @param [in]  alloc_cb    Callback that will be used to allocate page directory,
 *                           which is the basic building block of the page table
 *                           data structure. This may allow the page table functions
 *                           to be safe to use from memory allocation context.
 * @param [in]  release_cb  Callback to release memory which was allocated by alloc_cb.
 */
ucs_status_t ucs_pgtable_init(ucs_pgtable_t *pgtable,
                              ucs_pgt_dir_alloc_callback_t alloc_cb,
                              ucs_pgt_dir_release_callback_t release_cb);

/**
 * Cleanup the page table and release all associated memory.
 *
 * @param [in]  pgtable     Page table to initialize.
 */
void ucs_pgtable_cleanup(ucs_pgtable_t *pgtable);


/**
 * Add a memory region to the page table.
 *
 * @param [in]  pgtable     Page table to insert the region to.
 * @param [in]  region      Memory region to insert. The region must remain valid
 *                           and unchanged s long as it's in the page table.
 *
 * @return UCS_OK - region was added.
 *         UCS_ERR_INVALID_PARAM - memory region address in invalid (misaligned or empty)
 *         UCS_ERR_ALREADY_EXISTS - the region overlaps with existing region.
 *
 */
ucs_status_t ucs_pgtable_insert(ucs_pgtable_t *pgtable, ucs_pgt_region_t *region);


/**
 * Remove a memory region from the page table.
 *
 * @param [in]  pgtable     Page table to remove the region from.
 * @param [in]  region      Memory region to remove. This must be the same pointer
 *                           passed to @ref ucs_pgtable_insert.
 *
 * @return UCS_OK - region was removed.
 *         UCS_ERR_INVALID_PARAM - memory region address in invalid (misaligned or empty)
 *         UCS_ERR_ALREADY_EXISTS - the region overlaps with existing region.
 *
 */
ucs_status_t ucs_pgtable_remove(ucs_pgtable_t *pgtable, ucs_pgt_region_t *region);


/*
 * Find a region which contains the given address.
 *
 * @param [in]  pgtable     Page table to search the address in.
 * @param [in]  address     Address to search.
 *
 * @return Region which contains 'address', or NULL if not found.
 */
ucs_pgt_region_t *ucs_pgtable_lookup(const ucs_pgtable_t *pgtable,
                                     ucs_pgt_addr_t address);


/**
 * Search for all regions overlapping with a given address range.
 *
 * @param [in]  pgtable     Page table to search the range in.
 * @param [in]  from        Lower bound of the range.
 * @param [in]  to          Upper bound of the range (inclusive).
 * @param [in]  cb          Callback to be called for every region found.
 *                           The callback must not modify the page table.
 * @param [in]  arg         User-defined argument to the callback.
 */
void ucs_pgtable_search_range(const ucs_pgtable_t *pgtable,
                              ucs_pgt_addr_t from, ucs_pgt_addr_t to,
                              ucs_pgt_search_callback_t cb, void *arg);


/**
 * Remove all regions from the page table and call the provided callback for each.
 *
 * @param [in]  pgtable     Page table to clean up.
 * @param [in]  cb          Callback to be called for every region, after it (and
 *                           all others) are removed.
 *                           The callback must not modify the page table.
 * @param [in]  arg         User-defined argument to the callback.
 */
void ucs_pgtable_purge(ucs_pgtable_t *pgtable, ucs_pgt_search_callback_t cb,
                       void *arg);


/**
 * Dump page table to log.
 *
 * @param [in]  pgtable      Page table to dump.
 * @param [in]  log_level    Which log level to use.
 */
void ucs_pgtable_dump(const ucs_pgtable_t *pgtable, ucs_log_level_t log_level);


/**
 * @return >Number of regions currently present in the page table.
 */
static inline unsigned ucs_pgtable_num_regions(const ucs_pgtable_t *pgtable)
{
    return pgtable->num_regions;
}


#endif
