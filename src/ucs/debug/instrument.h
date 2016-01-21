/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#ifndef INSTRUMENT_H_
#define INSTRUMENT_H_

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/sys/preprocessor.h>
#include <ucs/datastruct/list.h>
#include <ucs/time/time.h>


typedef enum ucs_instrumentation_types {
    UCS_INSTRUMENT_TYPE_IB_TX,
    UCS_INSTRUMENT_TYPE_IB_RX,

    UCS_INSTRUMENT_TYPE_LAST
} ucs_instrumentation_types_t;

/**
 * Initialize instrumentation.
 */
void ucs_instrument_init();

/**
 * Save and cleanup instrumentation.
 */
void ucs_instrument_cleanup();


/**
 * Instrumentation file header
 */
typedef struct ucs_instrument_header {
    struct {
        char                 path[1024];    /* UCS library path for location information */
        uint32_t             chksum;        /* UCS library checksum */
        unsigned long        base;          /* UCS library loading base */
    } ucs_lib;

    struct {
        char                 cmdline[1024]; /* Command line */
        int                  pid;           /* Process ID */
        char                 hostname[40];  /* Host name */
    } app;

    size_t                   num_records;   /* Number of records in the file */
    size_t                   num_locations; /* Number of locations in the file */
    size_t                   record_offset; /* File starts from the n-th record in the program */
    ucs_time_t               start_time;    /* Time when application has started */
    ucs_time_t               one_second;    /* How much time is one second on the sampled machine */
    ucs_time_t               one_record;    /* How much time is one record on the sampled machine */
} ucs_instrument_header_t;


typedef struct ucs_instrument_record {
    uint64_t                 timestamp;
    uint64_t                 lparam;
    uint32_t                 wparam;
    uint32_t                 location;
} ucs_instrument_record_t;


typedef struct ucs_instrument_location {
    uint32_t                 location;      /* Location identifier in records */
    char                     name[256];     /* Name of the location */
    ucs_list_link_t          list;
} ucs_instrument_location_t;


typedef struct ucs_instrument_context {
    unsigned                 enabled;
    uint32_t                 next_location_id;
    ucs_list_link_t          locations_head;
    ucs_time_t               start_time;
    ucs_time_t               one_record_time;
    ucs_instrument_record_t  *start, *end;
    ucs_instrument_record_t  *current;
    size_t                   count;
    int                      fd;
} ucs_instrument_context_t;


/* Expand a 32-bit location address to full size address, assuming library
 * size is <2G.
 * @param location  Abbreviated 32-bit location
 * @param base      UCS library load base address.
 */
static inline unsigned long ucs_instrument_expand_location(uint32_t location,
                                                           unsigned long base)
{
    uint32_t offset = (location - (uint32_t)base) & UCS_MASK(31);
    return base + offset;
}


#if HAVE_INSTRUMENTATION

/*
 * Global instrumentation context.
 */
extern ucs_instrument_context_t ucs_instr_ctx;

/*
 * Register an instrumentation record - should be called once per record
 * mentioned in the code, before the first record of each such mention is made.
 *
 * @param type        Instrumentation record type (to check if enabled).
 * @param type_name   Instrumentation record type as a string.
 * @param custom_name Instrumentation record custom name string.
 * @param file_name   Instrumentation record file name string.
 * @param line_number Instrumentation record line number (in the file).
 *
 * @return 0 for disabled record, positive instrumentation record id otherwise.
 */
uint32_t ucs_instrument_register(ucs_instrumentation_types_t type,
                                 char const *type_name,
                                 char const *custom_name,
                                 char const *file_name,
                                 unsigned line_number);

/*
 * Store a new record with the given data.
 *
 * @param location Location id (provided by @ref ucs_instrument_register .
 * @param lparam   64-bit user-defined data.
 * @param wparam   32-bit user-defined data.
 */
void ucs_instrument_record(uint32_t location, uint64_t lparam, uint32_t wparam);


/*
 * Helper macros
 */
#define UCS_INSTRUMENT_RECORD_(_type, _name, _lparam, _wparam, ...) \
    do { \
        static uint32_t instrument_id = (uint32_t)-1; \
        if (instrument_id) { \
            if (ucs_unlikely(instrument_id == (uint32_t)-1)) { \
                instrument_id = ucs_instrument_register(_type, #_type, \
                                                        _name, __FILE__,\
                                                        __LINE__); \
                if (instrument_id == 0) { \
                    break; \
                } \
            } \
            ucs_instrument_record(instrument_id, \
                                  (uint64_t)(_lparam), \
                                  (uint32_t)(_wparam)); \
        } \
    } while (0)

/*
 * Main instrumentation macro
 */
#define UCS_INSTRUMENT_RECORD(...) UCS_INSTRUMENT_RECORD_(__VA_ARGS__, 0, 0)

#else

#define UCS_INSTRUMENT_RECORD(...)  do {} while (0)

#endif

#endif
