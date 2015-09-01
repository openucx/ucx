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
#include <ucs/time/time.h>


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
    size_t                   record_offset; /* File starts from the n-th record in the program */
    ucs_time_t               start_time;    /* Time when application has started */
    ucs_time_t               one_second;    /* How much time is one second on the sampled machine */
} ucs_instrument_header_t;


typedef struct ucs_instrument_record {
    uint64_t                 timestamp;
    uint64_t                 lparam;
    uint32_t                 wparam;
    uint32_t                 location;
} ucs_instrument_record_t;


typedef struct ucs_instrument_context {
    int                      enable;
    ucs_time_t               start_time;
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


void __ucs_instrument_record(uint64_t location, uint64_t lparam, uint32_t wparam);


/*
 * Helper macros
 */
#define UCS_MAKE_LABEL(uniq) \
    label_##uniq
#define UCS_INSTRUMENT_RECORD_(uniq, lparam, wparam, ...) \
    do { \
        if (ucs_instr_ctx.enable) do { \
    UCS_MAKE_LABEL(uniq):\
            __ucs_instrument_record((uint64_t)&&UCS_MAKE_LABEL(uniq), (uint64_t)(lparam), (uint32_t)(wparam)); \
        } while (0); \
    } while (0)

/*
 * Main instrumentation macro
 */
#define UCS_INSTRUMENT_RECORD(...) \
    UCS_INSTRUMENT_RECORD_(UCS_PP_UNIQUE_ID, ## __VA_ARGS__, 0, 0)

#else

#define UCS_INSTRUMENT_RECORD(...)  do {} while (0)

#endif


#endif
