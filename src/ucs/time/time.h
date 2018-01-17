/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_TIME_H
#define UCS_TIME_H

#include <ucs/arch/cpu.h>
#include <ucs/time/time_def.h>
#include <ucs/sys/math.h>
#include <sys/time.h>
#include <limits.h>

BEGIN_C_DECLS

/**
 * Short time type
 * Used to represent short time intervals, and takes less memory.
 */
typedef uint32_t             ucs_short_time_t;

/**
 * Compare short time values
 */
#define UCS_SHORT_TIME_CMP  UCS_CIRCULAR_COMPARE32


#define UCS_TIME_INFINITY  ULLONG_MAX

#define UCS_MSEC_PER_SEC   1000ull       /* Milli */
#define UCS_USEC_PER_SEC   1000000ul     /* Micro */
#define UCS_NSEC_PER_SEC   1000000000ul  /* Nano */


double ucs_get_cpu_clocks_per_sec();


/**
 * @return The current time, in UCS time units.
 */
static inline ucs_time_t ucs_get_time()
{
    return (ucs_time_t)ucs_arch_read_hres_clock();
}

/**
 * @return The clock value of a single second.
 */
static inline double ucs_time_sec_value()
{
    return ucs_get_cpu_clocks_per_sec();
}


/**
 * Convert seconds to UCS time units.
 */
static inline ucs_time_t ucs_time_from_sec(double sec)
{
    return (ucs_time_t)(sec * ucs_time_sec_value() + 0.5);
}

/**
 * Convert seconds to UCS time units.
 */
static inline ucs_time_t ucs_time_from_msec(double msec)
{
    return ucs_time_from_sec(msec / UCS_MSEC_PER_SEC);
}

/**
 * Convert seconds to UCS time units.
 */
static inline ucs_time_t ucs_time_from_usec(double usec)
{
    return ucs_time_from_sec(usec / UCS_USEC_PER_SEC);
}

/**
 * Convert UCS time units to seconds.
 */
static inline double ucs_time_to_sec(ucs_time_t time)
{
    return time / ucs_time_sec_value();
}

/**
 * Convert UCS time units to milliseconds.
 */
static inline double ucs_time_to_msec(ucs_time_t time)
{
    return ucs_time_to_sec(time) * UCS_MSEC_PER_SEC;
}

/**
 * Convert UCS time units to microseconds.
 */
static inline double ucs_time_to_usec(ucs_time_t time)
{
    return ucs_time_to_sec(time) * UCS_USEC_PER_SEC;
}

/**
 * Convert UCS time units to nanoseconds.
 */
static inline double ucs_time_to_nsec(ucs_time_t time)
{
    return ucs_time_to_sec(time) * UCS_NSEC_PER_SEC;
}

/**
 * Convert UCS time interval (small) to nanoseconds.
 */
static inline double ucs_time_interval_to_nsec(ucs_time_t time)
{
    return ucs_time_to_sec(time * UCS_NSEC_PER_SEC);
}

/* Convert seconds to POSIX timeval */
static inline void ucs_sec_to_timeval(double seconds, struct timeval *tv)
{
    int64_t usec = (int64_t)( (seconds * UCS_USEC_PER_SEC) + 0.5 );
    tv->tv_sec  = usec / UCS_USEC_PER_SEC;
    tv->tv_usec = usec % UCS_USEC_PER_SEC;
}

/* Convert seconds to POSIX timespec */
static inline void ucs_sec_to_timespec(double seconds, struct timespec *ts)
{
    int64_t nsec = (int64_t)( (seconds * UCS_NSEC_PER_SEC) + 0.5 );
    ts->tv_sec  = nsec / UCS_NSEC_PER_SEC;
    ts->tv_nsec = nsec % UCS_NSEC_PER_SEC;
}

END_C_DECLS

#endif
