/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_STATS_FD_H_
#define UCS_STATS_FD_H_

#include <stdint.h>

#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS

typedef uint64_t                          ucs_stats_counter_t;        /* Stats counter*/
typedef struct ucs_stats_class            ucs_stats_class_t;          /* Stats class */
typedef struct ucs_stats_node             ucs_stats_node_t;           /* Stats node */
typedef struct ucs_stats_filter_node      ucs_stats_filter_node_t;    /* Stats filter node */

typedef enum {
    UCS_STATS_FULL,        /* Full statistics report */
    UCS_STATS_FULL_AGG,    /* Full statistics report */
    UCS_STATS_SUMMARY,     /* Summary statistics report */
    UCS_STATS_LAST
} ucs_stats_formats_t;

extern const char *ucs_stats_formats_names[];
ucs_stats_node_t * ucs_stats_get_root();

END_C_DECLS

#endif /* STATS_FD_H_ */
