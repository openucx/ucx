/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_LIBSTATS_H_
#define UCS_LIBSTATS_H_

#include <ucs/stats/stats_fwd.h>
#include <ucs/datastruct/list.h>
#include <ucs/type/status.h>
#include <ucs/sys/math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>


/*
 * Serialization options
 */
enum {
    UCS_STATS_SERIALIZE_INACTVIVE = UCS_BIT(0),  /* Use "inactive" tree */
    UCS_STATS_SERIALIZE_BINARY    = UCS_BIT(1),  /* Binary mode */
    UCS_STATS_SERIALIZE_COMPRESS  = UCS_BIT(2)   /* Compress */
};

#define UCS_STATS_DEFAULT_UDP_PORT 37873


#define UCS_STAT_NAME_MAX          31

#define UCS_STATS_NODE_FMT \
    "%s%s"
#define UCS_STATS_NODE_ARG(_node) \
    (_node)->cls->name, (_node)->name

#define UCS_STATS_INDENT(_is_sum, _indent)  _is_sum ? 0 : (_indent) * 2, ""
#define UCS_STATS_IS_LAST_COUNTER(_counters_bits, _current) \
    (_counters_bits > ((2ull<<_current) - 1))

typedef struct ucs_stats_server    *ucs_stats_server_h; /* Handle to server */
typedef struct ucs_stats_client    *ucs_stats_client_h; /* Handle to client */


typedef enum ucs_stats_children_sel {
    UCS_STATS_INACTIVE_CHILDREN,
    UCS_STATS_ACTIVE_CHILDREN,
    UCS_STATS_CHILDREN_LAST
} ucs_stats_children_sel_t;


/* Statistics class */
struct ucs_stats_class {
    const char           *name;
    unsigned             num_counters;
    const char*          counter_names[];
};

/*
 * ucs_stats_node is used to hold the counters, their classes and the
 * relationship between them.
 * ucs_stats_filter_node is a data structure used to filter the counters
 * on the report.
 * Therre are 3 types of filtering: Full, Aggregate, and summary
 * Following is an example of the data structures in aggregate mode:
 *
 *      ucs_stats_node                        ucs_stats_filter_node
 *      --------------                        ---------------------
 *
 *             +-----+                              +-----+
 *             | A-1 |............................> | A-1 |      
 *             +-----+                              +-----+     
 *               A A A                                A 
 *               | | |                                | 
 *               | | +----------+                     |
 *               | |            |                     |
 *            +--+ +--+      +--+--+................. |
 *            |       |      | B-1 |                : |
 *            |       |      +-----+                : | 
 *            |    +-----+....|..A................. : |
 *            |    | B-2 |    |  |                : : |
 *            |    +-----+ <--+  |                V V |
 *       +-----+......|..........|.............> +-----+        
 *       | B-3 |      ||         +---------------| B*  |
 *       +-----+ <----+                          +-----+
 *       cntr1                                   cntr1*
 *       cntr2                                   cntr2*
 *       cntr3                                   cntr3*   
 *                               
 * unfiltered statistics report:
 * 
 *  A-1:
 *     B-1:
 *        cntr1: 11111
 *        cntr2: 22222
 *        cntr3: 33333
 *     B-2:
 *        cntr1: 11111
 *        cntr2: 22222
 *        cntr3: 33333
 *     B-3:
 *        cntr1: 11111
 *        cntr2: 22222
 *        cntr3: 33333
 *
 * filtered statistics report:
 * 
 *  A-1:
 *     B*:
 *        cntr1: 33333
 *        cntr2: 66666
 *        cntr3: 99999
 *
 */

/* In-memory statistics node */

struct ucs_stats_node {
    ucs_stats_class_t        *cls;               /* Class info */
    ucs_stats_node_t         *parent;            /* Hierachy structure */
    char                     name[UCS_STAT_NAME_MAX + 1];
                                                 /* instance name */
    ucs_list_link_t          list;               /* nodes sharing same parent */
    ucs_list_link_t          children[UCS_STATS_CHILDREN_LAST];
                                                 /* children list head */
    ucs_list_link_t          type_list;          /* nodes with same class/es
                                                    hierarchy */
    ucs_stats_filter_node_t  *filter_node;       /* ptr to type list head */
    ucs_stats_counter_t      counters[1];        /* instance counters */
};

struct ucs_stats_filter_node {
    ucs_stats_filter_node_t   *parent;
    ucs_list_link_t           list;               /* nodes sharing same parent.*/
    ucs_list_link_t           children;
    ucs_list_link_t           type_list_head;     /* nodes with same ancestors classes */
    int                       type_list_len;      /* length of list */
    int                       ref_count;          /* report node when non zero */
    uint64_t                  counters_bitmask;   /* which counters to print */
};

/**
 * Initialize statistics node.
 *
 * @param node  Node to initialize.
 * @param cls   Node class.
 * @param name  Node name format string.
 * @param ap    Name formatting arguments.
 */
ucs_status_t ucs_stats_node_initv(ucs_stats_node_t *node, ucs_stats_class_t *cls,
                                 const char *name, va_list ap);


/**
 * Serialize statistics.
 *
 * @param stream   Destination
 * @param root     Statistics node root.
 * @param options  Serialization options.
 */
ucs_status_t ucs_stats_serialize(FILE *stream, ucs_stats_node_t *root, int options);


/**
 * De-serialize statistics.
 *
 * @param stream   Source data.
 * @param p_roo    Filled with tatistics node root.
 *
 * @return UCS_ERR_NO_ELEM if hit EOF.
 */
ucs_status_t ucs_stats_deserialize(FILE *stream, ucs_stats_node_t **p_root);


/**
 * Release stats returned by ucs_stats_deserialize().
 * @param root     Stats to release.
 */
void ucs_stats_free(ucs_stats_node_t *root);


/**
 * Initialize statistics client.
 *
 * @param server_addr  Address of server machine.
 * @param port         Port number on server.
 * @param p_client     Filled with handle to the client.
 */
ucs_status_t ucs_stats_client_init(const char *server_addr, int port,
                                   ucs_stats_client_h *p_client);


/**
 * Destroy statistics client.
 */
void ucs_stats_client_cleanup(ucs_stats_client_h client);


/**
 * Send statistics.
 *
 * @param client     Client handle.
 * @param root       Statistics tree root.
 * @param timestamp  Current statistics timestamp, identifies every "snapshot".
 */
ucs_status_t ucs_stats_client_send(ucs_stats_client_h client, ucs_stats_node_t *root,
                                  uint64_t timestamp);


/**
 * Start a thread running a server which receives statistics.
 *
 * @param port       Port number to listen on. 0 - random available port.
 * @param verbose    Verbose level.
 * @param p_server   Filled with handle to the server.
 */
ucs_status_t ucs_stats_server_start(int port, ucs_stats_server_h *p_server);


/**
 * Stop statistics server.
 * @param server   Handle to statistics server.
 */
void ucs_stats_server_destroy(ucs_stats_server_h server);


/**
 * Get port number used by the server, useful if we started it on a random port.
 *
 * @param server   Handle to statistics server.
 *
 * @return Port number.
 */
int ucs_stats_server_get_port(ucs_stats_server_h server);


/**
 * Get current statistics gathered by the server. The data is valid until the next
 * call to any of the following functions:
 *  - ucs_stats_server_purge_stats
 *  - ucs_stats_server_cleanup
 *  - ucs_stats_server_get_stats
 *
 * @param server   Handle to statistics server.
 * @return A list of stat trees for all entities gathered by the server.
 */
ucs_list_link_t *ucs_stats_server_get_stats(ucs_stats_server_h server);


/**
 * Clean up existing statistics.
 */
void ucs_stats_server_purge_stats(ucs_stats_server_h server);


/**
 * @return Number of packets received by the server.
 */
unsigned long ucs_stats_server_rcvd_packets(ucs_stats_server_h server);


#endif /* LIBSTATS_H_ */
