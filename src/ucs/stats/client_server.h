/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_STATS_CLIENT_SERVER_H_
#define UCS_STATS_CLIENT_SERVER_H_

#include <ucs/datastruct/list.h>

#include <arpa/inet.h>
#include <pthread.h>
#include <stdint.h>
#include <sys/time.h>

BEGIN_C_DECLS

#define UCS_STATS_ENTITY_HASH_SIZE 997


/* An entity which reports statistics */
typedef struct stats_entity stats_entity_t;
struct stats_entity {
    struct sockaddr_in in_addr; /* Entity address */
    uint64_t           timestamp; /* Current timestamp */
    size_t             buffer_size; /* Buffer size */
    void               *inprogress_buffer; /* Fragment assembly buffer */
    ucs_list_link_t    holes; /* List of holes in the buffer */
    stats_entity_t     *next; /* Hash link */

    pthread_mutex_t    lock;
    volatile unsigned  refcount;
    void               *completed_buffer; /* Completed buffer */
    struct timeval     update_time;
};


static UCS_F_ALWAYS_INLINE int
stats_entity_cmp(stats_entity_t *e1, stats_entity_t *e2)
{
    uint32_t a1 = ntohl(e1->in_addr.sin_addr.s_addr);
    uint32_t a2 = ntohl(e2->in_addr.sin_addr.s_addr);

    /* Cannot use subtraction with a1, a2: uint32_t difference may overflow int,
     * using direct comparison instead. */
    if (a1 > a2) {
        return 1;
    } else if (a1 < a2) {
        return -1;
    } else {
        return (int)ntohs(e1->in_addr.sin_port) -
               (int)ntohs(e2->in_addr.sin_port);
    }
}

static UCS_F_ALWAYS_INLINE int stats_entity_hash(stats_entity_t *e)
{
    return (((uint64_t)e->in_addr.sin_addr.s_addr << 16) +
            (uint64_t)ntohs(e->in_addr.sin_port)) %
           UCS_STATS_ENTITY_HASH_SIZE;
}

END_C_DECLS

#endif
