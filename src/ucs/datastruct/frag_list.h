/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_FRAG_LIST_H
#define UCS_FRAG_LIST_H

#include <ucs/debug/log.h>
#include <ucs/datastruct/queue.h>
#include <ucs/sys/math.h>
#include <ucs/stats/stats.h>


/*
 * The "frag list" is a data structure containing elements ordered by sequence number.
 * Elements can be added to in any order, and removed from the head (dequeued)
 * in strict serial number order.
 * It is used for ordering packets according to sequence number.
 *
 * Complexity:
 *  - O(1) for getting head element
 *  - O(Nelems) for memory, with the hard bound of sendwindowsize. In order insertion uses no memory. 
 *  - O(k) insertion, where k is number of holes. Number of holes is expected to be
 *  something like SendWindowSize/BurstPacketSize. With win 1024 and burst 16 we
 *  get to 64 holes. In reality the number should be much less because:
 *  - each route send 'bursts' in order
 *  - it takes roughly the same time for each route
 *  - number of routes (burst generatos is expected to be small)
 *  
 *  so in the end number of holes is proportional to number of routes and time difference 
 *  between alternative paths. Better math is welcome :P
 *
 *  Organization
 *
 *     min_sn
 *     head =list1-[hole]->list2-[hole]...->listn
 *      |       |             |               |
 *    ready  elemlist     elemlist          elemlist
 *    list
 *
 *   elemlists and ready list are sorted and continuos - no holes
 *   ready list contains elements that can be easily pulled: head->sn = read_list.last_sn
 */

/* Out-of-order handling type */
typedef enum {
    UCS_FRAG_LIST_INSERT_FAST,   /* in order insert, list empty */
    UCS_FRAG_LIST_INSERT_FIRST,  /* in order insert, list not empty, must try pull */
    UCS_FRAG_LIST_INSERT_SLOW,   /* out of order insert, can not pull elems from list */
    UCS_FRAG_LIST_INSERT_DUP,    /* old element, can not pull */
    UCS_FRAG_LIST_INSERT_READY,   /* in order insert, while we can still pull elems from list */
    UCS_FRAG_LIST_INSERT_FAIL    /* insert failed for some reason */
} ucs_frag_list_ooo_type_t;

/* Sequence number type */
/* NOTE: it must be same type as UD transport psn */
typedef uint16_t   ucs_frag_list_sn_t;
#define UCS_FRAG_LIST_SN_CMP UCS_CIRCULAR_COMPARE16

/**
 * C standard specifies that short integer is promoted to int
 * if there is an overflow. The following will be false when
 * uint16_t is used for serial number:
 * sn1=0; sn2=0xFFFF; sn1 == sn2+1
 *
 * So we must always use compare macro
 */

#define UCS_FRAG_LIST_NEXT_SN(sn) ((ucs_frag_list_sn_t)((sn)+1))
/* part of skb */
typedef struct ucs_frag_list_head {
    ucs_queue_head_t       list;
    ucs_frag_list_sn_t first_sn;
    ucs_frag_list_sn_t last_sn;
} ucs_frag_list_head_t;

typedef struct ucs_frag_list_elem_t {
    ucs_queue_elem_t         list;
    ucs_frag_list_head_t head;
} ucs_frag_list_elem_t;


/* part of connection */
typedef struct ucs_frag_list {
    ucs_queue_head_t       list;
    ucs_queue_head_t       ready_list;
    ucs_frag_list_sn_t     head_sn;
    unsigned               elem_count;   /* total number of list elements */
    unsigned               list_count;   /* number of independent lists */
    int                    max_holes;    /* do not allow insertion if ucs_list_count >= max_holes */
    UCS_STATS_NODE_DECLARE(stats);
#ifdef ENABLE_STATS
    ucs_frag_list_sn_t prev_sn;      /*  needed to detect busrts */
#endif
} ucs_frag_list_t;

/* stat counters */
enum {
    UCS_FRAG_LIST_STAT_GAPS,
    UCS_FRAG_LIST_STAT_GAP_LEN,
    UCS_FRAG_LIST_STAT_GAP_OUT,
    UCS_FRAG_LIST_STAT_BURSTS,
    UCS_FRAG_LIST_STAT_BURST_LEN,
    UCS_FRAG_LIST_STAT_LAST
};
 

/**
 * Initialize the frag_list.
 *
 * @param frag_list   frag_list to initialize.
 * @param initial_sn  Sequence number to start with. This first inserted element
 *                    should have this SN.
 * @param max_holes   Max number number of holes to allow on the list.
 *                    Currently we support:
 *                    0 - allow no holes, only check sn. Out of order insert
 *                    will result either in DUP or FAIL
 *                    -1 - infinite number of holes
 *
 */
ucs_status_t ucs_frag_list_init(ucs_frag_list_sn_t initial_sn, ucs_frag_list_t *frag_list,
                               int max_holes
                               UCS_STATS_ARG(ucs_stats_node_t *stats_parent));

/**
 * Cleanup the frag_list.
 */
void ucs_frag_list_cleanup(ucs_frag_list_t *head);


/* Slow path insert */
ucs_frag_list_ooo_type_t ucs_frag_list_insert_slow(ucs_frag_list_t *head,
                                                   ucs_frag_list_elem_t *elem,
                                                   ucs_frag_list_sn_t sn);


/**
 * pull element from the list
 * @return  NULL if list is empty or it is impossible to pull anything
 */
ucs_frag_list_elem_t *ucs_frag_list_pull_slow(ucs_frag_list_t *head);


/**
 * Dump frag list structure for debug purposes.
 */
void ucs_frag_list_dump(ucs_frag_list_t *head, int how);


static inline ucs_frag_list_sn_t ucs_frag_list_sn(ucs_frag_list_t *head) 
{
    return head->head_sn;
}

static inline void ucs_frag_list_sn_inc(ucs_frag_list_t *head)
{
    head->head_sn++;
}

static inline unsigned ucs_frag_list_count(ucs_frag_list_t *head)
{
    return head->elem_count;
}

static inline int ucs_frag_list_empty(ucs_frag_list_t *head)
{
    return ucs_queue_is_empty(&head->list) && ucs_queue_is_empty(&head->ready_list);
}

static inline ucs_frag_list_ooo_type_t
ucs_frag_list_insert(ucs_frag_list_t *head, ucs_frag_list_elem_t *elem,
                     ucs_frag_list_sn_t sn)
{
#if ENABLE_STATS
    ucs_frag_list_ooo_type_t ret;

    if (UCS_FRAG_LIST_SN_CMP(sn, >, head->head_sn)) {
        if (UCS_FRAG_LIST_SN_CMP(head->prev_sn + 1, !=,sn)) {
            UCS_STATS_UPDATE_COUNTER(head->stats, UCS_FRAG_LIST_STAT_BURSTS, 1);
        } else if (ucs_unlikely(UCS_STATS_GET_COUNTER(head->stats, UCS_FRAG_LIST_STAT_BURST_LEN) == 0)) {
            /* initial burst */
            UCS_STATS_SET_COUNTER(head->stats, UCS_FRAG_LIST_STAT_BURSTS, 1);
        }
        UCS_STATS_UPDATE_COUNTER(head->stats, UCS_FRAG_LIST_STAT_BURST_LEN, 1);
        head->prev_sn = sn;
    }
#endif
    /* in order arrival on empty list - inc sn and do nothing */
    if (ucs_likely(UCS_FRAG_LIST_SN_CMP(sn, ==, head->head_sn + 1) && (head->elem_count == 0))) {
        head->head_sn = sn;
        return UCS_FRAG_LIST_INSERT_FAST;
    }

    /* return either dup or slow */
#if ENABLE_STATS
    ret = ucs_frag_list_insert_slow(head, elem, sn);
    UCS_STATS_UPDATE_COUNTER(head->stats, UCS_FRAG_LIST_STAT_GAP_OUT, 
                             ret != UCS_FRAG_LIST_INSERT_DUP ? head->list_count : 0);
    return ret;
#else
    return ucs_frag_list_insert_slow(head, elem, sn);
#endif
}

static inline ucs_frag_list_elem_t *ucs_frag_list_pull(ucs_frag_list_t *head)
{
    if (!ucs_queue_is_empty(&head->ready_list)) {
        --head->elem_count;
        return ucs_queue_pull_elem_non_empty(&head->ready_list, ucs_frag_list_elem_t, list);
    } else if (!ucs_queue_is_empty(&head->list)) {
        return ucs_frag_list_pull_slow(head);
    } else {
        return NULL;
    }
}

#endif
