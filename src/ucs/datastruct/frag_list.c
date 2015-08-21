
/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "frag_list.h"

#if ENABLE_STATS

static ucs_stats_class_t ucs_frag_list_stats_class = {
    .name = "frag_list",
    .num_counters = UCS_FRAG_LIST_STAT_LAST,
    .counter_names = {
            [UCS_FRAG_LIST_STAT_GAPS]              = "gaps",
            [UCS_FRAG_LIST_STAT_GAP_LEN]           = "gap_len",
            [UCS_FRAG_LIST_STAT_GAP_OUT]           = "gap_out",
            [UCS_FRAG_LIST_STAT_BURSTS]            = "bursts",
            [UCS_FRAG_LIST_STAT_BURST_LEN]         = "burst_len",
    }
};
#endif

ucs_status_t ucs_frag_list_init(ucs_frag_list_sn_t initial_sn, ucs_frag_list_t *frag_list,
                        int max_holes
                        UCS_STATS_ARG(ucs_stats_node_t *stats_parent)
                        )
{
    ucs_status_t status;

    ucs_assert(max_holes == 0 || max_holes == -1);
    frag_list->head_sn = initial_sn;
    frag_list->elem_count = 0;
    frag_list->list_count = 0;
    frag_list->max_holes = max_holes;
    ucs_queue_head_init(&frag_list->list);
    ucs_queue_head_init(&frag_list->ready_list);

#if ENABLE_STATS
    frag_list->prev_sn = initial_sn;
#endif
    status = UCS_STATS_NODE_ALLOC(&frag_list->stats, &ucs_frag_list_stats_class,
                                 stats_parent);
    return status;
}

void ucs_frag_list_cleanup(ucs_frag_list_t *frag_list)
{
    ucs_assert(frag_list->elem_count == 0);
    ucs_assert(frag_list->list_count == 0);
    ucs_assert(ucs_queue_is_empty(&frag_list->list));
    ucs_assert(ucs_queue_is_empty(&frag_list->ready_list));
    UCS_STATS_NODE_FREE(frag_list->stats);
}

/*
 prevh--- h --- .. --- 
          |
          e
          |
          e
 replace h with new_h:

prevh --- new_h --- .. --- 
          |
          h
          |
          e
          |
          e

 */
static inline void
frag_list_replace_head(ucs_frag_list_t *frag_list, ucs_frag_list_elem_t *prevh,
                       ucs_frag_list_elem_t *h, ucs_frag_list_elem_t *new_h)
{
    ucs_frag_list_elem_t UCS_V_UNUSED *e;

    ucs_trace_data("replace=%u %u", (unsigned)h->head.first_sn-1,
                   (unsigned)h->head.last_sn);

    new_h->head.first_sn = h->head.first_sn-1;
    new_h->head.last_sn = h->head.last_sn;
    /* add new_h before h in holes list */ 
    /* take h from holes list */
    if (prevh == NULL) {
        e = ucs_queue_pull_elem_non_empty(&frag_list->list, ucs_frag_list_elem_t, list);
        ucs_assert(e == h);
        ucs_queue_push_head(&frag_list->list, &new_h->list);
    } else {
        prevh->list.next = &new_h->list;
        new_h->list.next = h->list.next;
        if (frag_list->list.ptail == &h->list.next) {
            frag_list->list.ptail = &new_h->list.next;
        }
    }

    /* chain h to the new hole head */
    ucs_queue_head_init(&new_h->head.list);
    ucs_queue_splice(&new_h->head.list, &h->head.list);
    ucs_queue_push_head(&new_h->head.list, &h->list);
}

/*
    ..--- h --- .. --- 
          |
          e

  add new element to h:

    ..--- h --- .. --- 
          |
          |
          e
          |
          elem

 */
static inline void frag_list_add_tail(ucs_frag_list_elem_t *h, ucs_frag_list_elem_t *elem)
{
    h->head.last_sn++;
    ucs_trace_data("add_tail=%u %u", (unsigned)h->head.first_sn, (unsigned)h->head.last_sn);

    /* chain h to the new hole head */
    ucs_queue_push(&h->head.list, &elem->list);
}

/* 
  merge h2 into h1. Before:

    ..--- h1 --- h2 --- 
          |     |
          e     e2
 after:
    ..--- h1 --- .. --- 
          |      |
          e      e
          |
          h2
          |
          e2
 */
static inline void frag_list_merge_heads(ucs_frag_list_t *head, ucs_frag_list_elem_t *h1, ucs_frag_list_elem_t *h2)
{
    ucs_trace_data("merge_heads=%u %u", (unsigned)h1->head.first_sn, (unsigned)h2->head.last_sn);

    h1->head.last_sn = h2->head.last_sn;
    h1->list.next = h2->list.next;
    if (head->list.ptail == &h2->list.next) {
        head->list.ptail = &h1->list.next;
    }

    /* turn h2 into queue element */
    ucs_queue_push_head(&h2->head.list, &h2->list);
    ucs_queue_splice(&h1->head.list, &h2->head.list);
}

/* 
  insert new_h into h1. Before:
 prevh--- h --- .. --- 
          |     |
          e     e
          |

 after:

 prevh--- new_h --- h --- ... ---
                    |      |
                    e      e
 */
static inline void frag_list_insert_head(ucs_frag_list_t *head, 
        ucs_frag_list_elem_t *prevh, ucs_frag_list_elem_t *h, ucs_frag_list_elem_t *new_h,  ucs_frag_list_sn_t sn)
{

    ucs_trace_data("insert_head=%u prevh=%p", (unsigned)sn, prevh);
    new_h->head.first_sn = new_h->head.last_sn = sn;
    ucs_queue_head_init(&new_h->head.list);

    if (prevh == NULL) {
        ucs_queue_push_head(&head->list, &new_h->list);
    }
    else {
        prevh->list.next = &new_h->list;
        new_h->list.next = &h->list;
    }
}


/* 
  insert new_h into h1. Before:
   ..--- prevh --- h --- 
          |        |
          e        e
          |

 after:

     ---.. ---  h --- new_h
                |      |
                e      e
 */

static inline void frag_list_insert_tail(ucs_frag_list_t *head,
                                         ucs_frag_list_elem_t *new_h,
                                         ucs_frag_list_sn_t sn)
{
    ucs_trace_data("insert_tail=%u", (unsigned)sn);
    new_h->head.first_sn = new_h->head.last_sn = sn;
    ucs_queue_head_init(&new_h->head.list);
    ucs_queue_push(&head->list, &new_h->list);
}

/**
 * special case of insert where sn == head->head_sn
 */
ucs_frag_list_ooo_type_t
ucs_frag_list_insert_head(ucs_frag_list_t *head, ucs_frag_list_elem_t *elem,
                          ucs_frag_list_sn_t sn)
{
     ucs_frag_list_elem_t *h;

     /* next two ifs will not happen if we always pull all possible elems
      * on INSERT_FIRST
      */

     /* check that we are not hitting element on the first frag list */
     if (!ucs_queue_is_empty(&head->list)) {
         h = ucs_queue_head_elem_non_empty(&head->list, ucs_frag_list_elem_t, list);
         if (UCS_FRAG_LIST_SN_CMP(sn, >=, h->head.first_sn)) {
             return UCS_FRAG_LIST_INSERT_DUP;
         }
     }
     else {
         h = NULL;
     }

     head->head_sn++;
     if (!ucs_queue_is_empty(&head->ready_list)) {
         ucs_queue_push(&head->ready_list, &elem->list);
         return UCS_FRAG_LIST_INSERT_READY;
     }

     if (h != NULL && UCS_FRAG_LIST_SN_CMP(h->head.first_sn, ==, sn + 1)) {
         /* do not enqueue. let know that more elems may
          * be pulled from the list.
          * Ex of arrivals: 2 3 1
          */
         return UCS_FRAG_LIST_INSERT_FIRST;
     }

     return UCS_FRAG_LIST_INSERT_FAST;
}

ucs_frag_list_ooo_type_t
ucs_frag_list_insert_slow(ucs_frag_list_t *head, ucs_frag_list_elem_t *elem,
                          ucs_frag_list_sn_t sn)
{
    ucs_frag_list_elem_t *h, *prevh, *nexth;

    if (UCS_FRAG_LIST_SN_CMP(sn, ==, head->head_sn + 1)) {
        return ucs_frag_list_insert_head(head, elem, sn);
    }

    if (UCS_FRAG_LIST_SN_CMP(sn, <=, head->head_sn)) {
        return UCS_FRAG_LIST_INSERT_DUP;
    }

    if (head->max_holes == 0) {
        return UCS_FRAG_LIST_INSERT_FAIL;
    }

    prevh = NULL;
    /* find right list to insert */
    ucs_queue_for_each(h, &head->list, list) {
        /* trying to insert duplicate. retransmission or packet duplication */
        if (UCS_FRAG_LIST_SN_CMP(sn, >=, h->head.first_sn) &&
            UCS_FRAG_LIST_SN_CMP(sn, <=,  h->head.last_sn)) {
            return UCS_FRAG_LIST_INSERT_DUP;
        }

        if (UCS_FRAG_LIST_SN_CMP(sn+1, ==, h->head.first_sn)) {
            frag_list_replace_head(head, prevh, h, elem);
            /* no need to check merge here. merge iff prev->last_sn+1==sn & sn+1 == h->first_sn 
             * the condition is handled in next if */
            head->elem_count++;
            return UCS_FRAG_LIST_INSERT_SLOW;
        }

        /* todo: mark as likely */
        if (UCS_FRAG_LIST_SN_CMP(h->head.last_sn+1, ==, sn)) {
            /* add tail, check merge with next list */
            frag_list_add_tail(h, elem);
            nexth = ucs_container_of(h->list.next, ucs_frag_list_elem_t, list);

            if (nexth != NULL && nexth->head.first_sn == sn + 1) {
                frag_list_merge_heads(head, h, nexth);
                head->list_count--;
            }
            head->elem_count++;
            return UCS_FRAG_LIST_INSERT_SLOW;
        }

        if (UCS_FRAG_LIST_SN_CMP(sn, <, h->head.first_sn)) {
            /* new hole, see above comment on merge */
            if (prevh) {
                ucs_assert(UCS_FRAG_LIST_SN_CMP(prevh->head.last_sn+1, <, sn));
            }
            UCS_STATS_UPDATE_COUNTER(head->stats, UCS_FRAG_LIST_STAT_GAP_LEN, 
                                     prevh ? sn-prevh->head.last_sn : sn-head->head_sn);
            UCS_STATS_UPDATE_COUNTER(head->stats, UCS_FRAG_LIST_STAT_GAPS, 1);
            frag_list_insert_head(head, prevh, h, elem, sn);
            head->elem_count++;
            head->list_count++;
            return UCS_FRAG_LIST_INSERT_SLOW;
        }

        /* if we got here following must hold */
        ucs_assert(UCS_FRAG_LIST_SN_CMP(h->head.last_sn+1, <, sn));
        prevh = h;
    }

    frag_list_insert_tail(head, elem, sn);

    head->elem_count++;
    head->list_count++;
    UCS_STATS_UPDATE_COUNTER(head->stats, UCS_FRAG_LIST_STAT_GAP_LEN, 
                             sn-head->head_sn);
    UCS_STATS_UPDATE_COUNTER(head->stats, UCS_FRAG_LIST_STAT_GAPS, 1);
    return UCS_FRAG_LIST_INSERT_SLOW;
}


/*
  head->h->...->
        |
        e

 * mode of action
 *  - check if we have elements on ready list, if we do take one from there 
 *  - see if h is ready for extraction (sn check), extract firt, move rest to the ready list
 */

ucs_frag_list_elem_t *ucs_frag_list_pull_slow(ucs_frag_list_t *head)
{
    ucs_frag_list_elem_t *h;

    h = ucs_queue_head_elem_non_empty(&head->list, ucs_frag_list_elem_t, list);
    if (UCS_FRAG_LIST_SN_CMP(h->head.first_sn, !=, head->head_sn+1)) {
        ucs_trace_data("first_sn(%u) != head_sn(%u) + 1", (unsigned)h->head.first_sn,
                       (unsigned)head->head_sn);
        return NULL;
    }

    ucs_trace_data("ready list %d to %d", (unsigned)head->head_sn,
                   (unsigned)h->head.last_sn);
    head->head_sn = h->head.last_sn;
    head->elem_count--;
    head->list_count--;

    h = ucs_queue_pull_elem_non_empty(&head->list, ucs_frag_list_elem_t, list);
    ucs_queue_splice(&head->ready_list, &h->head.list);
    return h;
}

void ucs_frag_list_dump(ucs_frag_list_t *head, int how)
{
    ucs_frag_list_elem_t *h, *e;
    int list_count, elem_count;
    int cnt;

    list_count = 0;
    elem_count = 0;

    ucs_queue_for_each(e, &head->ready_list, list) {
       elem_count++; 
    }

    ucs_queue_for_each(h, &head->list, list) {
        list_count++;
        cnt = 0;
        ucs_queue_for_each(e, &h->head.list, list) {
           cnt++; 
           elem_count++;
        }
        elem_count++;
        if (how == 1) {
            ucs_trace_data("%d: %d-%d %d/%d", list_count, h->head.first_sn,
                           h->head.last_sn, h->head.last_sn - h->head.first_sn,
                           cnt);
        }
    }

    if (how == 1) {
        ucs_trace_data("elem count(expected/real)=%d/%d list_count(expected/real)=%d/%d\n",
                       head->elem_count, elem_count,
                       head->list_count, list_count);
    }

    ucs_assert(head->elem_count == elem_count);
    ucs_assert(head->list_count == list_count);
}

