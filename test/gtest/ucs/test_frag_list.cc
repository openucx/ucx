/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/datastruct/frag_list.h>
}

#include <time.h>

class frag_list : public ucs::test {
protected:
    struct pkt {
       uint32_t sn;
       ucs_frag_list_elem_t elem; 
    };
    ucs_frag_list_t m_frags;
    // @override
    virtual void init();
    virtual void cleanup();

    void init_pkts(pkt *packets, int n);
    void permute_array(int *arr, int n);

};

void frag_list::permute_array(int *arr, int n)
{
    
    int i;
    int idx;
    int tmp;

    for (i = 0; i < n; i++) {
        arr[i] = i;
    }

    for (i = 0; i < n - 1; i++) {
        idx = i + ucs::rand() % (n - i);
        tmp = arr[i];
        arr[i] = arr[idx];
        arr[idx] = tmp;
    }
}

void frag_list::init_pkts(pkt *packets, int n)
{
    int i;

    for (i = 0; i < n; i++) {
        packets[i].sn = i;
    }
}

void frag_list::init()
{
    ::srand(::time(NULL));
    ucs_stats_cleanup();
#if ENABLE_STATS
    push_config();
    modify_config("STATS_DEST", "stdout");
    modify_config("STATS_TRIGGER", "");
#endif
    ucs_stats_init();
    ucs_frag_list_init(0, &m_frags,
                       -1 UCS_STATS_ARG(ucs_stats_get_root()));
}

void frag_list::cleanup()
{
    ucs_frag_list_cleanup(&m_frags);
    ucs_stats_cleanup();
#if ENABLE_STATS
    pop_config();
#endif
    ucs_stats_init();
}


/* next four tests cover  all possible insertions and removals. */

/**
 * rcv in order
 */
UCS_TEST_F(frag_list, in_order_rcv) {
    ucs_frag_list_elem_t pkt;
    unsigned i;
    int err;

    err = ucs_frag_list_insert(&m_frags, &pkt, 0);
    EXPECT_EQ(UCS_FRAG_LIST_INSERT_DUP, err);
    err = ucs_frag_list_insert(&m_frags, &pkt, (ucs_frag_list_sn_t)(-1));
    EXPECT_EQ(UCS_FRAG_LIST_INSERT_DUP, err);

    for (i = 1; i < 10; i++) {
        err = ucs_frag_list_insert(&m_frags, &pkt, i);
        EXPECT_EQ(UCS_FRAG_LIST_INSERT_FAST, err);
    }
#if ENABLE_STATS
    EXPECT_EQ((ucs_stats_counter_t)1, UCS_STATS_GET_COUNTER(m_frags.stats, UCS_FRAG_LIST_STAT_BURSTS));
    EXPECT_EQ((ucs_stats_counter_t)9, UCS_STATS_GET_COUNTER(m_frags.stats, UCS_FRAG_LIST_STAT_BURST_LEN));
    EXPECT_EQ((ucs_stats_counter_t)0, UCS_STATS_GET_COUNTER(m_frags.stats, UCS_FRAG_LIST_STAT_GAPS));
    EXPECT_EQ((ucs_stats_counter_t)0, UCS_STATS_GET_COUNTER(m_frags.stats, UCS_FRAG_LIST_STAT_GAP_LEN));
    EXPECT_EQ((ucs_stats_counter_t)0, UCS_STATS_GET_COUNTER(m_frags.stats, UCS_FRAG_LIST_STAT_GAP_OUT));
#endif
}

/**
 * one hole in front
 */
UCS_TEST_F(frag_list, one_hole) {
    pkt pkts[10], *out;
    ucs_frag_list_elem_t *elem;
    unsigned i;
    int err;

    init_pkts(pkts, 10);

    for (i = 5; i < 10; i++) {
        err = ucs_frag_list_insert(&m_frags, &pkts[i].elem, i);
        EXPECT_EQ(UCS_FRAG_LIST_INSERT_SLOW, err);
    }

    /* try to pull - should fail */
    elem = ucs_frag_list_pull(&m_frags);
    EXPECT_EQ((void *)elem, (void *)NULL);
    
    /* insert 1-3: no need to pull more elems from list
     * insert 4: more elems can be pulled
     */
    for (i = 1; i < 5; i++) {
        err = ucs_frag_list_insert(&m_frags, &pkts[i].elem, i);
        if (i < 4) {
            EXPECT_EQ(UCS_FRAG_LIST_INSERT_FAST, err);
        }
        else {
            EXPECT_EQ(UCS_FRAG_LIST_INSERT_FIRST, err);
        }
    }

    /* sn 5 already in - next one must fail */
    err = ucs_frag_list_insert(&m_frags, &pkts[i].elem, 5);
    EXPECT_EQ(UCS_FRAG_LIST_INSERT_DUP, err);


    i = 0;
    /* elem 5..9 must be on the list now */
    while((elem = ucs_frag_list_pull(&m_frags)) != NULL) {
        out = ucs_container_of(elem, pkt, elem);
        EXPECT_EQ(out->sn, i+5);
        i++;
    }
    EXPECT_EQ((unsigned)5, i);
#if ENABLE_STATS
    EXPECT_EQ((ucs_stats_counter_t)2, UCS_STATS_GET_COUNTER(m_frags.stats, UCS_FRAG_LIST_STAT_BURSTS));
    EXPECT_EQ((ucs_stats_counter_t)10, UCS_STATS_GET_COUNTER(m_frags.stats, UCS_FRAG_LIST_STAT_BURST_LEN));
    EXPECT_EQ((ucs_stats_counter_t)1, UCS_STATS_GET_COUNTER(m_frags.stats, UCS_FRAG_LIST_STAT_GAPS));
    EXPECT_EQ((ucs_stats_counter_t)5, UCS_STATS_GET_COUNTER(m_frags.stats, UCS_FRAG_LIST_STAT_GAP_LEN));
    EXPECT_EQ((ucs_stats_counter_t)9, UCS_STATS_GET_COUNTER(m_frags.stats, UCS_FRAG_LIST_STAT_GAP_OUT));
#endif
}

UCS_TEST_F(frag_list, two_holes_basic) {
    pkt pkts[20], *out;
    ucs_frag_list_elem_t *elem;
    unsigned i;
    int err;

    init_pkts(pkts, 20);


    for (i = 15; i < 20; i++) {
        err = ucs_frag_list_insert(&m_frags, &pkts[i].elem, i);
        EXPECT_EQ(UCS_FRAG_LIST_INSERT_SLOW, err);
    }

    /* try to pull - should fail */
    elem = ucs_frag_list_pull(&m_frags);
    EXPECT_EQ((void *)NULL, (void *)elem);

    for (i = 5; i < 10; i++) {
        err = ucs_frag_list_insert(&m_frags, &pkts[i].elem, i);
        EXPECT_EQ(err, UCS_FRAG_LIST_INSERT_SLOW);
    }

    /* try to pull - should fail */
    elem = ucs_frag_list_pull(&m_frags);
    EXPECT_EQ((void *)NULL, (void *)elem);

    for (i = 4; i > 1; i--) {
        err = ucs_frag_list_insert(&m_frags, &pkts[i].elem, i);
        EXPECT_EQ(err, UCS_FRAG_LIST_INSERT_SLOW);
    }

    err = ucs_frag_list_insert(&m_frags, &pkts[1].elem, 1);
    EXPECT_EQ(err, UCS_FRAG_LIST_INSERT_FIRST);

    i = 2;
    while((elem = ucs_frag_list_pull(&m_frags)) != NULL) {
        out = ucs_container_of(elem, pkt, elem);
        EXPECT_EQ(out->sn, i);
        i++;
    }
    EXPECT_EQ(i, (unsigned)10);

    for (i = 10; i < 15; i++) {
        err = ucs_frag_list_insert(&m_frags, &pkts[i].elem, i);
        if (i < 14) {
            EXPECT_EQ(UCS_FRAG_LIST_INSERT_FAST, err);
        }
        else {
            EXPECT_EQ(UCS_FRAG_LIST_INSERT_FIRST, err);
        }
    }

    while((elem = ucs_frag_list_pull(&m_frags)) != NULL) {
        out = ucs_container_of(elem, pkt, elem);
        EXPECT_EQ(out->sn, i);
        i++;
    }
    EXPECT_EQ((unsigned)20, i);
#if ENABLE_STATS
    EXPECT_EQ((ucs_stats_counter_t)7, UCS_STATS_GET_COUNTER(m_frags.stats, UCS_FRAG_LIST_STAT_BURSTS));
    EXPECT_EQ((ucs_stats_counter_t)19, UCS_STATS_GET_COUNTER(m_frags.stats, UCS_FRAG_LIST_STAT_BURST_LEN));
    EXPECT_EQ((ucs_stats_counter_t)2, UCS_STATS_GET_COUNTER(m_frags.stats, UCS_FRAG_LIST_STAT_GAPS));
    EXPECT_EQ((ucs_stats_counter_t)20, UCS_STATS_GET_COUNTER(m_frags.stats, UCS_FRAG_LIST_STAT_GAP_LEN));
    EXPECT_EQ((ucs_stats_counter_t)28, UCS_STATS_GET_COUNTER(m_frags.stats, UCS_FRAG_LIST_STAT_GAP_OUT));
#endif
}

/**
 * two holes 
 */
UCS_TEST_F(frag_list, two_holes_advanced) {
    pkt pkts[20], *out;
    ucs_frag_list_elem_t *elem;
    unsigned i;
    int err;

    init_pkts(pkts, 20);

    for (i = 5; i < 10; i++) {
        err = ucs_frag_list_insert(&m_frags, &pkts[i].elem, i);
        EXPECT_EQ(UCS_FRAG_LIST_INSERT_SLOW, err);
    }

    /* try to pull - should fail */
    elem = ucs_frag_list_pull(&m_frags);
    EXPECT_EQ((void *)NULL, (void *)elem);
    
    for (i = 13; i < 18; i++) {
        err = ucs_frag_list_insert(&m_frags, &pkts[i].elem, i);
        EXPECT_EQ(UCS_FRAG_LIST_INSERT_SLOW, err);
    }

    for (i = 19; i >= 18; i--) {
        err = ucs_frag_list_insert(&m_frags, &pkts[i].elem, i);
        EXPECT_EQ(UCS_FRAG_LIST_INSERT_SLOW, err);
    }

    for (i = 12; i >= 10; i--) {
        err = ucs_frag_list_insert(&m_frags, &pkts[i].elem, i);
        EXPECT_EQ(UCS_FRAG_LIST_INSERT_SLOW, err);
    }

    for (i = 4; i > 1; i--) {
        err = ucs_frag_list_insert(&m_frags, &pkts[i].elem, i);
        EXPECT_EQ(UCS_FRAG_LIST_INSERT_SLOW, err);
    }

    err = ucs_frag_list_insert(&m_frags, &pkts[1].elem, 1);
    EXPECT_EQ(UCS_FRAG_LIST_INSERT_FIRST, err);

    i = 2;
    while((elem = ucs_frag_list_pull(&m_frags)) != NULL) {
        out = ucs_container_of(elem, pkt, elem);
        EXPECT_EQ(out->sn, i);
        i++;
    }
    EXPECT_EQ((unsigned)20, i);
}

/** 
 *
 * random arrival. Send/recv 10k packets in random order
 */
#define FRAG_LIST_N_PKTS 10000

UCS_TEST_F(frag_list, random_arrival) {
    std::vector<pkt> pkts(FRAG_LIST_N_PKTS + 1);
    pkt *out;
    ucs_frag_list_elem_t *elem;
    unsigned i;
    std::vector<int> idx(FRAG_LIST_N_PKTS);
    int err;
    int fast_inserts, slow_inserts, pulled;
    uint32_t last_sn = 0;
    uint32_t max_holes=0, max_elems=0;


    init_pkts(&pkts[0], FRAG_LIST_N_PKTS+1);
    permute_array(&idx[0], FRAG_LIST_N_PKTS);

    fast_inserts = slow_inserts = pulled = 0;
    for (i = 0; i < FRAG_LIST_N_PKTS; i++) {
        err = ucs_frag_list_insert(&m_frags, &pkts[idx[i]+1].elem, idx[i]+1);
        EXPECT_NE(err, UCS_FRAG_LIST_INSERT_DUP);
        if (err == UCS_FRAG_LIST_INSERT_FAST || err == UCS_FRAG_LIST_INSERT_FIRST) {
            fast_inserts++;
            EXPECT_EQ(last_sn+1, (uint32_t)idx[i]+1);
            last_sn = idx[i]+1;
        }
        else {
            slow_inserts++;
        }
        max_holes = ucs_max(m_frags.list_count, max_holes);
        max_elems = ucs_max(m_frags.elem_count, max_elems);
        while((elem = ucs_frag_list_pull(&m_frags)) != NULL) {
            out = ucs_container_of(elem, pkt, elem);
            pulled++;
            EXPECT_EQ(last_sn+1, out->sn);
            last_sn = out->sn;
        }
    }
    ucs_frag_list_dump(&m_frags, 0);
    UCS_TEST_MESSAGE << "max_holes=" << max_holes << " max_elems=" << max_elems;
    UCS_TEST_MESSAGE << "fast_ins=" << fast_inserts <<" slow_ins=" << slow_inserts << " pulled=" << pulled;
    while((elem = ucs_frag_list_pull(&m_frags)) != NULL) {
        out = ucs_container_of(elem, pkt, elem);
        EXPECT_EQ(last_sn+1, out->sn);
        last_sn = out->sn;
    }
}

