/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* Copyright (C) Huawei Technologies Co., Ltd. 2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <common/test.h>

extern "C" {
#include <ucs/debug/memtrack.h>
#include <ucs/datastruct/array.inl>
#include <ucs/datastruct/list.h>
#include <ucs/datastruct/hlist.h>
#include <ucs/datastruct/ptr_array.h>
#include <ucs/datastruct/ptr_map.inl>
#include <ucs/datastruct/queue.h>
#include <ucs/time/time.h>
#include <ucs/type/init_once.h>
#include <ucs/arch/cpu.h>
}

#include <vector>
#include <map>

class test_datatype : public ucs::test {
};

typedef struct {
    int               i;
    ucs_list_link_t   list;
    ucs_hlist_link_t  hlist;
    ucs_queue_elem_t  queue;
} elem_t;


UCS_TEST_F(test_datatype, list_basic) {

    ucs_list_link_t head;
    elem_t elem0, elem1;
    elem_t *iter, *tmp;

    ucs_list_head_init(&head);
    ASSERT_EQ((unsigned long)0, ucs_list_length(&head));
    ucs_list_insert_after(&head, &elem0.list);
    ucs_list_insert_before(&head, &elem1.list);

    std::vector<elem_t*> vec;
    ucs_list_for_each(iter, &head, list) {
        vec.push_back(iter);
    }
    ASSERT_EQ(2ul, vec.size());
    ASSERT_EQ(&elem0, vec[0]);
    ASSERT_EQ(&elem1, vec[1]);
    ASSERT_EQ((unsigned long)2, ucs_list_length(&head));

    ucs_list_for_each_safe(iter, tmp, &head, list) {
        ucs_list_del(&iter->list);
    }
    ASSERT_TRUE(ucs_list_is_empty(&head));
    ASSERT_EQ((unsigned long)0, ucs_list_length(&head));
}

UCS_TEST_F(test_datatype, list_splice) {

    ucs_list_link_t head1, head2;
    elem_t l1_elem0, l1_elem1, l1_elem2;
    elem_t l2_elem0, l2_elem1, l2_elem2;
    elem_t *iter;

    ucs_list_head_init(&head1);
    ucs_list_head_init(&head2);

    l1_elem0.i = 0;
    ucs_list_add_tail(&head1, &l1_elem0.list);
    l1_elem1.i = 1;
    ucs_list_add_tail(&head1, &l1_elem1.list);
    l1_elem2.i = 2;
    ucs_list_add_tail(&head1, &l1_elem2.list);

    l2_elem0.i = 3;
    ucs_list_add_tail(&head2, &l2_elem0.list);
    l2_elem1.i = 4;
    ucs_list_add_tail(&head2, &l2_elem1.list);
    l2_elem2.i = 5;
    ucs_list_add_tail(&head2, &l2_elem2.list);

    ucs_list_splice_tail(&head1, &head2);

    int i = 0;
    ucs_list_for_each(iter, &head1, list) {
        EXPECT_EQ(i, iter->i);
        ++i;
    }
}

UCS_TEST_F(test_datatype, hlist_basic) {
    elem_t elem1, elem2, elem3;
    ucs_hlist_head_t head;
    std::vector<int> v;
    elem_t *elem;

    elem1.i = 1;
    elem2.i = 2;
    elem3.i = 3;

    /* initialize list, should be empty */
    ucs_hlist_head_init(&head);
    EXPECT_TRUE(ucs_hlist_is_empty(&head));

    /* add one element to head */
    ucs_hlist_add_head(&head, &elem1.hlist);
    EXPECT_FALSE(ucs_hlist_is_empty(&head));

    EXPECT_EQ(&elem1, ucs_hlist_head_elem(&head, elem_t, hlist));

    /* test iteration over single-element list */
    v.clear();
    ucs_hlist_for_each(elem, &head, hlist) {
        v.push_back(elem->i);
    }
    ASSERT_EQ(1ul, v.size());
    EXPECT_EQ(1, v[0]);

    ucs_hlist_del(&head, &elem1.hlist);
    EXPECT_TRUE(ucs_hlist_is_empty(&head));

    /* when list is empty, extract_head should return NULL */
    ucs_hlist_link_t *helem = ucs_hlist_extract_head(&head);
    EXPECT_TRUE(helem == NULL);

    /* test iteration over empty list */
    v.clear();
    ucs_hlist_for_each(elem, &head, hlist) {
        v.push_back(elem->i);
    }
    ASSERT_EQ(0ul, v.size());

    /* add one element to head and extract it */
    ucs_hlist_add_head(&head, &elem1.hlist);
    elem = ucs_hlist_extract_head_elem(&head, elem_t, hlist);
    EXPECT_EQ(&elem1, elem);

    /* add 3 elements */
    ucs_hlist_add_tail(&head, &elem2.hlist);
    ucs_hlist_add_head(&head, &elem1.hlist);
    ucs_hlist_add_tail(&head, &elem3.hlist);

    /* iterate without extract */
    v.clear();
    ucs_hlist_for_each(elem, &head, hlist) {
        v.push_back(elem->i);
    }
    ASSERT_EQ(3ul, v.size());
    EXPECT_EQ(1, v[0]);
    EXPECT_EQ(2, v[1]);
    EXPECT_EQ(3, v[2]);

    /* iterate and extract */
    v.clear();
    ucs_hlist_for_each_extract(elem, &head, hlist) {
        v.push_back(elem->i);
    }
    ASSERT_EQ(3ul, v.size());
    EXPECT_EQ(1, v[0]);
    EXPECT_EQ(2, v[1]);
    EXPECT_EQ(3, v[2]);

    EXPECT_TRUE(ucs_hlist_is_empty(&head));
}

UCS_TEST_F(test_datatype, hlist_for_each_extract_if) {
    const size_t n_elems = 3;
    std::vector<elem_t*> v_elems;
    ucs_hlist_head_t head;
    std::vector<int> v;
    elem_t *elem;

    for (size_t i = 0; i < n_elems; ++i) {
        v_elems.push_back(new elem_t);
        v_elems[i]->i = i;
    }

    /* initialize list, should be empty */
    ucs_hlist_head_init(&head);
    EXPECT_TRUE(ucs_hlist_is_empty(&head));

    /* add one element to head */
    ucs_hlist_add_head(&head, &v_elems[0]->hlist);
    EXPECT_FALSE(ucs_hlist_is_empty(&head));

    EXPECT_EQ(v_elems[0], ucs_hlist_head_elem(&head, elem_t, hlist));

    /* test iteration over single-element list, don't remove */
    ucs_hlist_for_each_extract_if(elem, &head, hlist, false) {
        v.push_back(elem->i);
    }
    ASSERT_TRUE(v.empty());
    ASSERT_FALSE(ucs_hlist_is_empty(&head));

    /* test iteration over single-element list, remove */
    ucs_hlist_for_each_extract_if(elem, &head, hlist, true) {
        v.push_back(elem->i);
    }
    ASSERT_TRUE(ucs_hlist_is_empty(&head));
    ASSERT_EQ(1ul, v.size());
    EXPECT_EQ(0, v[0]);
    v.clear();

    /* when list is empty, extract_head should return NULL */
    ucs_hlist_link_t *helem = ucs_hlist_extract_head(&head);
    EXPECT_TRUE(helem == NULL);

    /* test iteration over empty list */
    v.clear();
    ucs_hlist_for_each_extract_if(elem, &head, hlist, true) {
        v.push_back(elem->i);
    }
    ASSERT_EQ(0ul, v.size());

    ucs_hlist_for_each_extract_if(elem, &head, hlist, false) {
        v.push_back(elem->i);
    }
    ASSERT_EQ(0ul, v.size());

    /* add 3 elements */
    ucs_hlist_add_tail(&head, &v_elems[1]->hlist);
    ucs_hlist_add_head(&head, &v_elems[0]->hlist);
    ucs_hlist_add_tail(&head, &v_elems[2]->hlist);

    /* iterate and extract 2 elements */
    v.clear();
    ucs_hlist_for_each_extract_if(elem, &head, hlist, elem->i < 2) {
        v.push_back(elem->i);
    }
    ASSERT_EQ(2ul, v.size());
    EXPECT_EQ(0, v[0]);
    EXPECT_EQ(1, v[1]);
    /* iterate and extract last element */
    ucs_hlist_for_each_extract_if(elem, &head, hlist, elem->i < 100) {
        v.push_back(elem->i);
    }
    EXPECT_EQ(2, v[2]);

    EXPECT_TRUE(ucs_hlist_is_empty(&head));

    /* add 3 elements */
    for (size_t i = 0; i < n_elems; ++i) {
        ucs_hlist_add_tail(&head, &v_elems[i]->hlist);
    }

    /* iterate and delete all the extracted elements */
    ucs_hlist_for_each_extract_if(elem, &head, hlist, true) {
        EXPECT_EQ(v_elems[0], elem);
        memset(elem, 0xff, sizeof(*elem));
        delete elem;
        v_elems.erase(v_elems.begin());
    }

    EXPECT_TRUE(ucs_hlist_is_empty(&head));
    EXPECT_TRUE(v_elems.empty());
}

UCS_TEST_F(test_datatype, queue) {

    ucs_queue_head_t head;
    elem_t elem0, elem1, elem2;
    elem_t *elem;

    ucs_queue_head_init(&head);
    EXPECT_TRUE(ucs_queue_is_empty(&head));

    elem0.i = 0;
    elem1.i = 1;
    elem2.i = 2;

    for (unsigned i = 0; i < 5; ++i) {
        ucs_queue_push(&head, &elem0.queue);
        EXPECT_FALSE(ucs_queue_is_empty(&head));
        EXPECT_EQ((unsigned long)1, ucs_queue_length(&head));
        EXPECT_TRUE(ucs_queue_is_tail(&head, &elem0.queue));

        ucs_queue_push(&head, &elem1.queue);
        EXPECT_EQ((unsigned long)2, ucs_queue_length(&head));
        EXPECT_TRUE(ucs_queue_is_tail(&head, &elem1.queue));

        EXPECT_EQ(&elem1, ucs_queue_tail_elem_non_empty(&head, elem_t, queue));

        elem = ucs_queue_pull_elem_non_empty(&head, elem_t, queue);
        EXPECT_EQ(&elem0, elem);
        EXPECT_EQ((unsigned long)1, ucs_queue_length(&head));

        ucs_queue_push(&head, &elem2.queue);
        EXPECT_EQ((unsigned long)2, ucs_queue_length(&head));

        elem = ucs_queue_pull_elem_non_empty(&head, elem_t, queue);
        EXPECT_EQ(&elem1, elem);
        EXPECT_EQ((unsigned long)1, ucs_queue_length(&head));

        elem = ucs_queue_pull_elem_non_empty(&head, elem_t, queue);
        EXPECT_EQ(&elem2, elem);
        EXPECT_TRUE(ucs_queue_is_empty(&head));
        EXPECT_TRUE(NULL == ucs_queue_pull(&head));

        /* Push to head now */

        ucs_queue_push_head(&head, &elem2.queue);
        EXPECT_EQ((unsigned long)1, ucs_queue_length(&head));

        ucs_queue_push_head(&head, &elem1.queue);
        ucs_queue_push_head(&head, &elem0.queue);
        EXPECT_EQ((unsigned long)3, ucs_queue_length(&head));

        elem = ucs_queue_pull_elem_non_empty(&head, elem_t, queue);
        EXPECT_EQ(&elem0, elem);

        elem = ucs_queue_pull_elem_non_empty(&head, elem_t, queue);
        EXPECT_EQ(&elem1, elem);

        elem = ucs_queue_pull_elem_non_empty(&head, elem_t, queue);
        EXPECT_EQ(&elem2, elem);

        EXPECT_TRUE(ucs_queue_is_empty(&head));
    }
}

UCS_TEST_F(test_datatype, queue_iter) {

    const int num_elems = 4;
    ucs_queue_head_t head;
    std::vector<elem_t> elems(num_elems);

    ucs_queue_head_init(&head);
    EXPECT_TRUE(ucs_queue_is_empty(&head));

    for (int i = 0; i < num_elems; ++i) {
        elems[i].i = i + 1;
        ucs_queue_push(&head, &elems[i].queue);
    }

    {
        std::vector<int> vec;
        elem_t *elem;

        ucs_queue_for_each(elem, &head, queue) {
            vec.push_back(elem->i);
        }
        ASSERT_EQ(static_cast<size_t>(num_elems), vec.size());
        EXPECT_EQ(1, vec[0]);
        EXPECT_EQ(2, vec[1]);
        EXPECT_EQ(3, vec[2]);
        EXPECT_EQ(4, vec[3]);
    }

    {
        std::vector<int> vec;
        ucs_queue_iter_t iter;
        elem_t *elem;

        ucs_queue_for_each_safe(elem, iter, &head, queue)
        {
            if (elem->i == 3 || elem->i == 4) {
                ucs_queue_del_iter(&head, iter);
                memset(elem, 0xff, sizeof(*elem));
            }
        }
        ASSERT_EQ((unsigned long)2, ucs_queue_length(&head));

        ucs_queue_for_each_safe(elem, iter, &head, queue) {
            vec.push_back(elem->i);
            ucs_queue_del_iter(&head, iter);
            memset(elem, 0xff, sizeof(*elem));
        }
        ASSERT_EQ(2u, vec.size());
        EXPECT_EQ(1, vec[0]);
        EXPECT_EQ(2, vec[1]);
    }

    /* foreach safe with next pointing to head */
    {
        elem_t e1, *elem;
        ucs_queue_iter_t iter;

        ucs_queue_push(&head, &e1.queue);
        e1.queue.next = &e1.queue;

        int count1 = 0;
        ucs_queue_for_each_safe(elem, iter, &head, queue) {
            EXPECT_EQ(&elem->queue, *iter);
            ++count1;
        }
        EXPECT_EQ(1, count1) << "Too many iterations on single element queue";

        int count2 = 0;
        ucs_queue_for_each_safe(elem, iter, &head, queue) {
            EXPECT_EQ(&elem->queue, *iter);
            ucs_queue_del_iter(&head, iter);
            ++count2;
            ASSERT_LE(count2, 2) << "Too many iterations on single element queue";
        }
    }
}

UCS_TEST_SKIP_COND_F(test_datatype, queue_perf,
                     (ucs::test_time_multiplier() > 1)) {
    const size_t count = 100000000ul;
    ucs_queue_head_t head;
    ucs_queue_elem_t elem;

    ucs_queue_head_init(&head);
    ucs_queue_push(&head, &elem);
    elem.next = NULL;

    ucs_time_t start_time = ucs_get_time();
    for (size_t i = 0; i < count; ++i) {
        ucs_queue_pull(&head);
        ucs_queue_push(&head, &elem);
    }
    ucs_time_t end_time = ucs_get_time();

    double lat = ucs_time_to_nsec(end_time - start_time) / count;
    UCS_TEST_MESSAGE << lat << " nsec per push+pull";

    if (ucs::perf_retry_count) {
        EXPECT_LT(lat, 15.0 * ucs::test_time_multiplier());
    } else {
        UCS_TEST_MESSAGE << "not validating performance";
    }
    EXPECT_EQ((unsigned long)1, ucs_queue_length(&head));
}

UCS_TEST_F(test_datatype, queue_splice) {
    ucs_queue_head_t head;
    elem_t elem0, elem1, elem2;
    elem_t *elem;

    elem0.i = 0;
    elem1.i = 1;
    elem2.i = 2;

    ucs_queue_head_init(&head);
    ucs_queue_push(&head, &elem0.queue);
    ucs_queue_push(&head, &elem1.queue);
    ucs_queue_push(&head, &elem2.queue);

    ucs_queue_head_t newq;
    ucs_queue_head_init(&newq);

    EXPECT_EQ((unsigned long)3, ucs_queue_length(&head));
    EXPECT_EQ((unsigned long)0, ucs_queue_length(&newq));

    ucs_queue_splice(&newq, &head);

    EXPECT_EQ((unsigned long)0, ucs_queue_length(&head));
    EXPECT_EQ((unsigned long)3, ucs_queue_length(&newq));

    elem = ucs_queue_pull_elem_non_empty(&newq, elem_t, queue);
    EXPECT_EQ(&elem0, elem);

    elem = ucs_queue_pull_elem_non_empty(&newq, elem_t, queue);
    EXPECT_EQ(&elem1, elem);

    elem = ucs_queue_pull_elem_non_empty(&newq, elem_t, queue);
    EXPECT_EQ(&elem2, elem);
}

typedef struct {
    ucs_queue_elem_t queue;
    uint16_t         sn;
} ucs_test_callbackq_elem_t;

static int ucs_test_callbackq_pull(ucs_queue_head_t *queue, uint16_t sn)
{
    ucs_test_callbackq_elem_t *elem;
    int count = 0;

    ucs_queue_for_each_extract(elem, queue, queue,
                               UCS_CIRCULAR_COMPARE16(elem->sn, <=, sn)) {
        elem->sn = 0;
        ++count;
    }

    return count;
}

UCS_TEST_F(test_datatype, queue_extract_if) {
    ucs_queue_head_t queue;
    ucs_test_callbackq_elem_t elem1, elem2, elem3;
    unsigned count;

    ucs_queue_head_init(&queue);

    elem1.sn = 1;
    elem2.sn = 2;
    elem3.sn = 3;

    ucs_queue_push(&queue, &elem1.queue);
    ucs_queue_push(&queue, &elem2.queue);
    ucs_queue_push(&queue, &elem3.queue);

    count = ucs_test_callbackq_pull(&queue, 0);
    EXPECT_EQ(0u, count);

    count = ucs_test_callbackq_pull(&queue, 1);
    EXPECT_EQ(1u, count);

    count = ucs_test_callbackq_pull(&queue, 2);
    EXPECT_EQ(1u, count);
    EXPECT_EQ(0u, elem1.sn); /* should be removed */
    EXPECT_EQ(0u, elem2.sn); /* should be removed */

    count = ucs_test_callbackq_pull(&queue, 10);
    EXPECT_EQ(1u, count);
    EXPECT_EQ(0u, elem3.sn); /* should be removed */
}

UCS_TEST_F(test_datatype, ptr_array_basic) {
    ucs_ptr_array_t pa;
    int a = 1, b = 2, c = 3, d = 4, e = 5, f = 6, g = 7;
    unsigned index;

    ucs_ptr_array_init(&pa, "ptr_array test");

    EXPECT_EQ(ucs_ptr_array_get_elem_count(&pa), 0);
    EXPECT_EQ(ucs_ptr_array_is_empty(&pa), 1);

    index = ucs_ptr_array_insert(&pa, &a);
    EXPECT_EQ(0u, index);

    EXPECT_EQ(ucs_ptr_array_get_elem_count(&pa), 1);
    EXPECT_EQ(ucs_ptr_array_is_empty(&pa), 0);

    index = ucs_ptr_array_insert(&pa, &b);
    EXPECT_EQ(1u, index);

    index = ucs_ptr_array_insert(&pa, &c);
    EXPECT_EQ(2u, index);

    ucs_ptr_array_set(&pa, 3, &d);

    index = ucs_ptr_array_insert(&pa, &e);
    EXPECT_EQ(4u, index);

    ucs_ptr_array_set(&pa, 6, &f);

    ucs_ptr_array_set(&pa, 100, &g);

    EXPECT_EQ(ucs_ptr_array_get_elem_count(&pa), 7);

    index = ucs_ptr_array_bulk_alloc(&pa, 200);
    EXPECT_EQ(ucs_ptr_array_get_elem_count(&pa), 207);

    EXPECT_EQ(301u, pa.size);
    EXPECT_EQ(101u, index);

    void *vc;
    int present = ucs_ptr_array_lookup(&pa, 2, vc);
    ASSERT_TRUE(present);
    EXPECT_EQ(&c, vc);

    vc = ucs_ptr_array_replace(&pa, 2, &d);
    EXPECT_EQ(&c, vc);

    present = ucs_ptr_array_lookup(&pa, 2, vc);
    EXPECT_EQ(&d, vc);

    ucs_ptr_array_set(&pa, 2, &g);
    present = ucs_ptr_array_lookup(&pa, 2, vc);
    EXPECT_EQ(&g, vc);

    present = ucs_ptr_array_lookup(&pa, 6, vc);
    EXPECT_EQ(&f, vc);

    present = ucs_ptr_array_lookup(&pa, 100, vc);
    EXPECT_EQ(&g, vc);

    present = ucs_ptr_array_lookup(&pa, 101, vc);
    EXPECT_EQ(NULL, vc);

    present = ucs_ptr_array_lookup(&pa, 301, vc);
    EXPECT_EQ(NULL, vc);

    EXPECT_FALSE(ucs_ptr_array_lookup(&pa, 5, vc));
    EXPECT_FALSE(ucs_ptr_array_lookup(&pa, 99, vc));
    EXPECT_FALSE(ucs_ptr_array_lookup(&pa, 302, vc));
    EXPECT_FALSE(ucs_ptr_array_lookup(&pa, 5005, vc));

    ucs_ptr_array_remove(&pa, 0);
    ucs_ptr_array_remove(&pa, 1);
    ucs_ptr_array_remove(&pa, 2);
    ucs_ptr_array_remove(&pa, 3);
    ucs_ptr_array_remove(&pa, 4);
    ucs_ptr_array_remove(&pa, 6);

    EXPECT_EQ(ucs_ptr_array_get_elem_count(&pa), 201);
    EXPECT_EQ(ucs_ptr_array_is_empty(&pa), 0);

    for (index = 100; index <= 300; index++) {
        ucs_ptr_array_remove(&pa, index);
    }

    EXPECT_EQ(ucs_ptr_array_get_elem_count(&pa), 0);
    EXPECT_EQ(ucs_ptr_array_is_empty(&pa), 1);

    ucs_ptr_array_cleanup(&pa, 1);
}

UCS_TEST_F(test_datatype, ptr_array_bulk_alloc) {
    ucs_ptr_array_t pa;
    unsigned idx, alloc1, alloc2;

    ucs_ptr_array_init(&pa, "ptr_array alloc test");

    alloc1 = ucs_ptr_array_bulk_alloc(&pa, 10);
    EXPECT_EQ(ucs_ptr_array_get_elem_count(&pa), 10);

    EXPECT_GE(pa.size, 10u);
    EXPECT_EQ(alloc1, 0u);

    alloc2 = ucs_ptr_array_bulk_alloc(&pa, 100);
    EXPECT_EQ(ucs_ptr_array_get_elem_count(&pa), 110);

    EXPECT_GE(pa.size, 110u);
    EXPECT_EQ(alloc2, alloc1 + 10u);

    for (idx = 0; idx < alloc2; idx++) {
        ucs_ptr_array_remove(&pa, idx);
    }

    EXPECT_EQ(ucs_ptr_array_get_elem_count(&pa), 100);
    EXPECT_EQ(alloc1, ucs_ptr_array_bulk_alloc(&pa, 10));
    EXPECT_EQ(ucs_ptr_array_get_elem_count(&pa), 110);

    for (idx = alloc1 + 10; idx > alloc1; idx--) {
        ucs_ptr_array_remove(&pa, idx - 1);
    }

    EXPECT_EQ(ucs_ptr_array_get_elem_count(&pa), 100);
    EXPECT_EQ(alloc1, ucs_ptr_array_bulk_alloc(&pa, 10));
    EXPECT_EQ(ucs_ptr_array_get_elem_count(&pa), 110);

    for (idx = alloc1; idx < alloc1 + 10; idx++) {
        ucs_ptr_array_remove(&pa, idx);
    }

    for (idx = alloc2; idx < alloc2 + 100; idx++) {
        ucs_ptr_array_remove(&pa, idx);
    }

    EXPECT_EQ(ucs_ptr_array_is_empty(&pa), 1);

    ucs_ptr_array_cleanup(&pa, 1);
}

UCS_TEST_F(test_datatype, ptr_array_set_first) {
    ucs_ptr_array_t pa;
    int a = 1;

    ucs_ptr_array_init(&pa, "ptr_array set-first test");

    EXPECT_EQ(0u, pa.size);

    ucs_ptr_array_set(&pa, 0, &a);

    EXPECT_GT(pa.size, 0u);

    ucs_ptr_array_remove(&pa, 0);

    ucs_ptr_array_cleanup(&pa, 1);
}

UCS_TEST_F(test_datatype, ptr_array_random) {
    const unsigned count = 10000 / ucs::test_time_multiplier();
    ucs_ptr_array_t pa;
    unsigned expeced_count = count;
    ucs_ptr_array_init(&pa, "ptr_array test");

    std::map<int, void*> map;

    /* Insert phase */
    for (unsigned i = 0; i < count; ++i) {
        void *ptr = malloc(0);
        unsigned index = ucs_ptr_array_insert(&pa, ptr);

        EXPECT_TRUE(map.end() == map.find(index));
        map[index] = ptr;
    }

    /* Remove + insert */
    for (unsigned i = 0; i < count / 10; ++i) {

        int remove_count = ucs::rand() % 10;
        expeced_count -= remove_count;
        for (int j = 0; j < remove_count; ++j) {
            unsigned to_remove = ucs::rand() % map.size();
            std::map<int, void*>::iterator iter = map.begin();
            std::advance(iter, to_remove);
            unsigned index = iter->first;

            void *ptr = NULL;
            EXPECT_TRUE(ucs_ptr_array_lookup(&pa, index, ptr));
            EXPECT_EQ(ptr, map[index]);
            free(ptr);

            ucs_ptr_array_remove(&pa, index);
            EXPECT_FALSE(ucs_ptr_array_lookup(&pa, index, ptr));

            map.erase(index);
        }

        int insert_count = ucs::rand() % 10;
        expeced_count += insert_count;
        for (int j = 0; j < insert_count; ++j) {
            void *ptr = malloc(0);
            unsigned index = ucs_ptr_array_insert(&pa, ptr);

            EXPECT_TRUE(map.end() == map.find(index));
            map[index] = ptr;
        }
    }

    unsigned count_elements = 0;
    /* remove all */
    void *ptr;
    unsigned index;
    ucs_ptr_array_for_each(ptr, index, &pa) {
        EXPECT_EQ(ptr, map[index]);
        ucs_ptr_array_remove(&pa, index);
        free(ptr);
        count_elements++;
    }

    EXPECT_EQ(count_elements, expeced_count);

    ucs_ptr_array_cleanup(&pa, 1);
}

UCS_TEST_SKIP_COND_F(test_datatype, ptr_array_perf,
                     (ucs::test_time_multiplier() > 1)) {
    const unsigned count = 10000000;
    ucs_ptr_array_t pa;

    ucs_time_t insert_start_time = ucs_get_time();
    ucs_ptr_array_init(&pa, "ptr_array test");
    for (unsigned i = 0; i < count; ++i) {
        EXPECT_EQ(i, ucs_ptr_array_insert(&pa, NULL));
    }

    ucs_time_t lookup_start_time = ucs_get_time();
    for (unsigned i = 0; i < count; ++i) {
        void *ptr GTEST_ATTRIBUTE_UNUSED_;
        int present = ucs_ptr_array_lookup(&pa, i, ptr);
        ASSERT_TRUE(present);
    }

    ucs_time_t foreach_start_time = ucs_get_time();
    unsigned index;
    void *element;
    unsigned count_elements = 0;
    ucs_ptr_array_for_each(element, index, &pa) {
        void *ptr GTEST_ATTRIBUTE_UNUSED_;
        int present = ucs_ptr_array_lookup(&pa, index, ptr);
        element = NULL;
        ASSERT_TRUE(present);
        ASSERT_TRUE(element == NULL);
        count_elements++;
    }

    EXPECT_EQ(count_elements, count);

    ucs_time_t remove_start_time = ucs_get_time();
    for (unsigned i = 0; i < count; ++i) {
        ucs_ptr_array_remove(&pa, i);
    }

    ucs_time_t end_time = ucs_get_time();

    ucs_ptr_array_cleanup(&pa, 1);

    double insert_ns = ucs_time_to_nsec(lookup_start_time  - insert_start_time) / count;
    double lookup_ns = ucs_time_to_nsec(foreach_start_time - lookup_start_time) / count;
    double foreach_ns = ucs_time_to_nsec(remove_start_time - foreach_start_time) / count;
    double remove_ns = ucs_time_to_nsec(end_time           - remove_start_time) / count;

    UCS_TEST_MESSAGE << "Timings (nsec): insert " << insert_ns << " lookup: " <<
      lookup_ns << " remove: " << remove_ns << " Foreach: " << foreach_ns;

    if (ucs::perf_retry_count) {
        EXPECT_LT(insert_ns, 1000.0);
        EXPECT_LT(remove_ns, 1000.0);

        if (ucs_arch_get_cpu_vendor() != UCS_CPU_VENDOR_GENERIC_ARM) {
            EXPECT_LT(lookup_ns, 60.0);
        } else {
            EXPECT_LT(lookup_ns, 100.0);
        }
    }
}

UCS_TEST_F(test_datatype, ptr_status) {
    void *ptr1 = (void*)(UCS_BIT(63) + 10);
    EXPECT_TRUE(UCS_PTR_IS_PTR(ptr1));
    EXPECT_FALSE(UCS_PTR_IS_PTR(NULL));
    EXPECT_FALSE(UCS_PTR_IS_ERR(NULL));
    EXPECT_FALSE(UCS_PTR_IS_ERR(ptr1));

    void *ptr2 = (void*)(uintptr_t)(UCS_ERR_LAST + 1);
    EXPECT_TRUE(UCS_PTR_IS_ERR(ptr2));
}

UCS_TEST_F(test_datatype, ptr_array_locked_basic) {
    ucs_ptr_array_locked_t pa;
    int a = 1, b = 2, c = 3, d = 4, e = 5, f = 6, g = 7;
    unsigned index;

    ucs_ptr_array_locked_init(&pa, "ptr_array_locked test");

    index = ucs_ptr_array_locked_insert(&pa, &a);
    EXPECT_EQ(0u, index);

    index = ucs_ptr_array_locked_insert(&pa, &b);
    EXPECT_EQ(1u, index);

    index = ucs_ptr_array_locked_insert(&pa, &c);
    EXPECT_EQ(2u, index);

    ucs_ptr_array_locked_set(&pa, 3, &d);

    index = ucs_ptr_array_locked_insert(&pa, &e);
    EXPECT_EQ(4u, index);

    ucs_ptr_array_locked_set(&pa, 6, &f);

    ucs_ptr_array_locked_set(&pa, 100, &g);

    index = ucs_ptr_array_locked_bulk_alloc(&pa, 200);
    EXPECT_EQ(301u, pa.super.size);
    EXPECT_EQ(101u, index);

    void *vc;
    int present = ucs_ptr_array_locked_lookup(&pa, 2, &vc);
    ASSERT_TRUE(present);
    EXPECT_EQ(&c, vc);

    vc = ucs_ptr_array_locked_replace(&pa, 2, &d);
    EXPECT_EQ(&c, vc);

    present = ucs_ptr_array_locked_lookup(&pa, 2, &vc);
    EXPECT_EQ(&d, vc);

    ucs_ptr_array_locked_set(&pa, 2, &g);
    present = ucs_ptr_array_locked_lookup(&pa, 2, &vc);
    EXPECT_EQ(&g, vc);

    present = ucs_ptr_array_locked_lookup(&pa, 6, &vc);
    EXPECT_EQ(&f, vc);

    present = ucs_ptr_array_locked_lookup(&pa, 100, &vc);
    EXPECT_EQ(&g, vc);

    present = ucs_ptr_array_locked_lookup(&pa, 101, &vc);
    EXPECT_EQ(NULL, vc);

    present = ucs_ptr_array_locked_lookup(&pa, 301, &vc);
    EXPECT_EQ(NULL, vc);

    EXPECT_FALSE(ucs_ptr_array_locked_lookup(&pa, 5, &vc));
    EXPECT_FALSE(ucs_ptr_array_locked_lookup(&pa, 99, &vc));
    EXPECT_FALSE(ucs_ptr_array_locked_lookup(&pa, 302, &vc));
    EXPECT_FALSE(ucs_ptr_array_locked_lookup(&pa, 5005, &vc));

    ucs_ptr_array_locked_remove(&pa, 0);
    ucs_ptr_array_locked_remove(&pa, 1);
    ucs_ptr_array_locked_remove(&pa, 2);
    ucs_ptr_array_locked_remove(&pa, 3);
    ucs_ptr_array_locked_remove(&pa, 4);
    ucs_ptr_array_locked_remove(&pa, 6);
    ucs_ptr_array_locked_remove(&pa, 100);

    ucs_ptr_array_locked_cleanup(&pa, 0);
}

UCS_TEST_F(test_datatype, ptr_array_locked_random) {
    const unsigned count = 10000 / ucs::test_time_multiplier();
    ucs_ptr_array_locked_t pa;

    ucs_ptr_array_locked_init(&pa, "ptr_array test");

    std::map<int, void*> map;

    /* Insert phase */
    for (unsigned i = 0; i < count; ++i) {
        void *ptr = malloc(0);
        unsigned index = ucs_ptr_array_locked_insert(&pa, ptr);

        EXPECT_TRUE(map.end() == map.find(index));
        map[index] = ptr;
    }

    /* Remove + insert */
    for (unsigned i = 0; i < count / 10; ++i) {
        int remove_count = ucs::rand() % 10;
        for (int j = 0; j < remove_count; ++j) {
            unsigned to_remove = ucs::rand() % map.size();
            std::map<int, void*>::iterator iter = map.begin();
            std::advance(iter, to_remove);
            unsigned index = iter->first;

            void *ptr = NULL;
            EXPECT_TRUE(ucs_ptr_array_locked_lookup(&pa, index, &ptr));
            EXPECT_EQ(ptr, map[index]);
            free(ptr);

            ucs_ptr_array_locked_remove(&pa, index);

            EXPECT_FALSE(ucs_ptr_array_locked_lookup(&pa, index, &ptr));

            map.erase(index);
        }

        int insert_count = ucs::rand() % 10;
        for (int j = 0; j < insert_count; ++j) {
            void *ptr = malloc(0);
            unsigned index = ucs_ptr_array_locked_insert(&pa, ptr);

            EXPECT_TRUE(map.end() == map.find(index));
            map[index] = ptr;
        }
    }

    /* remove all */
    void *ptr;
    unsigned index;
    ucs_ptr_array_locked_for_each(ptr, index, &pa) {
        EXPECT_EQ(ptr, map[index]);
        ucs_ptr_array_locked_remove(&pa, index);
        free(ptr);
    }

    ucs_ptr_array_locked_cleanup(&pa, 1);
}

UCS_TEST_SKIP_COND_F(test_datatype, ptr_array_locked_perf,
                     (ucs::test_time_multiplier() > 1)) {
    const unsigned count = 10000000;
    ucs_ptr_array_locked_t pa;

    ucs_time_t insert_start_time = ucs_get_time();
    ucs_ptr_array_locked_init(&pa, "ptr_array test");
    for (unsigned i = 0; i < count; ++i) {
        EXPECT_EQ(i, ucs_ptr_array_locked_insert(&pa, NULL));
    }

    ucs_time_t lookup_start_time = ucs_get_time();
    for (unsigned i = 0; i < count; ++i) {
        void *ptr GTEST_ATTRIBUTE_UNUSED_;
        int present = ucs_ptr_array_locked_lookup(&pa, i, &ptr);
        ASSERT_TRUE(present);
    }

    ucs_time_t foreach_start_time = ucs_get_time();
    unsigned index;
    void *element;
    ucs_ptr_array_locked_for_each(element, index, &pa) {
        void *ptr GTEST_ATTRIBUTE_UNUSED_;
        int present = ucs_ptr_array_locked_lookup(&pa, index, &ptr);
        ASSERT_TRUE(present);
        ASSERT_TRUE(element == NULL);
    }

    ucs_time_t remove_start_time = ucs_get_time();
    for (unsigned i = 0; i < count; ++i) {
        ucs_ptr_array_locked_remove(&pa, i);
    }

    ucs_time_t end_time = ucs_get_time();

    ucs_ptr_array_locked_cleanup(&pa, 1);

    double insert_ns = ucs_time_to_nsec(lookup_start_time  - insert_start_time) / count;
    double lookup_ns = ucs_time_to_nsec(foreach_start_time - lookup_start_time) / count;
    double foreach_ns = ucs_time_to_nsec(remove_start_time - foreach_start_time) / count;
    double remove_ns = ucs_time_to_nsec(end_time           - remove_start_time) / count;

    UCS_TEST_MESSAGE << "Locked array timings (nsec): insert " << insert_ns << " lookup: " <<
      lookup_ns << " remove: " << remove_ns << " Foreach: " << foreach_ns;

    if (ucs::perf_retry_count) {
        EXPECT_LT(insert_ns, 1000.0);
        EXPECT_LT(remove_ns, 1000.0);

        if (ucs_arch_get_cpu_vendor() != UCS_CPU_VENDOR_GENERIC_ARM) {
            EXPECT_LT(lookup_ns, 60.0);
        } else {
            EXPECT_LT(lookup_ns, 100.0);
        }
    }
}

typedef struct {
    int  num1;
    int  num2;
} test_value_type_t;

static bool operator==(const test_value_type_t& v1, const test_value_type_t &v2) {
    return (v1.num1 == v2.num1) && (v1.num2 == v2.num2);
}

static std::ostream& operator<<(std::ostream& os, const test_value_type_t &v) {
    return os << "<" << v.num1 << "," << v.num2 << ">";
}


UCS_ARRAY_DEFINE_INLINE(test_2num, unsigned, test_value_type_t);
UCS_ARRAY_DEFINE_INLINE(test_1int, size_t, int);

class test_array : public test_datatype {
protected:
    void test_fixed(ucs_array_t(test_1int) *array, size_t capacity);
};

UCS_TEST_F(test_array, dynamic_array_2int_grow) {
    ucs_array_t(test_2num) test_array;
    test_value_type_t value;
    ucs_status_t status;

    ucs_array_init_dynamic(&test_array);
    EXPECT_FALSE(ucs_array_is_fixed(&test_array));

    /* grow the array enough to contain 'value_index' */
    unsigned value_index = 9;
    status = ucs_array_reserve(test_2num, &test_array, value_index + 1);
    ASSERT_UCS_OK(status);

    value.num1 = ucs::rand();
    value.num2 = ucs::rand();

    /* save the value in the array and check it's there */
    ucs_array_elem(&test_array, value_index) = value;
    EXPECT_EQ(value, ucs_array_elem(&test_array, value_index));

    /* grow the array to larger size, check the value is not changed */
    status = ucs_array_reserve(test_2num, &test_array, 40);
    ASSERT_UCS_OK(status);
    EXPECT_EQ(value, ucs_array_elem(&test_array, value_index));

    /* grow the array with smaller size, check the value is not changed */
    status = ucs_array_reserve(test_2num, &test_array, 30);
    ASSERT_UCS_OK(status);
    EXPECT_EQ(value, ucs_array_elem(&test_array, value_index));

    ucs_array_cleanup_dynamic(&test_array);
}

UCS_TEST_F(test_array, dynamic_array_int_append) {
    static const size_t NUM_ELEMS = 1000;

    ucs_array_t(test_1int) test_array;
    std::vector<int> vec;
    ucs_status_t status;

    ucs_array_init_dynamic(&test_array);
    EXPECT_FALSE(ucs_array_is_fixed(&test_array));

    /* push same elements to the array and the std::vector */
    for (size_t i = 0; i < NUM_ELEMS; ++i) {
        int value = ucs::rand();
        status = ucs_array_append(test_1int, &test_array);
        ASSERT_UCS_OK(status);
        EXPECT_EQ(i + 1, ucs_array_length(&test_array));
        *ucs_array_last(&test_array) = value;
        vec.push_back(value);
    }

    /* validate array size and capacity */
    EXPECT_EQ(NUM_ELEMS, ucs_array_length(&test_array));
    EXPECT_GE(ucs_array_capacity(&test_array), NUM_ELEMS);

    /* validate array contents  */
    for (size_t i = 0; i < NUM_ELEMS; ++i) {
        EXPECT_EQ(vec[i], ucs_array_elem(&test_array, i));
    }

    /* test for_each */
    int *array_iter;
    std::vector<int>::iterator vec_iter = vec.begin();
    ucs_array_for_each(array_iter, &test_array) {
        EXPECT_EQ(*vec_iter, *array_iter);
        ++vec_iter;
    }

    /* test set_length */
    size_t new_length = NUM_ELEMS * 2 / 3;
    ucs_array_set_length(&test_array, new_length);
    EXPECT_EQ(new_length, ucs_array_length(&test_array));

    ucs_array_set_length(&test_array, NUM_ELEMS);
    EXPECT_EQ(NUM_ELEMS, ucs_array_length(&test_array));

    /* set length to max capacity */
    new_length = ucs_array_capacity(&test_array);
    ucs_array_set_length(&test_array, new_length);
    EXPECT_EQ(new_length, ucs_array_length(&test_array));

    ucs_array_cleanup_dynamic(&test_array);
}

void test_array::test_fixed(ucs_array_t(test_1int) *array, size_t capacity)
{
    ucs_status_t status;

    /* check initial capacity */
    size_t initial_capacity = ucs_array_capacity(array);
    EXPECT_LE(initial_capacity, capacity);
    EXPECT_GE(initial_capacity, capacity - 1);

    /* append one element */
    status = ucs_array_append(test_1int, array);
    ASSERT_UCS_OK(status);

    size_t idx = ucs_array_length(array) - 1;
    ucs_array_elem(array, idx) = 17;
    EXPECT_EQ(0u, idx);
    EXPECT_EQ(1u, ucs_array_length(array));

    /* check end capacity */
    EXPECT_EQ(initial_capacity - 1, ucs_array_available_length(array));
    EXPECT_EQ(&ucs_array_elem(array, 1), ucs_array_end(array));
}

UCS_TEST_F(test_array, fixed_static) {
    const size_t num_elems = 100;
    int buffer[num_elems];
    ucs_array_t(test_1int) test_array =
             UCS_ARRAY_FIXED_INITIALIZER(buffer, num_elems);
    test_fixed(&test_array, num_elems);
}

UCS_TEST_F(test_array, fixed_init) {
    const size_t num_elems = 100;
    int buffer[num_elems];
    ucs_array_t(test_1int) test_array;

    ucs_array_init_fixed(&test_array, buffer, num_elems);
    test_fixed(&test_array, num_elems);
}

UCS_TEST_F(test_array, fixed_onstack) {
    const size_t num_elems = 100;
    UCS_ARRAY_DEFINE_ONSTACK(test_array, test_1int, num_elems);
    test_fixed(&test_array, num_elems);
}

class test_datatype_ptr_map : public test_datatype {
public:
    UCS_PTR_MAP_DEFINE(unsafe_put, 0);

    typedef std::map<ucs_ptr_map_key_t, int*> std_map_t;
    typedef std::vector<int> std_vec_t;

    static const size_t vec_size = 10 * (1 << 10);
};

UCS_TEST_F(test_datatype_ptr_map, unsafe_put) {
    UCS_PTR_MAP_T(unsafe_put) ptr_map;

    std_map_t std_map;
    std_vec_t std_vec(vec_size, 0);

    ASSERT_UCS_OK(UCS_PTR_MAP_INIT(unsafe_put, &ptr_map));

    for (auto it = std_vec.begin(); it != std_vec.end(); ++it) {
        bool indirect = ucs::rand() % 2;
        ucs_ptr_map_key_t ptr_key;
        ucs_status_t status = UCS_PTR_MAP_PUT(unsafe_put, &ptr_map, &(*it),
                                              indirect, &ptr_key);

        if (indirect) {
            ASSERT_UCS_OK(status);
            EXPECT_TRUE(ucs_ptr_map_key_indirect(ptr_key));
        } else {
            ASSERT_EQ(UCS_ERR_NO_PROGRESS, status);
            EXPECT_FALSE(ucs_ptr_map_key_indirect(ptr_key));
        }

        std_map[ptr_key] = &(*it);
    }

    for (auto i = std_map.begin(); i != std_map.end(); ++i) {
        bool indirect = ucs_ptr_map_key_indirect(i->first);
        bool extract  = ucs::rand() % 2;
        void *value;
        ucs_status_t status = UCS_PTR_MAP_GET(unsafe_put, &ptr_map, i->first,
                                              extract, &value);
        if (indirect) {
            ASSERT_EQ(UCS_OK, status);
        } else {
            ASSERT_EQ(UCS_ERR_NO_PROGRESS, status);
        }

        ASSERT_EQ(i->second, value);
        if (!extract) {
            status = UCS_PTR_MAP_DEL(unsafe_put, &ptr_map, i->first);
            if (indirect) {
                ASSERT_EQ(UCS_OK, status);
            } else {
                ASSERT_EQ(UCS_ERR_NO_PROGRESS, status);
            }
        }
    }

    UCS_PTR_MAP_DESTROY(unsafe_put, &ptr_map);
}

class test_datatype_ptr_map_safe : public test_datatype_ptr_map {
public:
    test_datatype_ptr_map_safe()
    {
        pthread_mutex_init(&m_mutex, NULL);
        ASSERT_UCS_OK(UCS_PTR_MAP_INIT(safe_put, &m_ptr_map));
    }

    ~test_datatype_ptr_map_safe()
    {
        UCS_PTR_MAP_DESTROY(safe_put, &m_ptr_map);
        pthread_mutex_destroy(&m_mutex);
    }

    UCS_PTR_MAP_DEFINE(safe_put, 1);

    static const unsigned num_threads = 4;

protected:
    pthread_mutex_t m_mutex;
    UCS_PTR_MAP_T(safe_put) m_ptr_map;
};


UCS_MT_TEST_F(test_datatype_ptr_map_safe, safe_put, num_threads)
{
    std_map_t std_map;
    std_vec_t std_vec(vec_size / num_threads, 0);

    for (auto it = std_vec.begin(); it != std_vec.end(); ++it) {
        bool indirect = ucs::rand() % 2;
        ucs_ptr_map_key_t ptr_key;
        ucs_status_t status = UCS_PTR_MAP_PUT(safe_put, &m_ptr_map, &(*it),
                                              indirect, &ptr_key);

        if (indirect) {
            ASSERT_UCS_OK(status);
            EXPECT_TRUE(ucs_ptr_map_key_indirect(ptr_key));
        } else {
            ASSERT_EQ(UCS_ERR_NO_PROGRESS, status);
            EXPECT_FALSE(ucs_ptr_map_key_indirect(ptr_key));
        }

        std_map[ptr_key] = &(*it);
    }

    {
        ucs::scoped_mutex_lock lock(m_mutex);
        for (auto i = std_map.begin(); i != std_map.end(); ++i) {
            bool indirect = ucs_ptr_map_key_indirect(i->first);
            bool extract  = ucs::rand() % 2;
            void *value;
            ucs_status_t status = UCS_PTR_MAP_GET(safe_put, &m_ptr_map,
                                                  i->first, extract, &value);
            if (indirect) {
                ASSERT_EQ(UCS_OK, status);
            } else {
                ASSERT_EQ(UCS_ERR_NO_PROGRESS, status);
            }

            ASSERT_EQ(i->second, value);
            if (!extract) {
                status = UCS_PTR_MAP_DEL(safe_put, &m_ptr_map, i->first);
                if (indirect) {
                    ASSERT_EQ(UCS_OK, status);
                } else {
                    ASSERT_EQ(UCS_ERR_NO_PROGRESS, status);
                }
            }
        }
    }
}
