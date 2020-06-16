/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* Copyright (C) Huawei Technologies Co., Ltd. 2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/datastruct/list.h>
#include <ucs/datastruct/hlist.h>
#include <ucs/datastruct/ptr_array.h>
#include <ucs/datastruct/queue.h>
#include <ucs/time/time.h>
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
    elem = ucs_list_extract_head_elem(&head, elem_t, hlist);
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

        ucs_queue_push(&head, &elem1.queue);
        EXPECT_EQ((unsigned long)2, ucs_queue_length(&head));

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

    index = ucs_ptr_array_insert(&pa, &a);
    EXPECT_EQ(0u, index);

    index = ucs_ptr_array_insert(&pa, &b);
    EXPECT_EQ(1u, index);

    index = ucs_ptr_array_insert(&pa, &c);
    EXPECT_EQ(2u, index);

    ucs_ptr_array_set(&pa, 3, &d);

    index = ucs_ptr_array_insert(&pa, &e);
    EXPECT_EQ(4u, index);

    ucs_ptr_array_set(&pa, 6, &f);

    ucs_ptr_array_set(&pa, 100, &g);

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

    EXPECT_FALSE(ucs_ptr_array_lookup(&pa, 5, vc));
    EXPECT_FALSE(ucs_ptr_array_lookup(&pa, 99, vc));
    EXPECT_FALSE(ucs_ptr_array_lookup(&pa, 101, vc));
    EXPECT_FALSE(ucs_ptr_array_lookup(&pa, 5005, vc));

    ucs_ptr_array_remove(&pa, 0);
    ucs_ptr_array_remove(&pa, 1);
    ucs_ptr_array_remove(&pa, 2);
    ucs_ptr_array_remove(&pa, 3);
    ucs_ptr_array_remove(&pa, 4);
    ucs_ptr_array_remove(&pa, 6);
    ucs_ptr_array_remove(&pa, 100);

    ucs_ptr_array_cleanup(&pa);
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

    ucs_ptr_array_cleanup(&pa);
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

    ucs_ptr_array_cleanup(&pa);

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

    EXPECT_FALSE(ucs_ptr_array_locked_lookup(&pa, 5, &vc));
    EXPECT_FALSE(ucs_ptr_array_locked_lookup(&pa, 99, &vc));
    EXPECT_FALSE(ucs_ptr_array_locked_lookup(&pa, 101, &vc));
    EXPECT_FALSE(ucs_ptr_array_locked_lookup(&pa, 5005, &vc));

    ucs_ptr_array_locked_remove(&pa, 0);
    ucs_ptr_array_locked_remove(&pa, 1);
    ucs_ptr_array_locked_remove(&pa, 2);
    ucs_ptr_array_locked_remove(&pa, 3);
    ucs_ptr_array_locked_remove(&pa, 4);
    ucs_ptr_array_locked_remove(&pa, 6);
    ucs_ptr_array_locked_remove(&pa, 100);

    ucs_ptr_array_locked_cleanup(&pa);
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

    ucs_ptr_array_locked_cleanup(&pa);
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

    ucs_ptr_array_locked_cleanup(&pa);

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

