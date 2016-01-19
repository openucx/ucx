/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>

extern "C" {
#include <ucs/datastruct/mpmc.h>
}
#include <pthread.h>


class test_mpmc : public ucs::test {
protected:
    static const unsigned MPMC_SIZE = 100;
    static const uint32_t SENTINEL  = 0x7fffffffu;
    static const unsigned NUM_THREADS = 4;


    static long elem_count() {
        return ucs_max((long)(100000.0 / (pow(ucs::test_time_multiplier(), NUM_THREADS))),
                       500l);
    }

    static void * producer_thread_func(void *arg) {
        ucs_mpmc_queue_t *mpmc = reinterpret_cast<ucs_mpmc_queue_t*>(arg);
        long count = elem_count();
        ucs_status_t status;

        for (uint32_t i = 0; i < count; ++i) {
            do {
                status = ucs_mpmc_queue_push(mpmc, i);
            } while (status == UCS_ERR_EXCEEDS_LIMIT);
            ASSERT_UCS_OK(status);
        }
        do {
            status = ucs_mpmc_queue_push(mpmc, SENTINEL);
        } while (status == UCS_ERR_EXCEEDS_LIMIT);
        return NULL;
    }

    static void * consumer_thread_func(void *arg) {
        ucs_mpmc_queue_t *mpmc = reinterpret_cast<ucs_mpmc_queue_t*>(arg);
        ucs_status_t status;
        uint32_t value;
        size_t count;

        count = 0;
        do {
            do {
                status = ucs_mpmc_queue_pull(mpmc, &value);
            } while (status == UCS_ERR_NO_PROGRESS);
            ASSERT_UCS_OK(status);
            ++count;
        } while (value != SENTINEL);

        return (void*)((uintptr_t)count - 1); /* return count except sentinel */
    }

};

UCS_TEST_F(test_mpmc, basic) {
    ucs_mpmc_queue_t mpmc;
    ucs_status_t status;

    status = ucs_mpmc_queue_init(&mpmc, MPMC_SIZE);
    ASSERT_UCS_OK(status);

    EXPECT_TRUE(ucs_mpmc_queue_is_empty(&mpmc));

    status = ucs_mpmc_queue_push(&mpmc, 124);
    ASSERT_UCS_OK(status);

    status = ucs_mpmc_queue_push(&mpmc, 125);
    ASSERT_UCS_OK(status);

    status = ucs_mpmc_queue_push(&mpmc, 126);
    ASSERT_UCS_OK(status);

    EXPECT_FALSE(ucs_mpmc_queue_is_empty(&mpmc));

    uint32_t value;

    status = ucs_mpmc_queue_pull(&mpmc, &value);
    ASSERT_UCS_OK(status);
    EXPECT_EQ(124u, value);

    status = ucs_mpmc_queue_pull(&mpmc, &value);
    ASSERT_UCS_OK(status);
    EXPECT_EQ(125u, value);

    status = ucs_mpmc_queue_pull(&mpmc, &value);
    ASSERT_UCS_OK(status);
    EXPECT_EQ(126u, value);

    EXPECT_TRUE(ucs_mpmc_queue_is_empty(&mpmc));

    ucs_mpmc_queue_cleanup(&mpmc);
}


UCS_TEST_F(test_mpmc, multi_threaded) {
    pthread_t producers[NUM_THREADS];
    pthread_t consumers[NUM_THREADS];

    ucs_mpmc_queue_t mpmc;
    ucs_status_t status;
    size_t total;
    void *retval;

    status = ucs_mpmc_queue_init(&mpmc, MPMC_SIZE);
    ASSERT_UCS_OK(status);

    for (unsigned i = 0; i < NUM_THREADS; ++i) {
        pthread_create(&producers[i], NULL, producer_thread_func, &mpmc);
        pthread_create(&consumers[i], NULL, consumer_thread_func, &mpmc);
    }

    total = 0;
    for (unsigned i = 0; i < NUM_THREADS; ++i) {
        pthread_join(producers[i], &retval);
        pthread_join(consumers[i], &retval);
        total += (uintptr_t)retval;
    }

    EXPECT_EQ(NUM_THREADS * elem_count(), (long)total);
    EXPECT_TRUE(ucs_mpmc_queue_is_empty(&mpmc));
    ucs_mpmc_queue_cleanup(&mpmc);
}
