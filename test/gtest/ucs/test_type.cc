/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2016. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
extern "C" {
#include <ucs/type/cpu_set.h>
#include <ucs/type/init_once.h>
#include <ucs/type/serialize.h>
#include <ucs/type/status.h>
#include <ucs/type/float8.h>
#include <ucs/type/rwlock.h>
}

#include <time.h>
#include <thread>
#include <chrono>
#include <vector>

class test_type : public ucs::test {
};

UCS_TEST_F(test_type, cpu_set) {
    ucs_cpu_set_t cpu_mask;

    UCS_CPU_ZERO(&cpu_mask);
    EXPECT_FALSE(ucs_cpu_is_set(117, &cpu_mask));
    EXPECT_FALSE(ucs_cpu_is_set(127, &cpu_mask));
    EXPECT_EQ(0, ucs_cpu_set_find_lcs(&cpu_mask));

    UCS_CPU_SET(127, &cpu_mask);
    UCS_CPU_SET(117, &cpu_mask);
    EXPECT_TRUE(ucs_cpu_is_set(117, &cpu_mask));
    EXPECT_TRUE(ucs_cpu_is_set(127, &cpu_mask));
    EXPECT_EQ(117, ucs_cpu_set_find_lcs(&cpu_mask));

    UCS_CPU_CLR(117, &cpu_mask);
    EXPECT_FALSE(ucs_cpu_is_set(117, &cpu_mask));
    EXPECT_TRUE(ucs_cpu_is_set(127, &cpu_mask));
    EXPECT_EQ(127, ucs_cpu_set_find_lcs(&cpu_mask));

    UCS_CPU_CLR(127, &cpu_mask);
    EXPECT_FALSE(ucs_cpu_is_set(117, &cpu_mask));
    EXPECT_FALSE(ucs_cpu_is_set(127, &cpu_mask));
    EXPECT_EQ(0, ucs_cpu_set_find_lcs(&cpu_mask));
}

UCS_TEST_F(test_type, status) {
    void *ptr = (void*)0xff00000000ul;
    EXPECT_TRUE(UCS_PTR_IS_PTR(ptr));
    EXPECT_FALSE(UCS_PTR_IS_PTR(NULL));
    EXPECT_NE(UCS_OK, UCS_PTR_STATUS(ptr));
}

UCS_TEST_F(test_type, serialize) {
    std::vector<uint8_t> data(100);
    const size_t raw_field_size = 3;

    std::vector<uint64_t> values;
    values.push_back(ucs::rand() % UINT8_MAX);
    values.push_back(ucs::rand() % UINT32_MAX);
    for (unsigned i = 0; i < 3; ++i) {
        values.push_back(ucs::rand() * ucs::rand());
    }
    values.push_back(ucs::rand() % UCS_BIT(raw_field_size * 8));

    /* Pack */
    uint64_t *p64;
    void *pack_ptr = &data[0];

    *ucs_serialize_next(&pack_ptr, uint8_t)  = values[0];
    *ucs_serialize_next(&pack_ptr, uint32_t) = values[1];
    *ucs_serialize_next(&pack_ptr, uint64_t) = values[2];
    p64  = ucs_serialize_next(&pack_ptr, uint64_t);
    *p64 = values[3];
    *ucs_serialize_next(&pack_ptr, uint64_t) = values[4];
    /* Pack raw 3-byte value */
    memcpy(ucs_serialize_next_raw(&pack_ptr, void, raw_field_size), &values[5],
           raw_field_size);
    EXPECT_EQ(1 + 4 + (3 * 8) + raw_field_size,
              UCS_PTR_BYTE_DIFF(&data[0], pack_ptr));

    /* Unpack */
    const void *unpack_ptr = &data[0];
    uint64_t value;
    value = *ucs_serialize_next(&unpack_ptr, const uint8_t);
    EXPECT_EQ(values[0], value);
    value = *ucs_serialize_next(&unpack_ptr, const uint32_t);
    EXPECT_EQ(values[1], value);
    for (unsigned i = 0; i < 3; ++i) {
        value = *ucs_serialize_next(&unpack_ptr, const uint64_t);
        EXPECT_EQ(values[2 + i], value);
    }
    /* Unpack raw 3-byte value */
    value = 0;
    memcpy(&value, ucs_serialize_next_raw(&unpack_ptr, void, raw_field_size),
           raw_field_size);
    EXPECT_EQ(values[5], value);

    EXPECT_EQ(pack_ptr, unpack_ptr);
}

/* Represents latency (in ns) */
UCS_FP8_DECLARE_TYPE(TEST_LATENCY, UCS_BIT(7), UCS_BIT(20))

UCS_TEST_F(test_type, pack_float) {
    const std::size_t values_size    = 10;
    double values_array[values_size] = {
        130, 135.1234, 140, 200, 400, 1000, 10000, 100000, 1000000, 1000000
    };
    std::vector<double> values(values_array, values_array + values_size);
    float unpacked;

    /* 0 -> 0 */
    unpacked = UCS_FP8_PACK_UNPACK(TEST_LATENCY, 0);
    EXPECT_EQ(unpacked, 0);

    /* NaN -> NaN */
    unpacked = UCS_FP8_PACK_UNPACK(TEST_LATENCY, NAN);
    EXPECT_TRUE(isnan(unpacked));

    /* Below min -> min */
    for (uint64_t min_val = UCS_BIT(0); min_val <= UCS_BIT(7); min_val <<= 1) {
        UCS_TEST_MESSAGE << " Pack/unpack " << min_val;
        EXPECT_EQ(
         UCS_FP8_PACK_UNPACK(TEST_LATENCY, UCS_BIT(7)),
         UCS_FP8_PACK_UNPACK(TEST_LATENCY, min_val));
    }

    /* Precision test throughout the whole range */
    for (std::vector<double>::const_iterator it = values.begin();
         it < values.end(); it++) {
        unpacked = UCS_FP8_PACK_UNPACK(TEST_LATENCY, *it);
        ucs_assert((UCS_FP8_PRECISION < unpacked / *it) &&
                   (unpacked / *it <= 1));
    }

    /* Above max -> max */
    EXPECT_EQ(UCS_FP8_PACK_UNPACK(TEST_LATENCY, UCS_BIT(20)),
              UCS_FP8_PACK_UNPACK(TEST_LATENCY, 200000000));
}

class test_rwlock : public ucs::test {
protected:
    void sleep()
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    void measure_one(int num, int writers, const std::function<void()> &r,
                     const std::function<void()> &w, const std::string &name)
    {
        std::vector<std::thread> tt;

        tt.reserve(num);
        auto start = std::chrono::high_resolution_clock::now();
        for (int c = 0; c < num; c++) {
            tt.emplace_back([&]() {
                unsigned seed = time(0);
                for (int i = 0; i < 1000000 / num; i++) {
                    if ((rand_r(&seed) % 256) < writers) {
                        w();
                    } else {
                        r();
                    }
                }
            });
        }


        for (auto &t : tt) {
            t.join();
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        UCS_TEST_MESSAGE << elapsed.count() * 1000 << " ms " << name << " "
                         << std::to_string(num) << " threads "
                         << std::to_string(writers) << " writers per 256 ";
    }

    void measure(const std::function<void()> &r,
                 const std::function<void()> &w, const std::string &name)
    {
        int m = std::thread::hardware_concurrency();
        std::vector<int> threads = {1, 2, 4, m};
        std::vector<int> writers_per_256 = {1, 25, 128, 250};

        for (auto t : threads) {
            for (auto writers : writers_per_256) {
                measure_one(t, writers, r, w, name);
            }
        }
    }
};

UCS_TEST_F(test_rwlock, lock) {
    ucs_rwlock_t lock = UCS_RWLOCK_STATIC_INITIALIZER;

    ucs_rwlock_read_lock(&lock);
    EXPECT_EQ(-EBUSY, ucs_rwlock_write_trylock(&lock));

    ucs_rwlock_read_lock(&lock); /* second read lock should pass */

    int write_taken = 0;
    std::thread w([&]() {
        ucs_rwlock_write_lock(&lock);
        write_taken = 1;
        ucs_rwlock_write_unlock(&lock);
    });
    sleep();
    EXPECT_FALSE(write_taken); /* write lock should wait for read lock release */

    ucs_rwlock_read_unlock(&lock);
    sleep();
    EXPECT_FALSE(write_taken); /* first read lock still holding lock */

    int read_taken = 0;
    std::thread r1([&]() {
        ucs_rwlock_read_lock(&lock);
        read_taken = 1;
        ucs_rwlock_read_unlock(&lock);
    });
    sleep();
    EXPECT_FALSE(read_taken); /* read lock should wait while write lock is waiting */

    ucs_rwlock_read_unlock(&lock);
    sleep();
    EXPECT_TRUE(write_taken); /* write lock should be taken */
    w.join();

    sleep();
    EXPECT_TRUE(read_taken); /* read lock should be taken */
    r1.join();

    EXPECT_EQ(0, ucs_rwlock_write_trylock(&lock));
    read_taken = 0;
    std::thread r2([&]() {
        ucs_rwlock_read_lock(&lock);
        read_taken = 1;
        ucs_rwlock_read_unlock(&lock);
    });
    sleep();
    EXPECT_FALSE(read_taken); /* read lock should wait for write lock release */

    ucs_rwlock_write_unlock(&lock);
    sleep();
    EXPECT_TRUE(read_taken); /* read lock should be taken */
    r2.join();
}

UCS_TEST_F(test_rwlock, perf) {
    ucs_rwlock_t lock = UCS_RWLOCK_STATIC_INITIALIZER;
    measure(
            [&]() {
                ucs_rwlock_read_lock(&lock);
                ucs_rwlock_read_unlock(&lock);
            },
            [&]() {
                ucs_rwlock_write_lock(&lock);
                ucs_rwlock_write_unlock(&lock);
            },
            "builtin");
}

UCS_TEST_F(test_rwlock, pthread) {
    pthread_rwlock_t plock;
    pthread_rwlock_init(&plock, NULL);
    measure(
            [&]() {
                pthread_rwlock_rdlock(&plock);
                pthread_rwlock_unlock(&plock);
            },
            [&]() {
                pthread_rwlock_wrlock(&plock);
                pthread_rwlock_unlock(&plock);
            },
            "pthread");
    pthread_rwlock_destroy(&plock);
}

class test_init_once: public test_type {
protected:
    test_init_once() : m_once(INIT_ONCE_INIT), m_count(0) {};

    /* counter is not atomic, we expect the lock of init_once will protect it */
    ucs_init_once_t m_once;
    int             m_count;

private:
    static const ucs_init_once_t INIT_ONCE_INIT;
};

const ucs_init_once_t test_init_once::INIT_ONCE_INIT = UCS_INIT_ONCE_INITIALIZER;

UCS_MT_TEST_F(test_init_once, init_once, 10) {

    for (int i = 0; i < 100; ++i) {
        UCS_INIT_ONCE(&m_once) {
            ++m_count;
        }
    }

    EXPECT_EQ(1, m_count);
}

