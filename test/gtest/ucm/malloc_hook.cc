/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#define __STDC_LIMIT_MACROS

#include <ucm/api/ucm.h>

#include <ucs/arch/atomic.h>
#include <ucs/type/status.h>
#include <common/test.h>
#include <common/test_helpers.h>
#include <pthread.h>
#include <sstream>
#include <stdint.h>

extern "C" {
#include <ucs/sys/sys.h>
#include <malloc.h>
}

class malloc_hook : public ucs::test {
protected:
    virtual void init() {
        size_t free_space, min_free_space, prev_free_space, alloc_size;
        struct mallinfo mi;

        malloc_trim(0);

        /* Take up free space so we would definitely get mmap events */
        min_free_space  = 0;
        prev_free_space = SIZE_MAX;
        alloc_size      = 0;
        do {
            mi = mallinfo();
            m_pts.push_back(malloc(small_alloc_size));
            alloc_size += small_alloc_size;
            free_space = mi.fsmblks + mi.fordblks;
            if (free_space > prev_free_space) {
                /* If the heap grew, the minimal size is the previous one */
                min_free_space = prev_free_space;
            }
            prev_free_space = free_space;
        } while (free_space >= min_free_space + (small_alloc_count * small_alloc_size));
        UCS_TEST_MESSAGE << "Reduced heap free space to minimum: " << free_space <<
                            " after allocating " << alloc_size << " bytes";
    }

public:
    static int          small_alloc_count;
    static const size_t small_alloc_size  = 10000;
    ucs::ptr_vector<void> m_pts;
};

int malloc_hook::small_alloc_count = 1000 / ucs::test_time_multiplier();

class test_thread {
public:
    test_thread(const std::string& name, int num_threads, pthread_barrier_t *barrier,
                malloc_hook *test) :
        m_name(name), m_num_threads(num_threads), m_barrier(barrier),
        m_map_size(0), m_unmap_size(0), m_test(test)
    {
        pthread_create(&m_thread, NULL, thread_func, reinterpret_cast<void*>(this));
    }

    ~test_thread() {
        join();
    }

    void join() {
        void *retval;
        pthread_join(m_thread, &retval);
    }

private:
    typedef std::pair<void*, void*> range;

    static void *thread_func(void *arg) {
        test_thread *self = reinterpret_cast<test_thread*>(arg);
        self->test();
        return NULL;
    }

    static void mem_event_callback(ucm_event_type_t event_type, ucm_event_t *event,
                                   void *arg)
    {
        test_thread *self = reinterpret_cast<test_thread*>(arg);
        self->mem_event(event_type, event);
    }

    bool is_ptr_in_range(void *ptr, size_t size, const std::vector<range> &ranges) {
        for (std::vector<range>::const_iterator iter = ranges.begin(); iter != ranges.end(); ++iter) {
            if ((ptr >= iter->first) && ((char*)ptr < iter->second)) {
                return true;
            }
        }
        return false;
    }

    void test();
    void mem_event(ucm_event_type_t event_type, ucm_event_t *event);

    static pthread_mutex_t   lock;
    static pthread_barrier_t barrier;

    std::string        m_name;
    int                m_num_threads;
    pthread_barrier_t  *m_barrier;
    pthread_t          m_thread;
    size_t             m_map_size;
    size_t             m_unmap_size;
    std::vector<range> m_map_ranges;
    std::vector<range> m_unmap_ranges;
    malloc_hook        *m_test;
};

pthread_mutex_t test_thread::lock = PTHREAD_MUTEX_INITIALIZER;

void test_thread::mem_event(ucm_event_type_t event_type, ucm_event_t *event)
{
    switch (event_type) {
    case UCM_EVENT_VM_MAPPED:
        m_map_ranges.push_back(range(event->vm_mapped.address,
                                     (char*)event->vm_mapped.address + event->vm_mapped.size));
        m_map_size += event->vm_mapped.size;
        break;
    case UCM_EVENT_VM_UNMAPPED:
        m_unmap_ranges.push_back(range(event->vm_unmapped.address,
                                       (char*)event->vm_unmapped.address + event->vm_unmapped.size));
        m_unmap_size += event->vm_unmapped.size;
        break;
    default:
        break;
    }
}

void test_thread::test() {
    static const size_t large_alloc_size = 40 * 1024 * 1024;
    ucs_status_t result;
    ucs::ptr_vector<void> old_ptrs;
    ucs::ptr_vector<void> new_ptrs;
    void *ptr_r;
    size_t small_map_size;
    int num_ptrs_in_range;
    static volatile uint32_t total_ptrs_in_range = 0;

    /* Allocate some pointers with old heap manager */
    for (unsigned i = 0; i < 10; ++i) {
        old_ptrs.push_back(malloc(m_test->small_alloc_size));
    }

    ptr_r = malloc(m_test->small_alloc_size);

    m_map_ranges.reserve  ((m_test->small_alloc_count * 8 + 10) * m_num_threads);
    m_unmap_ranges.reserve((m_test->small_alloc_count * 8 + 10) * m_num_threads);

    total_ptrs_in_range = 0;

    pthread_barrier_wait(m_barrier);

    /* Install memory hooks */
    result = ucm_set_event_handler(UCM_EVENT_VM_MAPPED|UCM_EVENT_VM_UNMAPPED,
                                   0, mem_event_callback,
                                   reinterpret_cast<void*>(this));
    ASSERT_UCS_OK(result);

    /* Allocate small pointers with new heap manager */
    for (int i = 0; i < m_test->small_alloc_count; ++i) {
        new_ptrs.push_back(malloc(m_test->small_alloc_size));
    }
    small_map_size = m_map_size;

    /* If this test runs more than once, then sbrk may not really allocate new
     * memory
     */
    EXPECT_GT(m_map_size, 0lu) << m_name;

    num_ptrs_in_range = 0;
    for (ucs::ptr_vector<void>::const_iterator iter = new_ptrs.begin();
                    iter != new_ptrs.end(); ++iter)
    {
        if (is_ptr_in_range(*iter, m_test->small_alloc_size, m_map_ranges)) {
            ++num_ptrs_in_range;
        }
    }

    /* Need at least one ptr in the mapped ranges in one the threads */
    ucs_atomic_add32(&total_ptrs_in_range, num_ptrs_in_range);
    pthread_barrier_wait(m_barrier);

    EXPECT_GT(total_ptrs_in_range, 0u);

    /* Allocate large chunk */
    void *ptr = malloc(large_alloc_size);
    EXPECT_GE(m_map_size, large_alloc_size + small_map_size) << m_name;
    EXPECT_TRUE(is_ptr_in_range(ptr, large_alloc_size, m_map_ranges)) << m_name;

    free(ptr);
    EXPECT_GE(m_unmap_size, large_alloc_size) << m_name;
    /* coverity[pass_freed_arg] */
    EXPECT_TRUE(is_ptr_in_range(ptr, large_alloc_size, m_unmap_ranges)) << m_name;

    /* Test strdup */
    void *s = strdup("test");
    free(s);

    /* Test setenv */
    pthread_mutex_lock(&lock);
    setenv("TEST", "VALUE", 1);
    EXPECT_EQ(std::string("VALUE"), getenv("TEST"));
    pthread_mutex_unlock(&lock);

    /* Test username */
    ucs_get_user_name();

    /* Test realloc */
    ptr_r = realloc(ptr_r, m_test->small_alloc_size / 2);
    free(ptr_r);

    /* Release old pointers (should not crash) */
    old_ptrs.clear();

    m_map_ranges.clear();
    m_unmap_ranges.clear();

    /* Don't release pointers before other threads exit, so they will map new memory
     * and not reuse memory from other threads.
     */
    pthread_barrier_wait(m_barrier);

    /* Release new pointers  */
    new_ptrs.clear();

    /* Call several malloc routines */
    malloc_trim(0);

    ptr = malloc(large_alloc_size);
    if (!RUNNING_ON_VALGRIND) {
        void *state = malloc_get_state();
        malloc_set_state(state);
    }
    free(ptr);

    pthread_mutex_lock(&lock);
    UCS_TEST_MESSAGE << m_name
                     << ": small mapped: " << small_map_size
                     <<  ", total mapped: " << m_map_size
                     <<  ", total unmapped: " << m_unmap_size;
    std::cout.flush();
    pthread_mutex_unlock(&lock);

    ucm_unset_event_handler(UCM_EVENT_VM_MAPPED|UCM_EVENT_VM_UNMAPPED,
                            mem_event_callback,
                            reinterpret_cast<void*>(this));
}

UCS_TEST_F(malloc_hook, single_thread) {
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, 1);
    {
        test_thread thread("single-thread", 1, &barrier, this);
    }
    pthread_barrier_destroy(&barrier);
}

UCS_TEST_F(malloc_hook, multi_threads) {
    static const int num_threads = 8;
    ucs::ptr_vector<test_thread> threads;
    pthread_barrier_t barrier;

    malloc_trim(0);

    pthread_barrier_init(&barrier, NULL, num_threads);
    for (int i = 0; i < num_threads; ++i) {
        std::stringstream ss;
        ss << "thread " << i << "/" << num_threads;
        threads.push_back(new test_thread(ss.str(), num_threads, &barrier, this));
    }

    threads.clear();
    pthread_barrier_destroy(&barrier);
}

UCS_TEST_F(malloc_hook, fork) {
    static const int num_processes = 4;
    pthread_barrier_t barrier;
    std::vector<pid_t> pids;
    pid_t pid;

    for (int i = 0; i < num_processes; ++i) {
        pid = fork();
        if (pid == 0) {
            pthread_barrier_init(&barrier, NULL, 1);
            {
                std::stringstream ss;
                ss << "process " << i << "/" << num_processes;
                test_thread thread(ss.str(), 1, &barrier, this);
            }
            pthread_barrier_destroy(&barrier);
            exit(HasFailure() ? 1 : 0);
        }
        pids.push_back(pid);
    }

    for (int i = 0; i < num_processes; ++i) {
        int status;
        waitpid(pids[i], &status, 0);
        EXPECT_EQ(0, WEXITSTATUS(status)) << "Process " << i << " failed";
    }
}
