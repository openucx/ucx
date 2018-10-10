/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
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
#include <dlfcn.h>
#include <libgen.h>

extern "C" {
#include <ucs/time/time.h>
#include <ucm/malloc/malloc_hook.h>
#include <ucm/bistro/bistro.h>
#include <ucs/sys/sys.h>
#include <malloc.h>
}

#if HAVE_MALLOC_SET_STATE && HAVE_MALLOC_GET_STATE
#  define HAVE_MALLOC_STATES 1
#endif /* HAVE_MALLOC_SET_STATE && HAVE_MALLOC_GET_STATE */

#define EXPECT_INCREASED(_value, _prev, _size, _name)  \
    {                                                  \
        EXPECT_GE(_value, (_prev) + (_size)) << _name; \
        _prev = _value;                                \
    }

template <class T>
class mhook_thread {
public:
    mhook_thread(T *test): m_test(test)
    {
        pthread_create(&m_thread, NULL, thread_func, reinterpret_cast<void*>(m_test));
    }

    ~mhook_thread() {
        join();
        delete m_test;
    }

    void join() {
        void *retval;
        pthread_join(m_thread, &retval);
    }

protected:
    T         *m_test;
    pthread_t m_thread;

    static void *thread_func(void *arg) {
        T *test = reinterpret_cast<T*>(arg);
        test->test();
        return NULL;
    }
};

template <class T>
class mmap_event {
public:
    mmap_event(T *test): m_test(test), m_events(0)
    {
    }

    ~mmap_event()
    {
        unset();
    }

    ucs_status_t set(int events)
    {
        ucs_status_t status;

        status = ucm_set_event_handler(events, 0, mem_event_callback,
                                       reinterpret_cast<void*>(m_test));
        ASSERT_UCS_OK(status);
        m_events |= events;
        return status;
    }

    void unset()
    {
        if (m_events) {
            ucm_unset_event_handler(m_events, mem_event_callback,
                                    reinterpret_cast<void*>(m_test));
            m_events = 0;
        }
    }

protected:
    T   *m_test;
    int m_events;

    static void mem_event_callback(ucm_event_type_t event_type,
                                   ucm_event_t *event,
                                   void *arg)
    {
        T *test = reinterpret_cast<T*>(arg);
        test->mem_event(event_type, event);
    }
};


class malloc_hook : public ucs::test {
    friend class mmap_event<malloc_hook>;
protected:
    void mem_event(ucm_event_type_t event_type, ucm_event_t *event)
    {
        m_got_event = 1;
    }

    virtual void init() {
        ucs_status_t status;
        mmap_event<malloc_hook> event(this);

        m_got_event = 0;
        ucm_malloc_state_reset(128 * 1024, 128 * 1024);
        malloc_trim(0);
        status = event.set(UCM_EVENT_VM_MAPPED);
        ASSERT_UCS_OK(status);

        for (;;) {
            void *ptr = malloc(small_alloc_size);
            if (m_got_event) {
                /* If the heap grew, the minimal size is the previous one */
                free(ptr);
                break;
            } else {
                m_pts.push_back(ptr);
            }
        }
        event.unset();
    }

    static int bistro_munmap_hook(void *addr, size_t length)
    {
        UCM_BISTRO_PROLOGUE;
        bistro_call_counter++;
        int res = (intptr_t)syscall(SYS_munmap, addr, length);
        UCM_BISTRO_EPILOGUE;
        return res;
    }

    void skip_on_bistro() {
        /* BISTRO is disabled under valgrind, we may run tests */
        if ((ucm_global_opts.mmap_hook_mode == UCM_MMAP_HOOK_BISTRO) &&
             !RUNNING_ON_VALGRIND) {
            UCS_TEST_SKIP_R("skipping on BISTRO hooks");
        }
    }

public:
    static int            small_alloc_count;
    static const size_t   small_alloc_size = 10000;
    ucs::ptr_vector<void> m_pts;
    int                   m_got_event;
    static volatile int   bistro_call_counter;
};

int malloc_hook::small_alloc_count            = 1000 / ucs::test_time_multiplier();
volatile int malloc_hook::bistro_call_counter = 0;

class test_thread {
public:
    test_thread(const std::string& name, int num_threads, pthread_barrier_t *barrier,
                malloc_hook *test, void (test_thread::*test_func)() = &test_thread::test) :
        m_name(name), m_num_threads(num_threads), m_barrier(barrier),
        m_map_size(0), m_unmap_size(0), m_test(test), m_event(this)
    {
        pthread_mutex_init(&m_stats_lock, NULL);
    }

    ~test_thread() {
        pthread_mutex_destroy(&m_stats_lock);
    }

    void test();
    void mem_event(ucm_event_type_t event_type, ucm_event_t *event);

private:
    typedef std::pair<void*, void*> range;

    bool is_ptr_in_range(void *ptr, size_t size, const std::vector<range> &ranges) {
        for (std::vector<range>::const_iterator iter = ranges.begin(); iter != ranges.end(); ++iter) {
            if ((ptr >= iter->first) && ((char*)ptr < iter->second)) {
                return true;
            }
        }
        return false;
    }

    static pthread_mutex_t   lock;
    static pthread_barrier_t barrier;

    std::string        m_name;
    int                m_num_threads;
    pthread_barrier_t  *m_barrier;

    pthread_mutex_t    m_stats_lock;
    size_t             m_map_size;
    size_t             m_unmap_size;
    std::vector<range> m_map_ranges;
    std::vector<range> m_unmap_ranges;

    malloc_hook        *m_test;
    mmap_event<test_thread> m_event;
};

pthread_mutex_t test_thread::lock = PTHREAD_MUTEX_INITIALIZER;

void test_thread::mem_event(ucm_event_type_t event_type, ucm_event_t *event)
{
    pthread_mutex_lock(&m_stats_lock);
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
    pthread_mutex_unlock(&m_stats_lock);
}

void test_thread::test() {
    static const size_t large_alloc_size = 40 * 1024 * 1024;
    ucs_status_t result;
    ucs::ptr_vector<void> old_ptrs;
    ucs::ptr_vector<void> new_ptrs;
    void *ptr_r;
    size_t small_map_size;
    const size_t small_alloc_size = malloc_hook::small_alloc_size;
    int num_ptrs_in_range;
    static volatile uint32_t total_ptrs_in_range = 0;

    /* Allocate some pointers with old heap manager */
    for (unsigned i = 0; i < 10; ++i) {
        old_ptrs.push_back(malloc(small_alloc_size));
    }

    ptr_r = malloc(small_alloc_size);

    m_map_ranges.reserve  ((m_test->small_alloc_count * 8 + 10) * m_num_threads);
    m_unmap_ranges.reserve((m_test->small_alloc_count * 8 + 10) * m_num_threads);

    total_ptrs_in_range = 0;

    pthread_barrier_wait(m_barrier);

    /* Install memory hooks */
    result = m_event.set(UCM_EVENT_VM_MAPPED|UCM_EVENT_VM_UNMAPPED);
    ASSERT_UCS_OK(result);

    /* Allocate small pointers with new heap manager */
    for (int i = 0; i < m_test->small_alloc_count; ++i) {
        new_ptrs.push_back(malloc(small_alloc_size));
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
        if (is_ptr_in_range(*iter, small_alloc_size, m_map_ranges)) {
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
    ptr_r = realloc(ptr_r, small_alloc_size / 2);
    free(ptr_r);

    /* Test C++ allocations */
    {
        std::vector<char> vec(large_alloc_size, 0);
        ptr = &vec[0];
        EXPECT_TRUE(is_ptr_in_range(ptr, large_alloc_size, m_map_ranges)) << m_name;
    }

    /* coverity[use_after_free] - we don't dereference ptr, just search it*/
    EXPECT_TRUE(is_ptr_in_range(ptr, large_alloc_size, m_unmap_ranges)) << m_name;

    /* Release old pointers (should not crash) */
    old_ptrs.clear();

    m_map_ranges.clear();
    m_unmap_ranges.clear();

    /* Don't release pointers before other threads exit, so they will map new memory
     * and not reuse memory from other threads.
     */
    pthread_barrier_wait(m_barrier);

    /* This must be done when all other threads are inactive, otherwise we may leak */
#if HAVE_MALLOC_STATES
    if (!RUNNING_ON_VALGRIND) {
        pthread_mutex_lock(&lock);
        void *state = malloc_get_state();
        malloc_set_state(state);
        free(state);
        pthread_mutex_unlock(&lock);
    }
#endif /* HAVE_MALLOC_STATES */

    pthread_barrier_wait(m_barrier);

    /* Release new pointers  */
    new_ptrs.clear();

    /* Call several malloc routines */
    malloc_trim(0);

    ptr = malloc(large_alloc_size);

    free(ptr);

    /* shmat/shmdt */
    size_t shm_seg_size = ucs_get_page_size() * 2;
    int shmid = shmget(IPC_PRIVATE, shm_seg_size, IPC_CREAT | SHM_R | SHM_W);
    EXPECT_NE(-1, shmid) << strerror(errno);

    ptr = shmat(shmid, NULL, 0);
    EXPECT_NE(MAP_FAILED, ptr) << strerror(errno);

    shmdt(ptr);
    shmctl(shmid, IPC_RMID, NULL);

    EXPECT_TRUE(is_ptr_in_range(ptr, shm_seg_size, m_unmap_ranges));

    /* Print results */
    pthread_mutex_lock(&lock);
    UCS_TEST_MESSAGE << m_name
                     << ": small mapped: " << small_map_size
                     <<  ", total mapped: " << m_map_size
                     <<  ", total unmapped: " << m_unmap_size;
    std::cout.flush();
    pthread_mutex_unlock(&lock);

    m_event.unset();
}

UCS_TEST_F(malloc_hook, single_thread) {
    skip_on_bistro();

    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, 1);
    {
        mhook_thread<test_thread>(new test_thread("single-thread", 1, &barrier, this));
    }
    pthread_barrier_destroy(&barrier);
}

UCS_TEST_F(malloc_hook, multi_threads) {
    typedef mhook_thread<test_thread> thread_t;

    static const int num_threads = 8;
    ucs::ptr_vector<thread_t> threads;
    pthread_barrier_t barrier;

    skip_on_bistro();

    malloc_trim(0);

    pthread_barrier_init(&barrier, NULL, num_threads);
    for (int i = 0; i < num_threads; ++i) {
        std::stringstream ss;
        ss << "thread " << i << "/" << num_threads;
        threads.push_back(new thread_t(new test_thread(ss.str(), num_threads, &barrier, this)));
    }

    threads.clear();
    pthread_barrier_destroy(&barrier);
}

UCS_TEST_F(malloc_hook, fork) {
    static const int num_processes = 4;
    pthread_barrier_t barrier;
    std::vector<pid_t> pids;
    pid_t pid;

    UCS_TEST_SKIP_R("broken");
    /* coverity[unreachable] */

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
            throw ucs::exit_exception(HasFailure());
        }
        pids.push_back(pid);
    }

    for (int i = 0; i < num_processes; ++i) {
        int status;
        waitpid(pids[i], &status, 0);
        EXPECT_EQ(0, WEXITSTATUS(status)) << "Process " << i << " failed";
    }
}

class malloc_hook_cplusplus : public malloc_hook {
public:

    malloc_hook_cplusplus() :
        m_mapped_size(0), m_unmapped_size(0),
        m_dynamic_mmap_config(ucm_global_opts.enable_dynamic_mmap_thresh),
        m_event(this) {
    }

    ~malloc_hook_cplusplus() {
        ucm_global_opts.enable_dynamic_mmap_thresh = m_dynamic_mmap_config;
    }

    void set() {
        ucs_status_t status;
        status = m_event.set(UCM_EVENT_VM_MAPPED|UCM_EVENT_VM_UNMAPPED);
        ASSERT_UCS_OK(status);
    }

    void unset() {
        m_event.unset();
    }

    void mem_event(ucm_event_type_t event_type, ucm_event_t *event)
    {
        switch (event_type) {
        case UCM_EVENT_VM_MAPPED:
            m_mapped_size   += event->vm_mapped.size;
            break;
        case UCM_EVENT_VM_UNMAPPED:
            m_unmapped_size += event->vm_unmapped.size;
            break;
        default:
            break;
        }
    }

protected:
    double measure_alloc_time(size_t size, unsigned iters)
    {
        ucs_time_t start_time = ucs_get_time();
        for (unsigned i = 0; i < iters; ++i) {
            void *ptr = malloc(size);
            /* prevent the compiler from optimizing-out the memory allocation */
            *(volatile char*)ptr = '5';
            free(ptr);
        }
        return ucs_time_to_sec(ucs_get_time() - start_time);
    }

    void test_dynamic_mmap_thresh()
    {
        const size_t size = 8 * UCS_MBYTE;

        set();

        std::vector<std::string> strs;

        m_mapped_size = 0;
        while (m_mapped_size < size) {
            strs.push_back(std::string(size, 't'));
        }

        m_unmapped_size = 0;
        strs.clear();
        EXPECT_GE(m_unmapped_size, size);

        m_mapped_size = 0;
        while (m_mapped_size < size) {
            strs.push_back(std::string());
            strs.back().resize(size);
        }

        m_unmapped_size = 0;
        strs.clear();
        if (ucm_global_opts.enable_dynamic_mmap_thresh) {
            EXPECT_EQ(0ul, m_unmapped_size);
        } else {
            EXPECT_GE(m_unmapped_size, size);
        }

        unset();
    }

    size_t m_mapped_size;
    size_t m_unmapped_size;
    int    m_dynamic_mmap_config;
    mmap_event<malloc_hook_cplusplus> m_event;
};


class mmap_hooks {
public:
    mmap_hooks(const std::string& name, int num_threads, pthread_barrier_t *barrier):
        m_num_threads(num_threads), m_mapped_size(0), m_unmapped_size(0),
        m_name(name), m_barrier(barrier), m_event(this)
    {
    }

    void mem_event(ucm_event_type_t event_type, ucm_event_t *event)
    {
        switch (event_type) {
        case UCM_EVENT_VM_MAPPED:
            m_mapped_size   += event->vm_mapped.size;
            break;
        case UCM_EVENT_VM_UNMAPPED:
            m_unmapped_size += event->vm_unmapped.size;
            break;
        default:
            break;
        }
    }

    void test()
    {
        /*
         * Test memory mapping functions which override an existing mapping
         */
        size_t size          = ucs_get_page_size() * 800;
        size_t mapped_size   = 0;
        size_t unmapped_size = 0;
        void *buffer;
        int shmid;
        ucs_status_t status;

        EXPECT_EQ(0u, m_mapped_size) << m_name;
        EXPECT_EQ(0u, m_unmapped_size) << m_name;

        status = m_event.set(UCM_EVENT_VM_MAPPED|UCM_EVENT_VM_UNMAPPED);
        ASSERT_UCS_OK(status);

        pthread_barrier_wait(m_barrier);

        /* 1. Map a large buffer */
        {
            buffer = mmap(NULL, size, PROT_READ|PROT_WRITE,
                                MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
            ASSERT_NE(MAP_FAILED, buffer) << strerror(errno);

            EXPECT_INCREASED(m_mapped_size, mapped_size, size, m_name);
            EXPECT_INCREASED(m_unmapped_size, unmapped_size, 0, m_name);
        }

        /*
         * 2. Map another buffer in the same place.
         *    Expected behavior: unmap event on the old buffer
         */
        {
            void *remap = mmap(buffer, size, PROT_READ|PROT_WRITE,
                               MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED, -1, 0);
            ASSERT_EQ(buffer, remap);

            EXPECT_INCREASED(m_mapped_size, mapped_size, size, m_name);
            EXPECT_INCREASED(m_unmapped_size, unmapped_size, size, m_name);
        }

        /* 3. Create a shared memory segment */
        {
            shmid = shmget(IPC_PRIVATE, size, IPC_CREAT | SHM_R | SHM_W);
            ASSERT_NE(-1, shmid) << strerror(errno) << m_name;
        }

        /*
         * 4. Attach the segment at the same buffer address.
         *    Expected behavior: unmap event on the old buffer
         */
        {
            void *shmaddr = shmat(shmid, buffer, SHM_REMAP);
            ASSERT_EQ(buffer, shmaddr) << m_name;

            EXPECT_INCREASED(m_mapped_size, mapped_size, size, m_name);
            EXPECT_INCREASED(m_unmapped_size, unmapped_size, size, m_name);
        }

        /* 5. Detach the sysv segment */
        {
            shmdt(buffer);

            EXPECT_INCREASED(m_unmapped_size, unmapped_size, size, m_name);
        }

        /* 6. Remove the shared memory segment */
        {
            int ret = shmctl(shmid, IPC_RMID, NULL);
            ASSERT_NE(-1, ret) << strerror(errno);
        }

        /* 7. Unmap the buffer */
        {
            munmap(buffer, size);

            EXPECT_INCREASED(m_unmapped_size, unmapped_size, size, m_name);
        }

        /* 8. sbrk call - single thread only */
        {
            if (!RUNNING_ON_VALGRIND && m_num_threads < 2) {
                /* valgrind failed when sbrk is called directly,
                 * also sbrk is not thread safe */

                /* sbrk call is used to extend/cut memory heap,
                 * don't add any evaluations between calls sbrk+/sbrk- - it
                 * may break heap */
                sbrk(size);
                sbrk(-size);

                EXPECT_INCREASED(m_mapped_size, mapped_size, size, m_name);
                EXPECT_INCREASED(m_unmapped_size, unmapped_size, size, m_name);
            }
        }
        pthread_barrier_wait(m_barrier);
    }

protected:
    int                     m_num_threads;
    size_t                  m_mapped_size;
    size_t                  m_unmapped_size;
    std::string             m_name;
    pthread_barrier_t       *m_barrier;
    mmap_event<mmap_hooks>  m_event;
};


UCS_TEST_F(malloc_hook_cplusplus, new_delete) {

    const size_t size = 8 * 1000 * 1000;

    set();

    {
        std::vector<char> vec1(size, 0);
        std::vector<char> vec2(size, 0);
        std::vector<char> vec3(size, 0);
    }

    {
        std::vector<char> vec1(size, 0);
        std::vector<char> vec2(size, 0);
        std::vector<char> vec3(size, 0);
    }

    malloc_trim(0);

    EXPECT_GE(m_unmapped_size, size * 3);

    unset();
}

UCS_TEST_F(malloc_hook_cplusplus, dynamic_mmap_enable) {
    if (RUNNING_ON_VALGRIND) {
        UCS_TEST_SKIP_R("skipping on valgrind");
    }
    skip_on_bistro();
    EXPECT_TRUE(ucm_global_opts.enable_dynamic_mmap_thresh);
    test_dynamic_mmap_thresh();
}

UCS_TEST_F(malloc_hook_cplusplus, dynamic_mmap_disable) {
    skip_on_bistro();

    ucm_global_opts.enable_dynamic_mmap_thresh = 0;

    test_dynamic_mmap_thresh();
}

extern "C" {
    int ucm_dlmallopt_get(int);
};

UCS_TEST_F(malloc_hook_cplusplus, mallopt) {

    int v;
    int trim_thresh, mmap_thresh;
    char *p;
    size_t size;

    skip_on_bistro();

    /* This test can not be run with the other
     * tests because it assumes that malloc hooks
     * are not initialized
     */
    p = getenv("MALLOC_TRIM_THRESHOLD_");
    if (p == NULL) {
        UCS_TEST_SKIP_R("MALLOC_TRIM_THRESHOLD_ is not set");
    }
    ASSERT_TRUE(p != NULL);
    trim_thresh = atoi(p);

    p = getenv("MALLOC_MMAP_THRESHOLD_");
    if (p == NULL) {
        UCS_TEST_SKIP_R("MALLOC_MMAP_THRESHOLD_ is not set");
    }
    ASSERT_TRUE(p != NULL);
    mmap_thresh = atoi(p);

    /* make sure that rcache is explicitly disabled so
     * that the malloc hooks are installed after the setenv()
     */
    p = getenv("UCX_IB_RCACHE");
    if ((p == NULL) || (p[0] != 'n')) {
        UCS_TEST_SKIP_R("rcache must be disabled");
    }

    set();

    v = ucm_dlmallopt_get(M_TRIM_THRESHOLD);
    EXPECT_EQ(trim_thresh, v);

    v = ucm_dlmallopt_get(M_MMAP_THRESHOLD);
    EXPECT_EQ(mmap_thresh, v);

    /* give a lot of extra space since the same block
     * can be also used by other allocations
     */
    if (trim_thresh > 0) {
        size = trim_thresh/2;
    } else if (mmap_thresh > 0) {
        size = mmap_thresh/2;
    } else {
        size = 10 * 1024 * 1024;
    }

    UCS_TEST_MESSAGE << "trim_thresh=" << trim_thresh << " mmap_thresh=" << mmap_thresh <<
                        " allocating=" << size;
    p = new char [size];
    ASSERT_TRUE(p != NULL);
    delete [] p;

    EXPECT_EQ(m_unmapped_size, size_t(0));

    unset();
}

UCS_TEST_F(malloc_hook_cplusplus, mmap_ptrs) {

    if (RUNNING_ON_VALGRIND) {
        UCS_TEST_SKIP_R("skipping on valgrind");
    }

    ucm_global_opts.enable_dynamic_mmap_thresh = 0;
    set();

    const size_t   size    = ucm_dlmallopt_get(M_MMAP_THRESHOLD) * 2;
    const size_t   max_mem = ucs_min(ucs_get_phys_mem_size() / 4, 4 * UCS_GBYTE);
    const unsigned count   = ucs_min(400000ul, max_mem / size);
    const unsigned iters   = 100000;

    std::vector< std::vector<char> > ptrs;

    size_t large_blocks = 0;

    /* Allocate until we get MMAP event
     * Lock memory to avoid going to swap and ensure consistet test results.
     */
    while (m_mapped_size == 0) {
        std::vector<char> str(size, 'r');
        ptrs.push_back(str);
        ++large_blocks;
    }

    /* Remove memory off the heap top, to ensure the following large allocations
     * will use mmap()
     */
    malloc_trim(0);

    /* Measure allocation time with "clear" heap state */
    double alloc_time = measure_alloc_time(size, iters);
    UCS_TEST_MESSAGE << "With " << large_blocks << " large blocks:"
                     << " allocated " << iters << " buffers of " << size
                     << " bytes in " << alloc_time << " sec";

    /* Allocate many large strings to trigger mmap() based allocation. */
    ptrs.resize(count);
    for (unsigned i = 0; i < count; ++i) {
        ptrs[i].resize(size, 't');
        ++large_blocks;
    }

    /* Measure allocation time with many large blocks on the heap */
    bool success = false;
    unsigned attempt = 0;
    while (!success && (attempt < 5)) {
        double alloc_time_with_ptrs = measure_alloc_time(size, iters);
        UCS_TEST_MESSAGE << "With " << large_blocks << " large blocks:"
                         << " allocated " << iters << " buffers of " << size
                         << " bytes in " << alloc_time_with_ptrs << " sec";

        /* Allow up to 75% difference */
        success = (alloc_time < 0.25) || (alloc_time_with_ptrs < (1.75 * alloc_time));
        ++attempt;
    }

    if (!success) {
        ADD_FAILURE() << "Failed after " << attempt << " attempts";
    }

    ptrs.clear();

    unset();

}

UCS_TEST_F(malloc_hook_cplusplus, remap_override_single_thread) {
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, 1);
    {
        mhook_thread<mmap_hooks>(new mmap_hooks("single-thread", 1, &barrier));
    }
    pthread_barrier_destroy(&barrier);
}

UCS_TEST_F(malloc_hook_cplusplus, remap_override_multi_threads) {
    typedef mhook_thread<mmap_hooks> thread_t;

    static const int num_threads = 8;
    ucs::ptr_vector<thread_t> threads;
    pthread_barrier_t barrier;

    pthread_barrier_init(&barrier, NULL, num_threads);
    for (int i = 0; i < num_threads; ++i) {
        std::stringstream ss;
        ss << "thread " << i << "/" << num_threads;
        threads.push_back(new thread_t(new mmap_hooks(ss.str(), num_threads, &barrier)));
    }

    threads.clear();
    pthread_barrier_destroy(&barrier);
}

typedef int (munmap_f_t)(void *addr, size_t len);

UCS_TEST_F(malloc_hook, bistro_patch) {
    const char *symbol = "munmap";
    ucm_bistro_restore_point_t *rp = NULL;
    ucs_status_t status;
    munmap_f_t *munmap_f;
    void *ptr;
    int res;
    uint64_t UCS_V_UNUSED patched;
    uint64_t UCS_V_UNUSED origin;

    if (RUNNING_ON_VALGRIND) {
        UCS_TEST_SKIP_R("skipping on valgrind");
    }

    /* set hook to mmap call */
    status = ucm_bistro_patch(symbol, (void*)bistro_munmap_hook, &rp);
    ASSERT_UCS_OK(status);
    EXPECT_NE((intptr_t)rp, NULL);

    munmap_f = (munmap_f_t*)ucm_bistro_restore_addr(rp);
    EXPECT_NE((intptr_t)munmap_f, NULL);

    /* save partial body of patched function */
    patched = *(uint64_t*)munmap_f;

    bistro_call_counter = 0;
    ptr = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
    EXPECT_NE(ptr, MAP_FAILED);

    /* try to call munmap, we should jump into munmap_hook instead */
    res = munmap_f(ptr, 4096);
    EXPECT_EQ(res, 0);
    /* due to cache coherency issues on ARM systems could be executed
     * original function body, so, skip counter evaluation */
    EXPECT_GT(bistro_call_counter, 0);

    /* restore original mmap body */
    status = ucm_bistro_restore(rp);
    ASSERT_UCS_OK(status);

    bistro_call_counter = 0;
    /* now try to call mmap, we should NOT jump into mmap_hook */
    ptr = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
    EXPECT_NE(ptr, MAP_FAILED);
    res = munmap_f(ptr, 4096);
    EXPECT_EQ(res, 0);
    EXPECT_EQ(bistro_call_counter, 0);  /* hook is not called */
    /* save partial body of restored function */
    origin = *(uint64_t*)munmap_f;

#if !defined (__powerpc64__)
    EXPECT_NE(patched, origin);
#endif
}

/* test for mmap events are fired from non-direct load modules
 * we are trying to load lib1, from lib1 load lib2, and
 * fire mmap event from lib2 */
UCS_TEST_F(malloc_hook, dlopen) {
#ifndef GTEST_UCM_HOOK_LIB_DIR
#  error "Missing build configuration"
#else
    typedef void (fire_mmap_f)(void);
    typedef void* (load_lib_f)(const char *path);

    const char *libdlopen_load = "/libdlopen_test_do_load.so";
    const char *libdlopen_mmap = "/libdlopen_test_do_mmap.so";
    const char *load_lib       = "load_lib";
    const char *fire_mmap      = "fire_mmap";

    std::string lib_load;
    std::string lib_mmap;
    void *lib;
    void *lib2;
    load_lib_f *load;
    fire_mmap_f *fire;
    ucs_status_t status;
    mmap_event<malloc_hook> event(this);

    status = event.set(UCM_EVENT_VM_MAPPED);
    ASSERT_UCS_OK(status);

    lib_load = std::string(GTEST_UCM_HOOK_LIB_DIR) + libdlopen_load;
    lib_mmap = std::string(GTEST_UCM_HOOK_LIB_DIR) + libdlopen_mmap;

    UCS_TEST_MESSAGE << "Loading " << lib_load;
    UCS_TEST_MESSAGE << "Loading " << lib_mmap;

    lib = dlopen(lib_load.c_str(), RTLD_NOW);
    EXPECT_NE((uintptr_t)lib, (uintptr_t)NULL);
    if (!lib) {
        goto no_lib;
    }

    load = (load_lib_f*)dlsym(lib, load_lib);
    EXPECT_NE((uintptr_t)load, (uintptr_t)NULL);
    if (!load) {
        goto no_load;
    }

    lib2 = load(lib_mmap.c_str());
    EXPECT_NE((uintptr_t)lib2, (uintptr_t)NULL);
    if (!lib2) {
        goto no_load;
    }

    fire = (fire_mmap_f*)dlsym(lib2, fire_mmap);
    EXPECT_NE((uintptr_t)fire, (uintptr_t)NULL);
    if (!fire) {
        goto no_fire;
    }

    m_got_event = 0;
    fire();
    EXPECT_GT(m_got_event, 0);

no_fire:
    dlclose(lib2);
no_load:
    dlclose(lib);
no_lib:
    event.unset();
#endif /* GTEST_UCM_HOOK_LIB_DIR */
}
