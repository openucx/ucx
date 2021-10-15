/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2012.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/sys/sys.h>
#include <ucs/time/time.h>
#include <ucs/profile/profile.h>
}

#include <pthread.h>
#include <fstream>

#ifdef HAVE_PROFILING

class scoped_profile {
public:
    scoped_profile(ucs::test_base& test, const std::string &file_name,
                   const char *mode) : m_test(test), m_file_name(file_name) {
        ucs_profile_reset_locations_id(ucs_profile_default_ctx);
        ucs_profile_cleanup(ucs_profile_default_ctx);
        m_test.push_config();
        m_test.modify_config("PROFILE_MODE", mode);
        m_test.modify_config("PROFILE_FILE", m_file_name.c_str());
        ucs_profile_init(ucs_global_opts.profile_mode,
                         ucs_global_opts.profile_file,
                         ucs_global_opts.profile_log_size,
                         &ucs_profile_default_ctx);
    }

    std::string read() {
        ucs_profile_dump(ucs_profile_default_ctx);
        std::ifstream f(m_file_name.c_str());
        return std::string(std::istreambuf_iterator<char>(f),
                           std::istreambuf_iterator<char>());
    }

    ~scoped_profile() {
        ucs_profile_reset_locations_id(ucs_profile_default_ctx);
        ucs_profile_cleanup(ucs_profile_default_ctx);
        unlink(m_file_name.c_str());
        m_test.pop_config();
        ucs_profile_init(ucs_global_opts.profile_mode,
                         ucs_global_opts.profile_file,
                         ucs_global_opts.profile_log_size,
                         &ucs_profile_default_ctx);
    }
private:
    ucs::test_base&   m_test;
    const std::string m_file_name;
};

class test_profile : public testing::TestWithParam<int>,
                     public ucs::test_base {
public:
    test_profile();
    ~test_profile();

    UCS_TEST_BASE_IMPL;

protected:
    static const int      MIN_LINE;
    static const int      MAX_LINE;
    static const unsigned NUM_LOCAITONS;
    static const char*    PROFILE_FILENAME;


    std::set<int>      m_tids;
    pthread_spinlock_t m_tids_lock;

    struct thread_param {
        test_profile *test;
        int          iters;
    };

    void add_tid(int tid);

    static void *profile_thread_func(void *arg);

    int num_threads() const;

    void run_profiled_code(int num_iters);

    void test_header(const ucs_profile_header_t *hdr, unsigned exp_mode,
                     const void **ptr);

    void test_locations(const ucs_profile_location_t *locations,
                        unsigned num_locations, const void **ptr);

    void test_thread_locations(const ucs_profile_thread_header_t *thread_hdr,
                               unsigned num_locations, uint64_t exp_count,
                               unsigned exp_num_records, const void **ptr);

    void test_nesting(const ucs_profile_location_t *loc, int nesting,
                      const std::string &exp_name, int exp_nesting);

    void do_test(unsigned int_mode, const std::string &str_mode);
};

static int sum(int a, int b)
{
    return a + b;
}

const int test_profile::MIN_LINE = __LINE__;

static void *test_request = &test_request;

UCS_PROFILE_FUNC_VOID(profile_test_func1, ())
{
    UCS_PROFILE_REQUEST_NEW(test_request, "allocate", 10);
    UCS_PROFILE_REQUEST_EVENT(test_request, "work", 0);
    UCS_PROFILE_REQUEST_FREE(test_request);
    UCS_PROFILE_CODE("code") {
        UCS_PROFILE_SAMPLE("sample");
    }
}

UCS_PROFILE_FUNC(int, profile_test_func2, (a, b), int a, int b)
{
    return UCS_PROFILE_CALL(sum, a, b);
}

const int test_profile::MAX_LINE           = __LINE__;
const unsigned test_profile::NUM_LOCAITONS = 12u;
const char* test_profile::PROFILE_FILENAME = "test.prof";

test_profile::test_profile()
{
    pthread_spin_init(&m_tids_lock, 0);
}

test_profile::~test_profile()
{
    pthread_spin_destroy(&m_tids_lock);
}

void test_profile::add_tid(int tid)
{
    pthread_spin_lock(&m_tids_lock);
    m_tids.insert(tid);
    pthread_spin_unlock(&m_tids_lock);
}

void *test_profile::profile_thread_func(void *arg)
{
    const thread_param *param = (const thread_param*)arg;

    param->test->add_tid(ucs_get_tid());

    for (int i = 0; i < param->iters; ++i) {
        profile_test_func1();
        profile_test_func2(1, 2);
    }

    return NULL;
}

int test_profile::num_threads() const
{
    return GetParam();
}

void test_profile::run_profiled_code(int num_iters)
{
    int ret;
    thread_param param;

    param.iters = num_iters;
    param.test  = this;

    if (num_threads() == 1) {
        profile_thread_func(&param);
    } else {
        std::vector<pthread_t> threads;

        for (int i = 0; i < num_threads(); ++i) {
            pthread_t profile_thread;
            ret = pthread_create(&profile_thread, NULL, profile_thread_func,
                                 (void*)&param);
            if (ret < 0) {
                ADD_FAILURE() << "pthread_create failed: " << strerror(errno);
                break;
            }

            threads.push_back(profile_thread);
        }

        while (!threads.empty()) {
            void *result;
            ret = pthread_join(threads.back(), &result);
            if (ret < 0) {
                ADD_FAILURE() << "pthread_join failed: " << strerror(errno);
            }

            threads.pop_back();
        }
    }
}

void test_profile::test_header(const ucs_profile_header_t *hdr, unsigned exp_mode,
                               const void **ptr)
{
    EXPECT_EQ(UCS_PROFILE_FILE_VERSION,         hdr->version);
    EXPECT_EQ(std::string(ucs_get_host_name()), std::string(hdr->hostname));
    EXPECT_EQ(getpid(),                         (pid_t)hdr->pid);
    EXPECT_EQ(exp_mode,                         hdr->mode);
    EXPECT_EQ(NUM_LOCAITONS,                    hdr->num_locations);
    EXPECT_EQ((uint32_t)num_threads(),          hdr->num_threads);
    EXPECT_NEAR(hdr->one_second / ucs_time_from_sec(1.0), 1.0, 0.01);

    *ptr = hdr + 1;
}

void test_profile::test_locations(const ucs_profile_location_t *locations,
                                  unsigned num_locations, const void **ptr)
{
    std::set<std::string> loc_names;
    for (unsigned i = 0; i < num_locations; ++i) {
        const ucs_profile_location_t *loc = &locations[i];
        EXPECT_EQ(std::string(basename(__FILE__)), std::string(loc->file));
        EXPECT_GE(loc->line, MIN_LINE);
        EXPECT_LE(loc->line, MAX_LINE);
        loc_names.insert(loc->name);
    }

    EXPECT_NE(loc_names.end(), loc_names.find("profile_test_func1"));
    EXPECT_NE(loc_names.end(), loc_names.find("profile_test_func2"));
    EXPECT_NE(loc_names.end(), loc_names.find("code"));
    EXPECT_NE(loc_names.end(), loc_names.find("sample"));
    EXPECT_NE(loc_names.end(), loc_names.find("sum"));
    EXPECT_NE(loc_names.end(), loc_names.find("allocate"));
    EXPECT_NE(loc_names.end(), loc_names.find("work"));

    *ptr = locations + num_locations;
}

void test_profile::test_thread_locations(
                               const ucs_profile_thread_header_t *thread_hdr,
                               unsigned num_locations, uint64_t exp_count,
                               unsigned exp_num_records, const void **ptr)
{
    const ucs_profile_thread_location_t *loc;

    EXPECT_NE(m_tids.end(),    m_tids.find(thread_hdr->tid));
    EXPECT_EQ(exp_num_records, thread_hdr->num_records);

    EXPECT_LE(thread_hdr->end_time,   ucs_get_time());
    EXPECT_LE(thread_hdr->start_time, thread_hdr->end_time);
    EXPECT_LE(thread_hdr->end_time - thread_hdr->start_time,
              ucs_time_from_sec(1.0) * ucs::test_time_multiplier() * (1 + exp_count));

    for (unsigned i = 0; i < num_locations; ++i) {
        loc = &reinterpret_cast<const ucs_profile_thread_location_t*>
                        (thread_hdr + 1)[i];
        EXPECT_EQ(exp_count, loc->count);
        EXPECT_LE(loc->total_time,
                  ucs_time_from_sec(1.0) * ucs::test_time_multiplier() * exp_count);
    }

    *ptr = reinterpret_cast<const ucs_profile_thread_location_t*>(thread_hdr + 1) +
           num_locations;
}

void test_profile::test_nesting(const ucs_profile_location_t *loc, int nesting,
                                const std::string &exp_name, int exp_nesting)
{
    if (loc->name == exp_name) {
        EXPECT_EQ(exp_nesting, nesting)
                << "nesting level of " << exp_name << " is wrong";
    }
}

void test_profile::do_test(unsigned int_mode, const std::string& str_mode)
{
    const int ITER           = 5;
    uint64_t exp_count       = (int_mode & UCS_BIT(UCS_PROFILE_MODE_ACCUM)) ?
                               ITER : 0;
    uint64_t exp_num_records = (int_mode & UCS_BIT(UCS_PROFILE_MODE_LOG)) ?
                               (NUM_LOCAITONS * ITER) : 0;


    scoped_profile p(*this, PROFILE_FILENAME, str_mode.c_str());
    run_profiled_code(ITER);

    std::string data = p.read();
    const void *ptr  = &data[0];

    /* Read and test file header */
    const ucs_profile_header_t *hdr =
                    reinterpret_cast<const ucs_profile_header_t*>(ptr);
    test_header(hdr, int_mode, &ptr);

    /* Read and test global locations */
    const ucs_profile_location_t *locations =
                    reinterpret_cast<const ucs_profile_location_t*>(ptr);
    test_locations(locations, hdr->num_locations, &ptr);

    /* Read and test threads */
    for (int i = 0; i < num_threads(); ++i) {
        const ucs_profile_thread_header_t *thread_hdr =
                        reinterpret_cast<const ucs_profile_thread_header_t*>(ptr);

        test_thread_locations(thread_hdr, hdr->num_locations, exp_count,
                              exp_num_records, &ptr);

        const ucs_profile_record_t *records =
                reinterpret_cast<const ucs_profile_record_t*>(ptr);
        uint64_t prev_ts = records[0].timestamp;
        int nesting      = 0;

        for (uint64_t i = 0; i < thread_hdr->num_records; ++i) {
            const ucs_profile_record_t *rec = &records[i];

            /* test location index */
            EXPECT_GE(rec->location, 0u);
            EXPECT_LT(rec->location, uint32_t(NUM_LOCAITONS));

            /* test timestamp */
            EXPECT_GE(rec->timestamp, prev_ts);
            prev_ts = rec->timestamp;

            /* test param64 */
            const ucs_profile_location_t *loc = &locations[rec->location];
            switch (loc->type) {
            case UCS_PROFILE_TYPE_REQUEST_NEW:
            case UCS_PROFILE_TYPE_REQUEST_EVENT:
            case UCS_PROFILE_TYPE_REQUEST_FREE:
                EXPECT_EQ((uintptr_t)&test_request, rec->param64);
                break;
            case UCS_PROFILE_TYPE_SCOPE_BEGIN:
                ++nesting;
                break;
            case UCS_PROFILE_TYPE_SCOPE_END:
                --nesting;
                break;
            default:
                break;
            };

            test_nesting(loc, nesting, "profile_test_func1", 0);
            test_nesting(loc, nesting, "code", 1);
            test_nesting(loc, nesting, "sample", 2);
            test_nesting(loc, nesting, "profile_test_func2", 0);
            test_nesting(loc, nesting, "sum", 1);
        }

        ptr = records + thread_hdr->num_records;
    }

    EXPECT_EQ(&data[data.size()], ptr) << data.size();
}

UCS_TEST_P(test_profile, accum) {
    do_test(UCS_BIT(UCS_PROFILE_MODE_ACCUM), "accum");
}

UCS_TEST_P(test_profile, log) {
    do_test(UCS_BIT(UCS_PROFILE_MODE_LOG), "log");
}

UCS_TEST_P(test_profile, log_accum) {
    do_test(UCS_BIT(UCS_PROFILE_MODE_LOG) | UCS_BIT(UCS_PROFILE_MODE_ACCUM),
            "log,accum");
}

INSTANTIATE_TEST_SUITE_P(st, test_profile, ::testing::Values(1));
INSTANTIATE_TEST_SUITE_P(mt, test_profile, ::testing::Values(2, 4, 8));

class test_profile_perf : public test_profile {
};

UCS_TEST_SKIP_COND_P(test_profile_perf, overhead, RUNNING_ON_VALGRIND) {

#if defined(__x86_64__) || defined(__powerpc64__)
    const double EXP_OVERHEAD_NSEC = 100.0;
#else
    const double EXP_OVERHEAD_NSEC = 150.0;
#endif
    const int ITERS                = 100;
    const int WARMUP_ITERS         = 5;
    const int COUNT                = 100000;
    double overhead_nsec           = 0.0;

    scoped_profile p(*this, PROFILE_FILENAME, "accum");

    for (int retry = 0; retry < (ucs::perf_retry_count + 1); ++retry) {
        ucs_time_t  time_profile_on  = 0;
        ucs_time_t  time_profile_off = 0;

        for (int i = 0; i < WARMUP_ITERS + ITERS; ++i) {
            ucs_time_t t;

            t = ucs_get_time();
            for (volatile int j = 0; j < COUNT;) {
                ++j;
            }
            if (i > WARMUP_ITERS) {
                time_profile_off += ucs_get_time() - t;
            }

            t = ucs_get_time();
            for (volatile int j = 0; j < COUNT;) {
                UCS_PROFILE_CODE("test") {
                    ++j;
                }
            }
            if (i > WARMUP_ITERS) {
                time_profile_on += ucs_get_time() - t;
            }
        }

        overhead_nsec = ucs_time_to_nsec(time_profile_on - time_profile_off) /
                        COUNT / ITERS;
        UCS_TEST_MESSAGE << "overhead: " << overhead_nsec << " nsec";

        if (!ucs::perf_retry_count) {
            UCS_TEST_MESSAGE << "not validating performance";
            return; /* Success */
        } else if (overhead_nsec < EXP_OVERHEAD_NSEC) {
            return; /* Success */
        } else {
            ucs::safe_sleep(ucs::perf_retry_interval);
        }
    }

    EXPECT_LT(overhead_nsec, EXP_OVERHEAD_NSEC) << "Profiling overhead is too high";
}

INSTANTIATE_TEST_SUITE_P(st, test_profile_perf, ::testing::Values(1));

#endif
