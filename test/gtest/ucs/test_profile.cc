/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2012.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/sys/sys.h>
#include <ucs/debug/profile.h>
}

#include <fstream>
#include <set>


#if HAVE_PROFILING

class scoped_profile {
public:
    scoped_profile(ucs::test_base& test, const std::string &file_name,
                   const char *mode) : m_test(test), m_file_name(file_name)
{
        ucs_profile_global_cleanup();
        m_test.push_config();
        m_test.modify_config("PROFILE_MODE", mode);
        m_test.modify_config("PROFILE_FILE", m_file_name.c_str());
        ucs_profile_global_init();
    }

    std::string read() {
        ucs_profile_dump();
        std::ifstream f(m_file_name.c_str());
        return std::string(std::istreambuf_iterator<char>(f),
                           std::istreambuf_iterator<char>());
    }

    ~scoped_profile() {
        ucs_profile_global_cleanup();
        unlink(m_file_name.c_str());
        m_test.pop_config();
        ucs_profile_global_init();
    }
private:
    ucs::test_base&   m_test;
    const std::string m_file_name;
};

class test_profile : public ucs::test {
public:
    static const char* UCS_PROFILE_FILENAME;
    static const int   MIN_LINE;
    static const int   MAX_LINE;

    void test_header(ucs_profile_header_t *hdr, unsigned exp_mode);
    void test_locations(ucs_profile_location_t *locations, unsigned num_locations,
                        uint64_t exp_count);
};

const char* test_profile::UCS_PROFILE_FILENAME = "test.prof";

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

const int test_profile::MAX_LINE = __LINE__;

void test_profile::test_header(ucs_profile_header_t *hdr, unsigned exp_mode)
{
    EXPECT_EQ(std::string(ucs_get_host_name()), std::string(hdr->hostname));
    EXPECT_EQ(getpid(),                         (pid_t)hdr->pid);
    EXPECT_EQ(exp_mode,                         hdr->mode);
    EXPECT_NEAR(hdr->one_second / ucs_time_from_sec(1.0), 1.0, 0.01);
}

void test_profile::test_locations(ucs_profile_location_t *locations,
                                  unsigned num_locations, uint64_t exp_count)
{
    std::set<std::string> loc_names;
    for (unsigned i = 0; i < num_locations; ++i) {
        ucs_profile_location_t *loc = &locations[i];
        EXPECT_EQ(std::string(basename(__FILE__)), std::string(loc->file));
        EXPECT_GE(loc->line, MIN_LINE);
        EXPECT_LE(loc->line, MAX_LINE);
        EXPECT_LT(loc->total_time, ucs_time_from_sec(1.0) * ucs::test_time_multiplier());
        EXPECT_EQ(exp_count, locations[i].count);
        loc_names.insert(loc->name);
    }

    EXPECT_NE(loc_names.end(), loc_names.find("profile_test_func1"));
    EXPECT_NE(loc_names.end(), loc_names.find("profile_test_func2"));
    EXPECT_NE(loc_names.end(), loc_names.find("code"));
    EXPECT_NE(loc_names.end(), loc_names.find("sample"));
    EXPECT_NE(loc_names.end(), loc_names.find("sum"));
    EXPECT_NE(loc_names.end(), loc_names.find("allocate"));
    EXPECT_NE(loc_names.end(), loc_names.find("work"));
}

UCS_TEST_F(test_profile, accum) {
    scoped_profile p(*this, UCS_PROFILE_FILENAME, "accum");
    profile_test_func1();
    profile_test_func2(1, 2);

    std::string data = p.read();
    ucs_profile_header_t *hdr = reinterpret_cast<ucs_profile_header_t*>(&data[0]);
    test_header(hdr, UCS_BIT(UCS_PROFILE_MODE_ACCUM));

    EXPECT_EQ(12u, hdr->num_locations);
    test_locations(reinterpret_cast<ucs_profile_location_t*>(hdr + 1),
                   hdr->num_locations,
                   1);

    EXPECT_EQ(0u, hdr->num_records);
}

UCS_TEST_F(test_profile, log) {
    static const int ITER = 3;
    scoped_profile p(*this, UCS_PROFILE_FILENAME, "log");
    for (int i = 0; i < ITER; ++i) {
        profile_test_func1();
        profile_test_func2(1, 2);
    }

    std::string data = p.read();
    ucs_profile_header_t *hdr = reinterpret_cast<ucs_profile_header_t*>(&data[0]);
    test_header(hdr, UCS_BIT(UCS_PROFILE_MODE_LOG));

    EXPECT_EQ(12u, hdr->num_locations);
    ucs_profile_location_t *locations = reinterpret_cast<ucs_profile_location_t*>(hdr + 1);
    test_locations(locations, hdr->num_locations, 0);

    EXPECT_EQ(12 * ITER, (int)hdr->num_records);
    ucs_profile_record_t *records = reinterpret_cast<ucs_profile_record_t*>(locations +
                                                                            hdr->num_locations);
    uint64_t prev_ts = records[0].timestamp;
    for (uint64_t i = 0; i < hdr->num_records; ++i) {
        ucs_profile_record_t *rec = &records[i];
        EXPECT_GE(rec->location, 0u);
        EXPECT_LT(rec->location, 12u);
        EXPECT_GE(rec->timestamp, prev_ts);
        prev_ts = rec->timestamp;
        ucs_profile_location_t *loc = &locations[rec->location];
        if ((loc->type == UCS_PROFILE_TYPE_REQUEST_NEW) ||
            (loc->type == UCS_PROFILE_TYPE_REQUEST_EVENT) ||
            (loc->type == UCS_PROFILE_TYPE_REQUEST_FREE))
        {
            EXPECT_EQ((uintptr_t)&test_request, rec->param64);
        }
    }
}

#endif
