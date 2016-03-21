/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2012.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/sys/preprocessor.h>
#include <ucs/sys/sys.h>
#include <ucs/debug/debug.h>
#include <ucs/debug/instrument.h>
}

#include <fstream>

#if HAVE_INSTRUMENTATION

class instrument : public ucs::test {
protected:
    virtual void init() {
        ucs_instrument_cleanup();
        push_config();
        modify_config("INSTRUMENT_FILE", UCS_INSTR_FILENAME);
        modify_config("INSTRUMENT_TYPES", "ib-tx");
        /* - where ucs_instrumentation_types_t == 0 */
    }

    virtual void cleanup() {
        unlink(UCS_INSTR_FILENAME);
        pop_config();
        ucs_instrument_init();
    }

protected:
    static const char* UCS_INSTR_FILENAME;
};

const char* instrument::UCS_INSTR_FILENAME = "ucs.instr";

#define UCS_TEST_LPARAM   0xdeadbeefdeadbeefull
#define UCS_TEST_WPARAM   0x1ee7feed

static void test_func()
{
    UCS_INSTRUMENT_RECORD(UCS_INSTRUMENT_TYPE_IB_TX,
                          "test_func", UCS_TEST_LPARAM, UCS_TEST_WPARAM);
}


UCS_TEST_F(instrument, record) {
    ucs_instrument_header_t hdr;
    ucs_time_t start_time, end_time;

    /* Perform instrumentation */
    ucs_instrument_init();
    start_time = ucs_get_time();
    test_func();
    end_time = ucs_get_time();
    ucs_instrument_cleanup();

    /* Read the file */
    ucs_instrument_record_t record;
    ucs_instrument_location_t location;
    std::ifstream fi(UCS_INSTR_FILENAME);
    ASSERT_FALSE(fi.bad());

    fi.read(reinterpret_cast<char*>(&hdr), sizeof(ucs_instrument_header_t));
    fi.read(reinterpret_cast<char*>(&location),
            offsetof(ucs_instrument_location_t, list));
    fi.read(reinterpret_cast<char*>(&record), sizeof(record));

    if (ucs::test_time_multiplier() == 1) {
        EXPECT_GE(record.timestamp, start_time);
        EXPECT_LE(record.timestamp, end_time);
    }

    ASSERT_EQ((uint32_t)1, hdr.num_locations);
    EXPECT_EQ(UCS_TEST_LPARAM, record.lparam);
    EXPECT_EQ((uint32_t)UCS_TEST_WPARAM, record.wparam);
    EXPECT_GE(record.location, (uint32_t)1);
    EXPECT_EQ(location.location, record.location);
    UCS_TEST_MESSAGE << "Generated location name is: " << location.name;
}

UCS_TEST_F(instrument, overhead) {
    static const size_t count = 10000000;
    ucs_time_t elapsed1, elapsed2;

    if (ucs::test_time_multiplier() > 1) {
        UCS_TEST_SKIP;
    }

    {
        ucs_time_t start_time = ucs_get_time();
        for (size_t i = 0; i < count; ++i) {
        }
        elapsed1 = ucs_get_time() - start_time;
    }

    {
        ucs_instrument_init();
        ucs_time_t start_time = ucs_get_time();
        for (size_t i = 0; i < count; ++i) {
            UCS_INSTRUMENT_RECORD((ucs_instrumentation_types_t)0,
                                  "test_func", 0xdeadbeef);
        }
        elapsed2 = ucs_get_time() - start_time;
        ucs_instrument_cleanup();
    }

    double overhead_nsec = ucs_time_to_nsec(elapsed2 - elapsed1) / count;
    UCS_TEST_MESSAGE << "Overhead is " << overhead_nsec << " nsec";
}

#endif
