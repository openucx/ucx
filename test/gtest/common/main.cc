/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif
#include <ucs/config/parser.h>
#include <ucs/sys/sys.h>
#include <ucm/api/ucm.h>
#include "test_helpers.h"
#include "tap.h"

static int ucs_gtest_random_seed = -1;
int    ucs::perf_retry_count     = 0; /* 0 - don't check performance */
double ucs::perf_retry_interval  = 1.0;

void parse_test_opts(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "s:p:i:")) != -1) {
        switch (c) {
        case 's':
            ucs_gtest_random_seed = atoi(optarg);
            break;
        case 'p':
            ucs::perf_retry_count = atoi(optarg);
            break;
        case 'i':
            ucs::perf_retry_interval = atof(optarg);
            break;
        default:
            fprintf(stderr, "Usage: gtest [ -s rand-seed ] [ -p count ] [ -i interval ]\n");
            exit(1);
        }
    }
}

static void modify_config_for_valgrind(const char *name, const char *value)
{
    char full_name[128];

    snprintf(full_name, sizeof(full_name), "%s%s", UCS_CONFIG_PREFIX, name);

    if (getenv(full_name) == NULL) {
        UCS_TEST_MESSAGE << " Setting for valgrind: " << full_name << "=" << value;
        setenv(full_name, value, 1);
    }
}

int main(int argc, char **argv) {
	// coverity[fun_call_w_exception]: uncaught exceptions cause nonzero exit anyway, so don't warn.
	::testing::InitGoogleTest(&argc, argv);

    char *str = getenv("GTEST_TAP");
    /* Append TAP Listener */
    if (str) {
        if (0 < strtol(str, NULL, 0)) {
            testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
            if (1 == strtol(str, NULL, 0)) {
                delete listeners.Release(listeners.default_result_printer());
            }
            listeners.Append(new tap::TapListener());
        }
    }

    parse_test_opts(argc, argv);
    if (ucs_gtest_random_seed == -1) {
        ucs_gtest_random_seed = time(NULL) % 32768;
    }
    UCS_TEST_MESSAGE << "Using random seed of " << ucs_gtest_random_seed;
    srand(ucs_gtest_random_seed);
    if (RUNNING_ON_VALGRIND) {
        modify_config_for_valgrind("IB_RX_QUEUE_LEN", "512");
        modify_config_for_valgrind("IB_RX_BUFS_GROW", "512");
        modify_config_for_valgrind("MM_RX_BUFS_GROW", "128");
        modify_config_for_valgrind("IB_TX_QUEUE_LEN", "128");
        modify_config_for_valgrind("IB_TX_BUFS_GROW", "64");
        modify_config_for_valgrind("RC_TX_CQ_LEN", "256");
        modify_config_for_valgrind("CM_TIMEOUT", "600ms");
        ucm_config_modify("MALLOC_RELOC", "y"); /* Test reloc hooks with valgrind,
                                                   though it's generally unsafe. */
    }
    return RUN_ALL_TESTS();
}
