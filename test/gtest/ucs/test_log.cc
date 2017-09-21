/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/debug/log.h>
}

class log_test : public ucs::test {

public:
    virtual void init() {
        char ucs_log_spec[70];
        const char *default_tmp_dir = "/tmp";
        const char *tmp_dir;
        ucs::test::init();

        /* skip because logger does not support file
         * output on valgrind
         */
        if (RUNNING_ON_VALGRIND) {
            UCS_TEST_SKIP_R("skipping on valgrind");
        }

        ucs_log_cleanup();
        push_config();
        tmp_dir = getenv("TMPDIR");
        if (tmp_dir == NULL) {
            tmp_dir = default_tmp_dir;
        }
        snprintf(logfile, sizeof(logfile), "%s/gtest_ucs_log.%d", tmp_dir, getpid());
        /* coverity[tainted_string] */
        unlink(logfile);
        snprintf(ucs_log_spec, sizeof(ucs_log_spec), "file:%s", logfile);
        modify_config("LOG_FILE", ucs_log_spec);
        modify_config("LOG_LEVEL", "info");
        ucs_log_init();
    }

    virtual void cleanup() {
        ucs_log_cleanup();
        pop_config();
        check_log_file();
        unlink(logfile);
        ucs_log_init();
        ucs::test::cleanup();
    }

    virtual void check_log_file() {
        ADD_FAILURE();
    }

    int do_grep(const char *needle) {
        char cmd[128];

        snprintf(cmd, sizeof(cmd), "grep '%s' %s", needle, logfile);
        return system(cmd);
    }

protected:
    char logfile[64];
};

class log_test_info : public log_test {
    virtual void check_log_file() {
        if (do_grep("UCX  INFO  hello world")) {
            ADD_FAILURE();
        }
    }
};

UCS_TEST_F(log_test_info, hello) {
    ucs_info("hello world");
}


class log_test_print : public log_test {
    virtual void check_log_file() {
        if (do_grep("UCX  PRINT debug message")) {
            if (ucs_global_opts.log_print_enable) {
                /* not found but it should be there */
                ADD_FAILURE();
            }
        } else {
            if (!ucs_global_opts.log_print_enable) {
                /* found but prints disabled!!! */
                ADD_FAILURE();
            }
        }
    }
};

UCS_TEST_F(log_test_print, print_on, "LOG_PRINT_ENABLE=y") {
    ucs_print("debug message");
}

UCS_TEST_F(log_test_print, print_off) {
    ucs_print("debug message");
}

