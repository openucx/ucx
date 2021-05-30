/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <common/test.h>
#include <fstream>
#include <set>
#include <dirent.h>

extern "C" {
#include <ucs/debug/log.h>
#include <ucs/sys/compiler.h>
}

class log_test : private ucs::clear_dontcopy_regions, public ucs::test {
public:
    virtual void init() {
        /* skip because logger does not support file
         * output on valgrind
         */
        if (RUNNING_ON_VALGRIND) {
            UCS_TEST_SKIP_R("skipping on valgrind");
        }

        const char *default_tmp_dir = "/tmp";

        ucs::test::init();

        ucs_log_cleanup();
        push_config();
        const char *tmp_dir = getenv("TMPDIR");
        if (tmp_dir == NULL) {
            tmp_dir = default_tmp_dir;
        }

        tmp_dir_path  = tmp_dir;
        template_name = "gtest_ucs_log." + ucs::to_string(getpid());

        std::string logfile = template_grep_name =
            tmp_dir_path + "/" + template_name;

        /* add date/time to the log file name in order to track how many
         * different log files were created during testing */
        logfile += ".%t";

        /* add `*` to the template grep name to be able searching a test
         * string in all possible file names with the time specified */
        template_grep_name += ".*";

        /* remove already created files with the similar file name */
        log_files_foreach(&log_test::remove_file);

        std::string ucs_log_spec = "file:" + logfile;
        modify_config("LOG_FILE", ucs_log_spec);
        modify_config("LOG_LEVEL", "info");
        ucs_log_init();
    }

    virtual void cleanup() {
        ucs_log_cleanup();
        pop_config();
        check_log_file();
        unsigned files_count = log_files_foreach(&log_test::remove_file);
        EXPECT_LE(files_count, ucs_global_opts.log_file_rotate + 1);
        EXPECT_NE(0, files_count);
        ucs_log_init();
        ucs::test::cleanup();
    }

    void remove_file(const std::string &name, void *arg) {
        unlink(name.c_str());
    }

    typedef void (log_test::*log_file_foreach_cb)(const std::string &name,
                                                  void *arg);

    unsigned log_files_foreach(log_file_foreach_cb cb, void *arg = NULL) {
        DIR *dir = opendir(tmp_dir_path.c_str());
        struct dirent *entry;
        unsigned files_count = 0;

        while ((entry = readdir(dir)) != NULL) {
            if (strstr(entry->d_name, template_name.c_str()) != NULL) {
                std::string full_file_name = tmp_dir_path + "/" +
                                             ucs::to_string(entry->d_name);
                (this->*cb)(full_file_name, arg);
                files_count++;
            }
        }
        closedir(dir);

        return files_count;
    }

    void test_file_cur_size(const std::string &log_file_name, void *arg) {
        FILE *logfile_fp = fopen(log_file_name.c_str(), "r");
        ASSERT_TRUE(logfile_fp != NULL);

        ucs_log_flush();

        int ret = fseek(logfile_fp, 0, SEEK_END);
        EXPECT_EQ(0, ret);

        long cur_size = ftell(logfile_fp);
        EXPECT_LE(static_cast<size_t>(cur_size), ucs_global_opts.log_file_size);

        fclose(logfile_fp);

        m_log_files_set.insert(log_file_name);
    }

    virtual void check_log_file() {
        ADD_FAILURE() << read_logfile();
    }

    bool do_grep(const std::string &needle) {
        unsigned num_retries       = 0;
        std::string cmd_str        = "<none>";
        std::string system_ret_str = "<none>";

        while (num_retries++ < GREP_RETRIES) {
            /* if this is the last retry, allow printing the grep output */
            std::string grep_cmd = ucs_likely(num_retries != GREP_RETRIES) ?
                                   "grep -q" : "grep";
            cmd_str = grep_cmd + " '" + needle + "' " + template_grep_name;
            int ret = system(cmd_str.c_str());
            if (ret == 0) {
                return true;
            } else if (ret == -1) {
                system_ret_str = ucs::to_string(errno);
            } else {
                system_ret_str = ucs::exit_status_info(ret);
            }

            ucs_log_flush();
        }

        UCS_TEST_MESSAGE << "\"" << cmd_str << "\" failed after "
                         << num_retries - 1 << " iterations ("
                         << system_ret_str << ")";

        return false;
    }

    void read_logfile(const std::string &log_file_name, void *arg) {
        std::stringstream *ss = (std::stringstream*)arg;
        std::ifstream ifs(log_file_name.c_str());
        *ss << log_file_name << ":" << std::endl << ifs.rdbuf() << std::endl;
    }

    std::string read_logfile() {
        std::stringstream ss;
        log_files_foreach(&log_test::read_logfile, &ss);
        return ss.str();
    }

protected:
    std::string template_name;
    std::string template_grep_name;
    std::string tmp_dir_path;
    std::set<std::string> m_log_files_set;

    static const unsigned GREP_RETRIES = 20;
};

class log_test_info : public log_test {
protected:
    log_test_info() :
        m_spacer("  "), m_log_str("hello world"), m_exp_found(true)
    {
    }

    virtual void check_log_file()
    {
        std::string log_str = "UCX  INFO" + m_spacer + m_log_str;
        if (m_exp_found != do_grep(log_str)) {
            ADD_FAILURE() << read_logfile() << " Expected to "
                          << (m_exp_found ? "" : "not ") << "find " << log_str;
        }
    }

    void log_info()
    {
        ucs_info("%s", m_log_str.c_str());
    }

    std::string m_spacer;
    std::string m_log_str;
    bool m_exp_found;
};

UCS_TEST_F(log_test_info, hello) {
    log_info();
}

UCS_TEST_F(log_test_info, hello_indent) {
    ucs_log_indent(1);
    log_info();
    ucs_log_indent(-1);
    m_spacer += "  ";
}

UCS_TEST_F(log_test_info, filter_on, "LOG_FILE_FILTER=*test_lo*.cc") {
    log_info();
}

UCS_TEST_F(log_test_info, filter_off, "LOG_FILE_FILTER=file.c") {
    log_info();
    m_exp_found = false;
}

class log_test_print : public log_test {
    virtual void check_log_file() {
        if (!do_grep("UCX  PRINT debug message")) {
            if (ucs_global_opts.log_print_enable) {
                /* not found but it should be there */
                ADD_FAILURE() << read_logfile();
            }
        } else {
            if (!ucs_global_opts.log_print_enable) {
                /* found but prints disabled!!! */
                ADD_FAILURE() << read_logfile();
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


class log_test_file_size : public log_test {
protected:
    virtual void check_log_file() {
        unsigned files_count = log_files_foreach(&log_test_file_size::
                                                 test_file_cur_size);
        EXPECT_LE(files_count, ucs_global_opts.log_file_rotate + 1);
    }

    virtual void check_log_file(const std::string &test_str) {
        check_log_file();
        EXPECT_TRUE(do_grep(test_str));
    }

    void generate_random_str(std::string &s, size_t len) {
        static const char possible_vals[] =
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz";

        ASSERT_TRUE(len != 0);

        s.resize(len);

        for (size_t i = 0; i < len; ++i) {
            s[i] = possible_vals[ucs::rand() %
                                 (ucs_static_array_size(possible_vals) - 1)];
        }
    }

    void print_random_str() {
        size_t entry_size = ucs::rand() % ucs_log_get_buffer_size();
        if (entry_size == 0) {
            entry_size = 1;
        }

        std::string entry_buf;

        generate_random_str(entry_buf, entry_size);
        /* use %s here in order to satisfy the "format-security" compilation
         * flag requirements */
        ucs_info("%s", entry_buf.c_str());

        /* to not waste a lot of time grepping the test string */
        if (entry_size < 128) {
            check_log_file(entry_buf);
        } else {
            check_log_file();
        }
    }

    void test_log_file_max_size() {
        const unsigned num_iters = 4;

        for (unsigned i = 0; i < num_iters; i++) {
            size_t set_size, exp_files_count;

            do {
                print_random_str();

                set_size        = m_log_files_set.size();
                exp_files_count = (ucs_global_opts.log_file_rotate + 1);
            } while ((set_size == 0) ||
                     ((set_size % exp_files_count) != 0));
        }

        EXPECT_EQ(m_log_files_set.size(),
                  ucs_global_opts.log_file_rotate + 1);
    }
};

const std::string small_file_size = ucs::to_string(UCS_ALLOCA_MAX_SIZE);

UCS_TEST_F(log_test_file_size, small_file, "LOG_FILE_SIZE=" +
                                           small_file_size) {
    test_log_file_max_size();
}

UCS_TEST_F(log_test_file_size, large_file, "LOG_FILE_SIZE=8k") {
    test_log_file_max_size();
}

UCS_TEST_F(log_test_file_size, small_files, "LOG_FILE_SIZE=" +
                                            small_file_size,
                                            "LOG_FILE_ROTATE=4") {
    test_log_file_max_size();
}

UCS_TEST_F(log_test_file_size, large_files, "LOG_FILE_SIZE=8k",
                                            "LOG_FILE_ROTATE=4") {
    test_log_file_max_size();
}


class log_test_backtrace : public log_test {
    virtual void check_log_file() {
        if (!do_grep("print_backtrace")) {
            ADD_FAILURE() << read_logfile();
        }

#ifdef HAVE_DETAILED_BACKTRACE
        if (!do_grep("main")) {
            ADD_FAILURE() << read_logfile();
        }
#endif
    }
};

UCS_TEST_F(log_test_backtrace, backtrace) {
    ucs_log_print_backtrace(UCS_LOG_LEVEL_INFO);
}

class log_demo : public ucs::test {
};

UCS_MT_TEST_F(log_demo, indent, 4)
{
    ucs::scoped_log_level enable_debug(UCS_LOG_LEVEL_DEBUG);

    ucs_debug("scope begin");

    ucs_log_indent(1);
    EXPECT_EQ(1, ucs_log_get_current_indent());
    ucs_debug("nested log 1");

    ucs_log_indent(1);
    EXPECT_EQ(2, ucs_log_get_current_indent());
    ucs_debug("nested log 1.1");
    ucs_log_indent(-1);

    EXPECT_EQ(1, ucs_log_get_current_indent());
    ucs_debug("nested log 2");
    ucs_log_indent(-1);

    EXPECT_EQ(0, ucs_log_get_current_indent());
    ucs_debug("done");
}
