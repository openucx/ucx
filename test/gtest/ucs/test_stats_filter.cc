/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/stats/stats.h>
#include <ucs/sys/sys.h>
}

#include <sys/socket.h>
#include <netinet/in.h>

#if ENABLE_STATS

class stats_filter_test : public ucs::test {
public:

    stats_filter_test() {
        size_t size = sizeof(ucs_stats_class_t) +
                      NUM_COUNTERS * sizeof(m_data_stats_class->counter_names[0]);
        m_data_stats_class                   = (ucs_stats_class_t*)malloc(size);
        m_data_stats_class->name             = "data";
        m_data_stats_class->num_counters     = NUM_COUNTERS;
        m_data_stats_class->counter_names[0] = "counter0";
        m_data_stats_class->counter_names[1] = "counter1";
        m_data_stats_class->counter_names[2] = "counter2";
        m_data_stats_class->counter_names[3] = "counter3";

        cat_node = NULL;
        data_nodes[0] = data_nodes[1] = data_nodes[2] = NULL;
    }

    ~stats_filter_test() {
        free(m_data_stats_class);
    }

    virtual void init() {
        ucs::test::init();
        ucs_stats_cleanup();
        push_config();
        modify_config("STATS_DEST",    stats_dest_config().c_str());
        modify_config("STATS_TRIGGER", stats_trigger_config().c_str());
        modify_config("STATS_FORMAT", stats_format_config().c_str());
        ucs_stats_init();
        ASSERT_TRUE(ucs_stats_is_active());
    }

    virtual void cleanup() {
        ucs_stats_cleanup();
        pop_config();
        ucs_stats_init();
        ucs::test::cleanup();
    }

    virtual std::string stats_dest_config()    = 0;
    virtual std::string stats_trigger_config() = 0;
    virtual std::string stats_format_config() = 0;

    void prepare_nodes() {
        static ucs_stats_class_t category_stats_class = {
            "category", 0, {}
        };

        ucs_status_t status = UCS_STATS_NODE_ALLOC(&cat_node, &category_stats_class,
                                                   ucs_stats_get_root());
        ASSERT_UCS_OK(status);
        for (unsigned i = 0; i < NUM_DATA_NODES; ++i) {
            status = UCS_STATS_NODE_ALLOC(&data_nodes[i], m_data_stats_class,
                                         cat_node, "-%d", i);
            ASSERT_UCS_OK(status);

            UCS_STATS_UPDATE_COUNTER(data_nodes[i], 0, 10);
            UCS_STATS_UPDATE_COUNTER(data_nodes[i], 1, 20);
            UCS_STATS_UPDATE_COUNTER(data_nodes[i], 2, 30);
            UCS_STATS_UPDATE_COUNTER(data_nodes[i], 3, 40);
        }
    }

    void free_nodes() {
        for (unsigned i = 0; i < NUM_DATA_NODES; ++i) {
            UCS_STATS_NODE_FREE(data_nodes[i]);
        }
        UCS_STATS_NODE_FREE(cat_node);
    }

protected:    
    static const unsigned NUM_DATA_NODES = 3;
    static const unsigned NUM_COUNTERS   = 4;

    ucs_stats_class_t      *m_data_stats_class;
    ucs_stats_node_t       *cat_node;
    ucs_stats_node_t       *data_nodes[NUM_DATA_NODES];
};


class stats_filter_text_test : public stats_filter_test {
public:
    stats_filter_text_test() {
        m_pipefds[0] = -1;
        m_pipefds[1] = -1;
    }

    virtual void init() {
        /* Note: this test assumes data <64k, o/w stats dump will block forever */
        int ret = pipe(m_pipefds);
        ASSERT_EQ(0, ret);
        modify_config("STATS_FILTER",    "*counter*");
        stats_filter_test::init();
    }

    void close_pipes()
    {
        close(m_pipefds[0]);
        close(m_pipefds[1]);
        m_pipefds[0] = -1;
        m_pipefds[1] = -1;
    }

    virtual void cleanup() {
        stats_filter_test::cleanup();
        close_pipes();
    }

    virtual std::string stats_dest_config() {
        return "file:/dev/fd/" + ucs::to_string(m_pipefds[1]) + "";
    }

    std::string get_data() {
        std::string data(65536, '\0');
        ssize_t ret = read(m_pipefds[0], &data[0], data.size());
        EXPECT_GE(ret, 0);
        data.resize(ret);
        return data;
    }

    virtual std::string stats_trigger_config() {
        return "";
    }

protected:
    int m_pipefds[2];
};


class stats_filter_report : public stats_filter_text_test {
public:

    virtual std::string stats_format_config() {
        return "full";
    }

};

class stats_filter_agg : public stats_filter_text_test {
public:

    virtual std::string stats_format_config() {
        return "agg";
    }

};

class stats_filter_summary : public stats_filter_text_test {
public:

    virtual std::string stats_format_config() {
        return "summary";
    }

};


UCS_TEST_F(stats_filter_report, report) {
    prepare_nodes();
    ucs_stats_dump();
    free_nodes();

    std::string data = get_data();
    FILE *f = fmemopen(&data[0], data.size(), "rb");
    std::string output = "";
    char s[80];
    while (!feof(f)) {
	int term = fread(&s, 1, sizeof(s) - 1, f);
        if (term > 0) {
            s[term]=0;
            output += std::string(s);
        } else {
            break;
        }
    }

    std::string compared_string = std::string(ucs_get_host_name()) + ":" +
                                  ucs::to_string(getpid()) + ":" +
                                  "\n  category:\n" +
                                  "    data-0:\n" +
                                  "      counter0: 10\n" +
                                  "      counter1: 20\n" +
                                  "      counter2: 30\n" +
                                  "      counter3: 40\n" +
                                  "    data-1:\n" +
                                  "      counter0: 10\n" +
                                  "      counter1: 20\n" +
                                  "      counter2: 30\n" +
                                  "      counter3: 40\n" +
                                  "    data-2:\n" +
                                  "      counter0: 10\n" +
                                  "      counter1: 20\n" +
                                  "      counter2: 30\n" +
                                  "      counter3: 40\n\n";
    EXPECT_EQ(compared_string, output);
    fclose(f);
}

UCS_TEST_F(stats_filter_agg, report_agg) {

    prepare_nodes();
    ucs_stats_dump();
    free_nodes();

    std::string data = get_data();
    FILE *f = fmemopen(&data[0], data.size(), "rb");
    std::string output = "";
    char s[80];
    while (!feof(f)) {
	int term = fread(&s, 1, sizeof(s) - 1, f);
        if (term > 0) {
            s[term]=0;
            output += std::string(s);
        } else {
            break;
        }
    }

    std::string compared_string = std::string(ucs_get_host_name()) + ":" +
                                  ucs::to_string(getpid()) + ":" +
                                  "\n  category:\n"
                                  "    data*:\n"
                                  "      counter0: 30\n"
                                  "      counter1: 60\n"
                                  "      counter2: 90\n"
                                  "      counter3: 120\n\n";
    EXPECT_EQ(compared_string, output);
    fclose(f);
}

UCS_TEST_F(stats_filter_summary, summary) {
    prepare_nodes();
    ucs_stats_dump();
    free_nodes();

    std::string data = get_data();
    FILE *f = fmemopen(&data[0], data.size(), "rb");
    std::string output = "";
    char s[80];
    while (!feof(f)) {
	int term = fread(&s, 1, sizeof(s) - 1, f);
        if (term > 0) {
            s[term]=0;
            output += std::string(s);
        } else {
            break;
        }
    }
    std::string compared_string = std::string(ucs_get_host_name()) + ":" +
                                  ucs::to_string(getpid()) +
                                  ":data*:{counter0:30 counter1:60 " +
                                  "counter2:90 counter3:120} \n";
    EXPECT_EQ(compared_string, output);
    fclose(f);
}

#endif
