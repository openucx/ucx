/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* Copyright (C) Huawei Technologies Co., Ltd. 2021.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/stats/stats.h>
}

#include <sys/socket.h>
#include <sys/uio.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#ifdef ENABLE_STATS
#define NUM_DATA_NODES 20

/* The maximum number of UCX aggregate-sum counters */
#define UCS_STATS_NUM_AGGREGATE_SUM_COUNTERS_MAX 128

class stats_test : public ucs::test {
public:

    stats_test() {
        size_t size = sizeof(ucs_stats_class_t) +
                      NUM_COUNTERS * sizeof(m_data_stats_class->counter_names[0]);
        m_data_stats_class                   = (ucs_stats_class_t*)malloc(size);
        m_data_stats_class->name             = "data";
        m_data_stats_class->num_counters     = NUM_COUNTERS;
        m_data_stats_class->counter_names[0] = "counter0";
        m_data_stats_class->counter_names[1] = "counter1";
        m_data_stats_class->counter_names[2] = "counter2";
        m_data_stats_class->counter_names[3] = "counter3";
        m_data_stats_class->class_id         = UCS_STATS_CLASS_ID_INVALID;
    }

    ~stats_test() {
        free(m_data_stats_class);
    }

    virtual void init() {
        ucs::test::init();
        ucs_stats_cleanup();
        push_config();
        modify_config("STATS_DEST",    stats_dest_config().c_str());
        modify_config("STATS_TRIGGER", stats_trigger_config().c_str());
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

    void prepare_nodes(ucs_stats_node_t **cat_node,
                       ucs_stats_node_t *data_nodes[NUM_DATA_NODES]) {
        static ucs_stats_class_t category_stats_class = {
            "category", 0
        };

        ucs_status_t status = UCS_STATS_NODE_ALLOC(cat_node,
                                                   &category_stats_class,
                                                   ucs_stats_get_root(), "");
        ASSERT_UCS_OK(status);
        for (unsigned i = 0; i < NUM_DATA_NODES; ++i) {
            status = UCS_STATS_NODE_ALLOC(&data_nodes[i], m_data_stats_class,
                                          *cat_node, "-%d", i);
            ASSERT_UCS_OK(status);

            UCS_STATS_UPDATE_COUNTER(data_nodes[i], 0, 10);
            UCS_STATS_UPDATE_COUNTER(data_nodes[i], 1, 20);
            UCS_STATS_UPDATE_COUNTER(data_nodes[i], 2, 30);
            UCS_STATS_UPDATE_COUNTER(data_nodes[i], 3, 40);
        }

        /* make sure our original node is ok */
        check_cat_node(*cat_node, data_nodes);
    }

    void free_nodes(ucs_stats_node_t *cat_node,
                    ucs_stats_node_t *data_nodes[NUM_DATA_NODES]) {
        for (unsigned i = 0; i < NUM_DATA_NODES; ++i) {
            UCS_STATS_NODE_FREE(data_nodes[i]);
        }
        UCS_STATS_NODE_FREE(cat_node);
    }

    void check_tree(ucs_stats_node_t *root,
                    ucs_stats_node_t *data_nodes[NUM_DATA_NODES]) {
        EXPECT_EQ(1ul, ucs_list_length(&root->children[UCS_STATS_ACTIVE_CHILDREN]));
        check_cat_node(ucs_list_head(&root->children[UCS_STATS_ACTIVE_CHILDREN],
                                     ucs_stats_node_t, list), data_nodes);
    }

    void check_cat_node(ucs_stats_node_t *cat_node,
                        ucs_stats_node_t *data_nodes[NUM_DATA_NODES]) {
        EXPECT_EQ(std::string("category"), std::string(cat_node->cls->name));
        EXPECT_EQ((unsigned)0, cat_node->cls->num_counters);

        ucs_stats_node_t *data_node;
        ucs_list_for_each(data_node, &cat_node->children[UCS_STATS_ACTIVE_CHILDREN], list) {
            EXPECT_EQ(std::string("data"),     std::string(data_node->cls->name));
            EXPECT_EQ(unsigned(NUM_COUNTERS),  data_node->cls->num_counters);
            EXPECT_EQ(std::string("counter0"), std::string(data_node->cls->counter_names[0]));

            EXPECT_EQ((unsigned)10, data_node->counters[0]);
            EXPECT_EQ((unsigned)20, data_node->counters[1]);
            EXPECT_EQ((unsigned)30, data_node->counters[2]);
            EXPECT_EQ((unsigned)40, data_node->counters[3]);
        }
    }

protected:
    static const unsigned NUM_COUNTERS   = 4;

    ucs_stats_class_t *m_data_stats_class;
};

class stats_udp_test : public stats_test {
public:
    virtual void init() {
        ucs_status_t status = ucs_stats_server_start(0, &m_server);
        ASSERT_UCS_OK(status);
        stats_test::init();
    }

    virtual void cleanup() {
        stats_test::cleanup();
        ucs_stats_server_destroy(m_server);
    }

    void wait_for_stats() {
        do {
            usleep(1000 * ucs::test_time_multiplier());
        } while (ucs_stats_server_rcvd_packets(m_server) == 0);
    }

    virtual std::string stats_dest_config() {
        int port = ucs_stats_server_get_port(m_server);
        EXPECT_GT(port, 0);
        return "udp:localhost:" + ucs::to_string(port);
    }

    virtual std::string stats_trigger_config() {
        return "timer:0.1s";
    }

    void read_and_check_stats(ucs_stats_node_t *data_nodes[NUM_DATA_NODES]) {
        ucs_list_link_t *list = ucs_stats_server_get_stats(m_server);
        ASSERT_EQ(1ul, ucs_list_length(list));
        check_tree(ucs_list_head(list, ucs_stats_node_t, list), data_nodes);
        ucs_stats_server_purge_stats(m_server);
    }

protected:
    ucs_stats_server_h m_server;
};

class stats_file_test : public stats_test {
public:
    stats_file_test() {
        m_pipefds[0] = -1;
        m_pipefds[1] = -1;
    }

    virtual void init() {
        /* Note: this test assumes data <64k, o/w stats dump will block forever */
        int ret = pipe(m_pipefds);
        ASSERT_EQ(0, ret);
        stats_test::init();
    }

    void close_pipes()
    {
        close(m_pipefds[0]);
        close(m_pipefds[1]);
        m_pipefds[0] = -1;
        m_pipefds[1] = -1;
    }

    virtual void cleanup() {
        stats_test::cleanup();
        close_pipes();
    }

    virtual std::string stats_dest_config() {
        return "file:/dev/fd/" + ucs::to_string(m_pipefds[1]) + ":bin";
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

class stats_on_demand_test : public stats_udp_test {
public:
    virtual std::string stats_trigger_config() {
        return "";
    }
};

class stats_on_signal_test : public stats_udp_test {
public:
    virtual std::string stats_trigger_config() {
        return "signal:USR1";
    }
};

class stats_on_exit_test : public stats_file_test {
public:
    virtual std::string stats_dest_config() {
        return "file:/dev/fd/" + ucs::to_string(m_pipefds[1]);
    }

    /*
     * we check the dump-on-exit in cleanup method .
     */
    virtual void cleanup() {
        stats_test::cleanup();
        std::string data = get_data();
        size_t pos = 0;
        for (unsigned i = 0; i < NUM_DATA_NODES; ++i) {
            std::string node_name = " data-" + ucs::to_string(i) + ":";
            pos = data.find(node_name, pos);
            EXPECT_NE(pos, std::string::npos) << node_name << " not found";
            for (unsigned j = 0; j < NUM_COUNTERS; ++j) {
                std::string value = "counter" +
                                ucs::to_string(j) +
                                ": " +
                                ucs::to_string((j + 1) * 10);
                pos = data.find(value, pos);
                EXPECT_NE(pos, std::string::npos) << value << " not found";
            }
        }
        close_pipes();
    }

    virtual std::string stats_trigger_config() {
        return "exit";
    }
};

class stats_aggregate_sum_test : public stats_udp_test {
public:
    void
    read_and_check_aggrgt_sum_stats(ucs_stats_node_t *data_nodes[NUM_DATA_NODES])
    {
        const ucs_stats_aggrgt_counter_name_t *aggregate_cnt_names_db = NULL;
        ucs_stats_counter_t ucs_stats_aggregate_sum_counters
                [UCS_STATS_NUM_AGGREGATE_SUM_COUNTERS_MAX];
        size_t size;

        /* Test */
        size_t num_counters = ucs_stats_aggregate(ucs_stats_aggregate_sum_counters,
                                                  UCS_STATS_NUM_AGGREGATE_SUM_COUNTERS_MAX);

        ucs_stats_aggregate_get_counter_names(&aggregate_cnt_names_db, &size);

        EXPECT_EQ(num_counters, size);
        ASSERT_FALSE(aggregate_cnt_names_db == NULL);

        /* Example for processing the statistics */
        for (size_t i = 0; i < num_counters; i++) {
             ucs_print("[%zu] pid=%d | CNT: cls_name=<%s> Name=%s, value=%lu\n",
                       i,
                       getpid(),
                       aggregate_cnt_names_db[i].class_name,
                       aggregate_cnt_names_db[i].counter_name,
                       ucs_stats_aggregate_sum_counters[i]);
       }
    }
};

UCS_TEST_F(stats_on_demand_test, null_root) {
    ucs_stats_node_t       *cat_node;

    static ucs_stats_class_t category_stats_class = {
        "category", 0
    };
    ucs_status_t status                           =
            UCS_STATS_NODE_ALLOC(&cat_node, &category_stats_class, NULL, "");

    EXPECT_GE(status, UCS_ERR_INVALID_PARAM);
}

UCS_TEST_F(stats_udp_test, report) {
    ucs_stats_node_t       *cat_node;
    ucs_stats_node_t       *data_nodes[NUM_DATA_NODES] = {NULL};

    prepare_nodes(&cat_node, data_nodes);
    wait_for_stats();
    read_and_check_stats(data_nodes);
    free_nodes(cat_node, data_nodes);
}

UCS_TEST_F(stats_file_test, report) {
    ucs_stats_node_t       *cat_node;
    ucs_stats_node_t       *data_nodes[NUM_DATA_NODES] = {NULL};

    prepare_nodes(&cat_node, data_nodes);
    ucs_stats_dump();
    free_nodes(cat_node, data_nodes);

    std::string data = get_data();
    FILE *f = fmemopen(&data[0], data.size(), "rb");
    ucs_stats_node_t *root;
    ucs_status_t status = ucs_stats_deserialize(f, &root);
    ASSERT_UCS_OK(status);
    fclose(f);

    check_tree(root, data_nodes);
    ucs_stats_free(root);
}

UCS_TEST_F(stats_on_demand_test, report) {
    ucs_stats_node_t       *cat_node;
    ucs_stats_node_t       *data_nodes[NUM_DATA_NODES] = {NULL};

    prepare_nodes(&cat_node, data_nodes);
    ucs_stats_dump();
    wait_for_stats();
    read_and_check_stats(data_nodes);
    free_nodes(cat_node, data_nodes);
}

UCS_TEST_F(stats_on_signal_test, report) {
    ucs_stats_node_t       *cat_node;
    ucs_stats_node_t       *data_nodes[NUM_DATA_NODES] = {NULL};

    prepare_nodes(&cat_node, data_nodes);
    kill(getpid(), SIGUSR1);
    wait_for_stats();
    read_and_check_stats(data_nodes);
    free_nodes(cat_node, data_nodes);
}

UCS_TEST_F(stats_on_exit_test, dump) {
    ucs_stats_node_t       *cat_node;
    ucs_stats_node_t       *data_nodes[NUM_DATA_NODES] = {NULL};

    prepare_nodes(&cat_node, data_nodes);
    free_nodes(cat_node, data_nodes);
}

UCS_MT_TEST_F(stats_file_test, mt_add_remove, 10) {
    ucs_stats_node_t       *cat_node;
    ucs_stats_node_t       *data_nodes[NUM_DATA_NODES] = {NULL};
    unsigned i;

    for (i = 0; i < 100; i++) {
        prepare_nodes(&cat_node, data_nodes);
        free_nodes(cat_node, data_nodes);
    }
}

UCS_TEST_F(stats_aggregate_sum_test, report) {
    ucs_stats_node_t *cat_node;
    ucs_stats_node_t *data_nodes[NUM_DATA_NODES] = {NULL};

    prepare_nodes(&cat_node, data_nodes);
    read_and_check_aggrgt_sum_stats(data_nodes);
    free_nodes(cat_node, data_nodes);
}

class stats_entity_cmp_test : public stats_udp_test {
public:
    virtual void init() {
        stats_udp_test::init();

        static ucs_stats_class_t test_cls = {"test_entity", 0};
        ucs_stats_node_t *node;
        ucs_status_t status = UCS_STATS_NODE_ALLOC(&node, &test_cls,
                                                   ucs_stats_get_root(), "");
        ASSERT_UCS_OK(status);

        FILE *stream = open_memstream(&m_buffer, &m_buf_size);
        ASSERT_NE(nullptr, stream);
        status = ucs_stats_serialize(stream, node, UCS_STATS_SERIALIZE_BINARY);
        fclose(stream);
        ASSERT_UCS_OK(status);
        UCS_STATS_NODE_FREE(node);
    }

    virtual void cleanup() {
        free(m_buffer);
        stats_udp_test::cleanup();
    }

    virtual std::string stats_trigger_config() {
        return "";
    }

    int create_bound_udp(uint16_t src_port) {
        int fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        EXPECT_GE(fd, 0);

        struct sockaddr_in addr = {};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

        addr.sin_port = htons(src_port);
        if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
            close(fd);
            return -1;
        }

        addr.sin_port = htons(ucs_stats_server_get_port(m_server));
        if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
            close(fd);
            return -1;
        }
        return fd;
    }

    void send_raw_stats(int fd, uint64_t timestamp) {
        struct {
            char     magic[8];
            uint64_t timestamp;
            uint32_t total_size;
            uint32_t frag_offset;
            uint32_t frag_size;
        } UCS_S_PACKED hdr;

        const size_t max_frag = 1400 - sizeof(hdr);
        size_t offset = 0;

        memcpy(hdr.magic, "UCSSTAT1", 8);
        hdr.timestamp  = timestamp;
        hdr.total_size = m_buf_size;

        while (offset < m_buf_size) {
            size_t frag_size = std::min(max_frag, m_buf_size - offset);
            hdr.frag_offset  = offset;
            hdr.frag_size    = frag_size;

            struct iovec iov[2];
            iov[0].iov_base = &hdr;
            iov[0].iov_len  = sizeof(hdr);
            iov[1].iov_base = m_buffer + offset;
            iov[1].iov_len  = frag_size;

            ssize_t nsent = writev(fd, iov, 2);
            ASSERT_EQ((ssize_t)(sizeof(hdr) + frag_size), nsent);
            offset += frag_size;
        }
    }

protected:
    char   *m_buffer;
    size_t  m_buf_size;
};

/*
 * Verify that the stats server distinguishes two clients on the same IP but
 * different ports, even when both ports hash to the same bucket (difference
 * equals ENTITY_HASH_SIZE = 997).
 */
UCS_TEST_F(stats_entity_cmp_test, multi_client_same_hash_bucket) {
    const uint16_t port1 = 10000;
    const uint16_t port2 = port1 + 997;

    int fd1 = create_bound_udp(port1);
    int fd2 = create_bound_udp(port2);
    if (fd1 < 0 || fd2 < 0) {
        if (fd1 >= 0) close(fd1);
        if (fd2 >= 0) close(fd2);
        UCS_TEST_SKIP_R("cannot bind to test ports");
    }

    send_raw_stats(fd1, 1);
    send_raw_stats(fd2, 2);

    do {
        usleep(1000 * ucs::test_time_multiplier());
    } while (ucs_stats_server_rcvd_packets(m_server) < 2);

    ucs_list_link_t *stats_list = ucs_stats_server_get_stats(m_server);
    EXPECT_EQ(2ul, ucs_list_length(stats_list));
    ucs_stats_server_purge_stats(m_server);

    close(fd1);
    close(fd2);
}

#endif
