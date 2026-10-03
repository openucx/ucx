/**
 * Copyright (C) Intel Corporation, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <uct/uct_test.h>

extern "C" {
#include <uct/ze/base/ze_base.h>
#include <uct/ze/ze_ipc/ze_ipc_md.h>
}

#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>


/*
 * Endpoint-level RMA tests for ze_ipc with both entities in this process. The
 * transport takes its same-pid path, so these tests skip the pidfd_getfd fd
 * duplication and the reachability check; test_ze_ipc_rma_xproc covers them.
 */
class test_ze_ipc_rma : public uct_test {
protected:
    void init()
    {
        uct_test::init();

        m_receiver = uct_test::create_entity(0);
        m_entities.push_back(m_receiver);

        m_sender = uct_test::create_entity(0);
        m_entities.push_back(m_sender);

        m_sender->connect(0, *m_receiver, 0);
    }

    struct flush_comp {
        uct_completion_t super;
        uct_completion_t *copy_comp;
        int              copies_left;
    };

    static void copy_done(uct_completion_t *self)
    {
    }

    static void flush_done(uct_completion_t *self)
    {
        flush_comp *comp = ucs_container_of(self, flush_comp, super);

        comp->copies_left = comp->copy_comp->count;
    }

    entity *m_sender;
    entity *m_receiver;

    static const uint64_t SEED1 = 0xABClu;
    static const uint64_t SEED2 = 0xDEFlu;
};


UCS_TEST_P(test_ze_ipc_rma, put_zcopy)
{
    size_t length = 4096;

    mapped_buffer sendbuf(length, SEED1, *m_sender, 0,
                          UCS_MEMORY_TYPE_ZE_DEVICE);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0,
                          UCS_MEMORY_TYPE_ZE_DEVICE);

    ASSERT_UCS_OK_OR_INPROGRESS(uct_ep_put_zcopy(m_sender->ep(0), sendbuf.iov(),
                                                 1, recvbuf.addr(),
                                                 recvbuf.rkey(), NULL));
    m_sender->flush();
    recvbuf.pattern_check(SEED1);
}

UCS_TEST_P(test_ze_ipc_rma, get_zcopy)
{
    size_t length = 4096;

    mapped_buffer sendbuf(length, SEED1, *m_receiver, 0,
                          UCS_MEMORY_TYPE_ZE_DEVICE);
    mapped_buffer recvbuf(length, SEED2, *m_sender, 0,
                          UCS_MEMORY_TYPE_ZE_DEVICE);

    ASSERT_UCS_OK_OR_INPROGRESS(uct_ep_get_zcopy(m_sender->ep(0), recvbuf.iov(),
                                                 1, sendbuf.addr(),
                                                 sendbuf.rkey(), NULL));
    m_sender->flush();
    recvbuf.pattern_check(SEED1);
}

UCS_TEST_P(test_ze_ipc_rma, ep_flush_waits_for_copies)
{
    static const unsigned num_copies = 8;
    static const size_t length       = UCS_MBYTE;

    mapped_buffer sendbuf(length * num_copies, SEED1, *m_sender, 0,
                          UCS_MEMORY_TYPE_ZE_DEVICE);
    mapped_buffer recvbuf(length * num_copies, SEED2, *m_receiver, 0,
                          UCS_MEMORY_TYPE_ZE_DEVICE);
    uct_completion_t copy_comp = {copy_done, (int)num_copies, UCS_OK};
    flush_comp flush           = {{flush_done, 1, UCS_OK}, &copy_comp, -1};
    uct_iov_t iov              = *sendbuf.iov();

    iov.length = length;
    for (unsigned i = 0; i < num_copies; ++i) {
        iov.buffer = UCS_PTR_BYTE_OFFSET(sendbuf.ptr(), i * length);
        ASSERT_EQ(UCS_INPROGRESS,
                  uct_ep_put_zcopy(m_sender->ep(0), &iov, 1,
                                   recvbuf.addr() + (i * length),
                                   recvbuf.rkey(), &copy_comp));
    }

    /* Nothing has progressed the copies yet */
    ASSERT_EQ(UCS_INPROGRESS,
              uct_ep_flush(m_sender->ep(0), 0, &flush.super));
    wait_for_value(&flush.super.count, 0, true);
    ASSERT_EQ(0, flush.super.count);

    EXPECT_EQ(UCS_OK, flush.super.status);
    EXPECT_EQ(0, flush.copies_left);
    EXPECT_EQ(UCS_OK, copy_comp.status);
    recvbuf.pattern_check(SEED1);
}

UCS_TEST_P(test_ze_ipc_rma, ep_flush_pending_at_close, "ZE_IPC_MAX_POLL=1")
{
    static const size_t length = 4096;
    uct_completion_t copy_comp = {copy_done, 1, UCS_OK};
    flush_comp flush           = {{flush_done, 1, UCS_OK}, &copy_comp, -1};

    {
        mapped_buffer sendbuf(length, SEED1, *m_sender, 0,
                              UCS_MEMORY_TYPE_ZE_DEVICE);
        mapped_buffer recvbuf(length, SEED2, *m_receiver, 0,
                              UCS_MEMORY_TYPE_ZE_DEVICE);

        ASSERT_EQ(UCS_INPROGRESS,
                  uct_ep_put_zcopy(m_sender->ep(0), sendbuf.iov(), 1,
                                   recvbuf.addr(), recvbuf.rkey(),
                                   &copy_comp));
        ASSERT_EQ(UCS_INPROGRESS,
                  uct_ep_flush(m_sender->ep(0), 0, &flush.super));

        /* With MAX_POLL=1 a progress call completes at most one event, so the
         * flush marker is still queued when the copy completes */
        test_base::wait_for_value(&copy_comp.count, 0,
                                  [this]() { progress(); });
        ASSERT_EQ(0, copy_comp.count);
    }

    EXPECT_EQ(1, flush.super.count);
    m_sender->destroy_eps();
    m_entities.remove(m_sender);
    EXPECT_EQ(1, flush.super.count);
}

UCS_TEST_P(test_ze_ipc_rma, put_zcopy_rejects_multi_iov)
{
    size_t length = 64;
    mapped_buffer sendbuf(length, SEED1, *m_sender, 0,
                          UCS_MEMORY_TYPE_ZE_DEVICE);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0,
                          UCS_MEMORY_TYPE_ZE_DEVICE);
    uct_iov_t iov[2];

    iov[0] = *sendbuf.iov();
    iov[1] = *sendbuf.iov();

    ucs_status_t status;
    {
        scoped_log_handler slh(hide_errors_logger);
        status = uct_ep_put_zcopy(m_sender->ep(0), iov, 2, recvbuf.addr(),
                                  recvbuf.rkey(), NULL);
    }
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
}

UCS_TEST_P(test_ze_ipc_rma, put_zcopy_rejects_out_of_range_rkey)
{
    size_t length = 64;
    mapped_buffer sendbuf(length, SEED1, *m_sender, 0,
                          UCS_MEMORY_TYPE_ZE_DEVICE);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0,
                          UCS_MEMORY_TYPE_ZE_DEVICE);

    ucs_status_t status;
    {
        /* the rkey covers the whole driver allocation, which is page-rounded
         * and therefore larger than 'length' - go far enough past the base
         * address to land outside it */
        scoped_log_handler slh(hide_errors_logger);
        status = uct_ep_put_zcopy(m_sender->ep(0), sendbuf.iov(), 1,
                                  recvbuf.addr() + UCS_MBYTE, recvbuf.rkey(),
                                  NULL);
    }
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
}

_UCT_INSTANTIATE_TEST_CASE(test_ze_ipc_rma, ze_ipc)


/*
 * Cross-process RMA. The test process exports a device buffer, and an exec'd
 * copy of this gtest binary running the same test imports it and issues PUT
 * and GET. Only the control socket survives the exec, so the importer can
 * reach the exporter's dmabuf fd only through pidfd_getfd.
 */
class test_ze_ipc_rma_xproc :
    private ucs::clear_dontcopy_regions,
    public uct_test {
protected:
    enum {
        MSG_PUT_DONE,
        MSG_GET_READY,
        MSG_DONE
    };

    struct op_msg {
        uint32_t op;
        uint32_t round;
    };

    struct exporter_info {
        pid_t    pid;
        uint64_t address;
        uint32_t iface_addr_len;
        uint32_t dev_addr_len;
        uint32_t rkey_len;
    };

    struct peer_process {
        pid_t pid = -1;
        int   fd = -1;

        ~peer_process()
        {
            if (fd >= 0) {
                close(fd);
            }
            if (pid > 0) {
                kill(pid, SIGKILL);
                waitpid(pid, NULL, 0);
            }
        }
    };

    static const char *PEER_FD_ENV;
    static const int PEER_FD         = 3;
    static const int TIMEOUT_MS      = 60000;
    static const unsigned NUM_ROUNDS = 3;
    static const size_t LENGTH       = UCS_MBYTE;

    static uint64_t put_seed(unsigned round)
    {
        return 0x1000 + round;
    }

    static uint64_t get_seed(unsigned round)
    {
        return 0x2000 + round;
    }

    static void send_msg(int fd, const void *buffer, size_t length)
    {
        ssize_t ret = send(fd, buffer, length, MSG_NOSIGNAL);

        if (ret != (ssize_t)length) {
            UCS_TEST_ABORT("send() returned " << ret << ": "
                                              << strerror(errno));
        }
    }

    static size_t recv_msg(int fd, void *buffer, size_t max_length)
    {
        struct pollfd pfd = {fd, POLLIN, 0};
        ssize_t ret;

        do {
            ret = poll(&pfd, 1, TIMEOUT_MS * ucs::test_time_multiplier());
        } while ((ret < 0) && (errno == EINTR));

        if (ret != 1) {
            UCS_TEST_ABORT("no message from peer process: poll() returned "
                           << ret);
        }

        ret = recv(fd, buffer, max_length, 0);
        if (ret <= 0) {
            UCS_TEST_ABORT("peer process closed the socket: recv() returned "
                           << ret);
        }

        return ret;
    }

    static void send_op(int fd, uint32_t op, uint32_t round)
    {
        op_msg msg = {op, round};

        send_msg(fd, &msg, sizeof(msg));
    }

    static void recv_op(int fd, uint32_t op, uint32_t round)
    {
        op_msg msg = {};
        size_t length;

        length = recv_msg(fd, &msg, sizeof(msg));
        if ((length != sizeof(msg)) || (msg.op != op) || (msg.round != round)) {
            UCS_TEST_ABORT("got message op " << msg.op << " round " << msg.round
                                             << ", expected op " << op
                                             << " round " << round);
        }
    }

    void pack_exporter_info(const entity &e, const mapped_buffer &buffer,
                            std::vector<char> &info)
    {
        exporter_info hdr;
        uct_md_mkey_pack_params_t pack_params = {};

        hdr.pid            = getpid();
        hdr.address        = buffer.addr();
        hdr.iface_addr_len = e.iface_attr().iface_addr_len;
        hdr.dev_addr_len   = e.iface_attr().device_addr_len;
        hdr.rkey_len       = e.md_attr().rkey_packed_size;

        info.resize(sizeof(hdr) + hdr.iface_addr_len + hdr.dev_addr_len +
                    hdr.rkey_len);
        memcpy(&info[0], &hdr, sizeof(hdr));

        char *iface_addr = &info[sizeof(hdr)];
        char *dev_addr   = iface_addr + hdr.iface_addr_len;
        char *rkey_buf   = dev_addr + hdr.dev_addr_len;

        ASSERT_UCS_OK(uct_iface_get_address(e.iface(),
                                            (uct_iface_addr_t*)iface_addr));
        ASSERT_UCS_OK(
                uct_iface_get_device_address(e.iface(),
                                             (uct_device_addr_t*)dev_addr));
        ASSERT_UCS_OK(uct_md_mkey_pack_v2(e.md(), buffer.memh(), buffer.ptr(),
                                          buffer.length(), &pack_params,
                                          rkey_buf));
    }

    void start_importer(peer_process &peer)
    {
        const ::testing::TestInfo *test_info =
                ::testing::UnitTest::GetInstance()->current_test_info();
        std::string filter      = std::string("--gtest_filter=") +
                                  test_info->test_suite_name() + "." +
                                  test_info->name();
        std::string repeat      = "--gtest_repeat=1";
        std::string exe         = "/proc/self/exe";
        std::string fd_env      = std::string(PEER_FD_ENV) + "=" +
                                  std::to_string(PEER_FD);
        std::vector<char*> argv = {&exe[0], &filter[0], &repeat[0], NULL};
        std::vector<char*> envp;
        std::vector<int> close_fds;
        int fds[2];

        /* Sharding would make the child skip its only test and exit 0 */
        for (char **var = environ; *var != NULL; ++var) {
            if (strncmp(*var, "GTEST_TOTAL_SHARDS=", 19) &&
                strncmp(*var, "GTEST_SHARD_INDEX=", 18)) {
                envp.push_back(*var);
            }
        }
        envp.push_back(&fd_env[0]);
        envp.push_back(NULL);

        ASSERT_EQ(0, socketpair(AF_UNIX, SOCK_SEQPACKET, 0, fds));
        for (int fd : ucs::get_open_fds()) {
            if (fd > PEER_FD) {
                close_fds.push_back(fd);
            }
        }

        peer.pid = fork();
        if (peer.pid == 0) {
            /* Async-signal-safe calls only until execve() */
            if (dup2(fds[1], PEER_FD) == PEER_FD) {
                for (int fd : close_fds) {
                    close(fd);
                }
                execve(exe.c_str(), argv.data(), envp.data());
            }
            _exit(127);
        }

        close(fds[1]);
        peer.fd = fds[0];
        ASSERT_GT(peer.pid, 0) << "fork() failed: " << strerror(errno);
    }

    /* The driver closes the fd of an outstanding IPC handle when a device
     * other than the allocation's accesses the buffer */
    static void fill_from_other_device(const mapped_buffer &buffer)
    {
        ze_context_desc_t context_desc     = {};
        ze_command_queue_desc_t queue_desc = {};
        ze_command_list_handle_t cmdlist;
        ze_context_handle_t context;
        uint8_t value = 0;

        context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
        queue_desc.stype   = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
        queue_desc.mode    = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;

        ASSERT_EQ(ZE_RESULT_SUCCESS, zeContextCreate(uct_ze_base_get_driver(),
                                                     &context_desc, &context));
        ASSERT_EQ(ZE_RESULT_SUCCESS,
                  zeCommandListCreateImmediate(context,
                                               uct_ze_base_get_device(1),
                                               &queue_desc, &cmdlist));
        EXPECT_EQ(ZE_RESULT_SUCCESS,
                  zeCommandListAppendMemoryFill(cmdlist, buffer.ptr(), &value,
                                                sizeof(value), buffer.length(),
                                                NULL, 0, NULL));
        EXPECT_EQ(ZE_RESULT_SUCCESS, zeCommandListDestroy(cmdlist));
        EXPECT_EQ(ZE_RESULT_SUCCESS, zeContextDestroy(context));
    }

    void run_exporter(bool other_device)
    {
        entity *e = uct_test::create_entity(0);
        m_entities.push_back(e);

        mapped_buffer buffer(LENGTH, 0, *e, 0, UCS_MEMORY_TYPE_ZE_DEVICE);
        std::vector<char> info;
        peer_process peer;
        int status;

        ASSERT_NO_FATAL_FAILURE(pack_exporter_info(*e, buffer, info));
        ASSERT_NO_FATAL_FAILURE(start_importer(peer));
        if (other_device) {
            ASSERT_NO_FATAL_FAILURE(fill_from_other_device(buffer));
        }
        send_msg(peer.fd, info.data(), info.size());

        for (unsigned round = 0; round < NUM_ROUNDS; ++round) {
            recv_op(peer.fd, MSG_PUT_DONE, round);
            buffer.pattern_check(put_seed(round));
            buffer.pattern_fill(get_seed(round));
            send_op(peer.fd, MSG_GET_READY, round);
        }
        recv_op(peer.fd, MSG_DONE, NUM_ROUNDS);

        ASSERT_EQ(peer.pid, waitpid(peer.pid, &status, 0));
        peer.pid = -1;
        EXPECT_TRUE(WIFEXITED(status) && (WEXITSTATUS(status) == 0))
                << "importer exit status " << status;
    }

    void run_importer(int fd)
    {
        std::vector<char> info(UCS_KBYTE * 4);
        uct_iface_is_reachable_params_t params = {};
        uct_ep_params_t ep_params              = {};
        ucs::handle<uct_ep_h> ep;
        uct_rkey_bundle_t rkey;
        exporter_info hdr;
        char info_str[256];
        size_t length;

        length = recv_msg(fd, info.data(), info.size());
        ASSERT_GE(length, sizeof(hdr));
        memcpy(&hdr, info.data(), sizeof(hdr));
        ASSERT_EQ(sizeof(hdr) + hdr.iface_addr_len + hdr.dev_addr_len +
                          hdr.rkey_len,
                  length);

        const void *iface_addr = &info[sizeof(hdr)];
        const void *dev_addr   = &info[sizeof(hdr) + hdr.iface_addr_len];
        const void *rkey_buf =
                &info[sizeof(hdr) + hdr.iface_addr_len + hdr.dev_addr_len];

        entity *e = uct_test::create_entity(0);
        m_entities.push_back(e);

        ASSERT_NE(getpid(), hdr.pid);
        ASSERT_EQ(e->iface_attr().iface_addr_len, hdr.iface_addr_len);
        ASSERT_EQ(e->iface_attr().device_addr_len, hdr.dev_addr_len);
        ASSERT_EQ(hdr.pid, *(const pid_t*)iface_addr);

        params.field_mask         = UCT_IFACE_IS_REACHABLE_FIELD_DEVICE_ADDR |
                                    UCT_IFACE_IS_REACHABLE_FIELD_IFACE_ADDR |
                                    UCT_IFACE_IS_REACHABLE_FIELD_INFO_STRING |
                                    UCT_IFACE_IS_REACHABLE_FIELD_INFO_STRING_LENGTH;
        params.device_addr        = (const uct_device_addr_t*)dev_addr;
        params.iface_addr         = (const uct_iface_addr_t*)iface_addr;
        params.info_string        = info_str;
        params.info_string_length = sizeof(info_str);
        info_str[0]               = '\0';
        ASSERT_TRUE(uct_iface_is_reachable_v2(e->iface(), &params)) << info_str;

        ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE |
                               UCT_EP_PARAM_FIELD_DEV_ADDR |
                               UCT_EP_PARAM_FIELD_IFACE_ADDR;
        ep_params.iface      = e->iface();
        ep_params.dev_addr   = (const uct_device_addr_t*)dev_addr;
        ep_params.iface_addr = (const uct_iface_addr_t*)iface_addr;
        UCS_TEST_CREATE_HANDLE(uct_ep_h, ep, uct_ep_destroy, uct_ep_create,
                               &ep_params);

        ASSERT_UCS_OK(uct_rkey_unpack(GetParam()->component, rkey_buf, &rkey));
        ASSERT_EQ(hdr.pid, ((const uct_ze_ipc_key_t*)rkey.rkey)->pid);

        mapped_buffer sendbuf(LENGTH, 0, *e, 0, UCS_MEMORY_TYPE_ZE_DEVICE);
        mapped_buffer recvbuf(LENGTH, 0, *e, 0, UCS_MEMORY_TYPE_ZE_DEVICE);

        for (unsigned round = 0; round < NUM_ROUNDS; ++round) {
            sendbuf.pattern_fill(put_seed(round));
            ASSERT_UCS_OK_OR_INPROGRESS(uct_ep_put_zcopy(ep, sendbuf.iov(), 1,
                                                         hdr.address, rkey.rkey,
                                                         NULL));
            e->flush();
            send_op(fd, MSG_PUT_DONE, round);

            recv_op(fd, MSG_GET_READY, round);
            ASSERT_UCS_OK_OR_INPROGRESS(uct_ep_get_zcopy(ep, recvbuf.iov(), 1,
                                                         hdr.address, rkey.rkey,
                                                         NULL));
            e->flush();
            recvbuf.pattern_check(get_seed(round));
        }

        ASSERT_UCS_OK(uct_rkey_release(GetParam()->component, &rkey));
        send_op(fd, MSG_DONE, NUM_ROUNDS);
    }

    void test_put_get(bool other_device = false)
    {
        const char *peer_fd = getenv(PEER_FD_ENV);

        if (peer_fd == NULL) {
            run_exporter(other_device);
        } else {
            run_importer(atoi(peer_fd));
        }
    }
};

const char *test_ze_ipc_rma_xproc::PEER_FD_ENV = "UCT_ZE_IPC_TEST_PEER_FD";


UCS_TEST_P(test_ze_ipc_rma_xproc, put_get)
{
    test_put_get();
}

UCS_TEST_P(test_ze_ipc_rma_xproc, put_get_no_cache, "ZE_IPC_ENABLE_CACHE=n")
{
    test_put_get();
}

UCS_TEST_P(test_ze_ipc_rma_xproc, put_get_after_other_device_access)
{
    if (uct_ze_base_get_num_devices() < 2) {
        UCS_TEST_SKIP_R("needs two Level Zero devices");
    }

    test_put_get(true);
}

_UCT_INSTANTIATE_TEST_CASE(test_ze_ipc_rma_xproc, ze_ipc)
