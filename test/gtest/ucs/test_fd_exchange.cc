/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <common/test.h>

extern "C" {
#include <ucs/sys/fd_exchange.h>
#include <ucs/time/time.h>
}

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <set>
#include <thread>
#include <vector>


class test_fd_exchange : public ucs::test {
protected:
    void init() override
    {
        char dir[] = "/tmp/ucx_fd_exchange_XXXXXX";

        ucs::test::init();
        ASSERT_TRUE(mkdtemp(dir) != NULL) << strerror(errno);
        m_dir = dir;
    }

    void cleanup() override
    {
        /* The server removes everything it created in the directory */
        m_server.reset();
        EXPECT_EQ(0, rmdir(m_dir.c_str())) << strerror(errno);
        ucs::test::cleanup();
    }

    void create_server()
    {
        UCS_TEST_CREATE_HANDLE(ucs_fd_exchange_server_t*, m_server,
                               ucs_fd_exchange_server_destroy,
                               ucs_fd_exchange_server_create, m_dir.c_str());
    }

    const char *server_path() const
    {
        return ucs_fd_exchange_server_path(m_server);
    }

    static ucs_status_t import(const char *path, uint64_t key, int *fd_p,
                               double timeout_sec = 10.0)
    {
        return ucs_fd_exchange_import(path, key, ucs_time_from_sec(timeout_sec),
                                      fd_p);
    }

    static void init_sockaddr(struct sockaddr_un &addr, const std::string &path)
    {
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    }

    /* Send a datagram with optional descriptors attached */
    static void send_msg(int sock_fd, const struct sockaddr_un &dest,
                         const void *data, size_t length,
                         const std::vector<int> &fds)
    {
        std::vector<char> cbuf(CMSG_SPACE(sizeof(int) * (fds.size() + 1)));
        struct msghdr msgh = {};
        struct cmsghdr *cmsg;
        struct iovec iov;

        iov.iov_base     = const_cast<void*>(data);
        iov.iov_len      = length;
        msgh.msg_name    = const_cast<struct sockaddr_un*>(&dest);
        msgh.msg_namelen = sizeof(dest);
        msgh.msg_iov     = &iov;
        msgh.msg_iovlen  = 1;
        if (!fds.empty()) {
            msgh.msg_control    = cbuf.data();
            msgh.msg_controllen = CMSG_SPACE(sizeof(int) * fds.size());
            cmsg                = CMSG_FIRSTHDR(&msgh);
            cmsg->cmsg_level    = SOL_SOCKET;
            cmsg->cmsg_type     = SCM_RIGHTS;
            cmsg->cmsg_len      = CMSG_LEN(sizeof(int) * fds.size());
            memcpy(CMSG_DATA(cmsg), fds.data(), sizeof(int) * fds.size());
        }

        EXPECT_EQ((ssize_t)length, sendmsg(sock_fd, &msgh, 0))
                << strerror(errno);
    }

    /* A server directory, as a fake server or a stopped one leaves it */
    class server_dir {
    public:
        server_dir(const std::string &dir_path) : m_path(dir_path)
        {
            EXPECT_EQ(0, mkdir(dir_path.c_str(), S_IRWXU)) << strerror(errno);
        }

        ~server_dir()
        {
            EXPECT_EQ(0, rmdir(m_path.c_str())) << strerror(errno);
        }

        const std::string &path() const
        {
            return m_path;
        }

    private:
        std::string m_path;
    };

    /* A datagram socket bound to a path, which is removed with it. Reading
     * from it waits for a bounded time, so that a missing datagram fails
     * instead of hanging. */
    class bound_socket {
    public:
        bound_socket(const std::string &sock_path) : m_fd(-1), m_path(sock_path)
        {
            struct timeval tv = {10, 0};
            struct sockaddr_un addr;
            int sock_fd;

            init_sockaddr(addr, sock_path);
            sock_fd = socket(AF_UNIX, SOCK_DGRAM, 0);
            if (sock_fd < 0) {
                ADD_FAILURE() << "socket() failed: " << strerror(errno);
                return;
            }

            EXPECT_EQ(0, bind(sock_fd, (const struct sockaddr*)&addr,
                              sizeof(addr)))
                    << sock_path << ": " << strerror(errno);
            EXPECT_EQ(0, setsockopt(sock_fd, SOL_SOCKET, SO_RCVTIMEO, &tv,
                                    sizeof(tv)));
            m_fd = sock_fd;
        }

        ~bound_socket()
        {
            if (m_fd >= 0) {
                close(m_fd);
                unlink(m_path.c_str());
            }
        }

        int fd() const
        {
            return m_fd;
        }

        const std::string &path() const
        {
            return m_path;
        }

    private:
        int         m_fd;
        std::string m_path;
    };

    /* A datagram socket bound next to the server socket, like the reply
     * socket of an importer, which sends raw requests and reads replies */
    class raw_client {
    public:
        raw_client(const std::string &server_path, unsigned index) :
            m_sock(server_path.substr(0, server_path.rfind('/')) + "/rtest" +
                   std::to_string(index))
        {
            init_sockaddr(m_server_addr, server_path);
        }

        void send(const void *data, size_t length,
                  const std::vector<int> &fds = std::vector<int>())
        {
            send_msg(m_sock.fd(), m_server_addr, data, length, fds);
        }

        /* The reply byte, or -1 when there is none */
        int recv_reply(int flags = 0)
        {
            uint8_t reply;

            return (recv(m_sock.fd(), &reply, sizeof(reply), flags) ==
                    sizeof(reply)) ? reply : -1;
        }

    private:
        bound_socket       m_sock;
        struct sockaddr_un m_server_addr;
    };

    /* Answer one request on a bound server socket with a raw reply, like a
     * server of another version or a broken one would */
    static void fake_server(int sock_fd, const void *reply, size_t length,
                            const std::vector<int> &fds)
    {
        struct sockaddr_un client_addr;
        ucs_fd_exchange_request_t req;
        struct msghdr msgh = {};
        struct iovec iov;

        iov.iov_base     = &req;
        iov.iov_len      = sizeof(req);
        msgh.msg_name    = &client_addr;
        msgh.msg_namelen = sizeof(client_addr);
        msgh.msg_iov     = &iov;
        msgh.msg_iovlen  = 1;
        ASSERT_EQ((ssize_t)sizeof(req), recvmsg(sock_fd, &msgh, 0))
                << strerror(errno);
        EXPECT_EQ(UCS_FD_EXCHANGE_MAGIC, req.magic);
        EXPECT_EQ(UCS_FD_EXCHANGE_VERSION, req.version);
        EXPECT_EQ(uint64_t(KEY), req.key);

        send_msg(sock_fd, client_addr, reply, length, fds);
    }

    /* Import from a fake server which sends the given reply. Its socket is
     * bound before importing, so that the request finds it. */
    ucs_status_t import_fake_reply(const void *reply, size_t length,
                                   const std::vector<int> &fds, int *fd_p)
    {
        server_dir dir(m_dir + "/ucx-fake");
        bound_socket sock(dir.path() + "/s");
        ucs_status_t status;

        std::thread server(fake_server, sock.fd(), reply, length, fds);
        status = import(sock.path().c_str(), KEY, fd_p);
        server.join();
        return status;
    }

    static constexpr uint64_t KEY = 0x1234;

    std::string                           m_dir;
    ucs::handle<ucs_fd_exchange_server_t*> m_server;
};

UCS_TEST_F(test_fd_exchange, import) {
    int pipe_fds[2];
    char data;
    int fd;

    ASSERT_EQ(0, pipe(pipe_fds));
    create_server();
    EXPECT_EQ(0, strncmp(server_path(), m_dir.c_str(), m_dir.length()));

    ASSERT_UCS_OK(ucs_fd_exchange_server_add_fd(m_server, KEY, pipe_fds[0]));
    {
        scoped_log_handler wrap_err(wrap_errors_logger);
        EXPECT_EQ(UCS_ERR_ALREADY_EXISTS,
                  ucs_fd_exchange_server_add_fd(m_server, KEY, pipe_fds[1]));
    }

    /* The imported descriptor is a close-on-exec copy of the read end */
    ASSERT_UCS_OK(import(server_path(), KEY, &fd));
    EXPECT_NE(pipe_fds[0], fd);
    EXPECT_TRUE(fcntl(fd, F_GETFD) & FD_CLOEXEC);
    data = 'a';
    ASSERT_EQ(1, write(pipe_fds[1], &data, 1));
    data = 0;
    EXPECT_EQ(1, read(fd, &data, 1));
    EXPECT_EQ('a', data);
    close(fd);

    EXPECT_EQ(UCS_ERR_NO_ELEM, import(server_path(), KEY + 1, &fd));

    ucs_fd_exchange_server_remove_fd(m_server, KEY);
    EXPECT_EQ(UCS_ERR_NO_ELEM, import(server_path(), KEY, &fd));

    close(pipe_fds[0]);
    close(pipe_fds[1]);
}

UCS_TEST_F(test_fd_exchange, other_process) {
    int pipe_fds[2];
    int child_status;
    char data = 0;
    pid_t pid;
    int fd;

    ASSERT_EQ(0, pipe(pipe_fds));
    create_server();
    ASSERT_UCS_OK(ucs_fd_exchange_server_add_fd(m_server, KEY, pipe_fds[1]));

    /* The child imports the write end and signals through it */
    pid = fork();
    ASSERT_NE(-1, pid);
    if (pid == 0) {
        if (import(server_path(), KEY, &fd) != UCS_OK) {
            _exit(1);
        }

        data = 'b';
        _exit((write(fd, &data, 1) == 1) ? 0 : 2);
    }

    /* Wait first, so a child which could not import does not hang the read */
    ASSERT_EQ(pid, waitpid(pid, &child_status, 0));
    ASSERT_TRUE(WIFEXITED(child_status) && (WEXITSTATUS(child_status) == 0))
            << ucs::exit_status_info(child_status);
    EXPECT_EQ(1, read(pipe_fds[0], &data, 1));
    EXPECT_EQ('b', data);

    ucs_fd_exchange_server_remove_fd(m_server, KEY);
    close(pipe_fds[0]);
    close(pipe_fds[1]);
}

UCS_TEST_F(test_fd_exchange, no_server) {
    std::string dir = m_dir + "/ucx-00000000";
    std::string path = dir + "/s";
    int fd;

    /* Neither the server directory nor the socket exist */
    EXPECT_EQ(UCS_ERR_UNREACHABLE, import(path.c_str(), KEY, &fd));

    /* The server socket was removed, as when the process exited */
    ASSERT_EQ(0, mkdir(dir.c_str(), S_IRWXU));
    EXPECT_EQ(UCS_ERR_UNREACHABLE, import(path.c_str(), KEY, &fd));
    EXPECT_EQ(0, rmdir(dir.c_str()));
}

UCS_TEST_F(test_fd_exchange, timeout) {
    /* A socket which never answers, like a stopped server process */
    server_dir dir(m_dir + "/ucx-00000000");
    bound_socket sock(dir.path() + "/s");
    int fd;

    EXPECT_EQ(UCS_ERR_TIMED_OUT, import(sock.path().c_str(), KEY, &fd, 0.1));
}

UCS_TEST_F(test_fd_exchange, bad_requests) {
    ucs_fd_exchange_request_t req = {UCS_FD_EXCHANGE_MAGIC,
                                     UCS_FD_EXCHANGE_VERSION, KEY};
    uint64_t oversized[3]         = {0, 0, 0};
    const uint64_t zeros[2]       = {0, 0};
    std::set<int> open_fds;
    int pipe_fds[2];
    int fd;

    ASSERT_EQ(0, pipe(pipe_fds));
    create_server();
    ASSERT_UCS_OK(ucs_fd_exchange_server_add_fd(m_server, KEY, pipe_fds[0]));
    open_fds = ucs::get_open_fds();
    memcpy(oversized, &req, sizeof(req));

    {
        raw_client too_short(server_path(), 0);
        raw_client wrong_magic(server_path(), 1);
        raw_client with_fds(server_path(), 2);
        raw_client too_long(server_path(), 3);
        raw_client other_version(server_path(), 4);

        /* Requests which are not of this protocol get no reply at all */
        too_short.send(zeros, 3);
        wrong_magic.send(zeros, sizeof(zeros));
        with_fds.send(zeros, sizeof(zeros),
                      std::vector<int>(pipe_fds, pipe_fds + 2));

        /* Requests of another version of this protocol are told so */
        too_long.send(oversized, sizeof(oversized));
        req.version = UCS_FD_EXCHANGE_VERSION + 1;
        other_version.send(&req, sizeof(req));

        /* The server handles the datagrams in order, so a reply to a request
         * sent after them shows that it dropped or answered them already. The
         * key is a missing one, since the server closes its copy of a found
         * descriptor only after replying, which would race with the check of
         * the open descriptors below. */
        EXPECT_EQ(UCS_ERR_NO_ELEM, import(server_path(), KEY + 1, &fd));

        EXPECT_EQ(-1, too_short.recv_reply(MSG_DONTWAIT));
        EXPECT_EQ(-1, wrong_magic.recv_reply(MSG_DONTWAIT));
        EXPECT_EQ(-1, with_fds.recv_reply(MSG_DONTWAIT));
        EXPECT_EQ(UCS_FD_EXCHANGE_REPLY_UNSUPPORTED, too_long.recv_reply());
        EXPECT_EQ(UCS_FD_EXCHANGE_REPLY_UNSUPPORTED,
                  other_version.recv_reply());
    }

    /* The descriptors carried by the request were not left open */
    EXPECT_EQ(open_fds, ucs::get_open_fds());

    ucs_fd_exchange_server_remove_fd(m_server, KEY);
    close(pipe_fds[0]);
    close(pipe_fds[1]);
}

UCS_TEST_F(test_fd_exchange, bad_replies) {
    const uint8_t found[2]    = {UCS_FD_EXCHANGE_REPLY_FOUND, 0};
    const uint8_t unsupported = UCS_FD_EXCHANGE_REPLY_UNSUPPORTED;
    const uint8_t unknown     = 0xff;
    std::vector<int> fds;
    int pipe_fds[2];
    int fd;

    ASSERT_EQ(0, pipe(pipe_fds));
    fds.push_back(pipe_fds[0]);

    /* A reply which is too long is rejected, and the descriptor it carried is
     * closed, which the leak check of the test framework verifies */
    EXPECT_EQ(UCS_ERR_IO_ERROR,
              import_fake_reply(found, sizeof(found), fds, &fd));

    /* Found, but without a descriptor */
    EXPECT_EQ(UCS_ERR_IO_ERROR,
              import_fake_reply(found, 1, std::vector<int>(), &fd));

    EXPECT_EQ(UCS_ERR_IO_ERROR,
              import_fake_reply(&unknown, sizeof(unknown), fds, &fd));

    /* A server of another version */
    EXPECT_EQ(UCS_ERR_UNSUPPORTED,
              import_fake_reply(&unsupported, sizeof(unsupported),
                                std::vector<int>(), &fd));

    close(pipe_fds[0]);
    close(pipe_fds[1]);
}

UCS_TEST_F(test_fd_exchange, concurrent_add_remove) {
    static const unsigned num_imports = 200;
    std::atomic<bool> stop(false);
    unsigned num_found = 0;
    int pipe_fds[2];
    int fd;

    ASSERT_EQ(0, pipe(pipe_fds));
    create_server();

    /* Add and remove the key while importing it: an import either finds the
     * key and gets a usable descriptor, or does not find it */
    std::thread adder([&]() {
        while (!stop) {
            ucs_fd_exchange_server_add_fd(m_server, KEY, pipe_fds[0]);
            ucs_fd_exchange_server_remove_fd(m_server, KEY);
        }
    });

    for (unsigned i = 0; i < num_imports; ++i) {
        ucs_status_t status = import(server_path(), KEY, &fd);
        if (status == UCS_OK) {
            EXPECT_NE(-1, fcntl(fd, F_GETFD));
            close(fd);
            ++num_found;
        } else {
            EXPECT_EQ(UCS_ERR_NO_ELEM, status);
        }
    }

    stop = true;
    adder.join();
    UCS_TEST_MESSAGE << num_found << "/" << num_imports
                     << " imports found the key";

    close(pipe_fds[0]);
    close(pipe_fds[1]);
}

UCS_TEST_F(test_fd_exchange, invalid_path) {
    scoped_log_handler wrap_err(wrap_errors_logger);
    std::string long_path = "/" + std::string(100, 'x') + "/s";
    int fd;

    EXPECT_EQ(UCS_ERR_INVALID_PARAM, import("relative/dir/s", KEY, &fd));
    EXPECT_EQ(UCS_ERR_EXCEEDS_LIMIT, import(long_path.c_str(), KEY, &fd));
}

UCS_TEST_F(test_fd_exchange, invalid_dir) {
    scoped_log_handler wrap_err(wrap_errors_logger);
    std::string long_dir(200, 'x');
    ucs_fd_exchange_server_t *server;

    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              ucs_fd_exchange_server_create("relative/dir", &server));
    EXPECT_EQ(UCS_ERR_EXCEEDS_LIMIT,
              ucs_fd_exchange_server_create(("/" + long_dir).c_str(),
                                            &server));
    EXPECT_EQ(UCS_ERR_IO_ERROR,
              ucs_fd_exchange_server_create((m_dir + "/missing").c_str(),
                                            &server));
}
