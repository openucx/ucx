/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
extern "C" {
#include <ucs/sys/netlink.h>
}

#include <linux/rtnetlink.h>
#include <sys/socket.h>
#include <unistd.h>

#include <vector>


/*
 * Feed hand-made netlink responses to ucs_netlink_recv_response() through a
 * datagram socket pair, so the tests do not depend on how the running kernel
 * splits a dump into datagrams.
 */
class test_netlink : public ucs::test {
protected:
    typedef std::vector<char> datagram_t;

    /* Sequence number of the datagram that follows the tested response */
    static const uint32_t NEXT_RESPONSE_SEQ = 1000;

    virtual void init()
    {
        ucs::test::init();
        ASSERT_EQ(0, socketpair(AF_UNIX, SOCK_DGRAM, 0, m_fds));
    }

    virtual void cleanup()
    {
        close(m_fds[0]);
        close(m_fds[1]);
        ucs::test::cleanup();
    }

    static void add_msg(datagram_t &datagram, uint16_t type, uint32_t seq = 1)
    {
        size_t payload_len = (type == NLMSG_ERROR) ? sizeof(struct nlmsgerr) :
                                                     sizeof(struct rtmsg);
        struct {
            struct nlmsghdr nlh;
            union {
                struct rtmsg    rtm;
                struct nlmsgerr err;
            };
        } msg = {};

        msg.nlh.nlmsg_len   = NLMSG_LENGTH(payload_len);
        msg.nlh.nlmsg_type  = type;
        msg.nlh.nlmsg_flags = NLM_F_MULTI;
        msg.nlh.nlmsg_seq   = seq;
        if (type == NLMSG_ERROR) {
            msg.err.error = -EINVAL;
        }

        datagram.insert(datagram.end(), (const char*)&msg,
                        (const char*)&msg + NLMSG_SPACE(payload_len));
    }

    static datagram_t routes(unsigned count)
    {
        datagram_t datagram;

        for (unsigned i = 0; i < count; ++i) {
            add_msg(datagram, RTM_NEWROUTE);
        }

        return datagram;
    }

    void send_datagram(const datagram_t &datagram)
    {
        ASSERT_EQ((ssize_t)datagram.size(),
                  send(m_fds[1], datagram.data(), datagram.size(), 0));
    }

    static ucs_status_t count_routes_cb(const struct nlmsghdr *nlh, void *arg)
    {
        EXPECT_EQ(RTM_NEWROUTE, nlh->nlmsg_type);
        ++*(unsigned*)arg;
        return UCS_INPROGRESS;
    }

    static ucs_status_t
    parse_one_route_cb(const struct nlmsghdr *nlh, void *arg)
    {
        count_routes_cb(nlh, arg);
        return UCS_OK;
    }

    /*
     * Receive a response, then check that the receiver stopped at its end.
     * A datagram of the next response is queued behind the tested one, so a
     * receiver that misses the end of the response consumes it instead of
     * blocking forever.
     */
    ucs_status_t
    recv_response(unsigned short nlmsg_flags, unsigned *num_routes_p,
                  ucs_netlink_parse_cb_t parse_cb = count_routes_cb)
    {
        datagram_t next_response;
        struct nlmsghdr nlh = {};
        ucs_status_t status;

        add_msg(next_response, NLMSG_DONE, NEXT_RESPONSE_SEQ);
        send_datagram(next_response);

        *num_routes_p = 0;
        status = ucs_netlink_recv_response(m_fds[0], nlmsg_flags, parse_cb,
                                           num_routes_p);

        EXPECT_EQ((ssize_t)sizeof(nlh),
                  recv(m_fds[0], &nlh, sizeof(nlh), MSG_DONTWAIT))
                << "the next response was consumed";
        EXPECT_EQ(NEXT_RESPONSE_SEQ, nlh.nlmsg_seq);
        return status;
    }

    int m_fds[2];
};

const uint32_t test_netlink::NEXT_RESPONSE_SEQ;

UCS_TEST_F(test_netlink, done_in_separate_datagram) {
    datagram_t done;
    unsigned num_routes;

    add_msg(done, NLMSG_DONE);
    send_datagram(routes(3));
    send_datagram(done);

    EXPECT_UCS_OK(recv_response(NLM_F_DUMP, &num_routes));
    EXPECT_EQ(3u, num_routes);
}

UCS_TEST_F(test_netlink, done_after_last_route) {
    datagram_t last = routes(2);
    unsigned num_routes;

    /* Some kernels send NLMSG_DONE in the same datagram as the last routes */
    add_msg(last, NLMSG_DONE);
    send_datagram(routes(3));
    send_datagram(last);

    EXPECT_UCS_OK(recv_response(NLM_F_DUMP, &num_routes));
    EXPECT_EQ(5u, num_routes);
}

UCS_TEST_F(test_netlink, done_after_parse_complete) {
    datagram_t last = routes(2);
    unsigned num_routes;

    /* The dump ends at NLMSG_DONE even if the callback finished parsing at
     * an earlier message of the same datagram */
    add_msg(last, NLMSG_DONE);
    send_datagram(last);

    EXPECT_UCS_OK(recv_response(NLM_F_DUMP, &num_routes, parse_one_route_cb));
    EXPECT_EQ(1u, num_routes);
}

UCS_TEST_F(test_netlink, error_ends_dump) {
    datagram_t error;
    unsigned num_routes;

    add_msg(error, NLMSG_ERROR);
    send_datagram(routes(3));
    send_datagram(error);

    {
        scoped_log_handler slh(hide_errors_logger);
        EXPECT_EQ(UCS_ERR_IO_ERROR, recv_response(NLM_F_DUMP, &num_routes));
    }
    EXPECT_EQ(3u, num_routes);
}

UCS_TEST_F(test_netlink, non_dump_reads_one_datagram) {
    unsigned num_routes;

    /* The reply to a non-dump request is a single datagram, without
     * NLMSG_DONE */
    send_datagram(routes(1));

    EXPECT_UCS_OK(recv_response(0, &num_routes));
    EXPECT_EQ(1u, num_routes);
}
