/**
* Copyright (c) Google, LLC, 2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/sys/netlink.h>
#include <ucs/sys/sock.h>
}

#include <arpa/inet.h>
#include <linux/fib_rules.h>
#include <linux/rtnetlink.h>
#include <vector>

class test_netlink : public ucs::test {
protected:
    virtual void cleanup()
    {
        ucs_netlink_reset_routing_table_cache();
        ucs::test::cleanup();
    }

    static void append_rta(std::vector<uint8_t> &buf, unsigned short type,
                           const void *data, size_t payload_len)
    {
        size_t rta_len = RTA_LENGTH(payload_len), offset = buf.size();
        struct rtattr rt = {static_cast<unsigned short>(rta_len), type};

        buf.resize(offset + RTA_ALIGN(rta_len), 0);
        memcpy(&buf[offset], &rt, sizeof(rt));
        memcpy(&buf[offset + RTA_LENGTH(0)], data, payload_len);
    }

    static void append_nlmsg(std::vector<uint8_t> &stream, unsigned short type,
                             const void *hdr = NULL, size_t hdr_size = 0,
                             const std::vector<uint8_t> &attrs = {})
    {
        size_t msg_len = NLMSG_LENGTH(NLMSG_ALIGN(hdr_size) + attrs.size());
        size_t offset  = stream.size();
        struct nlmsghdr nlh = {static_cast<uint32_t>(msg_len), type,
                               NLM_F_MULTI, 0, 0};

        stream.resize(offset + NLMSG_ALIGN(msg_len), 0);
        memcpy(&stream[offset], &nlh, sizeof(nlh));
        if (hdr != NULL) {
            memcpy(&stream[offset + NLMSG_HDRLEN], hdr, hdr_size);
        }
        if (!attrs.empty()) {
            memcpy(&stream[offset + NLMSG_HDRLEN + NLMSG_ALIGN(hdr_size)],
                   &attrs[0], attrs.size());
        }
    }

    static void append_rule_msg(std::vector<uint8_t> &stream, uint32_t table_id,
                                uint32_t fwmark = 0, uint32_t fwmask = 0,
                                uint32_t flags = 0)
    {
        struct fib_rule_hdr frh = {};
        std::vector<uint8_t> attrs;

        frh.action = FR_ACT_TO_TBL;
        frh.flags  = flags;
        frh.table  = (table_id < 256) ? table_id : RT_TABLE_UNSPEC;
        if (table_id >= 256) {
            append_rta(attrs, FRA_TABLE, &table_id, sizeof(table_id));
        }
        if (fwmark != 0) {
            append_rta(attrs, FRA_FWMARK, &fwmark, sizeof(fwmark));
        }
        if (fwmask != 0) {
            append_rta(attrs, FRA_FWMASK, &fwmask, sizeof(fwmask));
        }
        append_nlmsg(stream, RTM_NEWRULE, &frh, sizeof(frh), attrs);
    }

    static void append_route_msg(std::vector<uint8_t> &stream, uint32_t table,
                                 int if_idx, const char *dst_ip,
                                 uint8_t rtm_type = RTN_UNICAST)
    {
        struct rtmsg rtm = {};
        std::vector<uint8_t> attrs;
        uint8_t dst_addr[sizeof(struct in6_addr)];

        rtm.rtm_family  = (strchr(dst_ip, ':') != NULL) ? AF_INET6 : AF_INET;
        rtm.rtm_dst_len = (rtm.rtm_family == AF_INET6) ? 128 : 32;
        rtm.rtm_type    = rtm_type;
        rtm.rtm_table   = (table < 256) ? table : RT_TABLE_UNSPEC;
        ASSERT_EQ(1, inet_pton(rtm.rtm_family, dst_ip, dst_addr));
        if (table >= 256) {
            append_rta(attrs, RTA_TABLE, &table, sizeof(table));
        }
        append_rta(attrs, RTA_OIF, &if_idx, sizeof(if_idx));
        append_rta(attrs, RTA_DST, dst_addr, rtm.rtm_dst_len / 8);
        append_nlmsg(stream, RTM_NEWROUTE, &rtm, sizeof(rtm), attrs);
    }

    static int check_route(int if_idx, const char *ip_str, int *mask_p = NULL)
    {
        struct sockaddr_storage ss = {};
        ucs_assert_always(ucs_sock_ipstr_to_sockaddr(ip_str, &ss) == UCS_OK);
        return ucs_netlink_route_exists(if_idx, (struct sockaddr*)&ss, mask_p);
    }

    static int get_local_ndev(const char *ip_str)
    {
        struct sockaddr_storage ss = {};
        ucs_assert_always(ucs_sock_ipstr_to_sockaddr(ip_str, &ss) == UCS_OK);
        return ucs_netlink_get_local_route_ndev_index((struct sockaddr*)&ss);
    }
};

UCS_TEST_F(test_netlink, parse_msg_and_send_request) {
    auto cb = [](const struct nlmsghdr*, void *arg) -> ucs_status_t {
        return (++(*static_cast<int*>(arg)) == 0) ? UCS_OK : UCS_INPROGRESS;
    };
    struct nlmsgerr err     = {-EAFNOSUPPORT};
    struct fib_rule_hdr frh = {};
    int count = 0, msg_done = 0;
    std::vector<uint8_t> stream, err_buf;
    ucs_netlink_rt_parse_ctx_t ctx;
    ucs_status_t status;

    append_rule_msg(stream, RT_TABLE_LOCAL);
    append_rule_msg(stream, RT_TABLE_MAIN);
    ASSERT_UCS_OK(ucs_netlink_parse_msg(&stream[0], stream.size(), cb, &count,
                                        UCS_LOG_LEVEL_ERROR, &msg_done));
    EXPECT_EQ(2, count);
    EXPECT_EQ(0, msg_done);
    count           = -1;
    size_t done_off = stream.size();
    append_nlmsg(stream, NLMSG_DONE);
    ASSERT_UCS_OK(ucs_netlink_parse_msg(&stream[0], stream.size(), cb, &count,
                                        UCS_LOG_LEVEL_ERROR, &msg_done));
    EXPECT_EQ(0, count);
    EXPECT_EQ(1, msg_done);

    for (size_t off : {size_t(0), done_off}) {
        count             = 0;
        auto *nlh         = reinterpret_cast<struct nlmsghdr*>(&stream[off]);
        nlh->nlmsg_flags |= NLM_F_DUMP_INTR;
        EXPECT_EQ(UCS_ERR_BUSY,
                  ucs_netlink_parse_msg(&stream[0], stream.size(), cb, &count,
                                        UCS_LOG_LEVEL_DEBUG, &msg_done));
        EXPECT_EQ(2, count);
        EXPECT_EQ(1, msg_done);
        nlh->nlmsg_flags &= ~NLM_F_DUMP_INTR;
    }

    append_nlmsg(err_buf, NLMSG_ERROR, &err, sizeof(err));
    EXPECT_EQ(UCS_ERR_IO_ERROR,
              ucs_netlink_parse_msg(&err_buf[0], err_buf.size(), cb, &count,
                                    UCS_LOG_LEVEL_DEBUG, &msg_done));
    EXPECT_EQ(0, msg_done);
    reinterpret_cast<struct nlmsghdr*>(&err_buf[0])->nlmsg_type = NLMSG_DONE;
    scoped_log_handler slh(wrap_errors_logger);
    EXPECT_EQ(UCS_ERR_IO_ERROR,
              ucs_netlink_parse_msg(&err_buf[0], err_buf.size(), cb, &count,
                                    UCS_LOG_LEVEL_ERROR, &msg_done));
    EXPECT_EQ(1, msg_done);
    EXPECT_EQ(1u, m_errors.size());
    kh_init_inplace(ucs_netlink_table_flags, &ctx);
    frh.family = AF_INET;
    status = ucs_netlink_send_request(NETLINK_ROUTE, RTM_GETRULE, NLM_F_DUMP,
                                      &frh, sizeof(frh),
                                      ucs_netlink_parse_rule_entry_cb, &ctx,
                                      UCS_LOG_LEVEL_DEBUG);
    EXPECT_TRUE((status == UCS_OK) || (status == UCS_ERR_IO_ERROR) ||
                (status == UCS_ERR_BUSY));
    EXPECT_EQ(UCS_ERR_IO_ERROR,
              ucs_netlink_send_request(NETLINK_ROUTE, RTM_GETRULE, 0, &frh,
                                       sizeof(frh), cb, &count,
                                       UCS_LOG_LEVEL_DEBUG));
    kh_destroy_inplace(ucs_netlink_table_flags, &ctx);
}

UCS_TEST_F(test_netlink, fwmark_route_filtering) {
    struct route_case {
        uint32_t   table_id;
        const char *dst_ip;
        bool       expect_cached;
    };
    const int if_idx         = INT_MAX;
    const route_case cases[] = {{RT_TABLE_MAIN, "198.18.0.1", true},
                                {200, "198.18.0.2", true},
                                {2004, "198.18.0.3", false},
                                {100, "198.18.0.4", true},
                                {210, "198.18.0.5", true},
                                {1001, "198.18.0.6", true},
                                {300, "198.18.0.7", true},
                                {301, "198.18.0.8", true},
                                {RT_TABLE_MAIN, "2001:db8::1", true},
                                {2005, "2001:db8::2", false}};
    std::vector<uint8_t> rules, rts, fbs;
    ucs_netlink_rt_parse_ctx_t ctx;
    int netmask_len, msg_done;

    scoped_log_handler slh(hide_errors_logger);
    EXPECT_EQ(0, check_route(if_idx, "198.18.0.254"));
    kh_init_inplace(ucs_netlink_table_flags, &ctx);
    append_rule_msg(rules, RT_TABLE_LOCAL);
    append_rule_msg(rules, RT_TABLE_MAIN);
    append_rule_msg(rules, 200);
    append_rule_msg(rules, 2004, 0x200, 0xf00);
    append_rule_msg(rules, 100, 0x100, 0x00f);
    append_rule_msg(rules, 210, 0x200, 0xf00);
    append_rule_msg(rules, 210);
    append_rule_msg(rules, 300, 0x200, 0xf00, FIB_RULE_INVERT);
    append_rule_msg(rules, 301, 0x0, 0xf00, FIB_RULE_INVERT);
    append_rule_msg(rules, 2005, 0x200);
    append_nlmsg(rules, NLMSG_DONE);
    ASSERT_UCS_OK(ucs_netlink_parse_msg(&rules[0], rules.size(),
                                        ucs_netlink_parse_rule_entry_cb, &ctx,
                                        UCS_LOG_LEVEL_DEBUG, &msg_done));
    EXPECT_EQ(1, msg_done);
    for (const auto &tc : cases) {
        append_route_msg(rts, tc.table_id, if_idx, tc.dst_ip);
    }
    append_route_msg(rts, RT_TABLE_LOCAL, if_idx, "198.18.0.10", RTN_LOCAL);
    append_route_msg(rts, 2004, if_idx, "198.18.0.11", RTN_LOCAL);
    append_nlmsg(rts, NLMSG_DONE);
    ASSERT_UCS_OK(ucs_netlink_parse_msg(&rts[0], rts.size(),
                                        ucs_netlink_parse_rt_entry_cb, &ctx,
                                        UCS_LOG_LEVEL_ERROR, &msg_done));
    EXPECT_EQ(1, msg_done);
    for (const auto &tc : cases) {
        int expected_len = (strchr(tc.dst_ip, ':') != NULL) ? 128 : 32;
        EXPECT_EQ(tc.expect_cached ? 1 : 0,
                  check_route(if_idx, tc.dst_ip, &netmask_len));
        EXPECT_EQ(tc.expect_cached ? expected_len : -1, netmask_len);
    }
    EXPECT_EQ(if_idx, get_local_ndev("198.18.0.10"));
    EXPECT_EQ(-1, get_local_ndev("198.18.0.11"));
    kh_clear(ucs_netlink_table_flags, &ctx);
    append_route_msg(fbs, 2004, if_idx, "198.18.0.9");
    append_nlmsg(fbs, NLMSG_DONE);
    ASSERT_UCS_OK(ucs_netlink_parse_msg(&fbs[0], fbs.size(),
                                        ucs_netlink_parse_rt_entry_cb, &ctx,
                                        UCS_LOG_LEVEL_ERROR, &msg_done));
    EXPECT_NE(0, check_route(if_idx, "198.18.0.9"));
    kh_destroy_inplace(ucs_netlink_table_flags, &ctx);
    ucs_netlink_reset_routing_table_cache();
    EXPECT_EQ(0, check_route(if_idx, "198.18.0.1"));
}
