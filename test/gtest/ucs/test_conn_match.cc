/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <common/test.h>
extern "C" {
#include <ucs/datastruct/conn_match.h>
}

#include <vector>

class test_conn_match : public ucs::test {
public:
    typedef struct {
        ucs_conn_match_t      conn_match;
        bool                  is_exp;
        ucs_conn_match_addr_t dest_addr;
        ucs_conn_sn_t         conn_sn;
    } conn_elem_t;

private:
    conn_elem_t *retrieve_common(ucs_conn_match_t *conn_match,
                                 const ucs_conn_match_addr_t &dest_addr,
                                 ucs_conn_sn_t conn_sn, bool is_exp) {
        if (conn_match == NULL) {
            return NULL;
        }

        conn_elem_t *conn_elem = ucs_container_of(conn_match, conn_elem_t,
                                                  conn_match);
        EXPECT_EQ(is_exp,           conn_elem->is_exp);
        EXPECT_EQ(conn_sn,          conn_elem->conn_sn);
        EXPECT_EQ(dest_addr.length, conn_elem->dest_addr.length);
        EXPECT_TRUE(!memcmp(dest_addr.addr, conn_elem->dest_addr.addr,
                            dest_addr.length));
        return conn_elem;
    }

    void insert_common(const ucs_conn_match_addr_t &dest_addr,
                       ucs_conn_sn_t conn_sn, bool is_exp,
                       conn_elem_t &elem) {
        elem.is_exp           = is_exp;
        elem.conn_sn          = conn_sn;
        elem.dest_addr.length = dest_addr.length;
        memcpy(elem.dest_addr.addr, dest_addr.addr, dest_addr.length);
    }

protected:
    static void get_addr(const ucs_conn_match_t *conn_match,
                         ucs_conn_match_addr_t *addr_p) {
        *addr_p = ucs_container_of(conn_match, conn_elem_t,
                                   conn_match)->dest_addr;
    }

    static ucs_conn_sn_t get_conn_sn(const ucs_conn_match_t *conn_match) {
        return ucs_container_of(conn_match, conn_elem_t,
                                conn_match)->conn_sn;
    }

    static const char *addr_str(const ucs_conn_match_addr_t *addr,
                                char *str, size_t max_size) {
        return ucs_strncpy_safe(str, static_cast<const char*>(addr->addr),
                                ucs_min(addr->length, max_size));
    }

    test_conn_match() {
        ucs_conn_match_ops_t conn_match_ops;

        conn_match_ops.get_addr    = get_addr;
        conn_match_ops.get_conn_sn = get_conn_sn;
        conn_match_ops.addr_str    = addr_str;

        ucs_conn_match_init(&m_conn_match_ctx, &conn_match_ops);
    }

    ~test_conn_match() {
        ucs_conn_match_cleanup(&m_conn_match_ctx);
    }

    void insert_exp(const ucs_conn_match_addr_t &dest_addr,
                    ucs_conn_sn_t conn_sn, conn_elem_t &elem) {
        ucs_conn_match_insert(&m_conn_match_ctx, &dest_addr,
                              conn_sn, &elem.conn_match,
                              UCS_CONN_MATCH_QUEUE_EXP);
        insert_common(dest_addr, conn_sn, true, elem);
    }

    void insert_unexp(const ucs_conn_match_addr_t &dest_addr,
                      ucs_conn_sn_t conn_sn, conn_elem_t &elem) {
        ucs_conn_match_insert(&m_conn_match_ctx, &dest_addr,
                              conn_sn, &elem.conn_match,
                              UCS_CONN_MATCH_QUEUE_UNEXP);
        insert_common(dest_addr, conn_sn, false, elem);
    }

    conn_elem_t *retrieve_exp(const ucs_conn_match_addr_t &dest_addr,
                              ucs_conn_sn_t conn_sn) {
        ucs_conn_match_t *conn_match =
            ucs_conn_match_retrieve(&m_conn_match_ctx, &dest_addr,
                                    conn_sn, UCS_CONN_MATCH_QUEUE_EXP);
        return retrieve_common(conn_match, dest_addr, conn_sn, true);
    }

    conn_elem_t *retrieve_unexp(const ucs_conn_match_addr_t &dest_addr,
                                ucs_conn_sn_t conn_sn) {
        ucs_conn_match_t *conn_match =
            ucs_conn_match_retrieve(&m_conn_match_ctx, &dest_addr,
                                    conn_sn, UCS_CONN_MATCH_QUEUE_UNEXP);
        return retrieve_common(conn_match, dest_addr, conn_sn, false);
    }

    void remove_conn(const ucs_conn_match_addr_t &dest_addr,
                     conn_elem_t &elem) {
        ucs_conn_match_remove_conn(&m_conn_match_ctx,
                                   &elem.conn_match,
                                   elem.is_exp ?
                                   UCS_CONN_MATCH_QUEUE_EXP :
                                   UCS_CONN_MATCH_QUEUE_UNEXP);
    }

    ucs_conn_sn_t get_next_sn(const ucs_conn_match_addr_t &dest_addr) {
        return ucs_conn_match_get_next_sn(&m_conn_match_ctx, &dest_addr);
    }

private:
    ucs_conn_match_ctx_t    m_conn_match_ctx;
};


UCS_TEST_F(test_conn_match, random_insert_retrieve) {
    const uint64_t max_uuids      = 128;
    const ucs_conn_sn_t max_conns = 128;
    const ucs_conn_sn_t min_conns = 1;
    std::vector<std::vector<conn_elem_t> > elems(max_uuids);

    for (uint64_t id = 0; id < max_uuids; id++) {
        ucs_conn_match_addr_t dest_addr;

        dest_addr.addr   = &id;
        dest_addr.length = sizeof(id);

        ucs_conn_sn_t num_conns =
            ucs::rand() % (max_conns + 1 - min_conns) + min_conns;
        elems[id].resize(num_conns);

        for (ucs_conn_sn_t conn = 0; conn < elems[id].size(); conn++) {
            conn_elem_t *conn_elem = &elems[id][conn];
            EXPECT_EQ(conn, get_next_sn(dest_addr));

            conn_elem->dest_addr.addr   = new char[dest_addr.length];
            conn_elem->dest_addr.length = dest_addr.length;

            if (ucs::rand() & 1) {
                insert_exp(dest_addr, conn, *conn_elem);
            } else {
                insert_unexp(dest_addr, conn, *conn_elem);
            }

            EXPECT_EQ(conn, conn_elem->conn_sn);
        }
    }

    for (uint64_t id = 0; id < max_uuids; id++) {
        ucs_conn_match_addr_t dest_addr;

        dest_addr.addr   = &id;
        dest_addr.length = sizeof(id);

        for (ucs_conn_sn_t conn = 0; conn < elems[id].size(); conn++) {
            conn_elem_t *conn_elem = &elems[id][conn];
            conn_elem_t *test_conn_elem;

            if (conn_elem->is_exp) {
                /* must not find this element in the unexpected queue */
                test_conn_elem = retrieve_unexp(dest_addr, conn_elem->conn_sn);
                EXPECT_EQ(NULL, test_conn_elem);

                test_conn_elem = retrieve_exp(dest_addr, conn_elem->conn_sn);
                EXPECT_EQ(conn_elem, test_conn_elem);

                /* subsequent retrieving of the same connection elemnt must return NULL */
                test_conn_elem = retrieve_exp(dest_addr, conn_elem->conn_sn);
                EXPECT_EQ(NULL, test_conn_elem);

                insert_unexp(dest_addr, conn_elem->conn_sn, *conn_elem);
            } else {
                /* must not find this element in the expected queue */
                test_conn_elem = retrieve_exp(dest_addr, conn_elem->conn_sn);
                EXPECT_EQ(NULL, test_conn_elem);

                test_conn_elem = retrieve_unexp(dest_addr, conn_elem->conn_sn);
                EXPECT_EQ(conn_elem, test_conn_elem);

                /* subsequent retrieving of the same connection elemnt must return NULL */
                test_conn_elem = retrieve_exp(dest_addr, conn_elem->conn_sn);
                EXPECT_EQ(NULL, test_conn_elem);

                insert_exp(dest_addr, conn_elem->conn_sn, *conn_elem);
            }
        }
    }

    for (uint64_t id = 0; id < max_uuids; id++) {
        ucs_conn_match_addr_t dest_addr;

        dest_addr.addr   = &id;
        dest_addr.length = sizeof(id);

        for (unsigned conn = 0; conn < elems[id].size(); conn++) {
            conn_elem_t *conn_elem = &elems[id][conn];

            EXPECT_EQ(conn, conn_elem->conn_sn);

            remove_conn(dest_addr, *conn_elem);

            delete[] (char*)conn_elem->dest_addr.addr;
        }
    }
}
