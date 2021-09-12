/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <common/test.h>

#include <ucs/datastruct/conn_match.h>
#include <ucs/sys/sys.h>

#include <vector>

class test_conn_match : public ucs::test {
public:
    typedef struct {
        ucs_conn_match_elem_t       elem;
        ucs_conn_match_queue_type_t queue_type;
        const void                  *dest_address;
        ucs_conn_sn_t               conn_sn;
    } conn_elem_t;

    test_conn_match() {
        m_address_length = 0;
        m_added_elems    = 0;
        m_removed_elems  = 0;
        m_purged_elems   = 0;
    }

private:
    void conn_match_init(size_t address_length) {
        ucs_conn_match_ops_t conn_match_ops;

        conn_match_ops.get_address = get_address;
        conn_match_ops.get_conn_sn = get_conn_sn;
        conn_match_ops.address_str = address_str;
        conn_match_ops.purge_cb    = purge_cb;

        ucs_conn_match_init(&m_conn_match_ctx, address_length,
                            UCS_CONN_MATCH_SN_MAX, &conn_match_ops);
        m_address_length = address_length;
    }

    void check_conn_elem(const conn_elem_t *conn_elem,
                         const void *dest_address, ucs_conn_sn_t conn_sn) {
        EXPECT_EQ(conn_sn, conn_elem->conn_sn);
        EXPECT_TRUE(!memcmp(dest_address, conn_elem->dest_address,
                            m_address_length));
    }

protected:
    static inline conn_elem_t*
    conn_elem_from_match_elem(const ucs_conn_match_elem_t *conn_match) {
        return ucs_container_of(conn_match, conn_elem_t, elem);
    }

    static const void *get_address(const ucs_conn_match_elem_t *conn_match) {
        return conn_elem_from_match_elem(conn_match)->dest_address;
    }

    static ucs_conn_sn_t get_conn_sn(const ucs_conn_match_elem_t *conn_match) {
        return conn_elem_from_match_elem(conn_match)->conn_sn;
    }

    static const char *address_str(const ucs_conn_match_ctx_t *conn_match_ctx,
                                   const void *address, char *str,
                                   size_t max_size) {
        EXPECT_EQ(&m_conn_match_ctx, conn_match_ctx);
        return ucs_strncpy_safe(str, static_cast<const char*>(address),
                                ucs_min(m_conn_match_ctx.address_length,
                                        max_size));
    }

    static void purge_cb(ucs_conn_match_ctx_t *conn_match_ctx,
                         ucs_conn_match_elem_t *conn_match) {
        EXPECT_EQ(&m_conn_match_ctx, conn_match_ctx);
        m_purged_elems++;
        delete conn_elem_from_match_elem(conn_match);
    }

    void init_new_address_length(size_t address_length) {
        ucs_conn_match_cleanup(&m_conn_match_ctx);
        conn_match_init(address_length);
    }

    void *alloc_address(size_t idx, size_t address_length) {
        void *address       = new uint8_t[address_length];
        std::string idx_str = ucs::to_string(idx);

        memcpy(address, idx_str.c_str(), idx_str.size());
        memset(UCS_PTR_BYTE_OFFSET(address, idx_str.size()), 'x',
               address_length - idx_str.size());

        return address;
    }

    void init() {
        conn_match_init(m_default_address_length);
    }

    void cleanup() {
        ucs_conn_match_cleanup(&m_conn_match_ctx);
        EXPECT_EQ(m_added_elems - m_removed_elems, m_purged_elems);
    }

    void insert(const void *dest_address, ucs_conn_sn_t conn_sn,
                ucs_conn_match_queue_type_t queue_type, conn_elem_t &elem) {
        ucs_conn_match_insert(&m_conn_match_ctx, dest_address, conn_sn,
                              &elem.elem, queue_type);
        elem.queue_type = queue_type;
        elem.conn_sn    = conn_sn;
        m_added_elems++;
    }

    conn_elem_t *retrieve(const void *dest_address, ucs_conn_sn_t conn_sn,
                          ucs_conn_match_queue_type_t queue_type) {
        ucs_conn_match_elem_t *conn_match =
            ucs_conn_match_get_elem(&m_conn_match_ctx, dest_address,
                                    conn_sn, queue_type, 1);
        if (conn_match == NULL) {
            return NULL;
        }

        conn_elem_t *conn_elem = conn_elem_from_match_elem(conn_match);
        EXPECT_EQ(queue_type, conn_elem->queue_type);
        check_conn_elem(conn_elem, dest_address, conn_sn);
        m_removed_elems++;

        return conn_elem;
    }

    conn_elem_t *lookup(const void *dest_address, ucs_conn_sn_t conn_sn,
                        ucs_conn_match_queue_type_t queue_type) {
        ucs_conn_match_elem_t *conn_match =
            ucs_conn_match_get_elem(&m_conn_match_ctx, dest_address,
                                    conn_sn, queue_type, 0);
        if (conn_match == NULL) {
            return NULL;
        }

        ucs_conn_match_elem_t *test_conn_match =
            ucs_conn_match_get_elem(&m_conn_match_ctx, dest_address,
                                    conn_sn, UCS_CONN_MATCH_QUEUE_ANY, 0);
        EXPECT_EQ(conn_match, test_conn_match);

        conn_elem_t *conn_elem = conn_elem_from_match_elem(conn_match);
        check_conn_elem(conn_elem, dest_address, conn_sn);

        return conn_elem;
    }

    void remove_conn(conn_elem_t &elem) {
        ucs_conn_match_remove_elem(&m_conn_match_ctx, &elem.elem,
                                   elem.queue_type);
        m_removed_elems++;
    }

    ucs_conn_sn_t get_next_sn(const void *dest_address) {
        return ucs_conn_match_get_next_sn(&m_conn_match_ctx,
                                          dest_address);
    }

private:
    static ucs_conn_match_ctx_t m_conn_match_ctx;
    size_t                      m_address_length;
    static const size_t         m_default_address_length;
    size_t                      m_added_elems;
    size_t                      m_removed_elems;
    static size_t               m_purged_elems;
};


const size_t test_conn_match::m_default_address_length = 64;
size_t test_conn_match::m_purged_elems                 = 0;
ucs_conn_match_ctx_t test_conn_match::m_conn_match_ctx = {};


UCS_TEST_F(test_conn_match, random_insert_retrieve) {
    const size_t max_conn_iters     =
            ucs_max(1, ucs_min(5, 128 / ucs::test_time_multiplier()));;       
    const size_t max_addresses      = max_conn_iters;
    const ucs_conn_sn_t max_conns   = max_conn_iters;
    const size_t min_address_length = ucs::to_string(max_addresses).size();
    const size_t max_address_length =
            ucs_max(min_address_length, 2048 / ucs::test_time_multiplier());
    const size_t max_iters          = 4;

    for (size_t it = 0; it < max_iters; it++) {
        size_t address_length = ucs::rand() %
                                (max_address_length - min_address_length + 1) +
                                min_address_length;
        std::vector<std::vector<conn_elem_t> > conn_elems(max_addresses);

        init_new_address_length(address_length);
        UCS_TEST_MESSAGE << "address length: " << address_length;

        for (size_t i = 0; i < max_addresses; i++) {
            ucs_conn_sn_t num_conns = (ucs::rand() % max_conns) + 1;
            void *dest_address      = alloc_address(i, address_length);

            conn_elems[i].resize(num_conns);

            for (ucs_conn_sn_t conn = 0; conn < num_conns; conn++) {
                conn_elem_t *conn_elem = &conn_elems[i][conn];
                EXPECT_EQ(conn, get_next_sn(dest_address));

                conn_elem->dest_address = dest_address;

                ucs_conn_match_queue_type_t queue_type = (ucs::rand() & 1) ?
                                                         UCS_CONN_MATCH_QUEUE_EXP :
                                                         UCS_CONN_MATCH_QUEUE_UNEXP;
                insert(dest_address, conn, queue_type, *conn_elem);
                EXPECT_EQ(queue_type, conn_elem->queue_type);
                EXPECT_EQ(conn, conn_elem->conn_sn);
            }
        }

        for (size_t i = 0; i < max_addresses; i++) {
            for (ucs_conn_sn_t conn = 0; conn < conn_elems[i].size(); conn++) {
                conn_elem_t *conn_elem                         = &conn_elems[i][conn];
                ucs_conn_match_queue_type_t another_queue_type =
                    (conn_elem->queue_type == UCS_CONN_MATCH_QUEUE_EXP) ?
                    UCS_CONN_MATCH_QUEUE_UNEXP : UCS_CONN_MATCH_QUEUE_EXP;
                conn_elem_t *test_conn_elem;

                /* must not find this element in the another queue */
                test_conn_elem = lookup(conn_elem->dest_address, conn_elem->conn_sn,
                                        another_queue_type);
                EXPECT_EQ(NULL, test_conn_elem);
                test_conn_elem = retrieve(conn_elem->dest_address, conn_elem->conn_sn,
                                          another_queue_type);
                EXPECT_EQ(NULL, test_conn_elem);

                test_conn_elem = lookup(conn_elem->dest_address, conn_elem->conn_sn,
                                        conn_elem->queue_type);
                EXPECT_EQ(conn_elem, test_conn_elem);
                test_conn_elem = retrieve(conn_elem->dest_address, conn_elem->conn_sn,
                                          conn_elem->queue_type);
                EXPECT_EQ(conn_elem, test_conn_elem);

                /* subsequent retrieving/lookup of the same connection element
                 * must return NULL */
                test_conn_elem = lookup(conn_elem->dest_address, conn_elem->conn_sn,
                                        conn_elem->queue_type);
                EXPECT_EQ(NULL, test_conn_elem);
                test_conn_elem = retrieve(conn_elem->dest_address, conn_elem->conn_sn,
                                          conn_elem->queue_type);
                EXPECT_EQ(NULL, test_conn_elem);

                insert(conn_elem->dest_address, conn_elem->conn_sn, another_queue_type,
                       *conn_elem);
            }
        }

        for (size_t i = 0; i < max_addresses; i++) {
            for (unsigned conn = 0; conn < conn_elems[i].size(); conn++) {
                conn_elem_t *conn_elem = &conn_elems[i][conn];
                conn_elem_t *test_conn_elem;

                test_conn_elem = lookup(conn_elem->dest_address, conn_elem->conn_sn,
                                        conn_elem->queue_type);
                EXPECT_EQ(conn_elem, test_conn_elem);

                EXPECT_EQ(conn, conn_elem->conn_sn);

                remove_conn(*conn_elem);

                /* subsequent retrieving/lookup of the same connection element
                 * must return NULL */
                test_conn_elem = lookup(conn_elem->dest_address, conn_elem->conn_sn,
                                        UCS_CONN_MATCH_QUEUE_EXP);
                EXPECT_EQ(NULL, test_conn_elem);
                test_conn_elem = lookup(conn_elem->dest_address, conn_elem->conn_sn,
                                        UCS_CONN_MATCH_QUEUE_UNEXP);
                EXPECT_EQ(NULL, test_conn_elem);
                test_conn_elem = retrieve(conn_elem->dest_address, conn_elem->conn_sn,
                                          UCS_CONN_MATCH_QUEUE_EXP);
                EXPECT_EQ(NULL, test_conn_elem);
                test_conn_elem = retrieve(conn_elem->dest_address, conn_elem->conn_sn,
                                          UCS_CONN_MATCH_QUEUE_UNEXP);
                EXPECT_EQ(NULL, test_conn_elem);
            }

            delete[] (uint8_t*)conn_elems[i][0].dest_address;
        }
    }
}

UCS_TEST_F(test_conn_match, purge_elems) {
    const size_t        max_addresses  = 128;
    const ucs_conn_sn_t max_conns      = 128;
    const size_t        address_length = 8;
    std::vector<std::vector<conn_elem_t*> > conn_elems(max_addresses);

    init_new_address_length(address_length);

    for (size_t i = 0; i < max_addresses; i++) {
        ucs_conn_sn_t num_conns = (ucs::rand() % max_conns) + 1;
        void *dest_address      = alloc_address(i, address_length);

        conn_elems[i].resize(num_conns);

        for (ucs_conn_sn_t conn = 0; conn < num_conns; conn++) {
            conn_elems[i][conn] = new conn_elem_t;

            conn_elem_t *conn_elem = conn_elems[i][conn];
            EXPECT_EQ(conn, get_next_sn(dest_address));

            conn_elem->dest_address = dest_address;

            ucs_conn_match_queue_type_t queue_type = (ucs::rand() & 1) ?
                                                     UCS_CONN_MATCH_QUEUE_EXP :
                                                     UCS_CONN_MATCH_QUEUE_UNEXP;
            insert(dest_address, conn, queue_type, *conn_elem);
            EXPECT_EQ(queue_type, conn_elem->queue_type);
            EXPECT_EQ(conn, conn_elem->conn_sn);
        }
    }

    /* remove some elements */
    for (size_t i = 0; i < max_addresses; i++) {
        const void *dest_address = conn_elems[i][0]->dest_address;

        for (ucs_conn_sn_t conn = 0; conn < conn_elems[i].size(); conn++) {
            conn_elem_t *conn_elem = conn_elems[i][conn];

            if (ucs::rand() & 1) {
                conn_elem_t *test_conn_elem = retrieve(conn_elem->dest_address,
                                                       conn_elem->conn_sn,
                                                       conn_elem->queue_type);
                EXPECT_EQ(conn_elem, test_conn_elem);
                delete test_conn_elem;
            } else {
                /* the elements that will be purged don't need the destination
                 * address anymore, so, the address will be deleted below */
                conn_elem->dest_address = NULL;
            }
        }

        delete[] (uint8_t*)dest_address;
    }
}
