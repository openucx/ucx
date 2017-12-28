/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <common/test.h>

extern "C" {
#include <ucp/core/ucp_ep.h>
#include <ucs/datastruct/khash.h>
#include <ucs/datastruct/sglib_wrapper.h>
#include <ucs/time/time.h>
#include <ucs/sys/sys.h>
}

#include <list>
#include <map>
#include <tr1/unordered_map>

typedef uint64_t key_type;
typedef ucp_ep_t hash_type;

#define MAX_COUNT 4194304
#define START_TIME ucs_time_t time = -ucs_get_time();
#define END_TIME (ucs_time_to_nsec(time + ucs_get_time()) / num_elem)
#define FIELD_ALIGN(_x_) ":" << std::right << std::setw(_x_)

static key_type keys[MAX_COUNT]; /* the keys array */

static void generate_keys()
{
    for (int i = 0; i < MAX_COUNT; ++i) {
        keys[i] = ucs_generate_uuid((key_type)keys);
    }
}

class perf_compare_base {
public:
    virtual ~perf_compare_base() {};

    ucs_time_t initialize(const size_t num_elem) {
        START_TIME
        init_storage();
        for (size_t i = 0; i < num_elem; ++i) {
            add_elem(keys[i]);
        }
        return (ucs_time_t)END_TIME;
    }

    ucs_time_t lookup(const size_t num_elem) const {
        START_TIME
        for (size_t i = 0; i < num_elem; ++i) {
            hash_type *ep = find_elem(keys[i]);
            EXPECT_TRUE(ep->dest_uuid == keys[i]);
        }
        return (ucs_time_t)END_TIME;
    }

    ucs_time_t cleanup(const size_t num_elem) {
        START_TIME
        del_elems();
        return (ucs_time_t)END_TIME;
    }

    virtual void        init_storage()                      = 0;
    virtual void        add_elem(const uint64_t key)        = 0;
    virtual hash_type  *find_elem(const uint64_t key) const = 0;
    virtual void        del_elems()                         = 0;
    virtual const char *get_name() const                    = 0;

    template <typename _type_to_alloc>
    _type_to_alloc *create_element(const uint64_t key) const {
        _type_to_alloc *ep = (_type_to_alloc *) ucs_malloc(sizeof(_type_to_alloc), get_name());
        EXPECT_TRUE(ep);
        ep->dest_uuid = key;
        return ep;
    }
};

class perf_compare_map : public perf_compare_base {
public:
    const char *get_name() const {
        return "map";
    }

    void init_storage() {}

    void add_elem(const uint64_t key) {
        obj[key] = create_element<hash_type>(key);
    }

    hash_type *find_elem(const uint64_t key) const {
        std::map<key_type, hash_type *>::const_iterator ep_found = obj.find(key);
        EXPECT_TRUE(ep_found != obj.end());
        return ep_found->second;
    }

    void del_elems() {
        for (std::map<key_type, hash_type *>::const_iterator it = obj.begin();
             it != obj.end(); ++it) {
            hash_type *ep = it->second;
            free(ep);
        }
        obj.clear();
    }

    std::map<key_type, hash_type *> obj;
};

class perf_compare_unordered_map : public perf_compare_base {
public:
    const char *get_name() const {
        return "unmap";
    }

    void init_storage() {}

    void add_elem(const uint64_t key) {
        obj[key] = create_element<hash_type>(key);
    }

    hash_type *find_elem(const uint64_t key) const {
        std::tr1::unordered_map<key_type, hash_type *>::const_iterator ep_found = obj.find(key);
        EXPECT_TRUE(ep_found != obj.end());
        return ep_found->second;
    }

    void del_elems() {
        for (std::tr1::unordered_map<key_type, hash_type *>::const_iterator it = obj.begin();
             it != obj.end(); ++it) {
            hash_type *ep = it->second;
            free(ep);
        }
        obj.clear();
    }

    std::tr1::unordered_map<key_type, hash_type *> obj;
};

class perf_compare_khash : public perf_compare_base {
public:
    const char *get_name() const {
        return "khash";
    }

    KHASH_MAP_INIT_INT64(khash_ep_hash, hash_type *);

    void init_storage() {
        kh_init_inplace(khash_ep_hash, &obj);
    }

    void add_elem(const uint64_t key) {
        int hash_extra_status = 0;

        khiter_t hash_it = kh_put(khash_ep_hash, &obj, key, &hash_extra_status);
        EXPECT_TRUE(hash_it != kh_end(&obj));
        kh_value(&obj, hash_it) = create_element<hash_type>(key);
    }

    hash_type *find_elem(const uint64_t key) const {
        khiter_t ep_found = kh_get(khash_ep_hash, &obj, key);
        EXPECT_TRUE(ep_found != kh_end(&obj));
        return kh_value(&obj, ep_found);
    }

    void del_elems() {
        for (khiter_t it = kh_begin(obj); it != kh_end(&obj); ++it) {
            if (!kh_exist(&obj, it)) {
                continue;
            }
            hash_type *ep = kh_value(&obj, it);
            free(ep);
        }
        kh_destroy_inplace(khash_ep_hash, &obj);
    }

    khash_t(khash_ep_hash) obj;
};

#define SGLIB_HASH_SIZE                 32767
#define test_sglib_compare(_ep1, _ep2)  ((int64_t)(_ep1)->dest_uuid - (int64_t)(_ep2)->dest_uuid)
#define test_sglib_hash(_ep)            ((_ep)->dest_uuid)

struct sglib_hash_type : public hash_type {
    sglib_hash_type *next;
};

SGLIB_DEFINE_LIST_PROTOTYPES(sglib_hash_type, test_sglib_compare, next);
SGLIB_DEFINE_LIST_FUNCTIONS(sglib_hash_type, test_sglib_compare, next);
SGLIB_DEFINE_HASHED_CONTAINER_PROTOTYPES(sglib_hash_type, SGLIB_HASH_SIZE, test_sglib_hash);
SGLIB_DEFINE_HASHED_CONTAINER_FUNCTIONS(sglib_hash_type, SGLIB_HASH_SIZE, test_sglib_hash);

class perf_compare_sglib : public perf_compare_base {
public:
    const char *get_name() const {
        return "sglib";
    }

    void init_storage() {
        obj = (sglib_hash_type **) ucs_malloc(sizeof(*obj) * SGLIB_HASH_SIZE, "sglib_hash");
        sglib_hashed_sglib_hash_type_init(obj);
    }

    void add_elem(const uint64_t key) {
        sglib_hash_type *ep = create_element<sglib_hash_type>(key);
        sglib_hashed_sglib_hash_type_add(obj, ep);
    }

    hash_type *find_elem(const uint64_t key) const {
        sglib_hash_type ep, *ep_found = 0;
        ep.dest_uuid = key;
        ep_found = sglib_hashed_sglib_hash_type_find_member(obj, &ep);
        EXPECT_TRUE(ep_found);
        return ep_found;
    }

    void del_elems() {
        struct sglib_hashed_sglib_hash_type_iterator it;
        for (sglib_hash_type *local_it = sglib_hashed_sglib_hash_type_it_init(&it, obj);
            local_it != NULL;
            local_it = sglib_hashed_sglib_hash_type_it_next(&it)) {
            sglib_hashed_sglib_hash_type_delete(obj, local_it);
            free(local_it);
        }
        free(obj);
    }

    sglib_hash_type **obj;
};

class test_hash_perf : public ucs::test {
protected:
    void check_lookup_perf(perf_compare_base* hash, size_t num_elems);
};

void test_hash_perf::check_lookup_perf(perf_compare_base* hash, size_t num_elems) {
    const ucs_time_t MAX_LOOKUP_NS_1024 = 400;
    for (int i = 0; i < (ucs::perf_retry_count + 1); ++i) {
        ucs_time_t lookup_ns = hash->lookup(num_elems);
        if (!ucs::perf_retry_count) {
            UCS_TEST_MESSAGE << "not validating performance";
            return; /* Skip */
        } else if (lookup_ns < MAX_LOOKUP_NS_1024) {
            return; /* Success */
        } else {
            ucs::safe_sleep(ucs::perf_retry_interval);
        }
    }
    ADD_FAILURE() << hash->get_name()  << " bad lookup performance";
}

UCS_TEST_F(test_hash_perf, perf_compare) {

    size_t trip_counts[] = {1, 2, 8, 128, 1024, 32768, 262144, 1048576, 0};

    if (ucs::test_time_multiplier() > 1) {
        UCS_TEST_SKIP_R("Long run expected. Skipped.");
    }
    perf_compare_base *perf_compare_khash_ptr = new perf_compare_khash;
    perf_compare_base *perf_compare_sglib_ptr = new perf_compare_sglib;
    perf_compare_base *hashes[] = {
        perf_compare_khash_ptr,
        perf_compare_sglib_ptr,
        new perf_compare_map,
        new perf_compare_unordered_map,
        NULL
    };

    generate_keys();

    UCS_TEST_MESSAGE << ":    elements   :init  :lookup:remove";
    for (int i = 0; hashes[i] != NULL; ++i) {
        perf_compare_base *cur_hash = hashes[i];
        for (int j = 0; trip_counts[j] > 0; ++j) {
            size_t num_elems = trip_counts[j];

            ucs_time_t insert_ns = cur_hash->initialize(num_elems);
            ucs_time_t lookup_ns = cur_hash->lookup(num_elems);

            if ((1024 == num_elems) &&
                ((cur_hash == perf_compare_khash_ptr) ||
                 (cur_hash == perf_compare_sglib_ptr)))
            {
                check_lookup_perf(cur_hash, num_elems);
            }

            ucs_time_t remove_ns = cur_hash->cleanup(num_elems);

            UCS_TEST_MESSAGE << FIELD_ALIGN(6) << cur_hash->get_name()
                             << FIELD_ALIGN(8) << num_elems
                             << FIELD_ALIGN(6) << insert_ns
                             << FIELD_ALIGN(6) << lookup_ns
                             << FIELD_ALIGN(6) << remove_ns;
        }
        delete cur_hash;
    }
}
