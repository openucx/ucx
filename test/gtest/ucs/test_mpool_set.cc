/**
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <common/test.h>
extern "C" {
#include <ucs/datastruct/mpool_set.h>
#include <ucs/datastruct/mpool_set.inl>
#include <ucs/datastruct/mpool.inl>
}

class test_mpool_set : public ucs::test {
protected:
    ucs_status_t create_mpool_set(ucs_mpool_set_t *mp_set, size_t *sizes,
                                  unsigned sizes_count,
                                  size_t max_size, size_t priv_size,
                                  size_t priv_elem_size, const char *name)
    {
        static ucs_mpool_ops_t ops = {
           ucs_mpool_chunk_malloc,
           ucs_mpool_chunk_free,
           NULL,
           NULL,
           NULL
        };

        return ucs_mpool_set_init(mp_set, sizes, sizes_count, max_size,
                                  priv_size, priv_elem_size, 0,
                                  UCS_SYS_CACHE_LINE_SIZE, 4, UINT_MAX, &ops,
                                  name);
    }
};

UCS_TEST_F(test_mpool_set, get)
{
    uint32_t sizes_map    = ucs::rand();
    unsigned num_sizes    = ucs_popcount(sizes_map);
    size_t *sizes         = (size_t*)ucs_alloca(num_sizes * sizeof(*sizes));
    size_t max_size       = ucs::rand() % UCS_BIT(18) + 1;
    std::string sizes_str = "Sizes: ";

    uint32_t bit;
    int i = 0;
    ucs_for_each_bit(bit, sizes_map) {
        sizes[i]   = UCS_BIT(bit);
        sizes_str += ucs::to_string(sizes[i]) + ", ";
        ++i;
    }

    sizes_str += "max_size " + ucs::to_string(max_size);

    UCS_TEST_MESSAGE << sizes_str;

    ucs_mpool_set_t mp_set;
    ASSERT_UCS_OK(create_mpool_set(&mp_set, sizes, num_sizes, max_size, 0, 0,
                                   "mpool_set"));

    const int max_iters = ucs_max(1, 7 / ucs::test_time_multiplier());
    ucs_mpool_t *mpools = reinterpret_cast<ucs_mpool_t*>(mp_set.data);
    for (int iters = 1; iters < max_iters; ++iters) {
        size_t size = ucs::rand() % max_size % (size_t)pow(10, iters);
        void *elem  = ucs_mpool_set_get_inline(&mp_set, size);

        for (int i = 0; i < ucs_popcount(mp_set.bitmap); ++i) {
            ucs_mpool_t *mpool = &mpools[i];
            size_t elem_size   = mpool->data->elem_size -
                                 sizeof(ucs_mpool_elem_t);

            if (elem_size >= (size + 1)) {
                EXPECT_EQ(ucs_mpool_obj_owner(elem), mpool);
                break;
            }
            EXPECT_NE(ucs_mpool_obj_owner(elem), mpool);
        }

        memset(elem, 0, size); // make sure the given memory is accessible

        ucs_mpool_set_put_inline(elem);
    }

    ucs_mpool_set_cleanup(&mp_set, 0);
}

UCS_TEST_F(test_mpool_set, name)
{
    size_t sizes[]        = {1, 8 * UCS_KBYTE};
    std::string test_name = "test mpool_set_name";
    ucs_mpool_set_t mp_set;

    ASSERT_UCS_OK(create_mpool_set(&mp_set, sizes, ucs_static_array_size(sizes),
                                   16 * UCS_KBYTE, 0, 0, test_name.c_str()));
    EXPECT_EQ(test_name, ucs_mpool_set_name(&mp_set));

    ucs_mpool_set_cleanup(&mp_set, 0);
}
