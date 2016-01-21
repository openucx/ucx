/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/debug/memtrack.h>
#include <ucs/type/component.h>

}


class test_component : public ucs::test {
};


/******* Base type *********/

typedef struct test_base {
    unsigned init_count;
    unsigned cleanup_count;
} test_base_t;


/******* First component *********/

typedef struct test_comp1_ctx {
    int initialized1;
} test_comp1_ctx_t;

ucs_status_t test_comp1_init(test_base_t *test_base)
{
    ++test_base->init_count;
    ucs_component_get(test_base, test1, test_comp1_ctx_t)->initialized1 = 1;
    return UCS_OK;
}

void test_comp1_cleanup(test_base_t *test_base)
{
    ++test_base->cleanup_count;
}

UCS_COMPONENT_DEFINE(test_base_t, test1, test_comp1_init, test_comp1_cleanup,
                     sizeof(test_comp1_ctx_t))


/******* Second component *********/

typedef struct test_comp2_ctx {
    int initialized2;
} test_comp2_ctx_t;

ucs_status_t test_comp2_init(test_base_t *test_base)
{
    ++test_base->init_count;
    ucs_component_get(test_base, test2, test_comp2_ctx_t)->initialized2 = 1;
    return UCS_OK;
}

void test_comp2_cleanup(test_base_t *test_base)
{
    ++test_base->cleanup_count;
}

UCS_COMPONENT_DEFINE(test_base_t, test2, test_comp2_init, test_comp2_cleanup,
                     sizeof(test_comp2_ctx_t))


/******* TEST *********/


UCS_TEST_F(test_component, init_cleanup) {
    ucs_status_t status;
    test_base_t *context;

    context = (test_base_t*)ucs_calloc(1, ucs_components_total_size(test_base_t),
                                       "test context");

    status = ucs_components_init_all(test_base_t, context);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(2u, context->init_count);
    EXPECT_EQ(0u, context->cleanup_count);

    EXPECT_TRUE(ucs_component_get(context, test1, test_comp1_ctx_t)->initialized1);
    EXPECT_TRUE(ucs_component_get(context, test2, test_comp2_ctx_t)->initialized2);

    ucs_components_cleanup_all(test_base_t, context);

    EXPECT_EQ(2u, context->cleanup_count);

    ucs_free(context);
}

UCS_COMPONENT_LIST_DEFINE(test_base_t);
