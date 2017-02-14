/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/type/class.h>
}

class test_class : public ucs::test {
};


typedef struct base {
    int            field1;
} base_t;
UCS_CLASS_DECLARE(base_t, int);

typedef struct derived {
    base_t         super;
    int            field2;
} derived_t;
UCS_CLASS_DECLARE(derived_t, int, int);

typedef struct derived2 {
    base_t         super;
    int            field2;
} derived2_t;
UCS_CLASS_DECLARE(derived2_t, int, int);

static int base_init_count = 0;
static int derived_init_count = 0;


/* Base impl */

UCS_CLASS_INIT_FUNC(base_t, int param)
{
    if (param < 0) {
        return UCS_ERR_INVALID_PARAM;
    }
    self->field1 = param;
    ++base_init_count;
    return UCS_OK;
}

UCS_CLASS_CLEANUP_FUNC(base_t)
{
    --base_init_count;
}

UCS_CLASS_DEFINE(base_t, void);

/* Derived impl */

UCS_CLASS_INIT_FUNC(derived_t, int param1, int param2)
{
    UCS_CLASS_CALL_SUPER_INIT(base_t, param1);

    if (param2 < 0) {
        return UCS_ERR_INVALID_PARAM;
    }
    self->field2 = param2;
    ++derived_init_count;
    return UCS_OK;
}

UCS_CLASS_CLEANUP_FUNC(derived_t)
{
    --derived_init_count;
}

UCS_CLASS_DEFINE(derived_t, base_t);

UCS_CLASS_DEFINE_NEW_FUNC(derived_t, derived_t, int, int);
UCS_CLASS_DEFINE_DELETE_FUNC(derived_t, derived_t);


/* Derived2 impl */

UCS_CLASS_INIT_FUNC(derived2_t, int param1, int param2)
{
    if (param2 < 0) {
        return UCS_ERR_INVALID_PARAM;
    }

    UCS_CLASS_CALL_SUPER_INIT(base_t, param1);

    self->field2 = param2;
    ++derived_init_count;
    return UCS_OK;
}

UCS_CLASS_CLEANUP_FUNC(derived2_t)
{
    --derived_init_count;
}

UCS_CLASS_DEFINE(derived2_t, base_t);


UCS_TEST_F(test_class, basic) {
    derived_t *derived;
    ucs_status_t status;

    ASSERT_EQ(0, base_init_count);
    ASSERT_EQ(0, derived_init_count);

    status = UCS_CLASS_NEW(derived_t, &derived, 1, 2);
    ASSERT_UCS_OK(status);

    /* coverity[uninit_use] */
    EXPECT_EQ(2, derived->field2);
    EXPECT_EQ(1, derived->super.field1);

    EXPECT_EQ(1, base_init_count);
    EXPECT_EQ(1, derived_init_count);

    UCS_CLASS_DELETE(derived_t, derived);

    EXPECT_EQ(0, base_init_count);
    EXPECT_EQ(0, derived_init_count);
}

UCS_TEST_F(test_class, create_destroy) {
    derived_t *derived;
    ucs_status_t status;

    ASSERT_EQ(0, base_init_count);
    ASSERT_EQ(0, derived_init_count);

    status = UCS_CLASS_NEW_FUNC_NAME(derived_t)(1, 2, &derived);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(2, derived->field2);
    EXPECT_EQ(1, derived->super.field1);

    EXPECT_EQ(1, base_init_count);
    EXPECT_EQ(1, derived_init_count);

    UCS_CLASS_DELETE_FUNC_NAME(derived_t)(derived);

    EXPECT_EQ(0, base_init_count);
    EXPECT_EQ(0, derived_init_count);
}

UCS_TEST_F(test_class, failure) {
    derived_t *derived;
    ucs_status_t status;

    ASSERT_EQ(0, base_init_count);
    ASSERT_EQ(0, derived_init_count);

    /* Should fail on base */
    derived = NULL;
    status = UCS_CLASS_NEW(derived_t, &derived, -1, 2);
    /* coverity[leaked_storage] */
    ASSERT_EQ(UCS_ERR_INVALID_PARAM, status);
    ASSERT_TRUE(NULL == derived);

    /* Should be properly cleaned up */
    EXPECT_EQ(0, base_init_count);
    EXPECT_EQ(0, derived_init_count);

    /* Should fail on derived */
    derived = NULL;
    status = UCS_CLASS_NEW(derived_t, &derived, 1, -2);
    /* coverity[leaked_storage] */
    ASSERT_EQ(UCS_ERR_INVALID_PARAM, status);
    ASSERT_TRUE(NULL == derived);

    /* Should be properly cleaned up */
    EXPECT_EQ(0, base_init_count);
    EXPECT_EQ(0, derived_init_count);
}

UCS_TEST_F(test_class, failure2) {
    derived2_t *derived;
    ucs_status_t status;

    ASSERT_EQ(0, base_init_count);
    ASSERT_EQ(0, derived_init_count);

    /* Should fail on base */
    derived = NULL;
    status = UCS_CLASS_NEW(derived2_t, &derived, -1, 2);
    /* coverity[leaked_storage] */
    ASSERT_EQ(UCS_ERR_INVALID_PARAM, status);
    ASSERT_TRUE(NULL == derived);

    /* Should be properly cleaned up */
    EXPECT_EQ(0, base_init_count);
    EXPECT_EQ(0, derived_init_count);

    /* Should fail on derived */
    derived = NULL;
    status = UCS_CLASS_NEW(derived2_t, &derived, 1, -2);
    /* coverity[leaked_storage] */
    ASSERT_EQ(UCS_ERR_INVALID_PARAM, status);
    ASSERT_TRUE(NULL == derived);

    /* Should be properly cleaned up */
    EXPECT_EQ(0, base_init_count);
    EXPECT_EQ(0, derived_init_count);
}
