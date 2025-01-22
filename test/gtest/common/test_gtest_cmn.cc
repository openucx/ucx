/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See file LICENSE for terms.
 */

#include "test.h"
#include "test_helpers.h"


class gtest_common : public ucs::test {
};


UCS_TEST_F(gtest_common, auto_ptr) {
    ucs::auto_ptr<int> p(new int);
}

