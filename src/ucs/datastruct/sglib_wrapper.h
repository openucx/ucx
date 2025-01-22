/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2001-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
*
* See file LICENSE for terms.
*/

#ifndef UCS_SGLIB_WRAPPER_H
#define UCS_SGLIB_WRAPPER_H

#include "sglib.h"

/*
 * Fix "unused variable"
 */
#undef SGLIB_LIST_LEN
#define SGLIB_LIST_LEN(type, list, next, result) {\
  (result) = 0;\
  SGLIB_LIST_MAP_ON_ELEMENTS(type, list, _ce_, next, (result)++);\
}

#endif
