/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_PREPROCESSOR_FUNC_H
#define UCS_PREPROCESSOR_FUNC_H

#include <ucs/sys/preprocessor.h>


/*
 * Define argument list with given types.
 */
#define UCS_FUNC_DEFINE_ARGS(...) \
    UCS_PP_FOREACH_SEP(_UCS_FUNC_ARG_DEFINE, _, \
                       UCS_PP_ZIP((UCS_PP_SEQ(UCS_PP_NUM_ARGS(__VA_ARGS__))), \
                                  (__VA_ARGS__)))


/*
 * Pass auto-generated arguments to a function call.
 */
#define UCS_FUNC_PASS_ARGS(...) \
    UCS_PP_FOREACH_SEP(_UCS_FUNC_ARG_PASS, _, \
                       UCS_PP_SEQ(UCS_PP_NUM_ARGS(__VA_ARGS__)))


/*
 * Helpers
 */
#define _UCS_FUNC_ARG_DEFINE(_, _bundle) \
    __UCS_FUNC_ARG_DEFINE(_, UCS_PP_TUPLE_0 _bundle, UCS_PP_TUPLE_1 _bundle)
#define __UCS_FUNC_ARG_DEFINE(_, _index, _type) \
    _type UCS_PP_TOKENPASTE(arg, _index)
#define _UCS_FUNC_ARG_PASS(_, _index) UCS_PP_TOKENPASTE(arg, _index)

#endif
