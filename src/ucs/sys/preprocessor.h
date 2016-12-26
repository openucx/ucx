/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2012.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_PREPROCESSOR_H
#define UCS_PREPROCESSOR_H

/* Convert token to string */
#define UCS_PP_QUOTE(x)                 # x

/* Paste two expanded tokens */
#define __UCS_TOKENPASTE_HELPER(x, y)   x ## y
#define UCS_PP_TOKENPASTE(x, y)         __UCS_TOKENPASTE_HELPER(x, y)

/* Unique value generator */
#ifdef __COUNTER__
#  define UCS_PP_UNIQUE_ID __COUNTER__
#else
#  define UCS_PP_UNIQUE_ID __LINE__
#endif

/* Creating unique identifiers, used for macros */
#define UCS_PP_APPEND_UNIQUE_ID(x)      UCS_PP_TOKENPASTE(x, UCS_PP_UNIQUE_ID)

/* Convert to string */
#define _UCS_PP_MAKE_STRING(x) #x
#define UCS_PP_MAKE_STRING(x) _UCS_PP_MAKE_STRING(x)

/*
 * Count number of macro arguments
 * e.g UCS_PP_NUM_ARGS(a,b) will expand to: 2
 */
#define UCS_PP_MAX_ARGS 20
#define _UCS_PP_NUM_ARGS(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,_17,_18,_19,_20,N,...) \
    N
#define UCS_PP_NUM_ARGS(...) \
    _UCS_PP_NUM_ARGS(, ## __VA_ARGS__,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)


/* Expand macro for each argument in the list
 * e.g
 * UCS_PP_FOREACH(macro, arg, a, b, c) will expand to: macro(arg, a) macro(arg, b) macro(arg, c)
 * UCS_PP_FOREACH_SEP(macro, arg, a, b, c) will expand to: macro(arg, a), macro(arg, b), macro(arg, c)
 * UCS_PP_ZIP((a, b, c), (1, 2, 3)) will expand to: (a, 1), (b, 2), (c, 3)
 */
#define UCS_PP_FOREACH(_macro, _arg, ...) \
    UCS_PP_TOKENPASTE(_UCS_PP_FOREACH_, UCS_PP_NUM_ARGS(__VA_ARGS__))(_macro, _arg, __VA_ARGS__)
#define UCS_PP_FOREACH_SEP(_macro, _arg, ...) \
    UCS_PP_TOKENPASTE(_UCS_PP_FOREACH_SEP_, UCS_PP_NUM_ARGS(__VA_ARGS__))(_macro, _arg, __VA_ARGS__)
#define UCS_PP_ZIP(_l1, _l2) \
    UCS_PP_TOKENPASTE(_UCS_PP_ZIP_, UCS_PP_NUM_ARGS _l1)(_l1, _l2)

#define _UCS_PP_FOREACH_0(_macro , _arg, ...)
#define _UCS_PP_FOREACH_1(_macro , _arg, _arg1, ...)  _macro(_arg, _arg1) _UCS_PP_FOREACH_0 (_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_2(_macro , _arg, _arg1, ...)  _macro(_arg, _arg1) _UCS_PP_FOREACH_1 (_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_3(_macro , _arg, _arg1, ...)  _macro(_arg, _arg1) _UCS_PP_FOREACH_2 (_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_4(_macro , _arg, _arg1, ...)  _macro(_arg, _arg1) _UCS_PP_FOREACH_3 (_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_5(_macro , _arg, _arg1, ...)  _macro(_arg, _arg1) _UCS_PP_FOREACH_4 (_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_6(_macro , _arg, _arg1, ...)  _macro(_arg, _arg1) _UCS_PP_FOREACH_5 (_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_7(_macro , _arg, _arg1, ...)  _macro(_arg, _arg1) _UCS_PP_FOREACH_6 (_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_8(_macro , _arg, _arg1, ...)  _macro(_arg, _arg1) _UCS_PP_FOREACH_7 (_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_9(_macro , _arg, _arg1, ...)  _macro(_arg, _arg1) _UCS_PP_FOREACH_8 (_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_10(_macro, _arg, _arg1, ...)  _macro(_arg, _arg1) _UCS_PP_FOREACH_9 (_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_11(_macro, _arg, _arg1, ...)  _macro(_arg, _arg1) _UCS_PP_FOREACH_10(_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_12(_macro, _arg, _arg1, ...)  _macro(_arg, _arg1) _UCS_PP_FOREACH_11(_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_13(_macro, _arg, _arg1, ...)  _macro(_arg, _arg1) _UCS_PP_FOREACH_12(_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_14(_macro, _arg, _arg1, ...)  _macro(_arg, _arg1) _UCS_PP_FOREACH_13(_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_15(_macro, _arg, _arg1, ...)  _macro(_arg, _arg1) _UCS_PP_FOREACH_14(_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_16(_macro, _arg, _arg1, ...)  _macro(_arg, _arg1) _UCS_PP_FOREACH_15(_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_17(_macro, _arg, _arg1, ...)  _macro(_arg, _arg1) _UCS_PP_FOREACH_16(_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_18(_macro, _arg, _arg1, ...)  _macro(_arg, _arg1) _UCS_PP_FOREACH_17(_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_19(_macro, _arg, _arg1, ...)  _macro(_arg, _arg1) _UCS_PP_FOREACH_18(_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_20(_macro, _arg, _arg1, ...)  _macro(_arg, _arg1) _UCS_PP_FOREACH_19(_macro, _arg, __VA_ARGS__)

#define _UCS_PP_FOREACH_SEP_0(_macro , _arg, _arg1, ...)
#define _UCS_PP_FOREACH_SEP_1(_macro , _arg, _arg1, ...)  _macro(_arg, _arg1)
#define _UCS_PP_FOREACH_SEP_2(_macro , _arg, _arg1, ...)  _macro(_arg, _arg1), _UCS_PP_FOREACH_SEP_1 (_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_SEP_3(_macro , _arg, _arg1, ...)  _macro(_arg, _arg1), _UCS_PP_FOREACH_SEP_2 (_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_SEP_4(_macro , _arg, _arg1, ...)  _macro(_arg, _arg1), _UCS_PP_FOREACH_SEP_3 (_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_SEP_5(_macro , _arg, _arg1, ...)  _macro(_arg, _arg1), _UCS_PP_FOREACH_SEP_4 (_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_SEP_6(_macro , _arg, _arg1, ...)  _macro(_arg, _arg1), _UCS_PP_FOREACH_SEP_5 (_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_SEP_7(_macro , _arg, _arg1, ...)  _macro(_arg, _arg1), _UCS_PP_FOREACH_SEP_6 (_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_SEP_8(_macro , _arg, _arg1, ...)  _macro(_arg, _arg1), _UCS_PP_FOREACH_SEP_7 (_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_SEP_9(_macro , _arg, _arg1, ...)  _macro(_arg, _arg1), _UCS_PP_FOREACH_SEP_8 (_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_SEP_10(_macro, _arg, _arg1, ...)  _macro(_arg, _arg1), _UCS_PP_FOREACH_SEP_9 (_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_SEP_11(_macro, _arg, _arg1, ...)  _macro(_arg, _arg1), _UCS_PP_FOREACH_SEP_10(_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_SEP_12(_macro, _arg, _arg1, ...)  _macro(_arg, _arg1), _UCS_PP_FOREACH_SEP_11(_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_SEP_13(_macro, _arg, _arg1, ...)  _macro(_arg, _arg1), _UCS_PP_FOREACH_SEP_12(_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_SEP_14(_macro, _arg, _arg1, ...)  _macro(_arg, _arg1), _UCS_PP_FOREACH_SEP_13(_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_SEP_15(_macro, _arg, _arg1, ...)  _macro(_arg, _arg1), _UCS_PP_FOREACH_SEP_14(_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_SEP_16(_macro, _arg, _arg1, ...)  _macro(_arg, _arg1), _UCS_PP_FOREACH_SEP_15(_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_SEP_17(_macro, _arg, _arg1, ...)  _macro(_arg, _arg1), _UCS_PP_FOREACH_SEP_16(_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_SEP_18(_macro, _arg, _arg1, ...)  _macro(_arg, _arg1), _UCS_PP_FOREACH_SEP_17(_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_SEP_19(_macro, _arg, _arg1, ...)  _macro(_arg, _arg1), _UCS_PP_FOREACH_SEP_18(_macro, _arg, __VA_ARGS__)
#define _UCS_PP_FOREACH_SEP_20(_macro, _arg, _arg1, ...)  _macro(_arg, _arg1), _UCS_PP_FOREACH_SEP_19(_macro, _arg, __VA_ARGS__)

#define _UCS_PP_ZIP_0(_l1, _l2)
#define _UCS_PP_ZIP_1(_l1, _l2)       _UCS_PP_ZIP_0(_l1, _l2)  (UCS_PP_TUPLE_0 _l1, UCS_PP_TUPLE_0 _l2)
#define _UCS_PP_ZIP_2(_l1, _l2)       _UCS_PP_ZIP_1(_l1, _l2), (UCS_PP_TUPLE_1 _l1, UCS_PP_TUPLE_1 _l2)
#define _UCS_PP_ZIP_3(_l1, _l2)       _UCS_PP_ZIP_2(_l1, _l2), (UCS_PP_TUPLE_2 _l1, UCS_PP_TUPLE_2 _l2)
#define _UCS_PP_ZIP_4(_l1, _l2)       _UCS_PP_ZIP_3(_l1, _l2), (UCS_PP_TUPLE_3 _l1, UCS_PP_TUPLE_3 _l2)
#define _UCS_PP_ZIP_5(_l1, _l2)       _UCS_PP_ZIP_4(_l1, _l2), (UCS_PP_TUPLE_4 _l1, UCS_PP_TUPLE_4 _l2)
#define _UCS_PP_ZIP_6(_l1, _l2)       _UCS_PP_ZIP_5(_l1, _l2), (UCS_PP_TUPLE_5 _l1, UCS_PP_TUPLE_5 _l2)
#define _UCS_PP_ZIP_7(_l1, _l2)       _UCS_PP_ZIP_6(_l1, _l2), (UCS_PP_TUPLE_6 _l1, UCS_PP_TUPLE_6 _l2)
#define _UCS_PP_ZIP_8(_l1, _l2)       _UCS_PP_ZIP_7(_l1, _l2), (UCS_PP_TUPLE_7 _l1, UCS_PP_TUPLE_7 _l2)
#define _UCS_PP_ZIP_9(_l1, _l2)       _UCS_PP_ZIP_8(_l1, _l2), (UCS_PP_TUPLE_8 _l1, UCS_PP_TUPLE_8 _l2)
#define _UCS_PP_ZIP_10(_l1, _l2)      _UCS_PP_ZIP_9(_l1, _l2), (UCS_PP_TUPLE_9 _l1, UCS_PP_TUPLE_9 _l2)


/* Extract elements from tuples
 */
#define UCS_PP_TUPLE_0(_0, ...)                                            _0
#define UCS_PP_TUPLE_1(_0, _1, ...)                                        _1
#define UCS_PP_TUPLE_2(_0, _1, _2, ...)                                    _2
#define UCS_PP_TUPLE_3(_0, _1, _2, _3, ...)                                _3
#define UCS_PP_TUPLE_4(_0, _1, _2, _3, _4, ...)                            _4
#define UCS_PP_TUPLE_5(_0, _1, _2, _3, _4, _5, ...)                        _5
#define UCS_PP_TUPLE_6(_0, _1, _2, _3, _4, _5, _6, ...)                    _6
#define UCS_PP_TUPLE_7(_0, _1, _2, _3, _4, _5, _6, _7, ...)                _7
#define UCS_PP_TUPLE_8(_0, _1, _2, _3, _4, _5, _6, _7, _8, ...)            _8
#define UCS_PP_TUPLE_9(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, ...)        _9
#define UCS_PP_TUPLE_10(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, ...)  _10
#define UCS_PP_TUPLE_BREAK(...)                                            __VA_ARGS__


/* Sequence of numbers
 */
#define _UCS_PP_SEQ_0
#define _UCS_PP_SEQ_1     _UCS_PP_SEQ_0   0
#define _UCS_PP_SEQ_2     _UCS_PP_SEQ_1 , 1
#define _UCS_PP_SEQ_3     _UCS_PP_SEQ_2 , 2
#define _UCS_PP_SEQ_4     _UCS_PP_SEQ_3 , 3
#define _UCS_PP_SEQ_5     _UCS_PP_SEQ_4 , 4
#define _UCS_PP_SEQ_6     _UCS_PP_SEQ_5 , 5
#define _UCS_PP_SEQ_7     _UCS_PP_SEQ_6 , 6
#define _UCS_PP_SEQ_8     _UCS_PP_SEQ_7 , 7
#define _UCS_PP_SEQ_9     _UCS_PP_SEQ_8 , 8
#define _UCS_PP_SEQ_10    _UCS_PP_SEQ_9 , 9
#define _UCS_PP_SEQ_11    _UCS_PP_SEQ_10, 10
#define _UCS_PP_SEQ_12    _UCS_PP_SEQ_11, 11
#define _UCS_PP_SEQ_13    _UCS_PP_SEQ_12, 12
#define _UCS_PP_SEQ_14    _UCS_PP_SEQ_13, 13
#define _UCS_PP_SEQ_15    _UCS_PP_SEQ_14, 14
#define _UCS_PP_SEQ_16    _UCS_PP_SEQ_15, 15
#define _UCS_PP_SEQ_17    _UCS_PP_SEQ_16, 16
#define _UCS_PP_SEQ_18    _UCS_PP_SEQ_17, 17
#define _UCS_PP_SEQ_19    _UCS_PP_SEQ_18, 18
#define _UCS_PP_SEQ_20    _UCS_PP_SEQ_19, 19
#define _UCS_PP_SEQ(_n)   _UCS_PP_SEQ_##_n
#define UCS_PP_SEQ(_n)    _UCS_PP_SEQ(_n)

#endif
