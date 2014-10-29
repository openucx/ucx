/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef _UCS_CHECK_H
#define _UCS_CHECK_H

#include <ucs/sys/compiler.h>
#include <ucs/sys/sys.h>

#if ENABLE_ASSERT
#  define UCS_REENTRY_GUARD(_enter_count) \
                ucs_assertv(_enter_count == 0, "%s called recursively", __FUNCTION__); \
                for (++(_enter_count); (_enter_count) > 0; --(_enter_count))
#else
#  define UCS_REENTRY_GUARD(_enter_count)
#endif

#endif
