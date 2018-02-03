/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <common/test.h>
#include <ucs/type/cpu_set.h>

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_request.h>
#include <ucp/core/ucp_types.h>
#include <uct/api/tl.h>
#include <uct/base/uct_iface.h>
#include <uct/sm/self/self_ep.h>
#include <uct/tcp/tcp.h>
#if HAVE_TL_RC
#  include <uct/ib/rc/verbs/rc_verbs.h>
#endif
#if HAVE_TL_DC
#  include <uct/ib/dc/base/dc_ep.h>
#  include <uct/ib/dc/base/dc_iface.h>
#  include <uct/ib/dc/verbs/dc_verbs.h>
#endif
#if HAVE_TL_UD
#  include <uct/ib/ud/base/ud_ep.h>
#  include <uct/ib/ud/verbs/ud_verbs.h>
#endif

class test_obj_size : public ucs::test {
};

#define EXPECTED_SIZE(_obj, _size) EXPECT_EQ(sizeof(_obj), (size_t)_size)

UCS_TEST_F(test_obj_size, size) {

#if ENABLE_DEBUG_DATA
   UCS_TEST_SKIP_R("Debug data");
#elif ENABLE_STATS
   UCS_TEST_SKIP_R("Statistic enabled");
#else
    EXPECTED_SIZE(ucp_ep_t, 104);
    EXPECTED_SIZE(ucp_request_t, 224);
    EXPECTED_SIZE(uct_ep_t, 8);
    EXPECTED_SIZE(uct_base_ep_t, 8);
    EXPECTED_SIZE(uct_rkey_bundle_t, 24);
    EXPECTED_SIZE(uct_self_ep_t, 8);
    EXPECTED_SIZE(uct_tcp_ep_t, 72);
#  if HAVE_TL_RC
#    if IBV_EXP_HW_TM
    EXPECTED_SIZE(uct_rc_ep_t, 88);
    EXPECTED_SIZE(uct_rc_verbs_ep_t, 96);
#    else
    EXPECTED_SIZE(uct_rc_ep_t, 80);
    EXPECTED_SIZE(uct_rc_verbs_ep_t, 88);
#    endif
#  endif
#  if HAVE_TL_DC
    EXPECTED_SIZE(uct_dc_ep_t, 24);
    EXPECTED_SIZE(uct_dc_verbs_ep_t, 40);
#  endif
#  if HAVE_TL_UD
    EXPECTED_SIZE(uct_ud_ep_t, 248);
    EXPECTED_SIZE(uct_ud_verbs_ep_t, 264);
#  endif
#endif
}

