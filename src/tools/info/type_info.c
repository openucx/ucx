/**_t
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucx_info.h"

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/async/async_int.h>
#include <ucs/async/pipe.h>
#include <ucs/datastruct/arbiter.h>
#include <ucs/datastruct/frag_list.h>
#include <ucs/datastruct/mpool.h>
#include <ucs/datastruct/pgtable.h>
#include <ucs/datastruct/ptr_array.h>
#include <ucs/sys/rcache.h>
#include <ucs/sys/rcache_int.h>
#include <ucs/time/timerq.h>
#include <ucs/time/timer_wheel.h>
#include <ucs/type/class.h>
#include <uct/base/uct_md.h>
#include <uct/base/uct_iface.h>
#include <uct/sm/self/self.h>
#include <uct/tcp/tcp.h>
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_mm.h>
#include <ucp/core/ucp_request.h>
#include <ucp/core/ucp_worker.h>
#include <ucp/wireup/wireup.h>

#if HAVE_IB
#  include <uct/ib/base/ib_device.h>
#  include <uct/ib/base/ib_iface.h>
#endif

#if HAVE_TL_RC
#  include <uct/ib/rc/base/rc_iface.h>
#  include <uct/ib/rc/base/rc_ep.h>
#  include <uct/ib/rc/verbs/rc_verbs.h>
#  if HAVE_MLX5_HW
#    include <uct/ib/rc/accel/rc_mlx5.h>
#  endif
#endif

#if HAVE_TL_DC
#  include <uct/ib/dc/base/dc_iface.h>
#  include <uct/ib/dc/base/dc_ep.h>
#  include <uct/ib/dc/verbs/dc_verbs.h>
#  if HAVE_MLX5_HW
#    include <uct/ib/dc/accel/dc_mlx5.h>
#  endif
#endif

#if HAVE_TL_UD
#  include <uct/ib/ud/base/ud_def.h>
#  include <uct/ib/ud/verbs/ud_verbs.h>
#  if HAVE_MLX5_HW_UD
#    include <uct/ib/ud/accel/ud_mlx5.h>
#  endif
#endif


#if HAVE_TL_UGNI
#  include <uct/ugni/base/ugni_ep.h>
#  include <uct/ugni/base/ugni_iface.h>
#  include <uct/ugni/base/ugni_device.h>
#  include <uct/ugni/smsg/ugni_smsg_ep.h>
#endif


static void print_size(const char *name, size_t size)
{
    int i;
    printf("    sizeof(%s)%n = ", name, &i);
    while (i++ < 40) {
        printf(".");
    }
    printf(" %-6lu\n", size);
}

#define PRINT_SIZE(type) print_size(UCS_PP_QUOTE(type), sizeof(type))


void print_type_info(const char * tl_name)
{
    if (tl_name == NULL) {
        printf("UCS:\n");
        PRINT_SIZE(ucs_mpool_t);
        PRINT_SIZE(ucs_mpool_chunk_t);
        PRINT_SIZE(ucs_mpool_elem_t);
        PRINT_SIZE(ucs_async_context_t);
        PRINT_SIZE(ucs_async_handler_t);
        PRINT_SIZE(ucs_async_ops_t);
        PRINT_SIZE(ucs_async_pipe_t);
        PRINT_SIZE(ucs_async_signal_context_t);
        PRINT_SIZE(ucs_async_thread_context_t);
        PRINT_SIZE(ucs_class_t);
        PRINT_SIZE(ucs_component_t);
        PRINT_SIZE(ucs_config_field_t);
        PRINT_SIZE(ucs_config_parser_t);
        PRINT_SIZE(ucs_frag_list_t);
        PRINT_SIZE(ucs_frag_list_elem_t);
        PRINT_SIZE(ucs_frag_list_head_t);
        PRINT_SIZE(ucs_ib_port_spec_t);
        PRINT_SIZE(ucs_list_link_t);
        PRINT_SIZE(ucs_memtrack_entry_t);
        PRINT_SIZE(ucs_mpmc_queue_t);
        PRINT_SIZE(ucs_callbackq_t);
        PRINT_SIZE(ucs_callbackq_elem_t);
        PRINT_SIZE(ucs_ptr_array_t);
        PRINT_SIZE(ucs_queue_elem_t);
        PRINT_SIZE(ucs_queue_head_t);
        PRINT_SIZE(ucs_spinlock_t);
        PRINT_SIZE(ucs_timer_t);
        PRINT_SIZE(ucs_timer_queue_t);
        PRINT_SIZE(ucs_twheel_t);
        PRINT_SIZE(ucs_wtimer_t);
        PRINT_SIZE(ucs_arbiter_t);
        PRINT_SIZE(ucs_arbiter_group_t);
        PRINT_SIZE(ucs_arbiter_elem_t);
        PRINT_SIZE(ucs_pgtable_t);
        PRINT_SIZE(ucs_pgt_entry_t);
        PRINT_SIZE(ucs_pgt_dir_t);
        PRINT_SIZE(ucs_pgt_region_t);
        PRINT_SIZE(ucs_rcache_t);
        PRINT_SIZE(ucs_rcache_region_t);

        printf("\nUCT:\n");
        PRINT_SIZE(uct_am_handler_t);
        PRINT_SIZE(uct_base_iface_t);
        PRINT_SIZE(uct_completion_t);
        PRINT_SIZE(uct_ep_t);
        PRINT_SIZE(uct_mem_h);
        PRINT_SIZE(uct_rkey_t);
        PRINT_SIZE(uct_iface_t);
        PRINT_SIZE(uct_iface_attr_t);
        PRINT_SIZE(uct_iface_config_t);
        PRINT_SIZE(uct_iface_mpool_config_t);
        PRINT_SIZE(uct_md_config_t);
        PRINT_SIZE(uct_iface_ops_t);
        PRINT_SIZE(uct_md_t);
        PRINT_SIZE(uct_md_attr_t);
        PRINT_SIZE(uct_md_ops_t);
        PRINT_SIZE(uct_tl_resource_desc_t);
        PRINT_SIZE(uct_rkey_bundle_t);
        PRINT_SIZE(uct_tcp_ep_t);
        PRINT_SIZE(uct_self_ep_t);

#if HAVE_TL_UGNI
        PRINT_SIZE(uct_sockaddr_ugni_t);
        PRINT_SIZE(uct_sockaddr_smsg_ugni_t);
        PRINT_SIZE(uct_devaddr_ugni_t);
#endif

#if HAVE_IB
        printf("\nIB:\n");
        PRINT_SIZE(uct_ib_address_t);
        PRINT_SIZE(uct_ib_device_t);
        PRINT_SIZE(uct_ib_md_t);
        PRINT_SIZE(uct_ib_mem_t);
        PRINT_SIZE(uct_ib_rcache_region_t);
        PRINT_SIZE(uct_ib_iface_t);
        PRINT_SIZE(uct_ib_iface_config_t);
        PRINT_SIZE(uct_ib_iface_recv_desc_t);
        PRINT_SIZE(uct_ib_recv_wr_t);
#endif
        printf("\n");
    }

#if HAVE_TL_RC
    if (tl_name == NULL || !strcasecmp(tl_name, "rc") ||
        !strcasecmp(tl_name, "rc_mlx5"))
    {
        printf("RC:\n");
        PRINT_SIZE(uct_rc_am_short_hdr_t);
        PRINT_SIZE(uct_rc_ep_t);
        PRINT_SIZE(uct_rc_hdr_t);
        PRINT_SIZE(uct_rc_iface_t);
        PRINT_SIZE(uct_rc_iface_config_t);
        PRINT_SIZE(uct_rc_iface_send_op_t);
        PRINT_SIZE(uct_rc_iface_send_desc_t);

        PRINT_SIZE(uct_rc_iface_send_desc_t);
        if (tl_name == NULL || !strcasecmp(tl_name, "rc")) {
            PRINT_SIZE(uct_rc_verbs_ep_t);
            PRINT_SIZE(uct_rc_verbs_iface_config_t);
            PRINT_SIZE(uct_rc_verbs_iface_t);
        }

#if HAVE_MLX5_HW
        if (tl_name == NULL || !strcasecmp(tl_name, "rc_mlx5")) {
            PRINT_SIZE(uct_rc_mlx5_ep_t);
            PRINT_SIZE(uct_rc_mlx5_iface_config_t);
            PRINT_SIZE(uct_rc_mlx5_iface_t);
        }
#endif
        printf("\n");
    }
#endif

#if HAVE_TL_DC
    if (tl_name == NULL || !strcasecmp(tl_name, "dc") ||
        !strcasecmp(tl_name, "dc_mlx5"))
    {
        printf("DC:\n");
        PRINT_SIZE(uct_dc_ep_t);
        PRINT_SIZE(uct_dc_iface_t);
        PRINT_SIZE(uct_dc_iface_config_t);

        if (tl_name == NULL || !strcasecmp(tl_name, "dc")) {
            PRINT_SIZE(uct_dc_verbs_ep_t);
            PRINT_SIZE(uct_dc_verbs_iface_t);
        }

#if HAVE_MLX5_HW
        if (tl_name == NULL || !strcasecmp(tl_name, "dc_mlx5")) {
            PRINT_SIZE(uct_dc_mlx5_ep_t);
            PRINT_SIZE(uct_dc_mlx5_grh_ep_t);
            PRINT_SIZE(uct_dc_mlx5_iface_t);
        }
#endif
        printf("\n");
    }
#endif

#if HAVE_TL_UD
    if (tl_name == NULL || !strcasecmp(tl_name, "ud") ||
        !strcasecmp(tl_name, "ud_mlx5"))
    {
        printf("UD:\n");
        PRINT_SIZE(uct_ud_ep_t);
        PRINT_SIZE(uct_ud_neth_t);
        PRINT_SIZE(uct_ud_iface_t);
        PRINT_SIZE(uct_ud_iface_config_t);
        PRINT_SIZE(uct_ud_ep_pending_op_t);
        PRINT_SIZE(uct_ud_send_skb_t);
        PRINT_SIZE(uct_ud_send_skb_inl_t);
        PRINT_SIZE(uct_ud_recv_skb_t);

        PRINT_SIZE(uct_rc_iface_send_desc_t);
        if (tl_name == NULL || !strcasecmp(tl_name, "ud")) {
            PRINT_SIZE(uct_ud_verbs_ep_t);
            PRINT_SIZE(uct_ud_verbs_iface_t);
        }

#if HAVE_MLX5_HW_UD
        if (tl_name == NULL || !strcasecmp(tl_name, "ud_mlx5")) {
            PRINT_SIZE(uct_ud_mlx5_ep_t);
            PRINT_SIZE(uct_ud_mlx5_iface_t);
            PRINT_SIZE(uct_rc_mlx5_iface_config_t);
        }
#endif
        printf("\n");
    }
#endif

#if HAVE_TL_UGNI
    if (tl_name == NULL || !strcasecmp(tl_name, "ugni")) {
        printf("UGNI:\n");
        PRINT_SIZE(uct_ugni_device_t);
        PRINT_SIZE(uct_ugni_ep_t);
        PRINT_SIZE(uct_ugni_iface_t);
        PRINT_SIZE(uct_ugni_md_t);
        PRINT_SIZE(uct_ugni_compact_smsg_attr_t);

        printf("\n");
    }
#endif

    printf("\nUCP:\n");
    PRINT_SIZE(ucp_context_t);
    PRINT_SIZE(ucp_worker_t);
    PRINT_SIZE(ucp_ep_t);
    PRINT_SIZE(ucp_ep_ext_gen_t);
    PRINT_SIZE(ucp_ep_ext_proto_t);
    PRINT_SIZE(ucp_ep_match_entry_t);
    PRINT_SIZE(ucp_ep_config_key_t);
    PRINT_SIZE(ucp_ep_config_t);
    PRINT_SIZE(ucp_request_t);
    PRINT_SIZE(ucp_recv_desc_t);
    PRINT_SIZE(ucp_tag_recv_info_t);
    PRINT_SIZE(ucp_mem_t);
    PRINT_SIZE(ucp_rkey_t);
    PRINT_SIZE(ucp_wireup_msg_t);

}
