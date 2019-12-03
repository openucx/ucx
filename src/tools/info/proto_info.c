/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
/**
*2019.12.30-Changed process for coll_ucx
*        Huawei Technologies Co., Ltd. 2019.
*/


#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucx_info.h"

#include <ucp/api/ucp.h>
#include <ucs/time/time.h>
#include <ucs/sys/string.h>
#include <sys/resource.h>
#include <dirent.h>
#include <string.h>
#include <errno.h>


typedef struct {
    ucs_time_t       time;
    long             memory;
    int              num_fds;
} resource_usage_t;


static int get_num_fds()
{
    static const char *fds_dir = "/proc/self/fd";
    struct dirent *entry;
    int num_fds;
    DIR *dir;

    dir = opendir(fds_dir);
    if (dir == NULL) {
        return -1;
    }

    num_fds = 0;
    for (;;) {
        errno = 0;
        entry = readdir(dir);
        if (entry == NULL) {
            closedir(dir);
            if (errno == 0) {
                return num_fds;
            } else {
                return -1;
            }
        }

        if (strncmp(entry->d_name, ".", 1)) {
            ++num_fds;
        }
    }
}

static void get_resource_usage(resource_usage_t *usage)
{
    struct rusage rusage;
    int ret;

    usage->time = ucs_get_time();

    ret = getrusage(RUSAGE_SELF, &rusage);
    if (ret == 0) {
        usage->memory = rusage.ru_maxrss * 1024;
    } else {
        usage->memory = -1;
    }

    usage->num_fds = get_num_fds();
}

static void print_resource_usage(const resource_usage_t *usage_before,
                                 const char *title)
{
    resource_usage_t usage_after;

    get_resource_usage(&usage_after);

    if ((usage_after.memory != -1) && (usage_before->memory != -1) &&
        (usage_after.num_fds != -1) && (usage_before->num_fds != -1))
    {
        printf("# memory: %.2fMB, file descriptors: %d\n",
               (usage_after.memory - usage_before->memory) / (1024.0 * 1024.0),
               (usage_after.num_fds - usage_before->num_fds));
    }
    printf("# create time: %.3f ms\n",
           ucs_time_to_msec(usage_after.time - usage_before->time));
    printf("#\n");
}

void print_ucp_info(int print_opts, ucs_config_print_flags_t print_flags,
                    uint64_t ctx_features, const ucp_ep_params_t *base_ep_params,
                    size_t estimated_num_eps, size_t estimated_num_ppn,
                    unsigned dev_type_bitmap, const char *mem_size
#if ENABLE_UCG
                    ,const char *planner_name,
                    ucg_group_member_index_t root_index,
                    ucg_group_member_index_t my_index,
                    const char *collective_type_name,
                    ucg_group_member_index_t peer_count[UCG_GROUP_MEMBER_DISTANCE_LAST]
#endif
                    )
{
    ucp_config_t *config;
    ucs_status_t status;
    ucs_status_ptr_t status_ptr;
    ucp_context_h context;
    ucp_worker_h worker;
    ucp_params_t params;
    ucp_worker_params_t worker_params;
    ucp_ep_params_t ep_params;
    ucp_address_t *address;
    size_t address_length;
    resource_usage_t usage;
    ucp_ep_h ep;

    status = ucp_config_read(NULL, NULL, &config);
    if (status != UCS_OK) {
        printf("<Failed to read UCP configuration>\n");
        return;
    }

    memset(&params, 0, sizeof(params));
    params.field_mask        = UCP_PARAM_FIELD_FEATURES |
                               UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
    params.features          = ctx_features;
    params.estimated_num_eps = estimated_num_eps;
    params.estimated_num_ppn = estimated_num_ppn;

    get_resource_usage(&usage);

    if (!(dev_type_bitmap & UCS_BIT(UCT_DEVICE_TYPE_SELF))) {
        ucp_config_modify(config, "SELF_DEVICES", "");
    }
    if (!(dev_type_bitmap & UCS_BIT(UCT_DEVICE_TYPE_SHM))) {
        ucp_config_modify(config, "SHM_DEVICES", "");
    }
    if (!(dev_type_bitmap & UCS_BIT(UCT_DEVICE_TYPE_NET))) {
        ucp_config_modify(config, "NET_DEVICES", "");
    }

#if ENABLE_UCG
    if (ctx_features & UCP_FEATURE_GROUPS) {
        status = ucg_init(&params, config, &context);
    } else
#endif
    status = ucp_init(&params, config, &context);
    if (status != UCS_OK) {
        printf("<Failed to create UCP context>\n");
        goto out_release_config;
    }

    if ((print_opts & PRINT_MEM_MAP) && (mem_size != NULL)) {
        ucp_mem_print_info(mem_size, context, stdout);
    }

    if (print_opts & PRINT_UCP_CONTEXT) {
        ucp_context_print_info(context, stdout);
        print_resource_usage(&usage, "UCP context");
    }

    if (!(print_opts & (PRINT_UCP_WORKER|PRINT_UCP_EP|PRINT_UCG|PRINT_UCG_TOPO))) {
        goto out_cleanup_context;
    }

    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;

    get_resource_usage(&usage);

    status = ucp_worker_create(context, &worker_params, &worker);
    if (status != UCS_OK) {
        printf("<Failed to create UCP worker>\n");
        goto out_cleanup_context;
    }

    if (print_opts & PRINT_UCP_WORKER) {
        ucp_worker_print_info(worker, stdout);
        print_resource_usage(&usage, "UCP worker");
    }

#if ENABLE_UCG
    if (print_opts & PRINT_UCG) {
        /* create a group with the generated paramters */
        enum ucg_group_member_distance distance = UCG_GROUP_MEMBER_DISTANCE_SELF;
        ucg_group_params_t group_params = {
                .distance = &distance,
                .member_count = 1
        };

        ucg_group_h group;
        ucs_status_t status = ucg_group_create(worker, &group_params, &group);
        if (status != UCS_OK) {
            printf("<Failed to create UCG group>\n");
            goto out_destroy_worker;
        }

        print_resource_usage(&usage, "UCG group");
        ucg_group_destroy(group);
    }

    if (print_opts & PRINT_UCG_TOPO) {
        ucg_group_member_index_t dist_len;
        enum ucg_group_member_distance* dist;
        if (UCS_OK == gen_ucg_topology(my_index, peer_count, &dist, &dist_len)) {
            print_ucg_topology(planner_name, worker, root_index, my_index,
                    collective_type_name, dist, dist_len, 1);
        }
    }
#endif

    if (print_opts & PRINT_UCP_EP) {
        status = ucp_worker_get_address(worker, &address, &address_length);
        if (status != UCS_OK) {
            printf("<Failed to get UCP worker address>\n");
            goto out_destroy_worker;
        }

        ep_params             = *base_ep_params;

        ep_params.field_mask |= UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
        ep_params.address     = address;

        status = ucp_ep_create(worker, &ep_params, &ep);
        ucp_worker_release_address(worker, address);
        if (status != UCS_OK) {
            printf("<Failed to create UCP endpoint>\n");
            goto out_destroy_worker;
        }

        ucp_ep_print_info(ep, stdout);

        status_ptr = ucp_disconnect_nb(ep);
        if (UCS_PTR_IS_PTR(status_ptr)) {
            do {
                ucp_worker_progress(worker);
                status = ucp_request_test(status_ptr, NULL);
            } while (status == UCS_INPROGRESS);
            ucp_request_release(status_ptr);
        }
    }

out_destroy_worker:
    ucp_worker_destroy(worker);
out_cleanup_context:
    ucp_cleanup(context);
out_release_config:
    ucp_config_release(config);
}
