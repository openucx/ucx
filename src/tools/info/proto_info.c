/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

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
                    size_t estimated_num_eps, unsigned dev_type_bitmap,
                    const char *mem_size)
{
    ucp_config_t *config;
    ucs_status_t status;
    ucs_status_ptr_t status_ptr;
    ucp_context_h context;
    ucp_worker_h worker;
    ucp_params_t params;
    ucp_worker_params_t worker_params;
    ucp_mem_map_params_t mem_params;
    ucp_ep_params_t ep_params;
    ucp_address_t *address;
    size_t address_length, mem_size_value;
    resource_usage_t usage;
    ucp_mem_h memh;
    ucp_ep_h ep;

    status = ucp_config_read(NULL, NULL, &config);
    if (status != UCS_OK) {
        return;
    }

    memset(&params, 0, sizeof(params));
    params.field_mask        = UCP_PARAM_FIELD_FEATURES |
                               UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
    params.features          = ctx_features;
    params.estimated_num_eps = estimated_num_eps;

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

    status = ucp_init(&params, config, &context);
    if (status != UCS_OK) {
        printf("<Failed to create UCP context>\n");
        goto out_release_config;
    }

    if (mem_size != NULL) {
        ucs_str_to_memunits(mem_size, &mem_size_value, NULL);
        mem_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                                UCP_MEM_MAP_PARAM_FIELD_LENGTH  |
                                UCP_MEM_MAP_PARAM_FIELD_FLAGS;
        mem_params.address    = NULL;
        mem_params.length     = mem_size_value;
        mem_params.flags      = UCP_MEM_MAP_ALLOCATE;

        status = ucp_mem_map(context, &mem_params, &memh);
        if (status != UCS_OK) {
            printf("<Failed to map memory of size %s>\n", mem_size);
            goto out_cleanup_context;
        }

        if (print_opts & PRINT_MEM_MAP) {
            ucp_mem_print_info(memh, context, stdout);
        }
    }

    if (print_opts & PRINT_UCP_CONTEXT) {
        ucp_context_print_info(context, stdout);
        print_resource_usage(&usage, "UCP context");
    }

    if (!(print_opts & (PRINT_UCP_WORKER|PRINT_UCP_EP))) {
        goto out_mem_unmap;
    }

    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;

    get_resource_usage(&usage);

    status = ucp_worker_create(context, &worker_params, &worker);
    if (status != UCS_OK) {
        printf("<Failed to create UCP worker>\n");
        goto out_mem_unmap;
    }

    if (print_opts & PRINT_UCP_WORKER) {
        ucp_worker_print_info(worker, stdout);
        print_resource_usage(&usage, "UCP worker");
    }

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
out_mem_unmap:
    if (mem_size != NULL) {
        status = ucp_mem_unmap(context, memh);
        if (status != UCS_OK) {
            printf("<Failed to unmap memory of size %s>\n", mem_size);
        }
    }
out_cleanup_context:
    ucp_cleanup(context);
out_release_config:
    ucp_config_release(config);
}
