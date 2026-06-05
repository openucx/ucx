/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
* Copyright (C) Shanghai Zhaoxin Semiconductor Co., Ltd. 2020. ALL RIGHTS RESERVED.
* Copyright (C) Tactical Computing Labs, LLC. 2022. ALL RIGHTS RESERVED.
* Copyright (C) Advanced Micro Devices, Inc. 2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucx_info.h"

#include <ucs/debug/table.h>
#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/math.h>
#include <ucs/time/time.h>
#include <ucs/config/parser.h>
#include <ucs/config/global_opts.h>
#include <sys/mman.h>
#include <string.h>


static double measure_memcpy_bandwidth(size_t size)
{
    ucs_time_t start_time, end_time;
    void *src, *dst;
    double result = 0.0;
    int iter;

    src = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (src == MAP_FAILED) {
        goto out;
    }

    dst = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (dst == MAP_FAILED) {
        goto out_unmap_src;
    }

    memset(dst, 0, size);
    memset(src, 0, size);
    memcpy(dst, src, size);

    iter = 0;
    start_time = ucs_get_time();
    do {
        ucs_memcpy_relaxed(dst, src, size, UCS_ARCH_MEMCPY_NT_NONE, size);
        end_time = ucs_get_time();
        ++iter;
    } while (end_time < start_time + ucs_time_from_sec(0.5));

    result = size * iter / ucs_time_to_sec(end_time - start_time);

    munmap(dst, size);
out_unmap_src:
    munmap(src, size);
out:
    return result;
}

/* Add an empty row used as vertical padding around content rows */
static void print_sys_topo_add_padding(ucs_table_t *table)
{
    ucs_table_row_h row;
    unsigned i;

    ucs_table_add_row(table, &row);

    for (i = 0; i < table->config.n_cols; ++i) {
        ucs_table_row_add_cell_empty(table, row, 1);
    }
}

/* Add a header row of the shape "<label> | dev0 | dev1 | ...", surrounded
 * by empty padding rows and followed by a separator. */
static void print_sys_topo_add_devices_header(ucs_table_t *table,
                                              const char *first_col_label,
                                              unsigned num_devices)
{
    ucs_table_row_h row;
    ucs_sys_device_t sys_dev;

    print_sys_topo_add_padding(table);

    ucs_table_add_row(table, &row);
    ucs_table_row_add_cell_fmt(table, row, 1, UCS_TABLE_ALIGN_RIGHT, "%s",
                               first_col_label);

    for (sys_dev = 0; sys_dev < num_devices; ++sys_dev) {
        ucs_table_row_add_cell_fmt(table, row, 1, UCS_TABLE_ALIGN_RIGHT, "%s",
                                   ucs_topo_sys_device_get_name(sys_dev));
    }

    print_sys_topo_add_padding(table);
    ucs_table_add_separator(table);
}

static void print_sys_topo_distances(unsigned num_devices)
{
    ucs_table_config_t cfg = {
        .n_cols       = 1 + num_devices,
        .row_prefix   = "# ",
        .equal_widths = 1,
    };
    ucs_sys_device_t sys_dev1, sys_dev2;
    ucs_sys_dev_distance_t distance;
    ucs_status_t dist_status;
    ucs_table_row_h row;
    ucs_table_t table;

    printf("#\n# System topology\n#\n");

    ucs_table_init(&table, &cfg);

    print_sys_topo_add_devices_header(&table, "MB/s", num_devices);

    for (sys_dev1 = 0; sys_dev1 < num_devices; ++sys_dev1) {
        if (sys_dev1 > 0) {
            ucs_table_add_separator(&table);
        }

        print_sys_topo_add_padding(&table);

        ucs_table_add_row(&table, &row);
        ucs_table_row_add_cell_fmt(&table, row, 1, UCS_TABLE_ALIGN_RIGHT, "%s",
                                   ucs_topo_sys_device_get_name(sys_dev1));

        for (sys_dev2 = 0; sys_dev2 < num_devices; ++sys_dev2) {
            if (sys_dev1 == sys_dev2) {
                /* Do not print distance of device to itself */
                ucs_table_row_add_cell_fmt(&table, row, 1,
                                           UCS_TABLE_ALIGN_CENTER, "%s", "-");
                continue;
            }

            dist_status = ucs_topo_get_distance(sys_dev1, sys_dev2, &distance);
            if (dist_status != UCS_OK) {
                ucs_table_row_add_cell_fmt(&table, row, 1,
                                           UCS_TABLE_ALIGN_RIGHT, "<%s>",
                                           ucs_status_string(dist_status));
            } else if (distance.bandwidth > UCS_PBYTE) {
                ucs_table_row_add_cell_fmt(&table, row, 1,
                                           UCS_TABLE_ALIGN_RIGHT, "%s", "inf");
            } else {
                ucs_table_row_add_cell_fmt(&table, row, 1,
                                           UCS_TABLE_ALIGN_RIGHT, "%.1f",
                                           distance.bandwidth / UCS_MBYTE);
            }
        }

        print_sys_topo_add_padding(&table);
    }

    ucs_table_print(&table);
    ucs_table_cleanup(&table);
}

static void print_sys_topo_memory_latency(unsigned num_devices)
{
    ucs_table_config_t cfg = {
        .n_cols       = 1 + num_devices,
        .row_prefix   = "# ",
        .equal_widths = 1,
    };
    ucs_sys_dev_distance_t distance;
    ucs_sys_device_t sys_dev;
    ucs_table_row_h row;
    ucs_table_t table;

    printf("#\n# NUMA memory latency\n#\n");

    ucs_table_init(&table, &cfg);

    print_sys_topo_add_devices_header(&table, "device", num_devices);
    print_sys_topo_add_padding(&table);

    ucs_table_add_row(&table, &row);
    ucs_table_row_add_cell_fmt(&table, row, 1, UCS_TABLE_ALIGN_RIGHT, "%s",
                               "nsec");

    for (sys_dev = 0; sys_dev < num_devices; ++sys_dev) {
        ucs_topo_get_memory_distance(sys_dev, &distance);
        ucs_table_row_add_cell_fmt(&table, row, 1, UCS_TABLE_ALIGN_RIGHT,
                                   "%.1f", distance.latency * UCS_NSEC_PER_SEC);
    }

    print_sys_topo_add_padding(&table);

    ucs_table_print(&table);

    printf("# Memory latency is calculated according to the CPU affinity\n");

    ucs_table_cleanup(&table);
}

static void print_sys_topo()
{
    const unsigned num_devices = ucs_topo_num_devices();

    print_sys_topo_distances(num_devices);
    print_sys_topo_memory_latency(num_devices);
}

static double measure_timer_accuracy()
{
    double elapsed, elapsed_accurate;
    ucs_time_t time0, time1;
    double sec0, sec1;

    sec0  = ucs_get_accurate_time();
    time0 = ucs_get_time();
    usleep(100000);
    sec1  = ucs_get_accurate_time();
    time1 = ucs_get_time();

    elapsed          = ucs_time_to_sec(time1 - time0);
    elapsed_accurate = sec1 - sec0;

    return ucs_min(elapsed, elapsed_accurate) /
           ucs_max(elapsed, elapsed_accurate);
}

void print_sys_info(int print_opts)
{
    size_t size;

    if (print_opts & PRINT_SYS_INFO) {
        printf("# Timer frequency: %.3f MHz\n",
               ucs_get_cpu_clocks_per_sec() / 1e6);
        printf("# Timer accuracy: %.3f %%\n", measure_timer_accuracy() * 100);
        printf("# %s: %s\n", UCS_CPU_VENDOR_LABEL, ucs_cpu_vendor_name());
        printf("# %s: %s\n", UCS_CPU_MODEL_LABEL, ucs_cpu_model_name());
        printf("# %s: %s\n", UCS_SYS_DMI_PRODUCT_NAME_LABEL,
               ucs_sys_dmi_product_name());
    }

    if (print_opts & PRINT_SYS_TOPO) {
        print_sys_topo();
    }

    if (print_opts & PRINT_MEMCPY_BW) {
        ucs_arch_print_memcpy_limits(&ucs_global_opts.arch);
        printf("# Memcpy bandwidth:\n");
        for (size = 4096; size <= 256 * UCS_MBYTE; size *= 2) {
            printf("#     %10zu bytes: %.3f MB/s\n", size,
                   measure_memcpy_bandwidth(size) / UCS_MBYTE);
        }
    }
}
