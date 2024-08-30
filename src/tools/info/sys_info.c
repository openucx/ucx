/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2015. ALL RIGHTS RESERVED.
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

#include <ucs/sys/string.h>
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

static void print_repeat_char(int ch, int count)
{
    int i;

    for (i = 0; i < count; ++i) {
        putchar(ch);
    }
}

static void print_row_separator(int column_width, int first_column_width,
                                int num_columns, int fill_char,
                                int separator_char)
{
    int i;

    printf("# %c", separator_char);
    print_repeat_char(fill_char, first_column_width);
    for (i = 0; i < num_columns; ++i) {
        putchar(separator_char);
        print_repeat_char(fill_char, column_width);
    }
    printf("%c\n", separator_char);
}

static void print_table_header(const char *title, const char *distance_unit,
                               int column_width, int first_column_width,
                               int num_columns)
{
    int column;

    printf("#\n");
    printf("# %s\n", title);
    printf("#\n");
    print_row_separator(column_width, first_column_width, num_columns, '-',
                        '+');
    print_row_separator(column_width, first_column_width, num_columns, ' ',
                        '|');
    printf("# |%*s ", first_column_width - 1, distance_unit);
    for (column = 0; column < num_columns; ++column) {
        printf("|%*s ", column_width - 1,
               ucs_topo_sys_device_get_name((ucs_sys_device_t)column));
    }

    printf("|\n");
    print_row_separator(column_width, first_column_width, num_columns, ' ',
                        '|');
    print_row_separator(column_width, first_column_width, num_columns, '-',
                        '+');
}

static void print_sys_topo()
{
    const unsigned num_devices              = ucs_topo_num_devices();
    static const int distance_width         = 10;
    static const char *system_topo_title    = "System topology";
    static const char *distance_unit        = "MB/s";
    static const char *memory_latency_title = "NUMA memory latency";
    static const char *numa_distance_unit   = "nsec";
    ucs_sys_device_t sys_dev1, sys_dev2;
    ucs_sys_dev_distance_t distance;
    char distance_str[20];
    ucs_status_t status;
    int name_width;

    /* Get maximal width of device name */
    name_width = 2 + strlen(distance_unit);
    for (sys_dev1 = 0; sys_dev1 < num_devices; ++sys_dev1) {
        name_width = ucs_max(
                name_width, 2 + strlen(ucs_topo_sys_device_get_name(sys_dev1)));
    }

    print_table_header(system_topo_title, distance_unit, distance_width,
                       name_width, num_devices);

    /* Print table content */
    for (sys_dev1 = 0; sys_dev1 < num_devices; ++sys_dev1) {
        print_row_separator(distance_width, name_width, num_devices, ' ', '|');

        printf("# |%*s ", name_width - 1,
               ucs_topo_sys_device_get_name(sys_dev1));
        for (sys_dev2 = 0; sys_dev2 < num_devices; ++sys_dev2) {
            if (sys_dev1 == sys_dev2) {
                /* Do not print distance of device to itself */
                strncpy(distance_str, "-", sizeof(distance_str));
            } else {
                status = ucs_topo_get_distance(sys_dev1, sys_dev2, &distance);
                if (status != UCS_OK) {
                    ucs_snprintf_safe(distance_str, sizeof(distance_str),
                                      "<%s>", ucs_status_string(status));
                } else if (distance.bandwidth > UCS_PBYTE) {
                    ucs_snprintf_safe(distance_str, sizeof(distance_str),
                                      "inf");
                } else {
                    ucs_snprintf_safe(distance_str, sizeof(distance_str),
                                      "%.1f", distance.bandwidth / UCS_MBYTE);
                }
            }
            printf("|%*s ", distance_width - 1, distance_str);
        }
        printf("|\n");

        print_row_separator(distance_width, name_width, num_devices, ' ', '|');
        print_row_separator(distance_width, name_width, num_devices, '-', '+');
    }

    print_table_header(memory_latency_title, "device", distance_width,
                       name_width, num_devices);
    print_row_separator(distance_width, name_width, num_devices, ' ', '|');

    printf("# |%*s ", name_width - 1, numa_distance_unit);
    printf("|");
    for (sys_dev1 = 0; sys_dev1 < num_devices; ++sys_dev1) {
        ucs_topo_get_memory_distance(sys_dev1, &distance);
        ucs_snprintf_safe(distance_str, sizeof(distance_str), "%.1f",
                          distance.latency * UCS_NSEC_PER_SEC);
        printf("%*s |", distance_width - 1, distance_str);
    }

    printf("\n");
    print_row_separator(distance_width, name_width, num_devices, ' ', '|');
    print_row_separator(distance_width, name_width, num_devices, '-', '+');

    printf("# Memory latency is calculated according to the CPU affinity\n");
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
