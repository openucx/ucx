/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <ucs/sys/compiler.h>
#include <ucs/arch/cpu.h>
#include <ucs/config/parser.h>
#include <ucs/debug/debug.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/profile/profile.h>
#include <ucs/stats/stats.h>
#include <ucs/async/async.h>
#include <ucs/sys/sys.h>


/* run-time CPU detection */
static UCS_F_NOOPTIMIZE void ucs_check_cpu_flags(void)
{
    char str[256];
    char *p_str;
    int cpu_flags;
    struct {
        const char* flag;
        ucs_cpu_flag_t value;
    } *p_flags,
    cpu_flags_array[] = {
        { "cmov", UCS_CPU_FLAG_CMOV },
        { "mmx", UCS_CPU_FLAG_MMX },
        { "mmx2", UCS_CPU_FLAG_MMX2 },
        { "sse", UCS_CPU_FLAG_SSE },
        { "sse2", UCS_CPU_FLAG_SSE2 },
        { "sse3", UCS_CPU_FLAG_SSE3 },
        { "ssse3", UCS_CPU_FLAG_SSSE3 },
        { "sse41", UCS_CPU_FLAG_SSE41 },
        { "sse42", UCS_CPU_FLAG_SSE42 },
        { "avx", UCS_CPU_FLAG_AVX },
        { "avx2", UCS_CPU_FLAG_AVX2 },
        { NULL, UCS_CPU_FLAG_UNKNOWN },
    };

    cpu_flags = ucs_arch_get_cpu_flag();
    if (UCS_CPU_FLAG_UNKNOWN == cpu_flags) {
        return ;
    }
    strncpy(str, UCS_PP_MAKE_STRING(CPU_FLAGS), sizeof(str) - 1);

    p_str = strtok(str, " |\t\n\r");
    while (p_str) {
        p_flags = cpu_flags_array;
        while (p_flags && p_flags->flag) {
            if (!strcmp(p_str, p_flags->flag)) {
                if (!(cpu_flags & p_flags->value)) {
                    fprintf(stderr, "[%s:%d] FATAL: UCX library was compiled with %s"
                            " but CPU does not support it.\n",
                            ucs_get_host_name(), getpid(), p_flags->flag);
                    exit(1);
                }
                break;
            }
            p_flags++;
        }
        if (NULL == p_flags->flag) {
            fprintf(stderr, "[%s:%d] FATAL: UCX library was compiled with %s"
                    " but CPU does not support it.\n",
                    ucs_get_host_name(), getpid(), p_str);
            exit(1);
        }
        p_str = strtok(NULL, " |\t\n\r");
    }
}

static void UCS_F_CTOR ucs_init()
{
    ucs_check_cpu_flags();
    ucs_log_early_init(); /* Must be called before all others */
    ucs_global_opts_init();
    ucs_log_init();
#if ENABLE_STATS
    ucs_stats_init();
#endif
    ucs_memtrack_init();
    ucs_debug_init();
    ucs_profile_global_init();
    ucs_async_global_init();
    ucs_debug("%s loaded at 0x%lx", ucs_debug_get_lib_path(),
              ucs_debug_get_lib_base_addr());
    ucs_debug("cmd line: %s", ucs_get_process_cmdline());
}

static void UCS_F_DTOR ucs_cleanup(void)
{
    ucs_config_parser_warn_unused_env_vars();
    ucs_async_global_cleanup();
    ucs_profile_global_cleanup();
    ucs_debug_cleanup();
    ucs_memtrack_cleanup();
#if ENABLE_STATS
    ucs_stats_cleanup();
#endif
    ucs_log_cleanup();
}
