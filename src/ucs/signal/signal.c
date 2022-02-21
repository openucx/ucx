/*
 * Copyright (C) 2022, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#define __USE_GNU /* for sighandler_t */

#include <ucs/debug/debug_int.h>

#include <signal.h>
#include <dlfcn.h>


#if !HAVE_SIGHANDLER_T
#if HAVE___SIGHANDLER_T
typedef __sighandler_t *sighandler_t;
#else
#error "Port me"
#endif
#endif


static void *ucs_debug_get_orig_func(const char *symbol)
{
    void *func_ptr;

    func_ptr = dlsym(RTLD_NEXT, symbol);
    if (func_ptr == NULL) {
        func_ptr = dlsym(RTLD_DEFAULT, symbol);
    }
    return func_ptr;
}

static sighandler_t ucs_orig_signal(int signum, sighandler_t handler)
{
    typedef sighandler_t (*sighandler_func_t)(int, sighandler_t);

    static sighandler_func_t orig = NULL;

    if (orig == NULL) {
        orig = (sighandler_func_t)ucs_debug_get_orig_func("signal");
    }

    return orig(signum, handler);
}

sighandler_t signal(int signum, sighandler_t handler)
{
    if (ucs_debug_is_error_signal(signum)) {
        return SIG_DFL;
    }

    return ucs_orig_signal(signum, handler);
}

int ucs_orig_sigaction(int signum, const struct sigaction *act,
                        struct sigaction *oact)
{
    typedef int (*sigaction_func_t)(int, const struct sigaction*, struct sigaction*);

    static sigaction_func_t orig = NULL;

    if (orig == NULL) {
        orig = (sigaction_func_t)ucs_debug_get_orig_func("sigaction");
    }

    return orig(signum, act, oact);
}

int sigaction(int signum, const struct sigaction *act, struct sigaction *oact)
{
    if (ucs_debug_is_error_signal(signum)) {
        return ucs_orig_sigaction(signum, NULL, oact); /* Return old, do not set new */
    }

    return ucs_orig_sigaction(signum, act, oact);
}

