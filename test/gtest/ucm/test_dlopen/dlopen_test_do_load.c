/**
 * Copyright (C) Mellanox Technologies Ltd.      2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucs/sys/compiler.h>

#include <dlfcn.h>

UCS_F_NOOPTIMIZE /* prevent using tail recursion unwind */
void* load_lib(const char *path, void* (*load_func)(const char*, int))
{
    return (load_func ? load_func : dlopen)(path, RTLD_NOW);
}
