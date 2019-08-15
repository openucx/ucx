/**
 * Copyright (C) Mellanox Technologies Ltd.      2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <dlfcn.h>

void* load_lib(const char *path, void* (*load_func)(const char*, int))
{
    return (load_func ? load_func : dlopen)(path, RTLD_NOW);
}
