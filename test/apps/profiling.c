/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#define HAVE_PROFILING 1
#include <ucs/debug/profile.h>

#include <stdio.h>

UCS_PROFILE_FUNC_VOID(my_func, ()) {
    UCS_PROFILE_CALL(printf, "Hello World!\n");
}

int main(int argc, char **argv)
{
    my_func();
    return 0;
}
