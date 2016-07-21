/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_API_ADDR_H_
#define UCT_API_ADDR_H_

#include <sys/socket.h>

#ifdef __SOCKADDR_COMMON
#  define UCT_SOCKADDR_COMMON              __SOCKADDR_COMMON
#else
#  define UCT_SOCKADDR_COMMON(sa_prefix)   sa_family_t sa_prefix##family
#endif


/*
 * Define additional address families for UCT addresses.
 */
enum {
    UCT_AF_PROCESS = AF_MAX + 1,  /**< Local process address */
    UCT_AF_UGNI,                  /**< Cray Gemini address */
    UCT_AF_MAX
};

#endif
