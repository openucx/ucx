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
    UCT_AF_INFINIBAND,            /**< Infiniband address */
    UCT_AF_INFINIBAND_SUBNET,     /**< Infiniband subnet address */
    UCT_AF_UGNI,                  /**< Cray Gemini address */
    UCT_AF_MAX
};


typedef struct uct_sockaddr_process {
    UCT_SOCKADDR_COMMON (sp_);
    uint64_t   node_guid;
    uint64_t   id;
    uintptr_t  vaddr;
} UCS_S_PACKED uct_sockaddr_process_t;


typedef struct uct_sockaddr_ib {
    UCT_SOCKADDR_COMMON (sib_);
    uint16_t   lid;
    uint32_t   qp_num;
    uint64_t   subnet_prefix;
    uint64_t   guid;
    uint32_t   id;
} UCS_S_PACKED uct_sockaddr_ib_t;


typedef struct uct_sockaddr_ib_subnet {
    UCT_SOCKADDR_COMMON (sib_);
    uint64_t   subnet_prefix;
} UCS_S_PACKED uct_sockaddr_ib_subnet_t;


typedef struct uct_sockaddr_ugni {
    UCT_SOCKADDR_COMMON (sgni_);
    uint32_t   nic_addr;
    uint32_t   domain_id;
} UCS_S_PACKED uct_sockaddr_ugni_t;


#endif
