/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_API_ADDR_H_
#define UCT_API_ADDR_H_

#include <sys/socket.h>


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
    __SOCKADDR_COMMON (sp_);
    uint64_t   node_guid;
    uint64_t   cookie;
} uct_sockaddr_process_t;


typedef struct uct_sockaddr_ib {
    __SOCKADDR_COMMON (sib_);
    uint16_t   lid;
    uint32_t   qp_num;
    uint64_t   subnet_prefix;
    uint64_t   guid;
    uint32_t   id;
} uct_sockaddr_ib_t;


typedef struct uct_sockaddr_ib_subnet {
    __SOCKADDR_COMMON (sib_);
    uint64_t   subnet_prefix;
} uct_sockaddr_ib_subnet_t;


typedef struct uct_sockaddr_ugni {
    __SOCKADDR_COMMON (sgni_);
    uint32_t   nic_addr;
    uint32_t   domain_id;
} uct_sockaddr_ugni_t;


#endif
