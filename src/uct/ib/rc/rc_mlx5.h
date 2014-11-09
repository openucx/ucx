/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_RC_MLX5
#define UCT_RC_MLX5

#include "rc_ep.h"

typedef struct {
    uct_rc_ep_t      super;
    unsigned         qpn_ds;

    struct {
        unsigned       sw_pi;
        unsigned       hw_ci;
        void           *seg;
        void           *bf_reg;
        unsigned long  bf_size;
        uint32_t       *dbrec;
        void           *qstart;
        void           *qend;
    } tx;
} uct_rc_mlx5_ep_t;



#endif
