/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_CM_H_
#define UCT_CM_H_

#include <uct/api/uct_def.h>
#include <uct/base/uct_md.h>


/**
 * Connection manager component operations
 */
typedef struct uct_cm_ops {
    void         (*close)(uct_cm_h cm);
} uct_cm_ops_t;


struct uct_cm {
    uct_cm_ops_t       *ops;
    uct_md_component_t *component;
};

#endif /* UCT_CM_H_ */
