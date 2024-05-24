/**
* Copyright (C) Tactical Computing Labs, LLC. 2022. ALL RIGHTS RESERVED.
* Copyright (C) Dandan Zhang, 2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_LOONGARCH64_GLOBAL_OPTS_H_
#define UCS_LOONGARCH64_GLOBAL_OPTS_H_

#include <stddef.h>

#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS

#define UCS_ARCH_GLOBAL_OPTS_INITALIZER {}

/* built-in memcpy config */
typedef struct ucs_arch_global_opts {
    char dummy;
} ucs_arch_global_opts_t;

END_C_DECLS

#endif
