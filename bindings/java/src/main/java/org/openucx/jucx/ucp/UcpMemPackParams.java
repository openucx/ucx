/*
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2023. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.openucx.jucx.ucp;

import org.openucx.jucx.UcxParams;

public class UcpMemPackParams extends UcxParams {
    private long flags;

    @Override
    public UcpMemPackParams clear() {
        super.clear();
        flags = 0;
        return this;
    }

    public UcpMemPackParams exported() {
        this.fieldMask |= UcpConstants.UCP_MEMH_PACK_PARAM_FIELD_FLAGS;
        flags |= UcpConstants.UCP_MEMH_PACK_FLAG_EXPORT;
        return this;
    }
}
