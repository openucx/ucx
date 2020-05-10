/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx;

import org.openucx.jucx.ucp.UcpRequest;

/**
 * Callback wrapper to notify successful or failure events from JNI.
 */

public class UcxCallback {
    public void onSuccess(UcpRequest request) {}

    public void onError(int ucsStatus, String errorMsg) {
        throw new UcxException(errorMsg);
    }
}
