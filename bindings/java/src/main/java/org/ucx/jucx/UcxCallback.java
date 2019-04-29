/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx;

/**
 * Callback wrapper to notify successful or failure events from JNI.
 */

public class UcxCallback {
    public void onSuccess() {}

    public void onError(int ucsStatus, String errorMsg) {}
}
