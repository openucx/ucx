/*
 * Copyright (C) Mellanox Technologies Ltd. 2020. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.openucx.jucx.ucp;

/**
 * Handler to process transport level failure.
 */
public interface UcpEndpointErrorHandler {
    /**
     * This callback routine is invoked when transport level error detected.
     * @param ep - Endpoint to handle transport level error. Upon return
     *             from the callback, this endpoint is no longer usable and
     *             all subsequent operations on this ep will fail with
     *             the error code passed in {@code status}.
     */
    void onError(UcpEndpoint ep, int status, String errorMsg) throws Exception;
}
