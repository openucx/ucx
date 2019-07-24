/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx.ucp;

import org.openucx.jucx.UcxException;
import org.openucx.jucx.UcxParams;

import java.net.InetSocketAddress;
import java.nio.ByteBuffer;

import static org.openucx.jucx.ucp.UcpConstants.*;

/**
 * Tuning parameters for the UCP endpoint.
 */
public class UcpEndpointParams extends UcxParams {

    @Override
    public UcpEndpointParams clear() {
        super.clear();
        ucpAddress = null;
        errorHandlingMode = 0;
        userData = null;
        flags = 0;
        socketAddress = null;
        return this;
    }

    private ByteBuffer ucpAddress;

    private int errorHandlingMode;

    private ByteBuffer userData;

    private long flags;

    private InetSocketAddress socketAddress;

    /**
     * Destination address in form of workerAddress.
     */
    public UcpEndpointParams setUcpAddress(ByteBuffer ucpAddress) {
        this.fieldMask |= UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
        this.ucpAddress = ucpAddress;
        return this;
    }

    /**
     * Guarantees that send requests are always completed (successfully or error) even in
     * case of remote failure, disables protocols and APIs which may cause a hang or undefined
     * behavior in case of peer failure, may affect performance and memory footprint.
     */
    public UcpEndpointParams setPeerErrorHadnlingMode() {
        this.fieldMask |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
        this.errorHandlingMode = UCP_ERR_HANDLING_MODE_PEER;
        return this;
    }

    /**
     * User data associated with an endpoint.
     */
    public UcpEndpointParams setUserData(ByteBuffer userData) {
        if (!userData.isDirect()) {
            throw new UcxException("User data must be of type DirectByteBuffer.");
        }
        this.fieldMask |= UCP_EP_PARAM_FIELD_USER_DATA;
        this.userData = userData;
        return this;
    }

    /**
     * Destination address in form of InetSocketAddress.
     */
    public UcpEndpointParams setSocketAddress(InetSocketAddress socketAddress) {
        this.fieldMask |= UCP_EP_PARAM_FIELD_SOCK_ADDR | UCP_EP_PARAM_FIELD_FLAGS;
        this.flags |= UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
        this.socketAddress = socketAddress;
        return this;
    }

    /**
     * Avoid connecting the endpoint to itself when connecting the endpoint
     * to the same worker it was created on. Affects protocols which send to a particular
     * remote endpoint, for example stream.
     */
    public UcpEndpointParams setNoLoopbackMode() {
        this.fieldMask |= UCP_EP_PARAM_FIELD_FLAGS;
        this.flags |= UCP_EP_PARAMS_FLAGS_NO_LOOPBACK;
        return this;
    }
}
