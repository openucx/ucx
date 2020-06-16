/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx.ucp;

import org.openucx.jucx.UcxParams;

import java.net.InetSocketAddress;
import java.nio.ByteBuffer;

/**
 * Tuning parameters for the UCP endpoint.
 */
public class UcpEndpointParams extends UcxParams {

    @Override
    public String toString() {
        String result = "UcpEndpointParams{";
        if (ucpAddress != null) {
            result += "ucpAddress,";
        }
        result += "errorHandlingMode="
            + ((errorHandlingMode == 0) ? "UCP_ERR_HANDLING_MODE_NONE," :
                                          "UCP_ERR_HANDLING_MODE_PEER,");

        if (socketAddress != null) {
            result += "socketAddress=" + socketAddress.toString() + ",";
        }

        if (connectionRequest != 0) {
            result += "connectionRequest,";
        }
        return result;
    }

    @Override
    public UcpEndpointParams clear() {
        super.clear();
        ucpAddress = null;
        errorHandlingMode = 0;
        flags = 0;
        socketAddress = null;
        connectionRequest = 0;
        errorHandler = null;
        return this;
    }

    private ByteBuffer ucpAddress;

    private int errorHandlingMode;

    private long flags;

    private InetSocketAddress socketAddress;

    private long connectionRequest;

    UcpEndpointErrorHandler errorHandler;

    /**
     * Destination address in form of workerAddress.
     */
    public UcpEndpointParams setUcpAddress(ByteBuffer ucpAddress) {
        this.fieldMask |= UcpConstants.UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
        this.ucpAddress = ucpAddress;
        return this;
    }

    /**
     * Guarantees that send requests are always completed (successfully or error) even in
     * case of remote failure, disables protocols and APIs which may cause a hang or undefined
     * behavior in case of peer failure, may affect performance and memory footprint.
     */
    public UcpEndpointParams setPeerErrorHandlingMode() {
        this.fieldMask |= UcpConstants.UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
        this.errorHandlingMode = UcpConstants.UCP_ERR_HANDLING_MODE_PEER;
        return this;
    }

    /**
     * Destination address in form of InetSocketAddress.
     */
    public UcpEndpointParams setSocketAddress(InetSocketAddress socketAddress) {
        this.fieldMask |= UcpConstants.UCP_EP_PARAM_FIELD_SOCK_ADDR |
                          UcpConstants.UCP_EP_PARAM_FIELD_FLAGS;
        this.flags |= UcpConstants.UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
        this.socketAddress = socketAddress;
        return this;
    }

    /**
     * Avoid connecting the endpoint to itself when connecting the endpoint
     * to the same worker it was created on. Affects protocols which send to a particular
     * remote endpoint, for example stream.
     */
    public UcpEndpointParams setNoLoopbackMode() {
        this.fieldMask |= UcpConstants.UCP_EP_PARAM_FIELD_FLAGS;
        this.flags |= UcpConstants.UCP_EP_PARAMS_FLAGS_NO_LOOPBACK;
        return this;
    }

    /**
     * Connection request from client.
     */
    public UcpEndpointParams setConnectionRequest(UcpConnectionRequest connectionRequest) {
        this.fieldMask |= UcpConstants.UCP_EP_PARAM_FIELD_CONN_REQUEST;
        this.connectionRequest = connectionRequest.getNativeId();
        return this;
    }

    /**
     * Handler to process transport level failure.
     */
    public UcpEndpointParams setErrorHandler(UcpEndpointErrorHandler errorHandler) {
        this.fieldMask |= UcpConstants.UCP_EP_PARAM_FIELD_ERR_HANDLER;
        this.errorHandler = errorHandler;
        return this;
    }
}
