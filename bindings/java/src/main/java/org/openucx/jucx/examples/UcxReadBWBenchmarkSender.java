/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx.examples;

import org.openucx.jucx.UcxException;
import org.openucx.jucx.ucp.*;
import org.openucx.jucx.UcxUtils;
import org.openucx.jucx.ucs.UcsConstants;

import java.net.ConnectException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;


public class UcxReadBWBenchmarkSender extends UcxBenchmark {

    public static void main(String[] args) throws Exception {
        if (!initializeArguments(args)) {
            return;
        }

        createContextAndWorker();

        String serverHost = argsMap.get("s");
        UcpEndpoint endpoint = worker.newEndpoint(new UcpEndpointParams()
            .setPeerErrorHandlingMode()
            .setErrorHandler((ep, status, errorMsg) -> {
                if (status == UcsConstants.STATUS.UCS_ERR_CONNECTION_RESET) {
                    throw new ConnectException(errorMsg);
                } else {
                    throw new UcxException(errorMsg);
                }
            })
            .setSocketAddress(new InetSocketAddress(serverHost, serverPort)));

        UcpMemory memory = context.memoryMap(allocationParams);
        resources.push(memory);
        ByteBuffer data = UcxUtils.getByteBufferView(memory.getAddress(),
            (int)Math.min(Integer.MAX_VALUE, totalSize));

        // Send worker and memory address and Rkey to receiver.
        ByteBuffer rkeyBuffer = memory.getRemoteKeyBuffer();

        // 24b = 8b buffer address + 8b buffer size + 4b rkeyBuffer size + 4b hashCode
        ByteBuffer sendData = ByteBuffer.allocateDirect(24 + rkeyBuffer.capacity());
        sendData.putLong(memory.getAddress());
        sendData.putLong(totalSize);
        sendData.putInt(rkeyBuffer.capacity());
        sendData.put(rkeyBuffer);
        sendData.putInt(data.hashCode());
        sendData.clear();

        // Send memory metadata
        endpoint.sendTaggedNonBlocking(sendData, null);

        // Wait until receiver will finish benchmark
        ByteBuffer recvBuffer = ByteBuffer.allocateDirect(0);
        UcpRequest recvRequest = worker.recvTaggedNonBlocking(recvBuffer, null);
        resources.push(recvRequest);
        Exception cause = null;
        try {
            while (!recvRequest.isCompleted()) {
                if (worker.progress() == 0) {
                    worker.waitForEvents();
                }
            }
        } catch (ConnectException e) {
            cause = e;
        } catch (UcxException e) {
            cause = e;
        }

        // If we could not receive the sync message, the test failed. Otherwise,
        // we can ignore the error.
        if (!recvRequest.isCompleted()) {
            throw new Exception("Receive sync message not completed", cause);
        }

        closeResources(endpoint);
    }
}
