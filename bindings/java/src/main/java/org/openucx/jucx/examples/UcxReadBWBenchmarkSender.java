/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx.examples;

import org.openucx.jucx.UcxCallback;
import org.openucx.jucx.ucp.UcpRequest;
import org.openucx.jucx.UcxUtils;
import org.openucx.jucx.ucp.UcpEndpoint;
import org.openucx.jucx.ucp.UcpEndpointParams;
import org.openucx.jucx.ucp.UcpMemory;

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
            .setPeerErrorHadnlingMode()
            .setSocketAddress(new InetSocketAddress(serverHost, serverPort)));

        UcpMemory memory = context.memoryMap(allocationParams);
        ByteBuffer data = UcxUtils.getByteBufferView(memory.getAddress(),
            (int)Math.min(Integer.MAX_VALUE, totalSize));

        // Send worker and memory address and Rkey to receiver.
        ByteBuffer rkeyBuffer = memory.getRemoteKeyBuffer();
        ByteBuffer workerAddress = worker.getAddress();

        ByteBuffer sendData = ByteBuffer.allocateDirect(28 + rkeyBuffer.capacity() +
            workerAddress.capacity());
        sendData.putLong(memory.getAddress());
        sendData.putLong(totalSize);
        sendData.putInt(rkeyBuffer.capacity());
        sendData.put(rkeyBuffer);
        sendData.putInt(workerAddress.capacity());
        sendData.put(workerAddress);
        sendData.putInt(data.hashCode());
        sendData.clear();

        endpoint.sendTaggedNonBlocking(sendData, null);

        ByteBuffer recvBuffer = ByteBuffer.allocateDirect(4096);
        UcpRequest recvRequest = worker.recvTaggedNonBlocking(recvBuffer,
            new UcxCallback() {
                @Override
                public void onSuccess(UcpRequest request) {
                    System.out.println("Received a message:");
                    System.out.println(recvBuffer.asCharBuffer().toString());
                }
            });

        worker.progressRequest(recvRequest);

        UcpRequest close = endpoint.closeNonBlockingFlush();
        worker.progressRequest(close);

        memory.deregister();
        closeResources();
    }
}
