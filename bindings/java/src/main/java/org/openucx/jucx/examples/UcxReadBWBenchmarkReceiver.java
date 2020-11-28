/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx.examples;

import org.openucx.jucx.UcxCallback;
import org.openucx.jucx.ucp.UcpRequest;
import org.openucx.jucx.UcxUtils;
import org.openucx.jucx.ucp.*;


import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicReference;

public class UcxReadBWBenchmarkReceiver extends UcxBenchmark {

    public static void main(String[] args) throws Exception {
        if (!initializeArguments(args)) {
            return;
        }

        createContextAndWorker();

        String serverHost = argsMap.get("s");
        InetSocketAddress sockaddr = new InetSocketAddress(serverHost, serverPort);
        AtomicReference<UcpConnectionRequest> connRequest = new AtomicReference<>(null);
        UcpListener listener = worker.newListener(
            new UcpListenerParams()
                .setConnectionHandler(connRequest::set)
                .setSockAddr(sockaddr));
        resources.push(listener);
        System.out.println("Waiting for connections on " + sockaddr + " ...");

        while (connRequest.get() == null) {
            worker.progress();
        }

        UcpEndpoint endpoint = worker.newEndpoint(new UcpEndpointParams()
            .setConnectionRequest(connRequest.get())
            .setPeerErrorHandlingMode());

        ByteBuffer recvBuffer = ByteBuffer.allocateDirect(4096);
        UcpRequest recvRequest = worker.recvTaggedNonBlocking(recvBuffer, null);

        worker.progressRequest(recvRequest);

        long remoteAddress = recvBuffer.getLong();
        long remoteSize = recvBuffer.getLong();
        int remoteKeySize = recvBuffer.getInt();
        int rkeyBufferOffset = recvBuffer.position();

        recvBuffer.position(rkeyBufferOffset + remoteKeySize);
        int remoteHashCode = recvBuffer.getInt();
        System.out.printf("Received connection. Will read %d bytes from remote address %d%n",
            remoteSize, remoteAddress);

        recvBuffer.position(rkeyBufferOffset);
        UcpRemoteKey remoteKey = endpoint.unpackRemoteKey(recvBuffer);
        resources.push(remoteKey);

        UcpMemory recvMemory = context.memoryMap(allocationParams);
        resources.push(recvMemory);
        ByteBuffer data = UcxUtils.getByteBufferView(recvMemory.getAddress(),
            (int)Math.min(Integer.MAX_VALUE, totalSize));
        for (int i = 0; i < numIterations; i++) {
            final int iterNum = i;
            UcpRequest getRequest = endpoint.getNonBlocking(remoteAddress, remoteKey,
                recvMemory.getAddress(), remoteSize,
                new UcxCallback() {
                    final long startTime = System.nanoTime();

                    @Override
                    public void onSuccess(UcpRequest request) {
                        long finishTime = System.nanoTime();
                        data.clear();
                        assert data.hashCode() == remoteHashCode;
                        double bw = getBandwithGbits(finishTime - startTime, remoteSize);
                        System.out.printf("Iteration %d, bandwidth: %.4f GB/s%n", iterNum, bw);
                    }
                });

            worker.progressRequest(getRequest);
            // To make sure we receive correct data each time to compare hashCodes
            data.put(0, (byte)1);
        }

        UcpRequest closeRequest = endpoint.closeNonBlockingFlush();
        worker.progressRequest(closeRequest);
        // Close request won't be return to pull automatically, since there's no callback.
        resources.push(closeRequest);

        closeResources();
    }
}
