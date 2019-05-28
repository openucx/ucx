/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx.examples;

import org.ucx.jucx.UcxCallback;
import org.ucx.jucx.UcxRequest;
import org.ucx.jucx.ucp.*;

import java.io.Closeable;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

public class UcxReadBWBenchmarkSender {

    public static void main(String[] args) throws IOException {
        Map<String, String> argsMap = new HashMap<>();
        for (String arg: args) {
            String[] parts = arg.split("=");
            argsMap.put(parts[0], parts[1]);
        }

        String serverHost = argsMap.get("s");
        int serverPort = Integer.parseInt(argsMap.get("p"));
        long totalSize = Long.parseLong(argsMap.get("t"));

        Stack<Closeable> resources = new Stack<>();

        UcpContext context = new UcpContext(new UcpParams().requestWakeupFeature()
            .requestRmaFeature().requestTagFeature());
        resources.push(context);
        UcpWorker worker = context.newWorker(new UcpWorkerParams());
        resources.push(worker);
        UcpEndpoint endpoint = worker.newEndpoint(new UcpEndpointParams()
            .setSocketAddress(new InetSocketAddress(serverHost, serverPort)));
        resources.push(endpoint);

        // In java ByteBuffer can be allocated up to 2GB (int max size).

        if (totalSize >= Integer.MAX_VALUE) {
            throw new IOException("Max size must be no greater then " + Integer.MAX_VALUE);
        }
        ByteBuffer data = ByteBuffer.allocateDirect((int)totalSize);
        byte b = Byte.MIN_VALUE;
        while (data.hasRemaining()) {
            data.put(b++);
        }
        data.clear();

        // Register allocated buffer
        UcpMemory memory = context.registerMemory(data);

        // Send worker and memory address and Rkey to receiver.
        ByteBuffer rkeyBuffer = memory.getRemoteKeyBuffer();
        ByteBuffer workerAddress = worker.getAddress();

        ByteBuffer sendData = ByteBuffer.allocateDirect(24 + rkeyBuffer.capacity() +
            workerAddress.capacity());
        sendData.putLong(memory.getAddress());
        sendData.putInt((int)totalSize);
        sendData.putInt(rkeyBuffer.capacity());
        sendData.put(rkeyBuffer);
        sendData.putInt(workerAddress.capacity());
        sendData.put(workerAddress);
        sendData.putInt(data.hashCode());

        endpoint.sendTaggedNonBlocking(sendData, null);

        ByteBuffer recvBuffer = ByteBuffer.allocateDirect(4096);
        UcxRequest recvRequest = worker.recvTaggedNonBlocking(recvBuffer,
            new UcxCallback() {
                @Override
                public void onSuccess(UcxRequest request) {
                    System.out.println("Received a message:");
                    System.out.println(recvBuffer.asCharBuffer().toString());
                }
            });

        while (!recvRequest.isCompleted()) {
            worker.progress();
        }

        memory.deregister();
        while (!resources.empty()) {
            resources.pop().close();
        }
    }
}
