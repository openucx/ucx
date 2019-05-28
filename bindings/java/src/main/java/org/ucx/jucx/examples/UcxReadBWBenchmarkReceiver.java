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

public class UcxReadBWBenchmarkReceiver {

    private static long bytesInGigabit = 1000L * 1000L * 1000L / 8;

    private static double getBandwith(long nanoTimeDelta, long size) {
        double sizeInGigabits = (double)size / bytesInGigabit;
        double secondsElapsed = nanoTimeDelta / 1_000_000_000.0;
        return sizeInGigabits / secondsElapsed;
    }

    public static void main(String[] args) throws IOException {
        Map<String, String> argsMap = new HashMap<>();
        for (String arg: args) {
            String[] parts = arg.split("=");
            argsMap.put(parts[0], parts[1]);
        }

        String serverHost = argsMap.get("s");
        int serverPort = Integer.parseInt(argsMap.getOrDefault("p", "55443"));
        int numIterations = Integer.parseInt(argsMap.getOrDefault("n", "5"));

        Stack<Closeable> resources = new Stack<>();

        UcpContext context = new UcpContext(new UcpParams().requestWakeupFeature()
            .requestRmaFeature().requestTagFeature());
        resources.push(context);
        UcpWorker worker = context.newWorker(new UcpWorkerParams());
        resources.push(worker);
        UcpListener listener = worker.newListener(
            new UcpListenerParams().setSockAddr(
                new InetSocketAddress(serverHost, serverPort)));
        resources.push(listener);

        ByteBuffer recvBuffer = ByteBuffer.allocateDirect(4096);
        UcxRequest recvRequest = worker.recvTaggedNonBlocking(recvBuffer, null);

        System.out.println("Waiting for connections ...");

        while (!recvRequest.isCompleted()) {
            worker.progress();
        }

        long remoteAddress = recvBuffer.getLong();
        long remoteSize = recvBuffer.getInt();
        int remoteKeySize = recvBuffer.getInt();
        ByteBuffer rkey = ByteBuffer.allocateDirect(remoteKeySize);
        for (int i = 0; i < remoteKeySize; i++) {
            rkey.put(recvBuffer.get());
        }
        int workerAdressSize = recvBuffer.getInt();
        ByteBuffer workerAddress = ByteBuffer.allocateDirect(workerAdressSize);
        for (int i = 0; i < workerAdressSize; i++) {
            workerAddress.put(recvBuffer.get());
        }
        int remoteHashCode = recvBuffer.getInt();
        System.out.printf("Received connection. Will read %d bytes from remote address %d\n",
            remoteSize, remoteAddress);

        UcpEndpoint endpoint = worker.newEndpoint(
            new UcpEndpointParams().setUcpAddress(workerAddress).setPeerErrorHadnlingMode());

        UcpRemoteKey remoteKey = endpoint.unpackRemoteKey(rkey);
        resources.push(remoteKey);
        // In java ByteBuffer can be allocated up to 2GB (int max size).
        // To get ByteBuffer of size > 2GB need to mmap file.
        ByteBuffer data = ByteBuffer.allocateDirect((int)remoteSize);

        for (int i = 0; i < numIterations; i++) {
            final int iterNum = i;
            UcxRequest getRequest = endpoint.getNonBlocking(remoteAddress, remoteKey, data,
                new UcxCallback() {
                    long startTime = System.nanoTime();

                    @Override
                    public void onSuccess(UcxRequest request) {
                        long finishTime = System.nanoTime();
                        data.clear();
                        assert data.hashCode() == remoteHashCode;
                        double bandwith = getBandwith(finishTime - startTime, remoteSize);
                        System.out.printf("Iteration %d, bandwith: %.4f GB/s\n", iterNum, bandwith);
                    }
                });

            while (!getRequest.isCompleted()) {
                worker.progress();
            }
            // To make sure we receive correct data each time to compare hashCodes
            data.put(0, (byte)1);
        }

        ByteBuffer sendBuffer = ByteBuffer.allocateDirect(100);
        sendBuffer.asCharBuffer().put("DONE");
        UcxRequest sent = endpoint.sendTaggedNonBlocking(sendBuffer,null);

        while (!sent.isCompleted()) {
            worker.progress();
        }

        while (!resources.empty()) {
            resources.pop().close();
        }
    }
}
