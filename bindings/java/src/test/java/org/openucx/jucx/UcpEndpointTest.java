/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx;

import org.junit.Test;
import org.openucx.jucx.ucp.*;

import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.NetworkInterface;
import java.net.SocketException;
import java.nio.ByteBuffer;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

public class UcpEndpointTest {
    @Test
    public void testConnectToListenerByWorkerAddr() {
        UcpContext context = new UcpContext(new UcpParams().requestStreamFeature());
        UcpWorker worker = context.newWorker(new UcpWorkerParams());
        UcpEndpointParams epParams = new UcpEndpointParams().setUcpAddress(worker.getAddress())
            .setPeerErrorHadnlingMode().setNoLoopbackMode();
        UcpEndpoint endpoint = worker.newEndpoint(epParams);
        assertNotNull(endpoint.getNativeId());

        endpoint.close();
        worker.close();
        context.close();
    }

    @Test
    public void testConnectToListenerBySocketAddr() throws SocketException {
        UcpContext context = new UcpContext(new UcpParams().requestStreamFeature());
        UcpWorker worker = context.newWorker(new UcpWorkerParams());
        // Iterate over each network interface - got it's sockaddr - try to instantiate listener
        // And pass this sockaddr to endpoint.
        Enumeration<NetworkInterface> interfaces = NetworkInterface.getNetworkInterfaces();
        boolean success = false;
        while (!success && interfaces.hasMoreElements()) {
            NetworkInterface networkInterface = interfaces.nextElement();
            Enumeration<InetAddress> inetAddresses = networkInterface.getInetAddresses();
            while (inetAddresses.hasMoreElements()) {
                InetAddress inetAddress = inetAddresses.nextElement();
                if (!inetAddress.isLoopbackAddress()) {
                    try {
                        InetSocketAddress addr = new InetSocketAddress(inetAddress,
                            UcpListenerTest.port);
                        UcpListener ucpListener = worker.newListener(
                            new UcpListenerParams().setSockAddr(addr));
                        UcpEndpointParams epParams =
                            new UcpEndpointParams().setSocketAddress(addr);
                        UcpEndpoint endpoint = worker.newEndpoint(epParams);
                        assertNotNull(endpoint.getNativeId());
                        success = true;
                        endpoint.close();
                        ucpListener.close();
                        break;
                    } catch (UcxException ex) {

                    }
                }
            }
        }

        worker.close();
        context.close();
    }

    @Test
    public void testGetNB() {
        // Crerate 2 contexts + 2 workers
        UcpParams params = new UcpParams().requestRmaFeature();
        UcpWorkerParams rdmaWorkerParams = new UcpWorkerParams().requestWakeupRMA();
        UcpContext context1 = new UcpContext(params);
        UcpContext context2 = new UcpContext(params);
        UcpWorker worker1 = context1.newWorker(rdmaWorkerParams);
        UcpWorker worker2 = context2.newWorker(rdmaWorkerParams);

        // Create endpoint worker1 -> worker2
        UcpEndpointParams epParams = new UcpEndpointParams().setPeerErrorHadnlingMode()
            .setUcpAddress(worker2.getAddress());
        UcpEndpoint endpoint = worker1.newEndpoint(epParams);

        // Allocate 2 source and 2 destination buffers, to perform 2 RDMA Read operations
        ByteBuffer src1 = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);
        ByteBuffer src2 = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);
        ByteBuffer dst1 = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);
        ByteBuffer dst2 = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);
        src1.asCharBuffer().put(UcpMemoryTest.RANDOM_TEXT);
        src2.asCharBuffer().put(UcpMemoryTest.RANDOM_TEXT + UcpMemoryTest.RANDOM_TEXT);

        // Register source buffers on context2
        UcpMemory memory1 = context2.registerMemory(src1);
        UcpMemory memory2 = context2.registerMemory(src2);

        UcpRemoteKey rkey1 = endpoint.unpackRemoteKey(memory1.getRemoteKeyBuffer());
        UcpRemoteKey rkey2 = endpoint.unpackRemoteKey(memory2.getRemoteKeyBuffer());

        AtomicInteger numCompletedRequests = new AtomicInteger(0);
        HashMap<UcpRequest, ByteBuffer> requestToData = new HashMap<>();
        UcxCallback callback = new UcxCallback() {
            @Override
            public void onSuccess(UcpRequest request) {
                // Here thread safety is guaranteed since worker progress is called after
                // request added to map. In multithreaded environment could be an issue that
                // callback is called, but request wasn't added yet to map.
                if (requestToData.get(request) == dst1) {
                    assertEquals(UcpMemoryTest.RANDOM_TEXT, dst1.asCharBuffer().toString().trim());
                    memory1.deregister();
                } else {
                    assertEquals(UcpMemoryTest.RANDOM_TEXT + UcpMemoryTest.RANDOM_TEXT,
                        dst2.asCharBuffer().toString().trim());
                    memory2.deregister();
                }
                numCompletedRequests.incrementAndGet();
            }
        };

        // Submit 2 get requests
        UcpRequest request1 = endpoint.getNonBlocking(memory1.getAddress(), rkey1, dst1, callback);
        UcpRequest request2 = endpoint.getNonBlocking(memory2.getAddress(), rkey2, dst2, callback);

        // Map each request to corresponding data buffer.
        requestToData.put(request1, dst1);
        requestToData.put(request2, dst2);

        // Wait for 2 get operations to complete
        while (numCompletedRequests.get() != 2) {
            worker1.progress();
        }

        assertTrue(request1.isCompleted() && request2.isCompleted());

        rkey1.close();
        rkey2.close();
        endpoint.close();
        worker1.close();
        worker2.close();
        context1.close();
        context2.close();
    }

    @Test
    public void testPutNB() {
        // Crerate 2 contexts + 2 workers
        UcpParams params = new UcpParams().requestRmaFeature();
        UcpWorkerParams rdmaWorkerParams = new UcpWorkerParams().requestWakeupRMA();
        UcpContext context1 = new UcpContext(params);
        UcpContext context2 = new UcpContext(params);
        UcpWorker worker1 = context1.newWorker(rdmaWorkerParams);
        UcpWorker worker2 = context2.newWorker(rdmaWorkerParams);

        ByteBuffer src = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);
        ByteBuffer dst = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);
        src.asCharBuffer().put(UcpMemoryTest.RANDOM_TEXT);

        // Register destination buffer on context2
        UcpMemory memory = context2.registerMemory(dst);
        UcpEndpoint ep =
            worker1.newEndpoint(new UcpEndpointParams().setUcpAddress(worker2.getAddress()));

        UcpRemoteKey rkey = ep.unpackRemoteKey(memory.getRemoteKeyBuffer());
        UcpRequest request = ep.putNonBlocking(src, memory.getAddress(), rkey,
            new UcxCallback() {
                @Override
                public void onSuccess(UcpRequest request) {
                    rkey.close();
                    memory.deregister();
                }
            });

        worker1.progressRequest(request);

        assertEquals(dst.asCharBuffer().toString().trim(), UcpMemoryTest.RANDOM_TEXT);

        ep.close();
        worker1.close();
        worker2.close();
        context1.close();
        context2.close();
    }

    @Test
    public void testSendRecv() throws Exception {
        // Crerate 2 contexts + 2 workers
        UcpParams params = new UcpParams().requestRmaFeature().requestTagFeature();
        UcpWorkerParams rdmaWorkerParams = new UcpWorkerParams().requestWakeupRMA();
        UcpContext context1 = new UcpContext(params);
        UcpContext context2 = new UcpContext(params);
        UcpWorker worker1 = context1.newWorker(rdmaWorkerParams);
        UcpWorker worker2 = context2.newWorker(rdmaWorkerParams);

        // Allocate 2 source and 2 destination buffers, to perform 2 RDMA Read operations
        UcpMemMapParams allocationParams = new UcpMemMapParams().allocate()
            .setLength(UcpMemoryTest.MEM_SIZE);
        UcpMemory memory1 = context1.memoryMap(allocationParams);
        UcpMemory memory2 = context1.memoryMap(allocationParams);
        ByteBuffer src1 = UcxUtils.getByteBufferView(memory1.getAddress(), UcpMemoryTest.MEM_SIZE);
        ByteBuffer src2 = UcxUtils.getByteBufferView(memory1.getAddress(), UcpMemoryTest.MEM_SIZE);
        ByteBuffer dst1 = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);
        ByteBuffer dst2 = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);
        src1.asCharBuffer().put(UcpMemoryTest.RANDOM_TEXT);
        src2.asCharBuffer().put(UcpMemoryTest.RANDOM_TEXT + UcpMemoryTest.RANDOM_TEXT);

        AtomicInteger receivedMessages = new AtomicInteger(0);
        worker2.recvTaggedNonBlocking(dst1, 0, 0, new UcxCallback() {
            @Override
            public void onSuccess(UcpRequest request) {
                assertEquals(dst1, src1);
                receivedMessages.incrementAndGet();
            }
        });

        worker2.recvTaggedNonBlocking(dst2, 1, -1, new UcxCallback() {
            @Override
            public void onSuccess(UcpRequest request) {
                assertEquals(dst2, src2);
                receivedMessages.incrementAndGet();
            }
        });

        UcpEndpoint ep = worker1.newEndpoint(new UcpEndpointParams()
            .setUcpAddress(worker2.getAddress()));

        ep.sendTaggedNonBlocking(src1, 0, null);
        ep.sendTaggedNonBlocking(src2, 1, null);

        while (receivedMessages.get() != 2) {
            worker1.progress();
            worker2.progress();
        }

        ep.close();
        memory1.deregister();
        memory2.deregister();
        worker1.close();
        worker2.close();
        context1.close();
        context2.close();
    }

    @Test
    public void testRecvAfterSend() {
        // Crerate 2 contexts + 2 workers
        UcpParams params = new UcpParams().requestRmaFeature().requestTagFeature()
            .setMtWorkersShared(true);
        UcpWorkerParams rdmaWorkerParams = new UcpWorkerParams().requestWakeupRMA()
            .requestThreadSafety();
        UcpContext context1 = new UcpContext(params);
        UcpContext context2 = new UcpContext(params);
        UcpWorker worker1 = context1.newWorker(rdmaWorkerParams);
        UcpWorker worker2 = context2.newWorker(rdmaWorkerParams);

        UcpEndpoint ep = worker1.newEndpoint(new UcpEndpointParams()
            .setPeerErrorHadnlingMode()
            .setUcpAddress(worker2.getAddress()));

        ByteBuffer src1 = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);
        ByteBuffer dst1 = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);

        ep.sendTaggedNonBlocking(src1, 0, null);

        Thread progressThread = new Thread() {
            @Override
            public void run() {
                while (!isInterrupted()) {
                    worker1.progress();
                    worker2.progress();
                }
            }
        };

        progressThread.setDaemon(true);
        progressThread.start();

        try {
            Thread.sleep(5);
        } catch (InterruptedException e) {

        }

        AtomicBoolean success = new AtomicBoolean(false);

        worker2.recvTaggedNonBlocking(dst1, 0, -1, new UcxCallback() {
            @Override
            public void onSuccess(UcpRequest request) {
                success.set(true);
            }
        });

        try {
            int count = 0;
            while ((++count < 100) && !success.get()) {
                Thread.sleep(50);
            }
        } catch (InterruptedException e) {

        }

        assertTrue(success.get());

        progressThread.interrupt();
        try {
            progressThread.join();
        } catch (InterruptedException e) {

        }

        UcpRequest close = ep.closeNonBlockingForce();
        while (!close.isCompleted()) {
            worker1.progress();
            worker2.progress();
        }

        worker2.close();
        worker1.close();
        context2.close();
        context1.close();
    }

    @Test
    public void testBufferOffset() {
        int msgSize = 200;
        int offset = 100;
        // Crerate 2 contexts + 2 workers
        UcpParams params = new UcpParams().requestTagFeature();
        UcpWorkerParams rdmaWorkerParams = new UcpWorkerParams().requestWakeupRMA();
        UcpContext context1 = new UcpContext(params);
        UcpContext context2 = new UcpContext(params);
        UcpWorker worker1 = context1.newWorker(rdmaWorkerParams);
        UcpWorker worker2 = context2.newWorker(rdmaWorkerParams);

        ByteBuffer bigRecvBuffer = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);
        ByteBuffer bigSendBuffer = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);

        bigRecvBuffer.position(offset).limit(offset + msgSize);
        UcpRequest recv = worker1.recvTaggedNonBlocking(bigRecvBuffer, 0,
            0, null);

        UcpEndpoint ep = worker2.newEndpoint(new UcpEndpointParams()
            .setUcpAddress(worker1.getAddress()));

        byte[] msg = new byte[msgSize];
        for (int i = 0; i < msgSize; i++) {
            msg[i] = (byte)i;
        }

        bigSendBuffer.position(offset).limit(offset + msgSize);
        bigSendBuffer.put(msg);
        bigSendBuffer.position(offset);

        UcpRequest sent = ep.sendTaggedNonBlocking(bigSendBuffer, 0, null);

        while (!sent.isCompleted() || !recv.isCompleted()) {
            worker1.progress();
            worker2.progress();
        }

        bigSendBuffer.position(offset).limit(offset + msgSize);
        bigRecvBuffer.position(offset).limit(offset + msgSize);
        final ByteBuffer sendData = bigSendBuffer.slice();
        final ByteBuffer recvData = bigRecvBuffer.slice();
        assertEquals("Send buffer not equals to recv buffer", sendData, recvData);

        ep.close();
        worker1.close();
        worker2.close();
        context1.close();
        context2.close();
    }

    @Test
    public void testFlushEp() {
        int numRequests = 10;
        // Crerate 2 contexts + 2 workers
        UcpParams params = new UcpParams().requestRmaFeature();
        UcpWorkerParams rdmaWorkerParams = new UcpWorkerParams().requestWakeupRMA();
        UcpContext context1 = new UcpContext(params);
        UcpContext context2 = new UcpContext(params);

        ByteBuffer src = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);
        src.asCharBuffer().put(UcpMemoryTest.RANDOM_TEXT);
        ByteBuffer dst = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);
        UcpMemory memory = context2.registerMemory(src);

        UcpWorker worker1 = context1.newWorker(rdmaWorkerParams);
        UcpWorker worker2 = context2.newWorker(rdmaWorkerParams);

        UcpEndpoint ep = worker1.newEndpoint(new UcpEndpointParams()
            .setUcpAddress(worker2.getAddress()).setPeerErrorHadnlingMode());
        UcpRemoteKey rkey = ep.unpackRemoteKey(memory.getRemoteKeyBuffer());

        int blockSize = UcpMemoryTest.MEM_SIZE / numRequests;
        for (int i = 0; i < numRequests; i++) {
            ep.getNonBlockingImplicit(memory.getAddress() + i * blockSize, rkey,
                UcxUtils.getAddress(dst) + i * blockSize, blockSize);
        }

        UcpRequest request = ep.flushNonBlocking(new UcxCallback() {
            @Override
            public void onSuccess(UcpRequest request) {
                rkey.close();
                memory.deregister();
                assertEquals(dst.asCharBuffer().toString().trim(), UcpMemoryTest.RANDOM_TEXT);
            }
        });

        while (request.isCompleted()) {
            worker1.progress();
            worker2.progress();
        }

        ep.close();
        worker1.close();
        worker2.close();
        context1.close();
        context2.close();
    }
}
