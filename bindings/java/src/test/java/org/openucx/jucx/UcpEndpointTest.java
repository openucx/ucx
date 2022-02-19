/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx;

import org.junit.Test;
import org.junit.experimental.theories.DataPoints;
import org.junit.experimental.theories.Theories;
import org.junit.experimental.theories.Theory;
import org.junit.runner.RunWith;
import org.openucx.jucx.ucp.*;
import org.openucx.jucx.ucs.UcsConstants;

import java.nio.ByteBuffer;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.*;

@RunWith(Theories.class)
public class UcpEndpointTest extends UcxTest {

    @DataPoints
    public static ArrayList<Integer> memTypes() {
        ArrayList<Integer> resut = new ArrayList<>();
        resut.add(UcsConstants.MEMORY_TYPE.UCS_MEMORY_TYPE_HOST);
        UcpContext testContext = new UcpContext(new UcpParams().requestTagFeature());
        long memTypeMask = testContext.getMemoryTypesMask();
        if (UcsConstants.MEMORY_TYPE.isMemTypeSupported(memTypeMask,
            UcsConstants.MEMORY_TYPE.UCS_MEMORY_TYPE_CUDA)) {
            resut.add(UcsConstants.MEMORY_TYPE.UCS_MEMORY_TYPE_CUDA);
        }
        if (UcsConstants.MEMORY_TYPE.isMemTypeSupported(memTypeMask,
            UcsConstants.MEMORY_TYPE.UCS_MEMORY_TYPE_CUDA_MANAGED)) {
            resut.add(UcsConstants.MEMORY_TYPE.UCS_MEMORY_TYPE_CUDA_MANAGED);
        }
        return resut;
    }

    @Test
    public void testConnectToListenerByWorkerAddr() {
        UcpContext context = new UcpContext(new UcpParams().requestStreamFeature());
        UcpWorker worker = context.newWorker(new UcpWorkerParams());
        UcpEndpointParams epParams = new UcpEndpointParams().setUcpAddress(worker.getAddress())
            .setPeerErrorHandlingMode().setNoLoopbackMode()
            .setName("testConnectToListenerByWorkerAddr");
        UcpEndpoint endpoint = worker.newEndpoint(epParams);
        assertNotNull(endpoint.getNativeId());

        Collections.addAll(resources, context, worker, endpoint);
        closeResources();
    }

    @Theory
    public void testGetNB(int memType) throws Exception {
        System.out.println("Running testGetNB with memType: " + memType);
        // Create 2 contexts + 2 workers
        UcpParams params = new UcpParams().requestRmaFeature().requestTagFeature();
        UcpWorkerParams rdmaWorkerParams = new UcpWorkerParams().requestWakeupRMA();
        UcpContext context1 = new UcpContext(params);
        UcpContext context2 = new UcpContext(params);
        UcpWorker worker1 = context1.newWorker(rdmaWorkerParams);
        UcpWorker worker2 = context2.newWorker(rdmaWorkerParams);

        // Create endpoint worker1 -> worker2
        UcpEndpointParams epParams = new UcpEndpointParams().setPeerErrorHandlingMode()
            .setName("testGetNB").setUcpAddress(worker2.getAddress());
        UcpEndpoint endpoint = worker1.newEndpoint(epParams);

        // Allocate 2 source and 2 destination buffers, to perform 2 RDMA Read operations
        MemoryBlock src1 = allocateMemory(context2, worker2, memType, UcpMemoryTest.MEM_SIZE);
        MemoryBlock src2 = allocateMemory(context2, worker2, memType, UcpMemoryTest.MEM_SIZE);
        MemoryBlock dst1 = allocateMemory(context1, worker1, memType, UcpMemoryTest.MEM_SIZE);
        MemoryBlock dst2 = allocateMemory(context1, worker1, memType, UcpMemoryTest.MEM_SIZE);

        src1.setData(UcpMemoryTest.RANDOM_TEXT);
        src2.setData(UcpMemoryTest.RANDOM_TEXT + UcpMemoryTest.RANDOM_TEXT);

        // Register source buffers on context2
        UcpMemory memory1 = src1.getMemory();
        UcpMemory memory2 = src2.getMemory();

        UcpRemoteKey rkey1 = endpoint.unpackRemoteKey(memory1.getRemoteKeyBuffer());
        UcpRemoteKey rkey2 = endpoint.unpackRemoteKey(memory2.getRemoteKeyBuffer());

        AtomicInteger numCompletedRequests = new AtomicInteger(0);

        UcxCallback callback = new UcxCallback() {
            @Override
            public void onSuccess(UcpRequest request) {
                numCompletedRequests.incrementAndGet();
            }
        };

        // Submit 2 get requests
        UcpRequest request1 = endpoint.getNonBlocking(memory1.getAddress(), rkey1,
            dst1.getMemory().getAddress(), dst1.getMemory().getLength(), callback,
            new UcpRequestParams().setMemoryHandle(memory1).setMemoryType(memType));
        UcpRequest request2 = endpoint.getNonBlocking(memory2.getAddress(), rkey2,
            dst2.getMemory().getAddress(), dst2.getMemory().getLength(), callback,
            new UcpRequestParams().setMemoryHandle(memory2).setMemoryType(memType));

        // Wait for 2 get operations to complete
        while (numCompletedRequests.get() != 2) {
            worker1.progress();
            worker2.progress();
        }

        assertEquals(src1.getData().asCharBuffer(), dst1.getData().asCharBuffer());
        assertEquals(src2.getData().asCharBuffer(), dst2.getData().asCharBuffer());
        assertTrue(request1.isCompleted() && request2.isCompleted());

        Collections.addAll(resources, context2, context1, worker2, worker1, endpoint, rkey2,
            rkey1, src1, src2, dst1, dst2);
        closeResources();
    }

    @Test
    public void testPutNB() throws Exception {
        // Create 2 contexts + 2 workers
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
        ep.putNonBlocking(src, memory.getAddress(), rkey, null);

        worker1.progressRequest(worker1.flushNonBlocking(null));

        assertEquals(UcpMemoryTest.RANDOM_TEXT, dst.asCharBuffer().toString().trim());

        Collections.addAll(resources, context2, context1, worker2, worker1, rkey, ep, memory);
        closeResources();
    }

    @Theory
    public void testSendRecv(int memType) throws Exception {
        System.out.println("Running testSendRecv with memType: " + memType);
        long tagSender = 0xFFFFFFFFFFFF0000L;
        // Create 2 contexts + 2 workers
        UcpParams params = new UcpParams().requestRmaFeature().requestTagFeature();
        UcpWorkerParams rdmaWorkerParams = new UcpWorkerParams().requestWakeupRMA();
        UcpContext context1 = new UcpContext(params.setTagSenderMask(tagSender));
        UcpContext context2 = new UcpContext(params);
        UcpWorker worker1 = context1.newWorker(rdmaWorkerParams);
        UcpWorker worker2 = context2.newWorker(rdmaWorkerParams);

        MemoryBlock src1 = allocateMemory(context1, worker1, memType, UcpMemoryTest.MEM_SIZE);
        MemoryBlock src2 = allocateMemory(context1, worker1, memType, UcpMemoryTest.MEM_SIZE);

        MemoryBlock dst1 = allocateMemory(context2, worker2, memType, UcpMemoryTest.MEM_SIZE);
        MemoryBlock dst2 = allocateMemory(context2, worker2, memType, UcpMemoryTest.MEM_SIZE);

        src1.setData(UcpMemoryTest.RANDOM_TEXT);
        src2.setData(UcpMemoryTest.RANDOM_TEXT + UcpMemoryTest.RANDOM_TEXT);

        AtomicInteger receivedMessages = new AtomicInteger(0);
        worker2.recvTaggedNonBlocking(dst1.getMemory().getAddress(), UcpMemoryTest.MEM_SIZE, 0, 0,
            new UcxCallback() {
                @Override
                public void onSuccess(UcpRequest request) {
                    receivedMessages.incrementAndGet();
                }
            }, new UcpRequestParams().setMemoryType(memType).setMemoryHandle(dst1.getMemory()));

        worker2.recvTaggedNonBlocking(dst2.getMemory().getAddress(), UcpMemoryTest.MEM_SIZE,
            1, tagSender, new UcxCallback() {
                @Override
                public void onSuccess(UcpRequest request) {
                    receivedMessages.incrementAndGet();
                }
            }, new UcpRequestParams().setMemoryType(memType).setMemoryHandle(dst2.getMemory()));

        UcpEndpoint ep = worker1.newEndpoint(new UcpEndpointParams().setName("testSendRecv")
            .setUcpAddress(worker2.getAddress()));

        ep.sendTaggedNonBlocking(src1.getMemory().getAddress(), UcpMemoryTest.MEM_SIZE, 0, null,
            new UcpRequestParams().setMemoryType(memType).setMemoryHandle(src1.getMemory()));
        ep.sendTaggedNonBlocking(src2.getMemory().getAddress(), UcpMemoryTest.MEM_SIZE, 1, null,
            new UcpRequestParams().setMemoryType(memType).setMemoryHandle(src2.getMemory()));

        while (receivedMessages.get() != 2) {
            worker1.progress();
            worker2.progress();
        }

        assertEquals(src1.getData().asCharBuffer(), dst1.getData().asCharBuffer());
        assertEquals(src2.getData().asCharBuffer(), dst2.getData().asCharBuffer());

        Collections.addAll(resources, context2, context1, worker2, worker1, ep,
            src1, src2, dst1, dst2);
        closeResources();
    }

    @Test
    public void testRecvAfterSend() {
        long sendTag = 4L;
        // Create 2 contexts + 2 workers
        UcpParams params = new UcpParams().requestRmaFeature().requestTagFeature()
            .setMtWorkersShared(true);
        UcpWorkerParams rdmaWorkerParams = new UcpWorkerParams().requestWakeupRMA()
            .requestThreadSafety();
        UcpContext context1 = new UcpContext(params);
        UcpContext context2 = new UcpContext(params);
        UcpWorker worker1 = context1.newWorker(rdmaWorkerParams);
        UcpWorker worker2 = context2.newWorker(rdmaWorkerParams);

        UcpEndpoint ep = worker1.newEndpoint(new UcpEndpointParams()
            .setPeerErrorHandlingMode().setName("testRecvAfterSend")
            .setErrorHandler((errEp, status, errorMsg) -> { })
            .setUcpAddress(worker2.getAddress()));

        ByteBuffer src1 = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);
        ByteBuffer dst1 = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);

        ep.sendTaggedNonBlocking(src1, sendTag, null);

        Thread progressThread = new Thread() {
            @Override
            public void run() {
                while (!isInterrupted()) {
                    try {
                        worker1.progress();
                        worker2.progress();
                    } catch (Exception ex) {
                        System.err.println(ex.getMessage());
                        ex.printStackTrace();
                    }
                }
            }
        };

        progressThread.setDaemon(true);
        progressThread.start();

        try {
            Thread.sleep(5);
        } catch (InterruptedException ignored) { }

        UcpRequest recv = worker2.recvTaggedNonBlocking(dst1, 0, 0, new UcxCallback() {
            @Override
            public void onSuccess(UcpRequest request) {
                assertEquals(UcpMemoryTest.MEM_SIZE, request.getRecvSize());
            }
        });

        try {
            int count = 0;
            while ((++count < 100) && !recv.isCompleted()) {
                Thread.sleep(50);
            }
        } catch (InterruptedException ignored) { }

        assertTrue(recv.isCompleted());
        assertEquals(sendTag, recv.getSenderTag());
        UcpRequest closeRequest = ep.closeNonBlockingForce();

        while (!closeRequest.isCompleted()) {
            try {
                // Wait until progress thread will close the endpoint.
                Thread.sleep(10);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        progressThread.interrupt();
        try {
            progressThread.join();
        } catch (InterruptedException ignored) { }

        Collections.addAll(resources, context1, context2, worker1, worker2);
        closeResources();
    }

    @Test
    public void testBufferOffset() throws Exception {
        int msgSize = 200;
        int offset = 100;
        // Create 2 contexts + 2 workers
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

        Collections.addAll(resources, context2, context1, worker2, worker1, ep);
        closeResources();
    }

    @Test
    public void testFlushEp() throws Exception {
        int numRequests = 10;
        // Create 2 contexts + 2 workers
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
            .setUcpAddress(worker2.getAddress()).setPeerErrorHandlingMode());
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

        while (!request.isCompleted()) {
            worker1.progress();
            worker2.progress();
        }

        Collections.addAll(resources, context2, context1, worker2, worker1, ep);
        closeResources();
    }

    @Test
    public void testRecvSize() throws Exception {
        UcpContext context1 = new UcpContext(new UcpParams().requestTagFeature());
        UcpContext context2 = new UcpContext(new UcpParams().requestTagFeature());

        UcpWorker worker1 = context1.newWorker(new UcpWorkerParams());
        UcpWorker worker2 = context2.newWorker(new UcpWorkerParams());

        UcpEndpoint ep = worker1.newEndpoint(
            new UcpEndpointParams().setUcpAddress(worker2.getAddress()));

        ByteBuffer sendBuffer = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);
        ByteBuffer recvBuffer = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);

        sendBuffer.limit(UcpMemoryTest.MEM_SIZE / 2);

        UcpRequest send = ep.sendTaggedNonBlocking(sendBuffer, null);
        UcpRequest recv = worker2.recvTaggedNonBlocking(recvBuffer, null);

        while (!send.isCompleted() || !recv.isCompleted()) {
            worker1.progress();
            worker2.progress();
        }

        assertEquals(UcpMemoryTest.MEM_SIZE / 2, recv.getRecvSize());

        Collections.addAll(resources, context1, context2, worker1, worker2, ep);
        closeResources();
    }

    @Test
    public void testStreamingAPI() throws Exception {
        UcpParams params = new UcpParams().requestStreamFeature().requestRmaFeature();
        UcpContext context1 = new UcpContext(params);
        UcpContext context2 = new UcpContext(params);

        UcpWorker worker1 = context1.newWorker(new UcpWorkerParams());
        UcpWorker worker2 = context2.newWorker(new UcpWorkerParams());

        UcpEndpoint clientToServer = worker1.newEndpoint(
            new UcpEndpointParams().setUcpAddress(worker2.getAddress()));

        UcpEndpoint serverToClient = worker2.newEndpoint(
            new UcpEndpointParams().setUcpAddress(worker1.getAddress()));

        ByteBuffer sendBuffer = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);
        sendBuffer.put(0, (byte) 1);
        ByteBuffer recvBuffer = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE * 2);

        UcpRequest[] sends = new UcpRequest[2];

        sends[0] = clientToServer.sendStreamNonBlocking(sendBuffer, new UcxCallback() {
            @Override
            public void onSuccess(UcpRequest request) {
                sendBuffer.put(0, (byte)2);
                sends[1] = clientToServer.sendStreamNonBlocking(sendBuffer, null);
            }
        });

        while (sends[1] == null || !sends[1].isCompleted()) {
            worker1.progress();
            worker2.progress();
        }

        AtomicBoolean received = new AtomicBoolean(false);
        serverToClient.recvStreamNonBlocking(
            UcxUtils.getAddress(recvBuffer), UcpMemoryTest.MEM_SIZE * 2L,
            UcpConstants.UCP_STREAM_RECV_FLAG_WAITALL,
            new UcxCallback() {
                @Override
                public void onSuccess(UcpRequest request) {
                    assertEquals(request.getRecvSize(), UcpMemoryTest.MEM_SIZE * 2);
                    assertEquals((byte)1, recvBuffer.get(0));
                    assertEquals((byte)2, recvBuffer.get(UcpMemoryTest.MEM_SIZE));
                    received.set(true);
                }
            });

        while (!received.get()) {
            worker1.progress();
            worker2.progress();
        }

        Collections.addAll(resources, context1, context2, worker1, worker2, clientToServer,
            serverToClient);
        closeResources();
    }

    @Test
    public void testIovOperations() throws Exception {
        int NUM_IOV = 6;
        long buffMultiplier = 10L;

        UcpMemMapParams memMapParams = new UcpMemMapParams().allocate();
        // Create 2 contexts + 2 workers
        UcpParams params = new UcpParams().requestTagFeature().requestStreamFeature();
        UcpWorkerParams workerParams = new UcpWorkerParams();
        UcpContext context1 = new UcpContext(params);
        UcpContext context2 = new UcpContext(params);
        UcpWorker worker1 = context1.newWorker(workerParams);
        UcpWorker worker2 = context2.newWorker(workerParams);

        UcpEndpoint ep = worker1.newEndpoint(
            new UcpEndpointParams().setUcpAddress(worker2.getAddress()));

        UcpEndpoint recvEp = worker2.newEndpoint(new UcpEndpointParams()
            .setUcpAddress(worker1.getAddress()));

        UcpMemory[] sendBuffers = new UcpMemory[NUM_IOV];
        long[] sendAddresses = new long[NUM_IOV];
        long[] sizes = new long[NUM_IOV];

        UcpMemory[] recvBuffers = new UcpMemory[NUM_IOV];
        long[] recvAddresses = new long[NUM_IOV];

        long totalSize = 0L;

        for (int i = 0; i < NUM_IOV; i++) {
            long bufferSize = (i + 1) * buffMultiplier;
            totalSize += bufferSize;
            memMapParams.setLength(bufferSize);

            sendBuffers[i] = context1.memoryMap(memMapParams);
            sendAddresses[i] = sendBuffers[i].getAddress();
            sizes[i] = bufferSize;

            ByteBuffer buf = UcxUtils.getByteBufferView(sendAddresses[i], (int)bufferSize);
            buf.putInt(0, (i + 1));

            recvBuffers[i] = context2.memoryMap(memMapParams);
            recvAddresses[i] = recvBuffers[i].getAddress();
        }

        ep.sendTaggedNonBlocking(sendAddresses, sizes, 0L, null);
        UcpRequest recv = worker2.recvTaggedNonBlocking(recvAddresses, sizes, 0L, 0L, null);

        while (!recv.isCompleted()) {
            worker1.progress();
            worker2.progress();
        }

        assertEquals(totalSize, recv.getRecvSize());

        for (int i = 0; i < NUM_IOV; i++) {
            ByteBuffer buf = UcxUtils.getByteBufferView(recvAddresses[i], (int)sizes[i]);
            assertEquals((i + 1), buf.getInt(0));
            recvBuffers[i].deregister();
        }

        // Test 6 send IOV to 3 recv IOV
        recvBuffers = new UcpMemory[NUM_IOV / 2];
        recvAddresses = new long[NUM_IOV / 2];
        long[] recvSizes = new long[NUM_IOV / 2];
        totalSize = 0L;

        for (int i = 0; i < NUM_IOV / 2; i++) {
            long bufferLength = (i + 1) * buffMultiplier * 2;
            totalSize += bufferLength;
            recvBuffers[i] = context2.memoryMap(memMapParams.setLength(bufferLength));
            recvAddresses[i] = recvBuffers[i].getAddress();
            recvSizes[i] = bufferLength;
        }

        ep.sendStreamNonBlocking(sendAddresses, sizes, null);
        recv = recvEp.recvStreamNonBlocking(recvAddresses, recvSizes, 0, null);

        while (!recv.isCompleted()) {
            worker1.progress();
            worker2.progress();
        }

        assertEquals(totalSize, recv.getRecvSize());
        ByteBuffer buf = UcxUtils.getByteBufferView(recvAddresses[0], (int)recvSizes[0]);
        assertEquals(1, buf.getInt(0));

        Collections.addAll(resources, context1, context2, worker1, worker2, ep);
        Collections.addAll(resources, sendBuffers);
        Collections.addAll(resources, recvBuffers);
        closeResources();
    }

    @Test
    public void testEpErrorHandler() throws Exception {
        // Create 2 contexts + 2 workers
        UcpParams params = new UcpParams().requestTagFeature();
        UcpWorkerParams workerParams = new UcpWorkerParams();
        UcpContext context1 = new UcpContext(params);
        UcpContext context2 = new UcpContext(params);
        UcpWorker worker1 = context1.newWorker(workerParams);
        UcpWorker worker2 = context2.newWorker(workerParams);

        ByteBuffer src = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);
        ByteBuffer dst = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);
        src.asCharBuffer().put(UcpMemoryTest.RANDOM_TEXT);

        AtomicBoolean errorHandlerCalled = new AtomicBoolean(false);
        UcpEndpointParams epParams = new UcpEndpointParams()
            .setPeerErrorHandlingMode()
            .setErrorHandler((ep, status, errorMsg) -> {
                errorHandlerCalled.set(true);
                assertNotNull(errorMsg);
            })
            .setUcpAddress(worker2.getAddress());
        UcpEndpoint ep =
            worker1.newEndpoint(epParams);

        UcpRequest recv = worker2.recvTaggedNonBlocking(dst, null);
        UcpRequest send = ep.sendTaggedNonBlocking(src, null);

        while (!send.isCompleted() || !recv.isCompleted()) {
            worker1.progress();
            worker2.progress();
        }

        // Closing receiver worker & context
        worker2.close();
        context2.close();
        assertNull(context2.getNativeId());

        AtomicBoolean errorCallabackCalled = new AtomicBoolean(false);

        ep.sendTaggedNonBlocking(src, null);
        worker1.progressRequest(ep.flushNonBlocking(new UcxCallback() {
            @Override
            public void onError(int ucsStatus, String errorMsg) {
                errorCallabackCalled.set(true);
            }
        }));

        assertTrue(errorHandlerCalled.get());
        assertTrue(errorCallabackCalled.get());

        ep.close();
        worker1.close();
        context1.close();
    }

    @Theory
    public void testActiveMessages(int memType) throws Exception {
        System.out.println("Running testActiveMessages with memType: " + memType);
        UcpParams params = new UcpParams().requestAmFeature().requestTagFeature();
        UcpContext context1 = new UcpContext(params);
        UcpContext context2 = new UcpContext(params);

        UcpWorker worker1 = context1.newWorker(new UcpWorkerParams());
        UcpWorker worker2 = context2.newWorker(new UcpWorkerParams());

        String headerString = "Hello";
        String dataString = "Active messages";
        long headerSize = headerString.length() * 2;
        long dataSize = UcpMemoryTest.MEM_SIZE;
        assertTrue(headerSize < worker1.getMaxAmHeaderSize());

        ByteBuffer header = ByteBuffer.allocateDirect((int) headerSize);
        header.asCharBuffer().append(headerString);

        header.rewind();

        MemoryBlock sendData = allocateMemory(context2, worker2, memType, dataSize);
        sendData.setData(dataString);

        MemoryBlock recvData = allocateMemory(context1, worker1, memType, dataSize);
        MemoryBlock recvEagerData = allocateMemory(context1, worker1, memType, dataSize);
        ByteBuffer recvHeader = ByteBuffer.allocateDirect((int) headerSize);
        UcpRequest[] requests = new UcpRequest[7];

        UcpEndpoint ep = worker2.newEndpoint(
            new UcpEndpointParams().setUcpAddress(worker1.getAddress()));

        Set<UcpEndpoint> cachedEp = new HashSet<>();

        // Test rndv flow
        worker1.setAmRecvHandler(0, (headerAddress, headerSize12, amData, replyEp) -> {
            assertFalse(amData.isDataValid());
            try {
                assertEquals(headerString,
                    UcxUtils.getByteBufferView(headerAddress, (int) headerSize12)
                        .asCharBuffer().toString().trim());
            } catch (Exception e) {
                e.printStackTrace();
            }

            requests[2] = replyEp.sendTaggedNonBlocking(header, null);
            requests[3] = amData.receive(recvData.getMemory().getAddress(), null);

            if (!cachedEp.isEmpty()) {
                assertTrue(cachedEp.contains(replyEp));
            } else {
                cachedEp.add(replyEp);
            }

            return UcsConstants.STATUS.UCS_OK;
        }, UcpConstants.UCP_AM_FLAG_WHOLE_MSG);

        // Test eager flow
        worker1.setAmRecvHandler(1, (headerAddress, headerSize1, amData, replyEp) -> {
            assertTrue(amData.isDataValid());
            try {
                assertEquals(dataString,
                    UcxUtils.getByteBufferView(amData.getDataAddress(), (int) amData.getLength())
                        .asCharBuffer().toString().trim());
            } catch (Exception e) {
                e.printStackTrace();
            }

            if (!cachedEp.isEmpty()) {
                assertTrue(cachedEp.contains(replyEp));
            } else {
                cachedEp.add(replyEp);
            }

            requests[6] = amData.receive(recvEagerData.getMemory().getAddress(), null);

            return UcsConstants.STATUS.UCS_OK;
        }, UcpConstants.UCP_AM_FLAG_WHOLE_MSG);

        AtomicReference<UcpAmData> persistantAmData = new AtomicReference<>(null);
        // Test amData persistence flow
        worker1.setAmRecvHandler(2, (headerAddress, headerSize1, amData, replyEp) -> {
            assertTrue(amData.isDataValid());
            assertTrue(amData.canPersist());
            persistantAmData.set(amData);
            return UcsConstants.STATUS.UCS_INPROGRESS;
        }, UcpConstants.UCP_AM_FLAG_WHOLE_MSG | UcpConstants.UCP_AM_FLAG_PERSISTENT_DATA);

        requests[0] = ep.sendAmNonBlocking(0,
            UcxUtils.getAddress(header), headerSize,
            sendData.getMemory().getAddress(), sendData.getMemory().getLength(),
            UcpConstants.UCP_AM_SEND_FLAG_REPLY | UcpConstants.UCP_AM_SEND_FLAG_RNDV,
            new UcxCallback() {
                @Override
                public void onSuccess(UcpRequest request) {
                    assertTrue(request.isCompleted());
                }
            }, new UcpRequestParams().setMemoryType(memType)
                .setMemoryHandle(sendData.getMemory()));

        requests[1] = worker2.recvTaggedNonBlocking(recvHeader, null);
        requests[4] = ep.sendAmNonBlocking(1, 0L, 0L,
            sendData.getMemory().getAddress(), dataSize,
            UcpConstants.UCP_AM_SEND_FLAG_REPLY | UcpConstants.UCP_AM_SEND_FLAG_EAGER, null,
            new UcpRequestParams().setMemoryType(memType)
                .setMemoryHandle(sendData.getMemory()));

        // Persistence data flow
        requests[5] = ep.sendAmNonBlocking(2, 0L, 0L,
            sendData.getMemory().getAddress(), 2L, UcpConstants.UCP_AM_FLAG_PERSISTENT_DATA, null,
            new UcpRequestParams().setMemoryType(memType).setMemoryHandle(sendData.getMemory()));

        while (!Arrays.stream(requests).allMatch(r -> (r != null) && r.isCompleted())) {
            worker1.progress();
            worker2.progress();
        }

        assertEquals(dataString,
            recvData.getData().asCharBuffer().toString().trim());

        assertEquals(dataString,
            recvEagerData.getData().asCharBuffer().toString().trim());

        assertEquals(headerString,
            recvHeader.asCharBuffer().toString().trim());

        assertEquals(dataString.charAt(0),
            UcxUtils.getByteBufferView(persistantAmData.get().getDataAddress(),
                persistantAmData.get().getLength()).getChar(0));
        persistantAmData.get().close();
        persistantAmData.set(null);

        // Reset AM callback
        worker1.removeAmRecvHandler(0);
        worker1.removeAmRecvHandler(1);
        worker1.removeAmRecvHandler(2);

        Collections.addAll(resources, context1, context2, worker1, worker2, ep,
            cachedEp.iterator().next(), sendData, recvData, recvEagerData);
        closeResources();
        cachedEp.clear();
    }
}
