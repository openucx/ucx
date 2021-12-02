/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx;

import org.junit.Test;
import org.openucx.jucx.ucp.*;
import org.openucx.jucx.ucs.UcsConstants;

import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.Assert.*;

public class UcpWorkerTest extends UcxTest {
    private static int numWorkers = Runtime.getRuntime().availableProcessors();

    @Test
    public void testSingleWorker() throws Exception {
        UcpContext context = new UcpContext(new UcpParams().requestTagFeature());
        assertEquals(2, UcsConstants.ThreadMode.UCS_THREAD_MODE_MULTI);
        assertNotEquals(context.getNativeId(), null);
        UcpWorker worker = context.newWorker(new UcpWorkerParams());
        assertNotNull(worker.getNativeId());
        assertEquals(0, worker.progress()); // No communications was submitted.
        worker.close();
        assertNull(worker.getNativeId());
        context.close();
    }

    @Test
    public void testMultipleWorkersWithinSameContext() {
        UcpContext context = new UcpContext(new UcpParams().requestTagFeature());
        assertNotEquals(context.getNativeId(), null);
        UcpWorker[] workers = new UcpWorker[numWorkers];
        UcpWorkerParams workerParam = new UcpWorkerParams();
        for (int i = 0; i < numWorkers; i++) {
            workerParam.clear().setCpu(i).requestThreadSafety();
            workers[i] = context.newWorker(workerParam);
            assertNotNull(workers[i].getNativeId());
        }
        for (int i = 0; i < numWorkers; i++) {
            workers[i].close();
        }
        context.close();
    }

    @Test
    public void testMultipleWorkersFromMultipleContexts() {
        UcpContext tcpContext = new UcpContext(new UcpParams().requestTagFeature());
        UcpContext rdmaContext = new UcpContext(new UcpParams().requestRmaFeature()
            .requestAtomic64BitFeature().requestAtomic32BitFeature());
        UcpWorker[] workers = new UcpWorker[numWorkers];
        UcpWorkerParams workerParams = new UcpWorkerParams();
        for (int i = 0; i < numWorkers; i++) {
            ByteBuffer userData = ByteBuffer.allocateDirect(100);
            workerParams.clear();
            if (i % 2 == 0) {
                userData.asCharBuffer().put("TCPWorker" + i);
                workerParams.requestWakeupRX().setUserData(userData);
                workers[i] = tcpContext.newWorker(workerParams);
            } else {
                userData.asCharBuffer().put("RDMAWorker" + i);
                workerParams.requestWakeupRMA().setCpu(i).setUserData(userData)
                    .requestThreadSafety();
                workers[i] = rdmaContext.newWorker(workerParams);
            }
        }
        for (int i = 0; i < numWorkers; i++) {
            workers[i].close();
        }
        tcpContext.close();
        rdmaContext.close();
    }

    @Test
    public void testGetWorkerAddress() {
        UcpContext context = new UcpContext(new UcpParams().requestTagFeature());
        UcpWorker worker = context.newWorker(new UcpWorkerParams());
        ByteBuffer workerAddress = worker.getAddress();
        assertNotNull(workerAddress);
        assertTrue(workerAddress.capacity() > 0);
        worker.close();
        context.close();
    }

    @Test
    public void testWorkerSleepWakeup() throws InterruptedException {
        UcpContext context = new UcpContext(new UcpParams()
            .requestRmaFeature().requestWakeupFeature());
        UcpWorker worker = context.newWorker(
            new UcpWorkerParams().requestWakeupRMA());

        AtomicBoolean success = new AtomicBoolean(false);
        Thread workerProgressThread = new Thread() {
            @Override
            public void run() {
                while (!isInterrupted()) {
                    try {
                        if (worker.progress() == 0) {
                            worker.waitForEvents();
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                success.set(true);
            }
        };

        workerProgressThread.start();

        workerProgressThread.interrupt();
        worker.signal();

        workerProgressThread.join();
        assertTrue(success.get());

        worker.close();
        context.close();
    }

    @Test
    public void testFlushWorker() throws Exception {
        int numRequests = 10;
        // Create 2 contexts + 2 workers
        UcpParams params = new UcpParams().requestRmaFeature();
        UcpWorkerParams rdmaWorkerParams = new UcpWorkerParams().requestWakeupRMA();
        UcpContext context1 = new UcpContext(params);
        UcpContext context2 = new UcpContext(params);

        ByteBuffer src = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);
        ByteBuffer dst = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);
        dst.asCharBuffer().put(UcpMemoryTest.RANDOM_TEXT);
        UcpMemory memory = context2.registerMemory(src);

        UcpWorker worker1 = context1.newWorker(rdmaWorkerParams);
        UcpWorker worker2 = context2.newWorker(rdmaWorkerParams);

        UcpEndpoint ep = worker1.newEndpoint( new UcpEndpointParams()
            .setUcpAddress(worker2.getAddress()).setPeerErrorHandlingMode());
        UcpRemoteKey rkey = ep.unpackRemoteKey(memory.getRemoteKeyBuffer());

        int blockSize = UcpMemoryTest.MEM_SIZE / numRequests;
        for (int i = 0; i < numRequests; i++) {
            ep.putNonBlockingImplicit(UcxUtils.getAddress(dst) + i * blockSize,
                blockSize, memory.getAddress() + i * blockSize, rkey);
        }

        UcpRequest request = worker1.flushNonBlocking(new UcxCallback() {
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

        assertTrue(request.isCompleted());
        Collections.addAll(resources, context1, context2, worker1, worker2, ep);
        closeResources();
    }

    @Test
    public void testTagProbe() throws Exception {
        UcpParams params = new UcpParams().requestTagFeature();
        UcpContext context1 = new UcpContext(params);
        UcpContext context2 = new UcpContext(params);

        UcpWorker worker1 = context1.newWorker(new UcpWorkerParams());
        UcpWorker worker2 = context2.newWorker(new UcpWorkerParams());
        ByteBuffer recvBuffer = ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE);

        UcpTagMessage message = worker1.tagProbeNonBlocking(0, 0, false);

        assertNull(message);

        UcpEndpoint endpoint = worker2.newEndpoint(
            new UcpEndpointParams().setUcpAddress(worker1.getAddress()));

        endpoint.sendTaggedNonBlocking(
            ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE), null);

        do {
            worker1.progress();
            worker2.progress();
            message = worker1.tagProbeNonBlocking(0, 0, true);
        } while (message == null);

        assertEquals(UcpMemoryTest.MEM_SIZE, message.getRecvLength());
        assertEquals(0, message.getSenderTag());

        UcpRequest recv = worker1.recvTaggedMessageNonBlocking(recvBuffer, message, null);

        worker1.progressRequest(recv);

        Collections.addAll(resources, context1, context2, worker1, worker2, endpoint);
    }
}
