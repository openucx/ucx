/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx;

import org.junit.Test;
import org.ucx.jucx.ucp.UcpContext;
import org.ucx.jucx.ucp.UcpParams;
import org.ucx.jucx.ucp.UcpWorker;
import org.ucx.jucx.ucp.UcpWorkerParams;
import org.ucx.jucx.ucs.UcsConstants;

import java.nio.ByteBuffer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

public class UcpWorkerTest {
    static int numWorkers = Runtime.getRuntime().availableProcessors();

    @Test
    public void testSingleWorker() {
        UcpContext context = new UcpContext(new UcpParams().requestTagFeature());
        assertEquals(2, UcsConstants.ThreadMode.UCS_THREAD_MODE_MULTI);
        assertNotEquals(context.getNativeId(), null);
        UcpWorker worker = new UcpWorker(context, new UcpWorkerParams());
        assertNotEquals(worker.getNativeId(), null);
        worker.close();
        assertEquals(worker.getNativeId(), null);
        context.close();
    }

    @Test
    public void testMultipleWorkersWithinSameContext() {
        UcpContext context = new UcpContext(new UcpParams().requestTagFeature());
        assertNotEquals(context.getNativeId(), null);
        UcpWorker workers[] = new UcpWorker[numWorkers];
        UcpWorkerParams workerParam = new UcpWorkerParams();
        for (int i = 0; i < numWorkers; i++) {
            workerParam.clear().setCpu(i).requestThreadSafety();
            workers[i] = new UcpWorker(context, workerParam);
            assertNotEquals(workers[i].getNativeId(), null);
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
        UcpWorker workers[] = new UcpWorker[numWorkers];
        UcpWorkerParams workerParams = new UcpWorkerParams();
        for (int i = 0; i < numWorkers; i++) {
            ByteBuffer userData = ByteBuffer.allocateDirect(100);
            workerParams.clear();
            if (i % 2 == 0) {
                userData.asCharBuffer().put("TCPWorker" + i);
                workerParams.requestWakeupRX().setUserData(userData);
                workers[i] = new UcpWorker(tcpContext, workerParams);
            } else {
                userData.asCharBuffer().put("RDMAWorker" + i);
                workerParams.requestWakeupRMA().setCpu(i).setUserData(userData)
                    .requestThreadSafety();
                workers[i] = new UcpWorker(rdmaContext, workerParams);
            }
        }
        for (int i = 0; i < numWorkers; i++) {
            workers[i].close();
        }
        tcpContext.close();
        rdmaContext.close();
    }
}
