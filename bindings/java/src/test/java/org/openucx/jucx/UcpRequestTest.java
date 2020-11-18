/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.openucx.jucx;

import org.junit.Test;
import org.openucx.jucx.ucp.*;

import java.nio.ByteBuffer;
import static org.junit.Assert.*;

public class UcpRequestTest {
    @Test
    public void testCancelRequest() throws Exception {
        UcpContext context = new UcpContext(new UcpParams().requestTagFeature());
        UcpWorker worker = context.newWorker(new UcpWorkerParams());
        UcpRequest recv = worker.recvTaggedNonBlocking(ByteBuffer.allocateDirect(100), null);
        worker.cancelRequest(recv);

        while (!recv.isCompleted()) {
            worker.progress();
        }

        assertTrue(recv.isCompleted());
        assertNull(recv.getNativeId());

        worker.close();
        context.close();
    }
}
