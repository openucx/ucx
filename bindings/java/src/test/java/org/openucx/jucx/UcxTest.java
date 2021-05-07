/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx;

import org.openucx.jucx.ucp.*;
import org.openucx.jucx.ucs.UcsConstants;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Stack;

abstract class UcxTest {
    protected static class MemoryBlock implements Closeable {
        private final UcpMemory memory;
        private UcpEndpoint selfEp;
        private ByteBuffer buffer;
        private final UcpWorker worker;

        protected MemoryBlock(UcpWorker worker, UcpMemory memory) {
            this.memory = memory;
            this.worker = worker;
            if (memory.getMemType() == UcsConstants.MEMORY_TYPE.UCS_MEMORY_TYPE_CUDA) {
                this.selfEp = worker.newEndpoint(
                    new UcpEndpointParams().setUcpAddress(worker.getAddress()));
            } else {
                buffer = UcxUtils.getByteBufferView(memory.getAddress(), memory.getLength());
            }
        }

        public UcpMemory getMemory() {
            return memory;
        }

        public void setData(String data) throws Exception {
            if (memory.getMemType() == UcsConstants.MEMORY_TYPE.UCS_MEMORY_TYPE_CUDA) {
                ByteBuffer srcBuffer = ByteBuffer.allocateDirect(data.length() * 2);
                srcBuffer.asCharBuffer().put(data);
                selfEp.sendTaggedNonBlocking(srcBuffer, 0, null);
                worker.progressRequest(worker.recvTaggedNonBlocking(memory.getAddress(),
                    data.length() * 2L, 0, 0, null));
            } else {
                buffer.asCharBuffer().put(data);
            }
        }

        public ByteBuffer getData() throws Exception {
            if (memory.getMemType() == UcsConstants.MEMORY_TYPE.UCS_MEMORY_TYPE_CUDA) {
                ByteBuffer dstBuffer = ByteBuffer.allocateDirect((int)memory.getLength());
                selfEp.sendTaggedNonBlocking(memory.getAddress(), memory.getLength(), 0, null);
                worker.progressRequest(worker.recvTaggedNonBlocking(dstBuffer, 0L, 0L, null));
                return dstBuffer;
            } else {
                return buffer;
            }
        }

        @Override
        public void close() {
            memory.close();
            if (selfEp != null) {
                selfEp.close();
            }
        }
    }

    // Stack of closable resources (context, worker, etc.) to be closed at the end.
    protected static Stack<Closeable> resources = new Stack<>();

    protected void closeResources() {
        while (!resources.empty()) {
            try {
                resources.pop().close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    protected static MemoryBlock allocateMemory(UcpContext context, UcpWorker worker, int memType,
                                                long length) {
        UcpMemMapParams memMapParams = new UcpMemMapParams().allocate().setLength(length)
            .setMemoryType(memType);
        return new MemoryBlock(worker, context.memoryMap(memMapParams));
    }
}
