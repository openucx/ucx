/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.openucx.jucx;

import org.junit.Test;

import org.openucx.jucx.ucp.*;
import org.openucx.jucx.ucs.UcsConstants;

import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.UUID;

import static java.nio.file.StandardOpenOption.*;
import static org.junit.Assert.*;
import static org.junit.Assume.assumeTrue;

public class UcpMemoryTest extends UcxTest {
    static int MEM_SIZE = 4096;
    static String RANDOM_TEXT = UUID.randomUUID().toString();

    @Test
    public void testMmapFile() throws Exception {
        UcpContext context = new UcpContext(new UcpParams().requestTagFeature());
        Path tempFile = Files.createTempFile("jucx", "test");
        // 1. Create FileChannel to file in tmp directory.
        FileChannel fileChannel = FileChannel.open(tempFile, CREATE, WRITE, READ, DELETE_ON_CLOSE);
        MappedByteBuffer buf = fileChannel.map(FileChannel.MapMode.READ_WRITE, 0, MEM_SIZE);
        buf.asCharBuffer().put(RANDOM_TEXT);
        buf.force();
        // 2. Register mmap buffer with ODP
        UcpMemory mmapedMemory = context.memoryMap(new UcpMemMapParams()
            .setAddress(UcxUtils.getAddress(buf)).setLength(MEM_SIZE).nonBlocking());

        assertEquals(mmapedMemory.getAddress(), UcxUtils.getAddress(buf));

        // 3. Test allocation
        UcpMemory allocatedMemory = context.memoryMap(new UcpMemMapParams()
            .allocate().setProtection(UcpConstants.UCP_MEM_MAP_PROT_LOCAL_READ)
            .setLength(MEM_SIZE).nonBlocking());
        assertEquals(allocatedMemory.getLength(), MEM_SIZE);

        allocatedMemory.deregister();
        mmapedMemory.deregister();
        fileChannel.close();
        context.close();
    }

    @Test
    public void testGetRkey() {
        UcpContext context = new UcpContext(new UcpParams().requestRmaFeature());
        ByteBuffer buf = ByteBuffer.allocateDirect(MEM_SIZE);
        UcpMemory mem = context.registerMemory(buf);
        ByteBuffer rkeyBuffer = mem.getRemoteKeyBuffer();
        assertTrue(rkeyBuffer.capacity() > 0);
        assertTrue(mem.getAddress() > 0);
        mem.deregister();
        context.close();
    }

    @Test
    public void testRemoteKeyUnpack() {
        UcpContext context = new UcpContext(new UcpParams().requestRmaFeature());
        UcpWorker worker1 = new UcpWorker(context, new UcpWorkerParams());
        UcpWorker worker2 = new UcpWorker(context, new UcpWorkerParams());
        UcpEndpoint endpoint = new UcpEndpoint(worker1,
            new UcpEndpointParams().setUcpAddress(worker2.getAddress()));
        ByteBuffer buf = ByteBuffer.allocateDirect(MEM_SIZE);
        UcpMemory mem = context.registerMemory(buf);
        UcpRemoteKey rkey = endpoint.unpackRemoteKey(mem.getRemoteKeyBuffer());
        assertNotNull(rkey.getNativeId());

        Collections.addAll(resources, context, worker1, worker2, endpoint, mem, rkey);
        closeResources();
    }
}
