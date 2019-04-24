/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx;

import org.junit.Test;

import org.ucx.jucx.ucp.*;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;

import static java.nio.file.StandardOpenOption.*;
import static org.junit.Assert.*;

public class UcpMemoryTest {
    public static int MEM_SIZE = 4096;
    public static String RANDOM_TEXT = UUID.randomUUID().toString();

    @Test
    public void testMmapFile() throws IOException {
        UcpContext context = new UcpContext(new UcpParams().requestTagFeature());
        Path tempFile = Files.createTempFile("jucx", "test");
        // 1. Create FileChannel to file in tmp directory.
        FileChannel fileChannel = FileChannel.open(tempFile, CREATE, WRITE, READ, DELETE_ON_CLOSE);
        ByteBuffer data = ByteBuffer.allocateDirect(MEM_SIZE);
        // 2. Put to file some random text.
        data.asCharBuffer().put(RANDOM_TEXT);
        fileChannel.write(data);
        fileChannel.force(true);
        // 3. MMap file channel.
        MappedByteBuffer buf = fileChannel.map(FileChannel.MapMode.READ_WRITE, 0, MEM_SIZE);
        buf.load();
        // 4. Make sure content of mmaped file has needed data
        assertEquals(RANDOM_TEXT, buf.asCharBuffer().toString().trim());
        // 5. Register mmaped buffer with UCX
        UcpMemory mem = context.registerMemory(buf);
        ByteBuffer memBuffer = mem.getData();
        // 6. Make sure registered buffer is the same as mapped buffer.
        assertEquals(RANDOM_TEXT, memBuffer.asCharBuffer().toString().trim());
        mem.deregister();
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
        rkey.close();
        mem.deregister();
        endpoint.close();
        worker1.close();
        worker2.close();
        context.close();
    }
}
