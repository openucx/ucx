/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import org.ucx.jucx.ucp.UcpContext;
import org.ucx.jucx.ucp.UcpMemory;
import org.ucx.jucx.ucp.UcpParams;

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
    private static int MEM_SIZE = 4096;
    private static String RANDOM_TEXT = UUID.randomUUID().toString();

    @Test
    public void testContextMemalloc() {
        UcpContext context = new UcpContext(new UcpParams().requestTagFeature());
        UcpMemory mem = context.allocateMemory(MEM_SIZE);
        assertNotNull(mem.getNativeId());
        assertEquals(mem.getData().capacity(), MEM_SIZE);
        mem.getData().asCharBuffer().put(RANDOM_TEXT);
        assertEquals(mem.getData().asCharBuffer().toString().trim(), RANDOM_TEXT);
        mem.free();
        assertNull(mem.getData());
        context.close();
    }

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
        mem.free();
        fileChannel.close();
        context.close();
    }

    @Test
    public void testGetRkey() {
        UcpContext context = new UcpContext(new UcpParams().requestRmaFeature());
        UcpMemory mem = context.allocateMemory(MEM_SIZE);
        ByteBuffer rkeyBuffer = mem.getrKeyBuffer();
        assertTrue(rkeyBuffer.capacity() > 0);
        mem.free();
        context.close();
    }
}
