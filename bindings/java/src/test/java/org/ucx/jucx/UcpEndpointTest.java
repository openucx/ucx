/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx;

import org.junit.Test;
import org.ucx.jucx.ucp.*;

import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.NetworkInterface;
import java.net.SocketException;
import java.nio.ByteBuffer;
import java.util.Enumeration;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

public class UcpEndpointTest {
    @Test
    public void testConnectToListenerByWorkerAddr() {
        UcpContext context = new UcpContext(new UcpParams().requestStreamFeature());
        UcpWorker worker = new UcpWorker(context, new UcpWorkerParams());
        UcpEndpointParams epParams = new UcpEndpointParams().setUcpAddress(worker.getAddress())
            .setPeerErrorHadnlingMode().setNoLoopbackMode();
        UcpEndpoint endpoint = new UcpEndpoint(worker, epParams);
        assertNotNull(endpoint.getNativeId());

        endpoint.close();
        worker.close();
        context.close();
    }

    @Test
    public void testConnectToListenerBySocketAddr() throws SocketException {
        UcpContext context = new UcpContext(new UcpParams().requestStreamFeature());
        UcpWorker worker = new UcpWorker(context, new UcpWorkerParams());
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
                        UcpListener ucpListener = new UcpListener(worker,
                            new UcpListenerParams().setSockAddr(addr));
                        UcpEndpointParams epParams =
                            new UcpEndpointParams().setSocketAddress(addr);
                        UcpEndpoint endpoint = new UcpEndpoint(worker, epParams);
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
        assertTrue(success);

        worker.close();
        context.close();
    }

    @Test
    public void testRdmaRead() {
        // Crerate 2 contexts + 2 workers and register
        UcpContext context1 = new UcpContext(new UcpParams().requestRmaFeature());
        UcpContext context2 = new UcpContext(new UcpParams().requestRmaFeature());
        UcpWorker worker1 = new UcpWorker(context1, new UcpWorkerParams().requestWakeupRMA());
        UcpWorker worker2 = new UcpWorker(context2, new UcpWorkerParams().requestWakeupRMA());

        // Create endpoint worker1 -> worker2
        UcpEndpointParams epParams = new UcpEndpointParams().setPeerErrorHadnlingMode()
            .setUcpAddress(worker2.getAddress());
        UcpEndpoint endpoint = new UcpEndpoint(worker1, epParams);

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

        AtomicInteger numCompletions = new AtomicInteger(0);

        UcpRemoteKey rkey1 = endpoint.unpackRemoteKey(memory1.getRemoteKeyBuffer());
        UcpRemoteKey rkey2 = endpoint.unpackRemoteKey(memory2.getRemoteKeyBuffer());

        endpoint.getNonBlocking(memory1.getAddress(), rkey1, dst1, new UcxCallback() {
            @Override
            public void onSuccess() {
                numCompletions.incrementAndGet();
            }
        });

        endpoint.getNonBlocking(memory2.getAddress(), rkey2, dst2, new UcxCallback() {
            @Override
            public void onSuccess() {
                numCompletions.incrementAndGet();
            }
        });

        // Wait for 2 get operations to complete
        while (numCompletions.get() != 2) {
            worker1.progress();
        }

        assertEquals(src1.asCharBuffer().toString().trim(), dst1.asCharBuffer().toString().trim());
        assertEquals(UcpMemoryTest.RANDOM_TEXT + UcpMemoryTest.RANDOM_TEXT,
            dst2.asCharBuffer().toString().trim());


        memory1.deregister();
        memory2.deregister();
        rkey1.close();
        rkey2.close();
        endpoint.close();
        worker1.close();
        worker2.close();
        context1.close();
        context2.close();
    }
}
