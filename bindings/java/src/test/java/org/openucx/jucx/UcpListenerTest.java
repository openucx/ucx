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
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.Assert.*;

public class UcpListenerTest  extends UcxTest {
    static final int port = Integer.parseInt(
        System.getenv().getOrDefault("JUCX_TEST_PORT", "55321"));

    @Test
    public void testCreateUcpListener() {
        UcpContext context = new UcpContext(new UcpParams().requestStreamFeature());
        UcpWorker worker = context.newWorker(new UcpWorkerParams());
        InetSocketAddress ipv4 = new InetSocketAddress("0.0.0.0", port);
        try {
            UcpListener ipv4Listener = worker.newListener(
                new UcpListenerParams().setSockAddr(ipv4));

            assertNotNull(ipv4Listener);
            ipv4Listener.close();
        } catch (UcxException ignored) { }

        try {
            InetSocketAddress ipv6 = new InetSocketAddress("::", port);
            UcpListener ipv6Listener = worker.newListener(
                new UcpListenerParams().setSockAddr(ipv6));

            assertNotNull(ipv6Listener);
            ipv6Listener.close();
        } catch (UcxException ignored) { }

        worker.close();
        context.close();
    }

    static Stream<NetworkInterface> getInterfaces() {
        try {
            return Collections.list(NetworkInterface.getNetworkInterfaces()).stream()
                .filter(iface -> {
                    try {
                        return iface.isUp() && !iface.isLoopback();
                    } catch (SocketException e) {
                        return false;
                    }
                });
        } catch (SocketException e) {
            return Stream.empty();
        }
    }

    /**
     * Iterates over network interfaces and tries to bind and create listener
     * on a specific socket address.
     */
    static UcpListener tryBindListener(UcpWorker worker, UcpListenerParams params) {
        UcpListener result = null;
        List<InetAddress> addresses = getInterfaces().flatMap(iface ->
            Collections.list(iface.getInetAddresses()).stream())
            .collect(Collectors.toList());
        for (InetAddress address : addresses) {
            try {
                result = worker.newListener(
                    params.setSockAddr(new InetSocketAddress(address, port)));
                break;
            } catch (UcxException ignored) { }
        }
        assertNotNull("Could not find socket address to start UcpListener", result);
        return result;
    }

    @Test
    public void testConnectionHandler() {
        UcpContext context1 = new UcpContext(new UcpParams().requestStreamFeature()
            .requestRmaFeature());
        UcpContext context2 = new UcpContext(new UcpParams().requestStreamFeature()
            .requestRmaFeature());
        UcpWorker serverWorker1 = context1.newWorker(new UcpWorkerParams());
        UcpWorker serverWorker2 = context1.newWorker(new UcpWorkerParams());
        UcpWorker clientWorker = context2.newWorker(new UcpWorkerParams());

        AtomicReference<UcpConnectionRequest> conRequest = new AtomicReference<>(null);

        // Create listener and set connection handler
        UcpListenerParams listenerParams = new UcpListenerParams()
            .setConnectionHandler(conRequest::set);
        UcpListener listener = tryBindListener(serverWorker1, listenerParams);

        UcpEndpoint clientToServer = clientWorker.newEndpoint(new UcpEndpointParams()
            .setSocketAddress(listener.getAddress()));

        while (conRequest.get() == null) {
            serverWorker1.progress();
            clientWorker.progress();
        }

        // Create endpoint from another worker from pool.
        UcpEndpoint serverToClient = serverWorker2.newEndpoint(
            new UcpEndpointParams().setConnectionRequest(conRequest.get()));
        
        // Temporary workaround until new connection establishment protocol in UCX.
        for (int i = 0; i < 10; i++) {
            serverWorker1.progress();
            serverWorker2.progress();
            clientWorker.progress();
            try {
                Thread.sleep(10);
            } catch (Exception ignored) { }
        }

        UcpRequest sent = serverToClient.sendStreamNonBlocking(
            ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE), null);

        // Progress all workers to make sure recv request will complete immediately
        for (int i = 0; i < 10; i++) {
            serverWorker1.progress();
            serverWorker2.progress();
            clientWorker.progress();
            try {
                Thread.sleep(2);
            } catch (Exception ignored) { }
        }

        UcpRequest recv = clientToServer.recvStreamNonBlocking(
            ByteBuffer.allocateDirect(UcpMemoryTest.MEM_SIZE), 0, null);

        while (!sent.isCompleted() || !recv.isCompleted()) {
            serverWorker1.progress();
            clientWorker.progress();
        }

        assertEquals(UcpMemoryTest.MEM_SIZE, recv.getRecvSize());

        Collections.addAll(resources, context2, context1, clientWorker, serverWorker1,
            serverWorker2, listener, serverToClient, clientToServer);
        closeResources();
    }
}
