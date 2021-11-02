/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.openucx.jucx;

import org.junit.Test;
import org.openucx.jucx.ucp.*;
import org.openucx.jucx.ucs.UcsConstants;

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
        Collections.reverse(addresses);
        for (InetAddress address : addresses) {
            result = worker.newListener(params.setSockAddr(new InetSocketAddress(address, 0)));
        }
        assertNotNull("Could not find socket address to start UcpListener", result);
        assertNotEquals(0, result.getAddress().getPort());
        System.out.println("Bound UcpListner on: " + result.getAddress());
        return result;
    }

    @Test
    public void testConnectionHandler() throws Exception {
        long clientId = 3L;
        UcpContext context1 = new UcpContext(new UcpParams().requestStreamFeature()
            .requestRmaFeature());
        UcpContext context2 = new UcpContext(new UcpParams().requestStreamFeature()
            .requestRmaFeature());
        UcpWorker serverWorker1 = context1.newWorker(new UcpWorkerParams());
        UcpWorker serverWorker2 = context1.newWorker(new UcpWorkerParams());
        UcpWorker clientWorker = context2.newWorker(new UcpWorkerParams().setClientId(clientId));

        AtomicReference<UcpConnectionRequest> connRequest = new AtomicReference<>(null);
        AtomicReference<UcpConnectionRequest> connReject = new AtomicReference<>(null);

        // Create listener and set connection handler
        UcpListenerParams listenerParams = new UcpListenerParams()
            .setConnectionHandler((UcpConnectionRequest connectionRequest) -> {
                if (connRequest.get() == null) {
                    connRequest.set(connectionRequest);
                } else {
                    connReject.set(connectionRequest);
                }
            });
        UcpListener serverListener = tryBindListener(serverWorker1, listenerParams);
        UcpListener clientListener = tryBindListener(clientWorker, listenerParams);

        UcpEndpoint clientToServer = clientWorker.newEndpoint(new UcpEndpointParams()
            .sendClientId().setErrorHandler((ep, status, errorMsg) ->
                System.err.println("clientToServer error: " + errorMsg))
            .setPeerErrorHandlingMode().setSocketAddress(serverListener.getAddress()));

        while (connRequest.get() == null) {
            serverWorker1.progress();
            clientWorker.progress();
        }

        assertEquals(clientId, connRequest.get().getClientId());
        assertNotNull(connRequest.get().getClientAddress());
        UcpEndpoint serverToClientListener = serverWorker2.newEndpoint(
            new UcpEndpointParams().setSocketAddress(connRequest.get().getClientAddress())
                                   .setPeerErrorHandlingMode()
                                   .setErrorHandler((errEp, status, errorMsg) ->
                                       System.err.println("serverToClientListener error: " +
                                           errorMsg)));
        serverWorker2.progressRequest(serverToClientListener.closeNonBlockingForce());

        // Create endpoint from another worker from pool.
        UcpEndpoint serverToClient = serverWorker2.newEndpoint(
            new UcpEndpointParams().setConnectionRequest(connRequest.get()));

        // Test connection handler persists
        for (int i = 0; i < 10; i++) {
            clientWorker.newEndpoint(new UcpEndpointParams()
                .setSocketAddress(serverListener.getAddress()).setPeerErrorHandlingMode()
                .setErrorHandler((ep, status, errorMsg) -> {
                    ep.close();
                    assertEquals(UcsConstants.STATUS.UCS_ERR_REJECTED, status);
                }));

            while (connReject.get() == null) {
                serverWorker1.progress();
                serverWorker2.progress();
                clientWorker.progress();
            }

            connReject.get().reject();
            connReject.set(null);

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
            serverWorker2.progress();
            clientWorker.progress();
        }

        assertEquals(UcpMemoryTest.MEM_SIZE, recv.getRecvSize());

        UcpRequest serverClose = serverToClient.closeNonBlockingFlush();
        UcpRequest clientClose = clientToServer.closeNonBlockingFlush();

        while (!serverClose.isCompleted() || !clientClose.isCompleted()) {
            serverWorker2.progress();
            clientWorker.progress();
        }

        Collections.addAll(resources, context2, context1, clientWorker, serverWorker1,
            serverWorker2, serverListener, clientListener);
        closeResources();
    }
}
