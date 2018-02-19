/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx.examples.helloworld;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;

public class HelloServer extends HelloWorld {
    private ServerSocket serv;

    public HelloServer() throws IOException {
        serv = new ServerSocket(port);
        serv.setReuseAddress(true);
    }

    @Override
    protected void exchangeWorkerAddress() throws IOException {
        // Print progress message only if not in quiet mode
        conditionalPrint("Waiting for connections...");

        Socket sock = serv.accept();
        serv.close();

        conditionalPrint("Connected to: " +
                         sock.getInetAddress().getHostAddress());

        // Send local Worker address through TCP socket
        sendLocalWorkerAddress(sock);

        // Received remote address through TCP socket
        @SuppressWarnings("unused")
        byte[] remoteWorkerAddress = recvRemoteWorkerAddress(sock);

        sock.close();
    }

    @Override
    protected void usage() {
        System.out.println("Usage: ./scripts/hello_world.sh server [OPTION]...");
        super.usage();
    }

    public static void main(String[] args) {
        try {
            HelloServer server = new HelloServer();
            server.run(args);
            server.conditionalPrint("[SUCCESS] Exiting...");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
