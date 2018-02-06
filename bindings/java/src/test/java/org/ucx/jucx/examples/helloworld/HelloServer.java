/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx.examples.helloworld;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;

public class HelloServer extends HelloWorld {
    @Override
    protected void exchangeWorkerAddress() throws IOException {
        ServerSocket serv = new ServerSocket(port);
        serv.setReuseAddress(true);

        // Print progress message only if not in quiet mode
        conditionalPrint(!quiet, "Waiting for connections...");

        Socket sock = serv.accept();
        serv.close();

        conditionalPrint(!quiet, "Connected to: " +
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
        System.out.println("Usage: ./runHelloWorld.sh server [OPTION]...");
        super.usage();
    }

    public static void main(String[] args) {
        HelloServer server = new HelloServer();
        try {
            server.run(args);
            server.conditionalPrint(!server.quiet, "[SUCCESS] Exiting...");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
