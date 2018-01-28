package org.ucx.jucx;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.fail;

import java.io.IOException;

import org.junit.Before;
import org.junit.Test;
import org.ucx.jucx.examples.helloworld.HelloClient;
import org.ucx.jucx.examples.helloworld.HelloServer;
import org.ucx.jucx.examples.helloworld.HelloWorld;

public class ExampleTest {
    private HelloServer server;
    private HelloClient client;
    private boolean     failed;

    @Before
    public void setUpHelloWorld() {
        server = new HelloServer();
        client = new HelloClient();
        failed = false;
    }

    @Test
    public void testHelloWorld() {
        Thread serverThread = new ExampleThread(server);
        Thread clientThread = new ExampleThread(client);

        serverThread.start();
        clientThread.start();

        try {
            serverThread.join();
            clientThread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
            fail("Example code execution failed");
        }

        assertFalse("Example code execution failed", failed);
    }


    private class ExampleThread extends Thread {
        private HelloWorld peer;

        private ExampleThread(HelloWorld peer) {
            this.peer = peer;
        }

        @Override
        public void run() {
            String[] args = { "-q" };
            try {
                peer.run(args);
            } catch (IOException e) {
                failed = true;
            }
        }
    }
}
