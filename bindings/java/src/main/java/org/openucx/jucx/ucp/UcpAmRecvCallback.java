/*
 * Copyright (C) Mellanox Technologies Ltd. 2020. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.openucx.jucx.ucp;

/**
 * Callback to process incoming Active Message sent by {@link UcpEndpoint#sendAmNonBlocking }
 * routine.
 *
 * The callback is always called from the progress context, therefore calling
 * {@link UcpWorker#progress()} is not allowed. It is recommended to define
 * callbacks with relatively short execution time to avoid blocking of
 * communication progress.
 */
public abstract class UcpAmRecvCallback {
    UcpWorker worker;

    void setWorker(UcpWorker worker) {
        this.worker = worker;
    }

    public abstract int onReceive(long headerAddress, long headerSize,
                                  UcpAmData amData, UcpEndpoint replyEp);
}
