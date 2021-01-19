/*
 * Copyright (C) Mellanox Technologies Ltd. 2021. ALL RIGHTS RESERVED.
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
public interface UcpAmRecvCallback {

    /**
     * The callback is always called from the progress context, therefore calling
     * {@link UcpWorker#progress()} is not allowed. It is recommended to define
     * callbacks with relatively short execution time to avoid blocking of communication progress.
     * @param headerAddress - User defined active message header. Can be 0.
     * @param headerSize - Active message header length in bytes. If this
     *                     value is 0, the headerAddress is undefined and should not be accessed.
     * @param amData     - Points to {@link UcpAmData} wrapper that has whether received data or
     *                     data descriptor to receive in {@link UcpWorker#recvAmDataNonBlocking}
     * @param replyEp    - Endpoint, which can be used for reply to this message.
     * @return           - {@link org.openucx.jucx.ucs.UcsConstants.STATUS#UCS_OK} -
     *                     data will not persist after the callback returns.
     *                     {@link org.openucx.jucx.ucs.UcsConstants.STATUS#UCS_INPROGRESS} -
     *                     The data will persist after the callback has returned.
     *                     To free the memory, need to call {@link UcpAmData#close()}
     */
    int onReceive(long headerAddress, long headerSize,
                  UcpAmData amData, UcpEndpoint replyEp);
}
