/*
 * Copyright (C) Mellanox Technologies Ltd. 2020. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx.ucp;

import org.openucx.jucx.UcxCallback;
import org.openucx.jucx.UcxNativeStruct;

/**
 * UCP Message descriptor is an opaque handle for a message returned by
 * {@link UcpWorker#tagProbeNonBlocking(long, long, boolean)}.
 * This handle can be passed to
 * {@link UcpWorker#recvTaggedMessageNonBlocking(long, long, UcpTagMessage, UcxCallback)}
 * in order to receive the message data to a specific buffer.
 */
public class UcpTagMessage extends UcxNativeStruct {
    private long recvLength;

    private long senderTag;

    private UcpTagMessage(long nativeId, long recvLength, long senderTag) {
        if (nativeId != 0) {
            setNativeId(nativeId);
        }
        this.recvLength = recvLength;
        this.senderTag = senderTag;
    }

    public long getRecvLength() {
        return recvLength;
    }

    public long getSenderTag() {
        return senderTag;
    }
}
