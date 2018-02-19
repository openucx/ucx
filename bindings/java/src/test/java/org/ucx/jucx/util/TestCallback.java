package org.ucx.jucx.util;

import org.ucx.jucx.Worker.Callback;
import org.ucx.jucx.Worker.CompletionStatus;

public class TestCallback implements Callback {
    @Override
    public void sendCompletionHandler(long requestId,
                                      CompletionStatus status) {}

    @Override
    public void recvCompletionHandler(long requestId,
                                      CompletionStatus status,
                                      int length) {}
}
