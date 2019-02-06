/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx.ucp;

import org.ucx.jucx.Bridge;
import org.ucx.jucx.UcxNativeStruct;

/**
 * UCP worker is an opaque object representing the communication context.  The
 * worker represents an instance of a local communication resource and progress
 * engine associated with it. Progress engine is a construct that is
 * responsible for asynchronous and independent progress of communication
 * directives. The progress engine could be implement in hardware or software.
 * The worker object abstract an instance of network resources such as a host
 * channel adapter port, network interface, or multiple resources such as
 * multiple network interfaces or communication ports. It could also represent
 * virtual communication resources that are defined across multiple devices.
 * Although the worker can represent multiple network resources, it is
 * associated with a single @ref ucp_context_h "UCX application context".
 * All communication functions require a context to perform the operation on
 * the dedicated hardware resource(s) and an @ref ucp_ep_h "endpoint" to address the
 * destination.
 *
 * <p>Worker are parallel "threading points" that an upper layer may use to
 * optimize concurrent communications.
 */
public class UcpWorker extends UcxNativeStruct {

  public UcpWorker(long nativeId) {
    super(nativeId);
  }

  public void release() {
    Bridge.releaseWorker(this);
    setNativeId(0L);
  }
}
