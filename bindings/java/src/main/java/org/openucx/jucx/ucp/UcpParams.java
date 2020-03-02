/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx.ucp;

import org.openucx.jucx.UcxParams;

import java.util.HashMap;
import java.util.Map;

/**
 * Tuning parameters for UCP library.
 * The structure defines the parameters that are used for
 * UCP library tuning during UCP library {@link UcpContext} "initialization".
 *
 * <p>UCP library implementation uses the "features"
 * parameter to optimize the library functionality that minimize memory
 * footprint. For example, if the application does not require send/receive
 * semantics UCP library may avoid allocation of expensive resources associated with
 * send/receive queues.
 */
public class UcpParams extends UcxParams {

    /**
     * UCP ucp_feature "features" that are used for library
     * initialization. It is recommended for applications only to request
     * the features that are required for an optimal functionality
     * This field must be specified.
     */
    private long features;

    private long tagSenderMask;

    private boolean mtWorkersShared;

    private long estimatedNumEps;

    private Map<String, String> config;

    @Override
    public UcpParams clear() {
        super.clear();
        features = 0L;
        tagSenderMask = 0L;
        mtWorkersShared = false;
        estimatedNumEps = 0L;
        config = null;
        return this;
    }

    /**
     * Mask which specifies particular bits of the tag which can uniquely
     * identify the sender (UCP endpoint) in tagged operations.
     * This field defaults to 0 if not specified.
     */
    public UcpParams setTagSenderMask(long tagSenderMask) {
        this.tagSenderMask = tagSenderMask;
        this.fieldMask |= UcpConstants.UCP_PARAM_FIELD_TAG_SENDER_MASK;
        return this;
    }

    /**
     * Indicates if this context is shared by multiple workers
     * from different threads. If so, this context needs thread safety
     * support; otherwise, the context does not need to provide thread
     * safety.
     * For example, if the context is used by single worker, and that
     * worker is shared by multiple threads, this context does not need
     * thread safety; if the context is used by worker 1 and worker 2,
     * and worker 1 is used by thread 1 and worker 2 is used by thread 2,
     * then this context needs thread safety.
     * Note that actual thread mode may be different from mode passed
     * to {@link UcpContext}.
     */
    public UcpParams setMtWorkersShared(boolean mtWorkersShared) {
        this.mtWorkersShared = mtWorkersShared;
        this.fieldMask |= UcpConstants.UCP_PARAM_FIELD_MT_WORKERS_SHARED;
        return this;
    }

    /**
     * An optimization hint of how many endpoints would be created on this context.
     * Does not affect semantics, but only transport selection criteria and the
     * resulting performance.
     * The value can be also set by UCX_NUM_EPS environment variable. In such case
     * it will override the number of endpoints set by {@link #setEstimatedNumEps}.
     */
    public UcpParams setEstimatedNumEps(long estimatedNumEps) {
        this.estimatedNumEps = estimatedNumEps;
        this.fieldMask |= UcpConstants.UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
        return this;
    }

    /**
     * Request tag matching support.
     */
    public UcpParams requestTagFeature() {
        this.fieldMask |= UcpConstants.UCP_PARAM_FIELD_FEATURES;
        this.features |= UcpConstants.UCP_FEATURE_TAG;
        return this;
    }

    /**
     * Request remote memory access support.
     */
    public UcpParams requestRmaFeature() {
        this.fieldMask |= UcpConstants.UCP_PARAM_FIELD_FEATURES;
        this.features |= UcpConstants.UCP_FEATURE_RMA;
        return this;
    }

    /**
     * Request 32-bit atomic operations support.
     */
    public UcpParams requestAtomic32BitFeature() {
        this.fieldMask |= UcpConstants.UCP_PARAM_FIELD_FEATURES;
        this.features |= UcpConstants.UCP_FEATURE_AMO32;
        return this;
    }

    /**
     * Request 64-bit atomic operations support.
     */
    public UcpParams requestAtomic64BitFeature() {
        this.fieldMask |= UcpConstants.UCP_PARAM_FIELD_FEATURES;
        this.features |= UcpConstants.UCP_FEATURE_AMO64;
        return this;
    }

    /**
     * Request interrupt notification support.
     */
    public UcpParams requestWakeupFeature() {
        this.fieldMask |= UcpConstants.UCP_PARAM_FIELD_FEATURES;
        this.features |= UcpConstants.UCP_FEATURE_WAKEUP;
        return this;
    }

    /**
     * Request stream support.
     */
    public UcpParams requestStreamFeature() {
        this.fieldMask |= UcpConstants.UCP_PARAM_FIELD_FEATURES;
        this.features |= UcpConstants.UCP_FEATURE_STREAM;
        return this;
    }

    /**
     * The routine sets runtime UCP library configuration.
     */
    public UcpParams setConfig(String key, String value) {
        if (config == null) {
            config = new HashMap<>();
        }
        config.put(key, value);
        return this;
    }
}
