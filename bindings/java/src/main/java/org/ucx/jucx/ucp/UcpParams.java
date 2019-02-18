/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx.ucp;

import static org.ucx.jucx.ucs.UcsBits.UCS_BIT;

/**
 * Tuning parameters for UCP library.
 *
 * The structure defines the parameters that are used for
 * UCP library tuning during UCP library {@link UcpContext} "initialization".
 *
 * <p>UCP library implementation uses the {@link UcpParams.UcpFeature} "features"
 * parameter to optimize the library functionality that minimize memory
 * footprint. For example, if the application does not require send/receive
 * semantics UCP library may avoid allocation of expensive resources associated with
 * send/receive queues.
 */
public class UcpParams {

    @Override
    public String toString() {
        return "UcpParams{"
            + "fieldMask=" + fieldMask
            + ", features=" + features
            + ", tagSenderMask=" + tagSenderMask
            + ", mtWorkersShared=" + mtWorkersShared
            + ", estimatedNumEps=" + estimatedNumEps
            + '}';
    }

    /**
     * UCP context parameters field mask.
     *
     * <p>The enumeration allows specifying which fields in {@link UcpParams} are
     * present. It is used for the enablement of backward compatibility support.
     */
    private enum UcpParamField {
        UCP_PARAM_FIELD_FEATURES(UCS_BIT(0L)),
        // UCP_PARAM_FIELD_REQUEST_SIZE(UCS_BIT(1L)),
        // UCP_PARAM_FIELD_REQUEST_INIT(UCS_BIT(2L)),
        // UCP_PARAM_FIELD_REQUEST_CLEANUP(UCS_BIT(3L)),
        UCP_PARAM_FIELD_TAG_SENDER_MASK(UCS_BIT(4L)),
        UCP_PARAM_FIELD_MT_WORKERS_SHARED(UCS_BIT(5L)),
        UCP_PARAM_FIELD_ESTIMATED_NUM_EPS(UCS_BIT(6L));

        private long value;

        UcpParamField(long value) {
            this.value = value;
        }
    }

    /**
     * UCP configuration features
     *
     * <p>The enumeration list describes the features supported by UCP.
     * An application can request the features using "UCP parameters"
     * during "UCP initialization" process.
     */
    private enum UcpFeature {
        UCP_FEATURE_TAG(UCS_BIT(0L)),
        UCP_FEATURE_RMA(UCS_BIT(1L)),
        UCP_FEATURE_AMO32(UCS_BIT(2L)),
        UCP_FEATURE_AMO64(UCS_BIT(3L)),
        UCP_FEATURE_WAKEUP(UCS_BIT(4L)),
        UCP_FEATURE_STREAM(UCS_BIT(5L));

        private long value;

        UcpFeature(long value) {
            this.value = value;
        }
    }

    /**
     * Mask of valid fields in this structure, using bits from {@link UcpParamField}
     * Fields not specified in this mask would be ignored.
     * Provides ABI compatibility with respect to adding new fields.
     */
    private long fieldMask;

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

    /**
     * Mask which specifies particular bits of the tag which can uniquely
     * identify the sender (UCP endpoint) in tagged operations.
     * This field defaults to 0 if not specified.
     */
    public UcpParams setTagSenderMask(long tagSenderMask) {
        this.tagSenderMask = tagSenderMask;
        this.fieldMask |= UcpParamField.UCP_PARAM_FIELD_TAG_SENDER_MASK.value;
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
        this.fieldMask |= UcpParamField.UCP_PARAM_FIELD_MT_WORKERS_SHARED.value;
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
        this.fieldMask |= UcpParamField.UCP_PARAM_FIELD_ESTIMATED_NUM_EPS.value;
        return this;
    }

    /**
     * Request tag matching support.
     */
    public UcpParams requestTagFeature() {
        this.fieldMask |= UcpParamField.UCP_PARAM_FIELD_FEATURES.value;
        this.features |= UcpFeature.UCP_FEATURE_TAG.value;
        return this;
    }

    /**
     * Request remote memory access support.
     */
    public UcpParams requestRmaFeature() {
        this.fieldMask |= UcpParamField.UCP_PARAM_FIELD_FEATURES.value;
        this.features |= UcpFeature.UCP_FEATURE_RMA.value;
        return this;
    }

    /**
     * Request 32-bit atomic operations support.
     */
    public UcpParams requestAtomic32BitFeature() {
        this.fieldMask |= UcpParamField.UCP_PARAM_FIELD_FEATURES.value;
        this.features |= UcpFeature.UCP_FEATURE_AMO32.value;
        return this;
    }

    /**
     * Request 64-bit atomic operations support.
     */
    public UcpParams requestAtomic64BitFeature() {
        this.fieldMask |= UcpParamField.UCP_PARAM_FIELD_FEATURES.value;
        this.features |= UcpFeature.UCP_FEATURE_AMO64.value;
        return this;
    }

    /**
     * Request interrupt notification support.
     */
    public UcpParams requestWakeupFeature() {
        this.fieldMask |= UcpParamField.UCP_PARAM_FIELD_FEATURES.value;
        this.features |= UcpFeature.UCP_FEATURE_WAKEUP.value;
        return this;
    }

    /**
     * Request stream support.
     */
    public UcpParams requestStreamFeature() {
        this.fieldMask |= UcpParamField.UCP_PARAM_FIELD_FEATURES.value;
        this.features |= UcpFeature.UCP_FEATURE_STREAM.value;
        return this;
    }
}

