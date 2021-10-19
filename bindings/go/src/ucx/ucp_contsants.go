/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucx

// #include <ucp/api/ucp.h>
import "C"

type UcpProtection uint32

const (
	UCP_MEM_MAP_PROT_LOCAL_READ   UcpProtection = C.UCP_MEM_MAP_PROT_LOCAL_READ
	UCP_MEM_MAP_PROT_LOCAL_WRITE  UcpProtection = C.UCP_MEM_MAP_PROT_LOCAL_WRITE
	UCP_MEM_MAP_PROT_REMOTE_READ  UcpProtection = C.UCP_MEM_MAP_PROT_REMOTE_READ
	UCP_MEM_MAP_PROT_REMOTE_WRITE UcpProtection = C.UCP_MEM_MAP_PROT_REMOTE_WRITE
)

type UcpContextAttr uint32

const (
	UCP_ATTR_FIELD_REQUEST_SIZE UcpContextAttr = C.UCP_ATTR_FIELD_REQUEST_SIZE
	UCP_ATTR_FIELD_THREAD_MODE  UcpContextAttr = C.UCP_ATTR_FIELD_THREAD_MODE
	UCP_ATTR_FIELD_MEMORY_TYPES UcpContextAttr = C.UCP_ATTR_FIELD_MEMORY_TYPES
	UCP_ATTR_FIELD_NAME         UcpContextAttr = C.UCP_ATTR_FIELD_NAME
)

type UcpMemAttribute uint32

const (
	UCP_MEM_ATTR_FIELD_ADDRESS  UcpMemAttribute = C.UCP_MEM_ATTR_FIELD_ADDRESS
	UCP_MEM_ATTR_FIELD_LENGTH   UcpMemAttribute = C.UCP_MEM_ATTR_FIELD_LENGTH
	UCP_MEM_ATTR_FIELD_MEM_TYPE UcpMemAttribute = C.UCP_MEM_ATTR_FIELD_MEM_TYPE
)

type UcpWakeupEvent uint32

const (
	UCP_WAKEUP_RMA      UcpWakeupEvent = C.UCP_WAKEUP_RMA
	UCP_WAKEUP_AMO      UcpWakeupEvent = C.UCP_WAKEUP_AMO
	UCP_WAKEUP_TAG_SEND UcpWakeupEvent = C.UCP_WAKEUP_TAG_SEND
	UCP_WAKEUP_TAG_RECV UcpWakeupEvent = C.UCP_WAKEUP_TAG_RECV
	UCP_WAKEUP_TX       UcpWakeupEvent = C.UCP_WAKEUP_TX
	UCP_WAKEUP_RX       UcpWakeupEvent = C.UCP_WAKEUP_RX
	UCP_WAKEUP_EDGE     UcpWakeupEvent = C.UCP_WAKEUP_EDGE
)

type UcpWorkerAttribute uint32

const (
	UCP_WORKER_ATTR_FIELD_THREAD_MODE     UcpWorkerAttribute = C.UCP_WORKER_ATTR_FIELD_THREAD_MODE
	UCP_WORKER_ATTR_FIELD_ADDRESS         UcpWorkerAttribute = C.UCP_WORKER_ATTR_FIELD_ADDRESS
	UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS   UcpWorkerAttribute = C.UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS
	UCP_WORKER_ATTR_FIELD_MAX_AM_HEADER   UcpWorkerAttribute = C.UCP_WORKER_ATTR_FIELD_MAX_AM_HEADER
	UCP_WORKER_ATTR_FIELD_NAME            UcpWorkerAttribute = C.UCP_WORKER_ATTR_FIELD_NAME
	UCP_WORKER_ATTR_FIELD_MAX_INFO_STRING UcpWorkerAttribute = C.UCP_WORKER_ATTR_FIELD_MAX_INFO_STRING
)

type UcpListenerAttribute uint32

const (
	UCP_LISTENER_ATTR_FIELD_SOCKADDR UcpListenerAttribute = C.UCP_LISTENER_ATTR_FIELD_SOCKADDR
)

type UcpAmSendFlags uint64

const (
	// Force relevant reply endpoint to be passed to the data callback on the receiver.
	UCP_AM_SEND_FLAG_REPLY UcpAmSendFlags = C.UCP_AM_SEND_FLAG_REPLY
	// Force UCP to use only eager protocol for AM sends.
	UCP_AM_SEND_FLAG_EAGER UcpAmSendFlags = C.UCP_AM_SEND_FLAG_EAGER
	// Force UCP to use only rendezvous protocol for AM sends.
	UCP_AM_SEND_FLAG_RNDV UcpAmSendFlags = C.UCP_AM_SEND_FLAG_RNDV
)

type UcpAmRecvAttrs uint64

const (
	UCP_AM_RECV_ATTR_FIELD_REPLY_EP UcpAmRecvAttrs = C.UCP_AM_RECV_ATTR_FIELD_REPLY_EP
	UCP_AM_RECV_ATTR_FLAG_DATA      UcpAmRecvAttrs = C.UCP_AM_RECV_ATTR_FLAG_DATA
	UCP_AM_RECV_ATTR_FLAG_RNDV      UcpAmRecvAttrs = C.UCP_AM_RECV_ATTR_FLAG_RNDV
)

type UcpAmCbFlags uint64

const (
	// Indicates that the entire message will be handled in one callback.
	UCP_AM_FLAG_WHOLE_MSG UcpAmCbFlags = C.UCP_AM_FLAG_WHOLE_MSG

	// Guarantees that the specified callback, will always be called
	// with UCP_AM_RECV_ATTR_FLAG_DATA flag set,so the data will be accessible outside the callback,
	// until UcpAmData.Close() is called.
	UCP_AM_FLAG_PERSISTENT_DATA UcpAmCbFlags = C.UCP_AM_FLAG_PERSISTENT_DATA
)

type UcpConnRequestAttribute uint32

const (
	UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR = C.UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR
	UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ID   = C.UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ID
)
