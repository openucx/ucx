/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucx

// #include <arpa/inet.h>
// #include <netinet/in.h>
// #include <stdlib.h>
// #include <string.h>
// #include <sys/socket.h>
// #include <ucp/api/ucp.h>
import "C"
import (
	"net"
	"runtime"
	"syscall"
	"unsafe"
)

func AllocateNativeMemory(size uint64) unsafe.Pointer {
	return C.malloc(C.ulong(size))
}

func FreeNativeMemory(pointer unsafe.Pointer) {
	C.free(pointer)
}

// Helper method to check whether name of an ucp object is set.
// if so, free previously set memory and finalizer
func freeParamsName(p interface{}) {
	var cstr *C.char
	switch p := p.(type) {
	case *UcpParams:
		cstr = p.params.name
	case *UcpWorkerParams:
		cstr = p.params.name
	case *UcpEpParams:
		cstr = p.params.name
	}

	if cstr != nil {
		runtime.SetFinalizer(p, nil)
		FreeNativeMemory(unsafe.Pointer(cstr))
	}
}

// Helper method to check whether sockaddr is set
// if so, free previously set memory and finalizer
func freeParamsAddress(p interface{}) {
	var caddr *C.struct_sockaddr
	switch p := p.(type) {
	case *UcpListenerParams:
		caddr = p.params.sockaddr.addr
	case *UcpEpParams:
		caddr = p.params.sockaddr.addr
	}

	if caddr != nil {
		runtime.SetFinalizer(p, nil)
		FreeNativeMemory(unsafe.Pointer(caddr))
	}
}

func CBytes(data []byte) unsafe.Pointer {
	return C.CBytes(data)
}

func GoBytes(p unsafe.Pointer, n uint64) []byte {
	return C.GoBytes(p, C.int(n))
}

func family(a *net.TCPAddr) int {
	if a == nil || len(a.IP) <= net.IPv4len {
		return syscall.AF_INET
	}
	if a.IP.To4() != nil {
		return syscall.AF_INET
	}
	return syscall.AF_INET6
}

// Convert GO TCP address to C
func toSockAddr(a *net.TCPAddr) (*C.ucs_sock_addr_t, error) {
	// We can't assing to ucs_sock_addr_t->addr reference to Go's memory,
	// so need to allocate
	sockaddrPtr := AllocateNativeMemory(C.sizeof_struct_sockaddr_storage)
	sa := (*C.struct_sockaddr_storage)(sockaddrPtr)
	var result C.ucs_sock_addr_t
	if a == nil {
		a = &net.TCPAddr{}
	}
	switch family(a) {
	case syscall.AF_INET:
		if len(a.IP) == 0 {
			a.IP = net.IPv4zero
		}
		ip4 := a.IP.To4()
		if ip4 == nil {
			return nil, &net.AddrError{Err: "non-IPv4 address", Addr: a.IP.String()}
		}
		sa.ss_family = C.AF_INET
		var sin *C.struct_sockaddr_in = (*C.struct_sockaddr_in)(unsafe.Pointer(sa))
		sin.sin_port = C.htons(C.ushort(a.Port))
		charBytes := C.CString(ip4.String())
		C.inet_pton(C.AF_INET, charBytes, unsafe.Pointer(&sin.sin_addr.s_addr))
		FreeNativeMemory(unsafe.Pointer(charBytes))
	case syscall.AF_INET6:
		// In general, an IP wildcard address, which is either
		// "0.0.0.0" or "::", means the entire IP addressing
		// space. For some historical reason, it is used to
		// specify "any available address" on some operations
		// of IP node.
		//
		// When the IP node supports IPv4-mapped IPv6 address,
		// we allow a listener to listen to the wildcard
		// address of both IP addressing spaces by specifying
		// IPv6 wildcard address.
		if len(a.IP) == 0 || a.IP.Equal(net.IPv4zero) {
			a.IP = net.IPv6zero
		}
		// We accept any IPv6 address including IPv4-mapped
		// IPv6 address.
		ip6 := a.IP.To16()
		if ip6 == nil {
			return nil, &net.AddrError{Err: "non-IPv6 address", Addr: a.IP.String()}
		}
		sa.ss_family = C.AF_INET6
		var sin6 *C.struct_sockaddr_in6 = (*C.struct_sockaddr_in6)(unsafe.Pointer(sa))
		sin6.sin6_port = C.htons(C.ushort(a.Port))
		charBytes := C.CString(ip6.String())
		C.inet_pton(C.AF_INET6, charBytes, unsafe.Pointer(&sin6.sin6_addr))
		FreeNativeMemory(unsafe.Pointer(charBytes))
	default:
		return nil, &net.AddrError{Err: "invalid address family", Addr: a.IP.String()}
	}
	result.addrlen = C.socklen_t(len(a.IP))
	result.addr = (*C.struct_sockaddr)(unsafe.Pointer(sa))
	return &result, nil
}

// Convert sockaddr_storage to go TCPAddr
func toTcpAddr(sockaddr *C.struct_sockaddr_storage) *net.TCPAddr {
	result := &net.TCPAddr{}
	if sockaddr.ss_family == C.AF_INET6 {
		var sin6 *C.struct_sockaddr_in6 = (*C.struct_sockaddr_in6)(unsafe.Pointer(sockaddr))
		result.Port = int(C.ntohs(sin6.sin6_port))
		C.memcpy(unsafe.Pointer(&result.IP), unsafe.Pointer(&sin6.sin6_addr), 16)
	} else {
		var sin *C.struct_sockaddr_in = (*C.struct_sockaddr_in)(unsafe.Pointer(sockaddr))
		result.Port = int(C.ntohs(sin.sin_port))
		C.memcpy(unsafe.Pointer(&result.IP), unsafe.Pointer(&sin.sin_addr), 4)
	}

	return result
}
